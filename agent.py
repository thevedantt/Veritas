"""Self-correcting agentic RAG: retrieve + rerank -> generate (or NOT_FOUND -> rewrite + retry) -> grounding check.

CLI: python agent.py "your question"
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph import END, StateGraph
from qdrant_client import models

from cache import EMBED_TTL, cache

load_dotenv()
# retry rounds; each round searches several rewritten queries in parallel, so 2 rounds cover what 3+ sequential
# single-query retries did, at about half the wall time
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", 2))
N_VARIANTS = 3
# retrieve K_FETCH candidates by hybrid search, keep the K_KEEP best after reranking (fewer, better passages
# also make generation faster and cheaper); the baseline keeps plain hybrid top-K_BASELINE
K_FETCH, K_KEEP, K_BASELINE = 10, 5, 6
RERANKER = os.getenv("RERANKER", "jinaai/jina-reranker-v1-turbo-en")  # "off" disables; 8K context for long tables
NOT_FOUND_MARK = "NOT_FOUND"
NOT_FOUND = "I couldn't find this in the documents."
MISSING_MARK = "MISSING:"  # last line of a partial answer: a search query for the part the passages lack
K_EXTRA = 2  # passages added per targeted search for a missing part (on top of the ones already kept)
# Gemini 3.x on the global endpoint has slow outliers (20-50s for a one-word reply); cap each call and
# retry, since a retried call is usually fast. The client's own retries skip read timeouts.
LLM_TIMEOUT, LLM_ATTEMPTS = 45, 3
# the generator gets one short try; if it stalls, FALLBACK_MODEL answers instead of waiting out retries
PRIMARY_TIMEOUT = int(os.getenv("PRIMARY_TIMEOUT", 15))
PRIMARY_COOLDOWN = int(os.getenv("PRIMARY_COOLDOWN", 30))  # seconds to skip a failed primary; longer = fewer
# timeouts during an outage, but more answers from the weaker fallback after a one-off stall
_primary_down_until = [0.0]  # shared by every graph in the process
# USD per 1M tokens (input, output incl. reasoning), Vertex AI global endpoint, list prices checked 2026-10-14:
# https://cloud.google.com/vertex-ai/generative-ai/pricing (3.8 Flash is an introductory rate until 2026-12-31).
# Models not listed report tokens without a cost.
# Hard monthly spend cap across chat, scanned-page reading and evals, tracked in SPEND_FILE from token usage.
# Models missing from PRICES are charged at UNKNOWN_PRICE, a deliberately high guess, so the cap never undercounts.
MONTHLY_BUDGET_USD = float(os.getenv("MONTHLY_BUDGET_USD", 5))
SPEND_FILE = Path(os.getenv("SPEND_FILE", "spend.json"))
UNKNOWN_PRICE = (2.50, 15.00)
# Embeddings are billed per character. Conservative upper bound (about 4x the Vertex list price as of 2026-10),
# so the cap overcounts rather than undercounts.
EMBED_PRICE_PER_1M_CHARS = float(os.getenv("EMBED_PRICE_PER_1M_CHARS", 0.10))
# A question is refused unless this much budget is left, so the last one can't push spend past the cap.
QUESTION_RESERVE_USD = 0.01
PRICES = {"gemini-3.8-flash": (0.75, 3.75), "gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-flash-lite": (0.10, 0.40),
          "gemini-2.5-pro": (1.25, 10.00)}
COLLECTION, QDRANT_PATH = "docs", "qdrant_data"
# ponytail: one lock around every vector-store call (embedded Qdrant is not thread-safe);
# switch to a Qdrant server for concurrent reads/writes
DB_LOCK = threading.Lock()

REWRITE = """The search query below did not retrieve enough relevant passages from a document collection.
Original question: {q}
Previous queries: {prev}

Write 3 different search queries that could find the answer: expand acronyms, add likely synonyms, table or
column names, or the specific terms a document would use. One query per line, no numbering, nothing else."""

ANSWER = """Answer the question using only the context. After each claim cite its source using the label in parentheses above the passage, e.g. (report.pdf p.3).
If the context contains part of the answer, give that part and say what is missing, then end with one line
"MISSING: <short search query for the missing part>". If it contains nothing that answers the question,
reply with exactly NOT_FOUND and nothing else. Write plain text: no LaTeX or math markup.
For a table lookup, use the table and column group the question names, find the row whose range contains
the value (at least <= value < but less than), and give every value from that row the answer needs (e.g. a
base amount plus a percentage of the excess over a threshold).

Context:
{ctx}

Question: {q}"""

GROUNDED = """Context:
{ctx}

Answer:
{a}

Is every factual claim in the answer supported by the context? An answer that says the context is
insufficient counts as supported. Reply "yes" or "no" only."""


CONDENSE = """Conversation so far:
{h}

Follow-up question: {q}

Rewrite the follow-up as one standalone question that keeps every detail it needs from the conversation
(names, numbers, products, documents). If it already stands alone, return it unchanged. Reply with the
question only."""
# words that make a question depend on earlier turns ("what about 250?", "and its price?")
FOLLOW_UP = re.compile(r"\b(it|its|they|them|their|this|that|these|those|he|she|his|her|same|above|previous|"
                       r"what about|how about|and for|also|instead|else|more)\b", re.I)


def needs_context(question: str) -> bool:
    return bool(FOLLOW_UP.search(question)) or len(question.split()) <= 4


def standalone(question: str, history: list[dict]) -> str:
    """Turn a follow-up into a self-contained question using the last few turns (one cheap critic call, only
    when the question looks like it depends on them)."""
    if not history or not needs_context(question):
        return question
    h = "\n".join(f"Q: {t['q'][:300]}\nA: {t['a'][:600]}" for t in history[-3:])
    model = os.getenv("CRITIC_MODEL", "gemini-2.5-flash-lite")
    msg = llm(model, 0).with_retry(stop_after_attempt=LLM_ATTEMPTS).invoke(CONDENSE.format(h=h, q=question))
    charge(cost_of({model: msg.usage_metadata or {}}))
    out = text(msg).strip().strip('"')
    return out if 0 < len(out) <= 2000 else question


class State(TypedDict, total=False):
    question: str
    query: str
    docs: list[Document]
    kept: list[Document]  # best docs found so far, carried across retries and re-ranked with new results
    found: int
    variants: list[str]  # rewritten queries searched in parallel on the next retrieve
    insufficient: bool
    missing: str | None  # search query for the part of the answer the passages lacked
    answer: str
    attempts: int
    grounded: bool
    queries: list[str]


def llm(model: str | None = None, thinking_budget: int | None = None, timeout: int = LLM_TIMEOUT,
        retries: int = 2):
    model = model or os.getenv("LLM_MODEL", "gemini-2.5-flash")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if model.startswith("claude"):
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex
        return ChatAnthropicVertex(model_name=model, project=project, location=os.getenv("CLAUDE_LOCATION", "us-east5"),
                                   temperature=0, timeout=timeout, max_retries=retries)
    from langchain_google_genai import ChatGoogleGenerativeAI  # replaces deprecated ChatVertexAI (HTTP, no gRPC)
    extra = {} if thinking_budget is None else {"thinking_budget": thinking_budget}
    return ChatGoogleGenerativeAI(model=model, vertexai=True, project=project,
                                  location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
                                  temperature=0, timeout=timeout, max_retries=retries, **extra)


def with_backoff(fn, *args, attempts: int = 4):
    """Retry rate-limit (429) and transient errors with exponential backoff: 2s, 4s, 8s."""
    for i in range(attempts):
        try:
            return fn(*args)
        except Exception as e:
            if i == attempts - 1 or not any(m in str(e) for m in ("429", "RESOURCE_EXHAUSTED", "503", "timed out")):
                raise
            time.sleep(2 ** (i + 1))


class QueryCachedEmbeddings(Embeddings):
    """Caches query embeddings (every search and rewrite embeds one); documents pass straight through
    because uploads are already de-duplicated by content hash. Both retry on rate limits."""

    def __init__(self, inner: Embeddings, name: str):
        self.inner, self.name = inner, name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = with_backoff(self.inner.embed_documents, texts)
        charge(sum(map(len, texts)) * EMBED_PRICE_PER_1M_CHARS / 1e6)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        key = f"emb:{self.name}:{hashlib.sha256(text.encode()).hexdigest()}"
        if (vector := cache.get(key)) is None:
            vector = with_backoff(self.inner.embed_query, text)
            charge(len(text) * EMBED_PRICE_PER_1M_CHARS / 1e6)
            cache.set(key, vector, EMBED_TTL)
        return vector


def embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # replaces deprecated VertexAIEmbeddings; same vectors
    name = os.getenv("EMBED_MODEL", "text-embedding-005")
    return QueryCachedEmbeddings(GoogleGenerativeAIEmbeddings(model=name, vertexai=True,
                                                              project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                                                              location=os.getenv("EMBED_LOCATION", "us-central1")), name)


def sparse_embeddings():
    from langchain_qdrant import FastEmbedSparse
    return FastEmbedSparse(model_name="Qdrant/bm25")


def store():
    """Open the hybrid collection, creating it if missing."""
    from langchain_qdrant import QdrantVectorStore, RetrievalMode
    return QdrantVectorStore.construct_instance(
        embedding=embeddings(), sparse_embedding=sparse_embeddings(), retrieval_mode=RetrievalMode.HYBRID,
        collection_name=COLLECTION,
        # QDRANT_URL (a Qdrant server) lets several app processes share one index; default is the embedded folder
        client_options={"url": os.environ["QDRANT_URL"]} if os.getenv("QDRANT_URL") else {"path": QDRANT_PATH},
        validate_collection_config=False)  # skips a billed "dummy" embedding call on every start


def label(d: Document) -> str:
    page = d.metadata.get("page")
    return f"{d.metadata.get('source')} p.{page}" if page else str(d.metadata.get("source"))


def fmt(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] ({label(d)})\n{d.page_content}" for i, d in enumerate(docs))


_executor = ThreadPoolExecutor(8)       # query embeddings
_search_pool = ThreadPoolExecutor(4)    # parallel searches; separate pool so nested embedding jobs can't deadlock
_reranker, _reranker_lock = None, threading.Lock()


def hybrid_search(vs, query: str, k: int) -> list[Document]:
    """Dense + BM25 search fused with RRF. Both query embeddings are computed in parallel and outside
    DB_LOCK, so one user's embedding call never makes another user's search wait."""
    dense = _executor.submit(vs.embeddings.embed_query, query)
    sparse = _executor.submit(vs.sparse_embeddings.embed_query, query)
    dense, sparse = dense.result(), sparse.result()
    with DB_LOCK:
        points = vs.client.query_points(COLLECTION, prefetch=[
            models.Prefetch(query=dense, using=vs.vector_name, limit=k),
            models.Prefetch(query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                            using=vs.sparse_vector_name, limit=k),
        ], query=models.FusionQuery(fusion=models.Fusion.RRF), limit=k, with_payload=True).points
    return [Document(page_content=p.payload["page_content"], metadata=p.payload.get("metadata") or {}) for p in points]


def rerank(question: str, docs: list[Document], keep: int) -> list[Document]:
    """Order passages with a local cross-encoder (~0.1-0.5s, no API call). It ranks well but its raw
    scores can't tell "relevant" from "not" reliably, so the generator decides sufficiency (NOT_FOUND)."""
    global _reranker
    if RERANKER == "off" or len(docs) <= 1:
        return docs[:keep]
    with _reranker_lock:
        if _reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _reranker = TextCrossEncoder(RERANKER)
    scores = list(_reranker.rerank(question, [d.page_content for d in docs]))
    # Fuse with the incoming (hybrid search) order: the small cross-encoder misjudges vague paraphrases
    # ("where each word sits" vs "positional encoding") and alone dropped hybrid's top hit
    by_score = sorted(range(len(docs)), key=lambda i: -scores[i])
    fused = {i: 1 / (60 + i) for i in range(len(docs))}
    for rank, i in enumerate(by_score):
        fused[i] += 1 / (60 + rank)
    return [docs[i] for i in sorted(fused, key=lambda i: -fused[i])[:keep]]


def could_be_marker(pending: str) -> bool:
    """True while the streamed answer so far could still turn out to be the NOT_FOUND marker."""
    return NOT_FOUND_MARK.startswith(pending.lstrip()[:len(NOT_FOUND_MARK)])


def split_missing(answer: str) -> tuple[str, str | None]:
    """Separate a partial answer's trailing "MISSING: <query>" line from the answer text."""
    head, _, last = answer.rstrip().rpartition("\n")
    if last.strip().startswith(MISSING_MARK):
        return head.rstrip(), last.strip()[len(MISSING_MARK):].strip() or None
    return answer, None


def held_tail(text: str) -> int:
    """Index from which streamed text is held back: a final line that is or could become the MISSING line."""
    start = text.rfind("\n") + 1
    line = text[start:].lstrip()
    return start - 1 if start and (line.startswith(MISSING_MARK) or MISSING_MARK.startswith(line)) else len(text)


def after_generate(s: State, max_attempts: int) -> str:
    if s.get("insufficient"):
        return "rewrite" if s["attempts"] < max_attempts else END  # "not found" makes no claims to check
    if s.get("missing") and s["attempts"] < max_attempts:
        return "rewrite"  # partial answer: search for the missing part, keep what was found
    return "check"


def after_check(s: State, max_attempts: int) -> str:
    return END if s["grounded"] or s["attempts"] >= max_attempts else "rewrite"


def text(msg) -> str:
    c = msg.content
    return c if isinstance(c, str) else "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)


def build(max_attempts: int = MAX_ATTEMPTS, vs=None):
    # generator answers; a different critic model rewrites and checks so no model judges itself
    # critic calls are short yes/no-style tasks: thinking off by default (thinking tokens bill as output)
    # generator thinking off by default too: reasoning tokens delay the first word by seconds
    think = int(os.getenv("GENERATOR_THINKING_BUDGET", 0))
    model = llm(None, think).with_retry(stop_after_attempt=LLM_ATTEMPTS)
    fallback = os.getenv("FALLBACK_MODEL")
    primary = llm(None, think, PRIMARY_TIMEOUT, retries=0) if fallback else None
    backup = llm(fallback, think).with_retry(stop_after_attempt=LLM_ATTEMPTS) if fallback else None

    def write(prompt):
        """Generator call. With a fallback: one short try on the primary, unless it failed in the last
        PRIMARY_COOLDOWN seconds (then the fallback answers straight away, so an outage costs one timeout)."""
        if not fallback:
            return text(model.invoke(prompt)).strip()
        if time.monotonic() >= _primary_down_until[0]:
            try:
                return text(primary.invoke(prompt)).strip()
            except Exception as e:
                # ponytail: a stall mid-answer restarts on the fallback, so a streamed draft can repeat its
                # opening; the final answer is clean
                print(f"primary model failed ({type(e).__name__}); fallback for {PRIMARY_COOLDOWN}s", file=sys.stderr)
                _primary_down_until[0] = time.monotonic() + PRIMARY_COOLDOWN
        return text(backup.invoke(prompt)).strip()

    critic = llm(os.getenv("CRITIC_MODEL", "gemini-2.5-flash-lite"),
                 int(os.getenv("CRITIC_THINKING_BUDGET", 0))).with_retry(stop_after_attempt=LLM_ATTEMPTS)
    vs = vs or store()
    ask = lambda prompt, m=critic: text(m.invoke(prompt)).strip()

    def retrieve(s):
        if max_attempts == 0:  # baseline: plain hybrid search, no reranking
            return {"docs": hybrid_search(vs, s["query"], K_BASELINE)}
        queries = s.get("variants") or [s["query"]]
        batches = _search_pool.map(lambda q: hybrid_search(vs, q, K_FETCH), queries)
        # interleave so the list order is a rank across queries (rerank fuses with it); dedupe keeps first position
        found = [d for row in zip_longest(*batches) for d in row if d]
        found = list({d.page_content: d for d in found}.values())
        kept = s.get("kept", [])
        seen = {d.page_content for d in kept}
        new = [d for d in found if d.page_content not in seen]
        if s.get("missing"):  # targeted search: add its best hits to what was kept, don't let rerank drop them
            docs = kept + new[:K_EXTRA]
        else:
            docs = rerank(s["question"], kept + new, K_KEEP)
        return {"docs": docs, "kept": docs, "found": len(found)}

    def rewrite(s):
        if s.get("missing"):  # the generator already named what to search for
            print(f"search for missing part #{s['attempts'] + 1}: {s['missing']}", file=sys.stderr)
            return {"query": s["missing"], "variants": [s["missing"]], "attempts": s["attempts"] + 1,
                    "queries": s.get("queries", []) + [s["missing"]]}
        out = ask(REWRITE.format(q=s["question"], prev=" | ".join(s.get("variants") or [s["query"]])))
        variants = [line.strip(" -*\u2022\"'") for line in out.splitlines() if line.strip()][:N_VARIANTS] or [s["query"]]
        print(f"rewrite #{s['attempts'] + 1}: {variants}", file=sys.stderr)
        return {"query": variants[0], "variants": variants, "attempts": s["attempts"] + 1,
                "queries": s.get("queries", []) + variants}

    def generate(s):
        answer = write(ANSWER.format(q=s["question"], ctx=fmt(s["docs"]) or "(no relevant context found)"))
        if answer.startswith(NOT_FOUND_MARK):
            return {"answer": NOT_FOUND, "insufficient": True, "missing": None}
        answer, missing = split_missing(answer)
        return {"answer": answer, "insufficient": False, "missing": missing}

    def check(s):
        verdict = ask(GROUNDED.format(ctx=fmt(s["docs"]) or "(none)", a=s["answer"]))
        return {"grounded": verdict.lower().startswith("yes")}

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.set_entry_point("retrieve")
    if max_attempts == 0:  # baseline: plain retrieve -> generate, no reranking or self-correction
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)
        return g.compile()
    g.add_node("rewrite", rewrite)
    g.add_node("check", check)
    g.add_edge("retrieve", "generate")
    g.add_conditional_edges("generate", lambda s: after_generate(s, max_attempts), ["check", "rewrite", END])
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("check", lambda s: after_check(s, max_attempts), [END, "rewrite"])
    return g.compile()


def tracing() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY"))


class BudgetExceeded(RuntimeError):
    pass


_spend_lock = threading.Lock()


def cost_of(by_model: dict) -> float:
    """USD for token usage by model; unknown models at UNKNOWN_PRICE (for the budget, never undercount)."""
    total = 0.0
    for name, u in by_model.items():
        i = u.get("input_tokens", 0)
        price_in, price_out = PRICES.get(name, UNKNOWN_PRICE)
        total += (i * price_in + (u.get("total_tokens", 0) - i) * price_out) / 1e6
    return total


def _ledger_lock():
    """Thread lock plus an OS file lock, so several server/eval processes can share one ledger file."""
    import portalocker
    return portalocker.Lock(str(SPEND_FILE) + ".lock", timeout=10)


def spent() -> float:
    """USD spent this calendar month (UTC)."""
    with _spend_lock, _ledger_lock():
        try:
            return json.loads(SPEND_FILE.read_text()).get(time.strftime("%Y-%m", time.gmtime()), 0.0)
        except (OSError, ValueError):
            return 0.0


def charge(usd: float):
    if usd <= 0:
        return
    month = time.strftime("%Y-%m", time.gmtime())
    with _spend_lock, _ledger_lock():
        try:
            ledger = json.loads(SPEND_FILE.read_text())
        except (OSError, ValueError):
            ledger = {}
        ledger[month] = round(ledger.get(month, 0.0) + usd, 6)
        SPEND_FILE.write_text(json.dumps(ledger, indent=1))


def check_budget():
    if spent() + QUESTION_RESERVE_USD > MONTHLY_BUDGET_USD:
        raise BudgetExceeded(f"monthly AI budget of ${MONTHLY_BUDGET_USD:.2f} reached")


def usage_summary(by_model: dict) -> dict:
    """Token totals and estimated cost; output counts reasoning tokens, which are billed as output."""
    tokens_in = tokens_out = 0
    cost, priced = 0.0, True
    for name, u in by_model.items():
        i = u.get("input_tokens", 0)
        o = u.get("total_tokens", 0) - i
        tokens_in, tokens_out = tokens_in + i, tokens_out + o
        if name in PRICES:
            cost += (i * PRICES[name][0] + o * PRICES[name][1]) / 1e6
        else:
            priced = False
    return {"input_tokens": tokens_in, "output_tokens": tokens_out,
            "cost_usd": round(cost, 6) if priced else None, "by_model": by_model}


STEP_INFO = {
    "retrieve": lambda u: {"found": u.get("found", len(u.get("docs", []))), "kept": len(u.get("docs", []))},
    "generate": lambda u: {"insufficient": u.get("insufficient", False), "missing": u.get("missing")},
    "rewrite": lambda u: {"query": " | ".join(u.get("variants") or [u.get("query", "")])},
    "check": lambda u: {"grounded": u.get("grounded")},
}


def stream(graph, question: str):
    """Run the graph, yielding progress events: {"type": "step", "node", ...}, {"type": "token", "text"}
    for the answer as it is written, and finally {"type": "done", "result": state}. The result carries
    trace_id (when Langfuse is configured) and usage (tokens and estimated cost)."""
    usage = UsageMetadataCallbackHandler()
    callbacks = [usage]
    if tracing():
        from langfuse.langchain import CallbackHandler
        callbacks.append(CallbackHandler())
    state: State = {}
    # hold back the start of each answer until it can't be the NOT_FOUND marker, and a final line that may be
    # the MISSING line, so neither flashes in the UI
    pending, released = "", False
    check_budget()
    try:
        for mode, chunk in graph.stream({"question": question, "query": question, "attempts": 0, "queries": []},
                                        config={"callbacks": callbacks, "run_name": "veritas"},
                                        stream_mode=["values", "updates", "messages"]):
            if mode == "values":
                state = chunk
            elif mode == "messages":
                msg, meta = chunk
                if meta.get("langgraph_node") == "generate" and (t := text(msg)):
                    pending += t
                    if not released and could_be_marker(pending):
                        continue
                    released = True
                    cut = held_tail(pending)
                    if cut:
                        yield {"type": "token", "text": pending[:cut]}
                    pending = pending[cut:]
            else:
                pending, released = "", False  # a node finished: the next generate pass starts fresh
                for node, update in chunk.items():
                    yield {"type": "step", "node": node, **STEP_INFO.get(node, lambda u: {})(update or {})}
    finally:
        charge(cost_of(usage.usage_metadata))  # billed even when a run fails part-way
    state["trace_id"] = callbacks[1].last_trace_id if len(callbacks) > 1 else None
    state["usage"] = usage_summary(usage.usage_metadata)
    yield {"type": "done", "result": state}


def run(graph, question: str) -> State:
    """Run to completion and return the final state (with trace_id and usage)."""
    for event in stream(graph, question):
        if event["type"] == "done":
            return event["result"]


def flush():
    if tracing():
        from langfuse import get_client
        get_client().flush()


if __name__ == "__main__":
    out = run(build(), " ".join(sys.argv[1:]) or "What BLEU score does the big Transformer get on EN-DE?")
    flush()
    print(out["answer"])
    u = out["usage"]
    print(f"\nattempts={out['attempts']} grounded={out.get('grounded')} trace={out['trace_id']}")
    print(f"tokens in={u['input_tokens']} out={u['output_tokens']} cost=${u['cost_usd']}")
