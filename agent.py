"""Self-correcting agentic RAG: retrieve + rerank -> generate (or NOT_FOUND -> rewrite + retry) -> grounding check.

CLI: python agent.py "your question"
"""
import os
import hashlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph import END, StateGraph
from qdrant_client import models

from cache import EMBED_TTL, cache

load_dotenv()
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", 3))
# retrieve K_FETCH candidates by hybrid search, keep the K_KEEP best after reranking (fewer, better passages
# also make generation faster and cheaper); the baseline keeps plain hybrid top-K_BASELINE
K_FETCH, K_KEEP, K_BASELINE = 10, 5, 6
RERANKER = os.getenv("RERANKER", "jinaai/jina-reranker-v1-turbo-en")  # "off" disables; 8K context for long tables
NOT_FOUND_MARK = "NOT_FOUND"
NOT_FOUND = "I couldn't find this in the documents."
# Gemini 3.x on the global endpoint has slow outliers (20-50s for a one-word reply); cap each call and
# retry, since a retried call is usually fast. The client's own retries skip read timeouts.
LLM_TIMEOUT, LLM_ATTEMPTS = 45, 3
# USD per 1M tokens (input, output incl. reasoning), Vertex AI global endpoint, list prices checked 2026-10-14:
# https://cloud.google.com/vertex-ai/generative-ai/pricing (3.8 Flash is an introductory rate until 2026-12-31).
# Models not listed report tokens without a cost.
PRICES = {"gemini-3.8-flash": (0.75, 3.75), "gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-pro": (1.25, 10.00)}
COLLECTION, QDRANT_PATH = "docs", "qdrant_data"
# ponytail: one lock around every vector-store call (embedded Qdrant is not thread-safe);
# switch to a Qdrant server for concurrent reads/writes
DB_LOCK = threading.Lock()

REWRITE = """The search query below did not retrieve enough relevant passages from a document collection.
Original question: {q}
Previous query: {prev}

Write one better search query: expand acronyms, add likely synonyms, table names or column terms.
Reply with the query only."""

ANSWER = """Answer the question using only the context. After each claim cite its source using the label in parentheses above the passage, e.g. (report.pdf p.3).
If the context contains part of the answer, give that part and say what is missing. If it contains nothing
that answers the question, reply with exactly NOT_FOUND and nothing else. Write plain text: no LaTeX or math markup.

Context:
{ctx}

Question: {q}"""

GROUNDED = """Context:
{ctx}

Answer:
{a}

Is every factual claim in the answer supported by the context? An answer that says the context is
insufficient counts as supported. Reply "yes" or "no" only."""


class State(TypedDict, total=False):
    question: str
    query: str
    docs: list[Document]
    kept: list[Document]  # best docs found so far, carried across retries and re-ranked with new results
    found: int
    insufficient: bool
    answer: str
    attempts: int
    grounded: bool
    queries: list[str]


def llm(model: str | None = None, thinking_budget: int | None = None):
    model = model or os.getenv("LLM_MODEL", "gemini-3.8-flash")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if model.startswith("claude"):
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex
        return ChatAnthropicVertex(model_name=model, project=project, location=os.getenv("CLAUDE_LOCATION", "us-east5"),
                                   temperature=0, timeout=LLM_TIMEOUT, max_retries=2)
    from langchain_google_genai import ChatGoogleGenerativeAI  # replaces deprecated ChatVertexAI (HTTP, no gRPC)
    extra = {} if thinking_budget is None else {"thinking_budget": thinking_budget}
    return ChatGoogleGenerativeAI(model=model, vertexai=True, project=project,
                                  location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
                                  temperature=0, timeout=LLM_TIMEOUT, max_retries=2, **extra)


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
        return with_backoff(self.inner.embed_documents, texts)

    def embed_query(self, text: str) -> list[float]:
        key = f"emb:{self.name}:{hashlib.sha256(text.encode()).hexdigest()}"
        if (vector := cache.get(key)) is None:
            vector = with_backoff(self.inner.embed_query, text)
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
        collection_name=COLLECTION, client_options={"path": QDRANT_PATH})


def label(d: Document) -> str:
    page = d.metadata.get("page")
    return f"{d.metadata.get('source')} p.{page}" if page else str(d.metadata.get("source"))


def fmt(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] ({label(d)})\n{d.page_content}" for i, d in enumerate(docs))


_executor = ThreadPoolExecutor(8)
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
    return [docs[i] for i in sorted(range(len(docs)), key=lambda i: -scores[i])[:keep]]


def could_be_marker(pending: str) -> bool:
    """True while the streamed answer so far could still turn out to be the NOT_FOUND marker."""
    return NOT_FOUND_MARK.startswith(pending.lstrip()[:len(NOT_FOUND_MARK)])


def after_generate(s: State, max_attempts: int) -> str:
    if s.get("insufficient"):
        return "rewrite" if s["attempts"] < max_attempts else END  # "not found" makes no claims to check
    return "check"


def after_check(s: State, max_attempts: int) -> str:
    return END if s["grounded"] or s["attempts"] >= max_attempts else "rewrite"


def text(msg) -> str:
    c = msg.content
    return c if isinstance(c, str) else "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)


def build(max_attempts: int = MAX_ATTEMPTS, vs=None):
    # generator answers; a different critic model grades, rewrites and checks so no model judges itself
    # critic calls are short yes/no-style tasks: thinking off by default (thinking tokens bill as output)
    # generator thinking off by default too: reasoning tokens delay the first word by seconds
    model = llm(None, int(os.getenv("GENERATOR_THINKING_BUDGET", 0))).with_retry(stop_after_attempt=LLM_ATTEMPTS)
    critic = llm(os.getenv("CRITIC_MODEL", "gemini-2.5-flash"),
                 int(os.getenv("CRITIC_THINKING_BUDGET", 0))).with_retry(stop_after_attempt=LLM_ATTEMPTS)
    vs = vs or store()
    ask = lambda prompt, m=critic: text(m.invoke(prompt)).strip()

    def retrieve(s):
        if max_attempts == 0:  # baseline: plain hybrid search, no reranking
            return {"docs": hybrid_search(vs, s["query"], K_BASELINE)}
        found = hybrid_search(vs, s["query"], K_FETCH)
        kept = s.get("kept", [])
        seen = {d.page_content for d in kept}
        docs = rerank(s["question"], kept + [d for d in found if d.page_content not in seen], K_KEEP)
        return {"docs": docs, "kept": docs, "found": len(found)}

    def rewrite(s):
        q = ask(REWRITE.format(q=s["question"], prev=s["query"]))
        print(f"rewrite #{s['attempts'] + 1}: {q}", file=sys.stderr)
        return {"query": q, "attempts": s["attempts"] + 1, "queries": s.get("queries", []) + [q]}

    def generate(s):
        answer = ask(ANSWER.format(q=s["question"], ctx=fmt(s["docs"]) or "(no relevant context found)"), model)
        if answer.startswith(NOT_FOUND_MARK):
            return {"answer": NOT_FOUND, "insufficient": True}
        return {"answer": answer, "insufficient": False}

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
    "generate": lambda u: {"insufficient": u.get("insufficient", False)},
    "rewrite": lambda u: {"query": u.get("query")},
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
    # hold back the start of each answer until it can't be the NOT_FOUND marker, so it never flashes in the UI
    pending, released = "", False
    for mode, chunk in graph.stream({"question": question, "query": question, "attempts": 0, "queries": []},
                                    config={"callbacks": callbacks, "run_name": "veritas"},
                                    stream_mode=["values", "updates", "messages"]):
        if mode == "values":
            state = chunk
        elif mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") == "generate" and (t := text(msg)):
                if released:
                    yield {"type": "token", "text": t}
                    continue
                pending += t
                if not could_be_marker(pending):
                    released = True
                    yield {"type": "token", "text": pending}
        else:
            pending, released = "", False  # a node finished: the next generate pass starts fresh
            for node, update in chunk.items():
                yield {"type": "step", "node": node, **STEP_INFO.get(node, lambda u: {})(update or {})}
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
