"""Self-correcting agentic RAG: retrieve -> grade -> (rewrite + retry) -> generate -> grounding check.

CLI: python agent.py "your question"
"""
import os
import hashlib
import re
import sys
import threading
import time
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph import END, StateGraph

from cache import EMBED_TTL, cache

load_dotenv()
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", 3))
MIN_RELEVANT = 1  # answers often sit in a single table chunk
K = 6
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

GRADE = """Question: {q}

Passages:
{ctx}

List the numbers of the passages that contain information useful for answering the question,
comma-separated (e.g. "0, 3"). Reply "none" if no passage is useful. Reply with numbers only."""

REWRITE = """The search query below did not retrieve enough relevant passages from a document collection.
Original question: {q}
Previous query: {prev}

Write one better search query: expand acronyms, add likely synonyms, table names or column terms.
Reply with the query only."""

ANSWER = """Answer the question using only the context. After each claim cite its source using the label in parentheses above the passage, e.g. (report.pdf p.3).
If the context is insufficient, say so plainly instead of guessing. Write plain text: no LaTeX or math markup.

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
    kept: list[Document]  # relevant docs found so far, carried across retries
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


def parse_indices(text: str, n: int) -> list[int]:
    return sorted({int(x) for x in re.findall(r"\d+", text) if int(x) < n})


def after_grade(s: State, max_attempts: int) -> str:
    return "generate" if len(s["docs"]) >= MIN_RELEVANT or s["attempts"] >= max_attempts else "rewrite"


def after_check(s: State, max_attempts: int) -> str:
    return END if s["grounded"] or s["attempts"] >= max_attempts else "rewrite"


def text(msg) -> str:
    c = msg.content
    return c if isinstance(c, str) else "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)


def build(max_attempts: int = MAX_ATTEMPTS, vs=None):
    # generator answers; a different critic model grades, rewrites and checks so no model judges itself
    # critic calls are short yes/no-style tasks: thinking off by default (thinking tokens bill as output)
    model = llm().with_retry(stop_after_attempt=LLM_ATTEMPTS)
    critic = llm(os.getenv("CRITIC_MODEL", "gemini-2.5-flash"),
                 int(os.getenv("CRITIC_THINKING_BUDGET", 0))).with_retry(stop_after_attempt=LLM_ATTEMPTS)
    vs = vs or store()
    ask = lambda prompt, m=critic: text(m.invoke(prompt)).strip()

    def retrieve(s):
        with DB_LOCK:
            return {"docs": vs.similarity_search(s["query"], k=K)}

    def grade(s):
        kept = s.get("kept", [])
        seen = {d.page_content for d in kept}
        new = [d for d in s["docs"] if d.page_content not in seen]
        keep = parse_indices(ask(GRADE.format(q=s["question"], ctx=fmt(new))), len(new)) if new else []
        kept = kept + [new[i] for i in keep]
        return {"docs": kept, "kept": kept}

    def rewrite(s):
        q = ask(REWRITE.format(q=s["question"], prev=s["query"]))
        print(f"rewrite #{s['attempts'] + 1}: {q}", file=sys.stderr)
        return {"query": q, "attempts": s["attempts"] + 1, "queries": s.get("queries", []) + [q]}

    def generate(s):
        return {"answer": ask(ANSWER.format(q=s["question"], ctx=fmt(s["docs"]) or "(no relevant context found)"), model)}

    def check(s):
        verdict = ask(GROUNDED.format(ctx=fmt(s["docs"]) or "(none)", a=s["answer"]))
        return {"grounded": verdict.lower().startswith("yes")}

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.set_entry_point("retrieve")
    if max_attempts == 0:  # baseline: plain retrieve -> generate, no grading or self-correction
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)
        return g.compile()
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("check", check)
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", lambda s: after_grade(s, max_attempts), ["generate", "rewrite"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "check")
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
    "retrieve": lambda u: {"found": len(u.get("docs", []))},
    "grade": lambda u: {"kept": len(u.get("docs", []))},
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
    for mode, chunk in graph.stream({"question": question, "query": question, "attempts": 0, "queries": []},
                                    config={"callbacks": callbacks, "run_name": "veritas"},
                                    stream_mode=["values", "updates", "messages"]):
        if mode == "values":
            state = chunk
        elif mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") == "generate" and (t := text(msg)):
                yield {"type": "token", "text": t}
        else:
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
