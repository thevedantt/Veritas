"""Self-correcting agentic RAG: retrieve -> grade -> (rewrite + retry) -> generate -> grounding check.

CLI: python agent.py "your question"
"""
import os
import re
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

load_dotenv()
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", 3))
MIN_RELEVANT = 2
K = 6
COLLECTION, QDRANT_PATH = "docs", "qdrant_data"

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

ANSWER = """Answer the question using only the context. Cite sources as (source p.page) after each claim.
If the context is insufficient, say so plainly instead of guessing.

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
    answer: str
    attempts: int
    grounded: bool
    queries: list[str]


def llm(model: str | None = None):
    model = model or os.getenv("LLM_MODEL", "gemini-3.8-flash")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if model.startswith("claude"):
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex
        return ChatAnthropicVertex(model_name=model, project=project,
                                   location=os.getenv("CLAUDE_LOCATION", "us-east5"), temperature=0)
    from langchain_google_vertexai import ChatVertexAI
    return ChatVertexAI(model_name=model, project=project,
                        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"), temperature=0)


def embeddings():
    from langchain_google_vertexai import VertexAIEmbeddings
    return VertexAIEmbeddings(model_name=os.getenv("EMBED_MODEL", "text-embedding-005"),
                              project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                              location=os.getenv("EMBED_LOCATION", "us-central1"))


def sparse_embeddings():
    from langchain_qdrant import FastEmbedSparse
    return FastEmbedSparse(model_name="Qdrant/bm25")


def store():
    from langchain_qdrant import QdrantVectorStore, RetrievalMode
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings(), sparse_embedding=sparse_embeddings(), retrieval_mode=RetrievalMode.HYBRID,
        collection_name=COLLECTION, path=QDRANT_PATH)


def fmt(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] ({d.metadata.get('source')} p.{d.metadata.get('page')})\n{d.page_content}"
                       for i, d in enumerate(docs))


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
    model, critic, vs = llm(), llm(os.getenv("CRITIC_MODEL", "gemini-3.7-flash")), vs or store()
    ask = lambda prompt, m=critic: text(m.invoke(prompt)).strip()

    def retrieve(s):
        return {"docs": vs.similarity_search(s["query"], k=K)}

    def grade(s):
        keep = parse_indices(ask(GRADE.format(q=s["question"], ctx=fmt(s["docs"]))), len(s["docs"]))
        return {"docs": [s["docs"][i] for i in keep]}

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


def run(graph, question: str) -> State:
    """Invoke the graph; when Langfuse keys are set every node and LLM call is traced. Adds trace_id."""
    callbacks = []
    if tracing():
        from langfuse.langchain import CallbackHandler
        callbacks = [CallbackHandler()]
    out = graph.invoke({"question": question, "query": question, "attempts": 0, "queries": []},
                       config={"callbacks": callbacks, "run_name": "veritas"})
    out["trace_id"] = callbacks[0].last_trace_id if callbacks else None
    return out


def flush():
    if tracing():
        from langfuse import get_client
        get_client().flush()


if __name__ == "__main__":
    out = run(build(), " ".join(sys.argv[1:]) or "What BLEU score does the big Transformer get on EN-DE?")
    flush()
    print(out["answer"])
    print(f"\nattempts={out['attempts']} grounded={out.get('grounded')} trace={out['trace_id']}")
