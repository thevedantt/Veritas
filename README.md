# Veritas

Veritas is a RAG agent for messy PDFs with tables. It grades its own retrieval, rewrites the query and retries when
the context is weak, checks that the answer is grounded, traces every step, and is scored against a
fixed test set.

| Piece | Tool | File |
|---|---|---|
| PDF → structured chunks (tables kept intact) | Docling `DocumentConverter` + `HybridChunker` | `ingest.py` |
| Dense + sparse retrieval, RRF fusion | Qdrant (embedded) via `langchain-qdrant`, Vertex `text-embedding-005` + BM25 | `ingest.py`, `agent.py` |
| Agent loop | LangGraph | `agent.py` |
| LLMs | Gemini (or Claude) on Vertex AI: separate generator, critic and judge models | `agent.py:llm()` |
| Quality scores | Ragas | `eval.py`, `testset.json` |
| Tracing | Langfuse | `agent.py:run()` |
| Chat UI | Streamlit | `app.py` |

No Docker needed: Qdrant runs as a local folder (`qdrant_data/`), Langfuse uses the cloud tier.

## Agent graph

```
retrieve ──► grade ──(≥2 relevant)──► generate ──► check ──(grounded)──► END
   ▲           │                                      │
   │     (too few relevant)                     (not grounded)
   │           ▼                                      │
   └──────── rewrite ◄────────────────────────────────┘
```

- **retrieve**: hybrid search (dense + BM25), k=6.
- **grade**: LLM keeps only passages useful for the question.
- **rewrite**: LLM rewrites the query (acronyms, synonyms, table/column terms); retried up to `MAX_ATTEMPTS`.
- **generate**: answers only from context, with `(source p.page)` citations.
- **check**: LLM verifies every claim is supported; if not, rewrite and try again.

Out of retries, the agent answers with what it has and says when context is insufficient.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
gcloud auth application-default login
copy .env.example .env      # set GOOGLE_CLOUD_PROJECT, models, optional Langfuse keys
```

## Run

```bash
python ingest.py data/                       # Docling parse + index (sample: "Attention Is All You Need")
python agent.py "What dropout does the big Transformer use?"
streamlit run app.py                         # chat UI with self-correction toggle, sources, trace links
python eval.py --baseline                    # before: plain retrieve -> generate
python eval.py                               # after: self-correcting agent; rewrites the table below
python test_routing.py                       # routing logic self-check
```

With Langfuse keys set, every run is a trace (retrieve, grade, rewrite, generate, check spans plus each LLM
call), and `eval.py` attaches the Ragas scores to each trace.

## Results: before / after

Baseline = hybrid retrieve then generate. Agent = the full graph above. Same test set, same models,
same judge. Generator `gemini-3.8-flash`, critic (grade, rewrite, grounding check) `gemini-3.7-flash`, Ragas judge `gemini-2.5-pro`: no model scores its own output.

<!-- scores:start -->
| Metric | Baseline (retrieve → generate) | Self-correcting agent | Δ |
|---|---|---|---|
| faithfulness | 0.806 | 0.752 | -0.054 |
| answer_relevancy | 0.755 | 0.801 | +0.046 |
| llm_context_precision_with_reference | 0.773 | 0.939 | +0.166 |
| context_recall | 1.000 | 0.967 | -0.033 |
| mean_attempts | 0.000 | 1.000 | +1.000 |

_15 questions from `testset.json`, judged by Ragas._
<!-- scores:end -->

Metrics: **faithfulness** (answer claims supported by context), **answer_relevancy** (answer addresses the
question), **llm_context_precision_with_reference** (relevant chunks ranked high), **context_recall**
(context covers the reference answer), **mean_attempts** (how often self-correction fired).
