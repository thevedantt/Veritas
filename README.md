# Veritas

Veritas is a RAG agent for messy PDFs with tables. It grades its own retrieval, rewrites the query and retries when
the context is weak, checks that the answer is grounded, traces every step, and is scored against a
fixed test set.

| Piece | Tool | File |
|---|---|---|
| Any document → structured chunks (tables kept intact, OCR for scans) | Docling `DocumentConverter` + `HybridChunker` | `ingest.py` |
| Dense + sparse retrieval, RRF fusion | Qdrant (embedded) via `langchain-qdrant`, Vertex `text-embedding-005` + BM25 | `ingest.py`, `agent.py` |
| Agent loop | LangGraph | `agent.py` |
| LLMs | Gemini (or Claude) on Vertex AI: separate generator, critic and judge models | `agent.py:llm()` |
| Quality scores | Ragas | `eval.py`, `testset.json` |
| Tracing | Langfuse | `agent.py:run()` |
| Web app: upload, library, chat | Python `http.server` + HTML/CSS/JS | `server.py`, `web/` |

No Docker needed: Qdrant runs as a local folder (`qdrant_data/`), Langfuse uses the cloud tier.

## Agent graph

```
retrieve ──► grade ──(≥1 relevant)──► generate ──► check ──(grounded)──► END
   ▲           │                                      │
   │     (too few relevant)                     (not grounded)
   │           ▼                                      │
   └──────── rewrite ◄────────────────────────────────┘
```

- **retrieve**: hybrid search (dense + BM25), k=6.
- **grade**: LLM keeps only passages useful for the question; relevant passages carry over across retries.
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
python server.py                             # web app at http://localhost:8000
```

Drop files onto the library panel to upload them: PDF, Word, PowerPoint, Excel, HTML, Markdown, CSV and
images (scans are OCR'd), up to 50 MB each. Each upload is parsed and indexed in the background; the same
file uploaded twice is detected by content hash and skipped. Remove a document from the library and its
chunks leave the index. Answers cite the file (and page, when the format has pages).

```bash
python ingest.py data/                       # add a folder from the command line (already-indexed files skipped)
python ingest.py --reset data/               # wipe the index and rebuild from the folder (stop the server first)
python agent.py "What dropout does the big Transformer use?"
python eval.py --baseline                    # before: plain retrieve -> generate
python eval.py                               # after: self-correcting agent; rewrites the table below
python test_routing.py                       # routing logic self-check
```

With Langfuse keys set, every run is a trace (retrieve, grade, rewrite, generate, check spans plus each LLM
call), and `eval.py` attaches the Ragas scores to each trace. Langfuse US-region accounts use
`LANGFUSE_HOST=https://us.cloud.langfuse.com`.

**Rate limits:** each question makes 3-9 model calls. Gemini 3.x preview models have low per-minute Vertex
quotas; when throttled (HTTP 429) each call is capped at 45 s and retried, and the web app says the model is
busy instead of hanging.

## Results: before / after

Baseline = hybrid retrieve then generate. Agent = the full graph above. Same test set, same models,
same judge. Generator `gemini-3.8-flash`, critic (grade, rewrite, grounding check) `gemini-3.7-flash`, Ragas judge `gemini-2.5-pro`: no model scores its own output.

<!-- scores:start -->
| Metric | Baseline (retrieve → generate) | Self-correcting agent | Δ |
|---|---|---|---|
| faithfulness | 0.845 | 0.869 | +0.024 |
| answer_relevancy | 0.792 | 0.811 | +0.019 |
| llm_context_precision_with_reference | 0.801 | 0.939 | +0.138 |
| context_recall | 1.000 | 0.933 | -0.067 |
| mean_attempts | 0.000 | 0.000 | +0.000 |

_15 questions from `testset.json`, judged by Ragas._
<!-- scores:end -->

Metrics: **faithfulness** (answer claims supported by context), **answer_relevancy** (answer addresses the
question), **llm_context_precision_with_reference** (relevant chunks ranked high), **context_recall**
(context covers the reference answer), **mean_attempts** (how often self-correction fired).
