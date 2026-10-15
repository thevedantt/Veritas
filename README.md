# Veritas

Veritas is a RAG agent for messy PDFs with tables. It reranks what it retrieves, rewrites the search and retries
when the passages don't hold the answer, checks that the answer is grounded, traces every step, and is scored
against a fixed test set.

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
retrieve ──► generate ──(answered)──► check ──(grounded)──► END
   ▲            │                        │
   │       (NOT_FOUND)             (not grounded)
   │            ▼                        │
   └──────── rewrite ◄───────────────────┘
```

- **retrieve**: hybrid search (dense + BM25) for 10 passages, reranked by a local cross-encoder
  (`jina-reranker-v1-turbo-en`) fused with the search order; the best 5 are kept and carried across retries.
- **generate**: answers only from those passages, with `(source p.page)` citations, or replies `NOT_FOUND`.
  A partial answer ends with `MISSING: <query>` for the part the passages lack (hidden from the UI).
- **rewrite**: for a partial answer, searches for exactly the missing part and adds its best 2 passages to
  the ones already kept; otherwise writes 3 new search queries (acronyms, synonyms, table/column terms),
  searched in parallel. Up to `MAX_ATTEMPTS` rounds.
- **check**: a different model verifies every claim is supported; if not, rewrite and try again.

Out of retries, the agent says it couldn't find the answer instead of guessing.

**Tables:** each chunk repeats its table's (shortened) header row, and rows under a section row such as
"Head of Household" are prefixed with that label, so a row keeps its meaning when a table spans chunks.
Scanned PDF pages (no text layer) are transcribed to Markdown by `SCAN_MODEL` (`gemini-2.5-flash`, about
$0.005 and 10 s per page): on a noisy scanned wage table Docling's OCR + table recognizer shifted values
between rows, while the vision model read every checked cell correctly. Other scans (image uploads, or
`SCAN_MODEL=off`) use Docling OCR, with common misreads in money tables (`S755`, `$O`) corrected.

**Models:** generator `gemini-2.5-flash`, critic (rewrite, grounding check) `gemini-2.5-flash-lite`, both with
thinking off. If the generator stalls for `PRIMARY_TIMEOUT` seconds or errors, `FALLBACK_MODEL` answers and
the generator is skipped for 2 minutes, so an outage costs one slow answer, not every answer.

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
python test_routing.py                       # routing and rerank self-check
python test_ingest.py                        # table labels, OCR cleanup, scanned-page split self-check
```

With Langfuse keys set, every run is a trace (retrieve, rewrite, generate, check spans plus each LLM
call), and `eval.py` attaches the Ragas scores to each trace. Langfuse US-region accounts use
`LANGFUSE_HOST=https://us.cloud.langfuse.com`.

**Rate limits:** each question makes 2-8 model calls. When throttled (HTTP 429) each call is capped at 45 s
and retried, and the web app says the model is busy instead of hanging.

## Results: before / after

Baseline = hybrid retrieve then generate. Agent = the full graph above. Same test set, same models,
same judge. Generator `gemini-2.5-flash`, critic `gemini-2.5-flash-lite`, Ragas judge `gemini-2.5-pro`: no
model scores its own output. The judge sees each passage with its source label, as the generator did.

<!-- scores:start -->
| Metric | Baseline (retrieve → generate) | Self-correcting agent | Δ |
|---|---|---|---|
| faithfulness | 0.879 | 0.875 | -0.004 |
| answer_relevancy | 0.744 | 0.712 | -0.032 |
| llm_context_precision_with_reference | 0.749 | 0.754 | +0.005 |
| context_recall | 0.933 | 0.967 | +0.034 |
| correct | 0.867 | 0.833 | -0.034 |
| faithfulness [direct] | 0.952 | 0.95 | -0.002 |
| context_recall [direct] | 0.867 | 1 | +0.133 |
| correct [direct] | 0.867 | 1 | +0.133 |
| faithfulness [vague] | 0.933 | 0.833 | -0.1 |
| context_recall [vague] | 1 | 0.8 | -0.2 |
| correct [vague] | 1 | 0.8 | -0.2 |
| faithfulness [messy] | 0.75 | 0.783 | +0.033 |
| context_recall [messy] | 1 | 1 | +0 |
| correct [messy] | 0.8 | 0.6 | -0.2 |
| abstention_rate [unanswerable] | 1 | 1 | +0 |
| rewrite_rate | 0 | 0.273 | +0.273 |
| mean_attempts | 0 | 0.39 | +0.39 |
| mean_seconds | 1.2 | 4.3 | +3.1 |
| mean_cost_usd | 0.00087 | 0.00054 | -0.00033 |

_33 questions from `testset.json` (direct, vague, messy scanned tables, unanswerable), judged by Ragas; unanswerable questions are scored on abstention instead._
<!-- scores:end -->

Metrics: **faithfulness** (answer claims supported by context), **answer_relevancy** (answer addresses the
question), **llm_context_precision_with_reference** (relevant chunks ranked high), **context_recall**
(context covers the reference answer), **correct** (judge: the answer matches the reference on what was
asked; faithfulness alone rewards answers that say little), **mean_attempts** (how often self-correction
fired).

### Choosing the models (2026-10-15)

Same index, same 33 questions, agent graph. Faithfulness re-scored with citations stripped; correct = answers
matching the reference (of 30 answerable).

| Generator / critic | Faithfulness | Correct | Cost / question | Seconds / question |
|---|---|---|---|---|
| **`gemini-2.5-flash` / `gemini-2.5-flash-lite` (chosen)** | 0.933 | 22/30 | $0.00139 | 11.2 |
| `gemini-2.5-flash-lite` / `gemini-2.5-flash` | 0.960 | 21/30 | $0.00101 | 6.3 |
| `gemini-3.7-flash` / `gemini-2.5-flash` | 0.910 | 24/30 | not on price list (~2×) | 25.9 |

Flash-lite as generator is cheapest but gives clipped answers ("37%" for a question asking for the full
withholding rule). 3.7 Flash was most accurate but slow and pricier. `gemini-3.8-flash` was not compared: it
timed out on most calls that day, and its price doubles to $1.50 / $7.50 per 1M tokens on 2027-01-01.
