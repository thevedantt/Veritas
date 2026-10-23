# Veritas

Veritas is a RAG agent for messy PDFs with tables. It reranks what it retrieves, rewrites the search and retries
when the passages don't hold the answer, checks that the answer is grounded, traces every step, and is scored
against a fixed test set. It runs under a hard $5/month AI budget.

**At a glance** (167-question benchmark over 20 documents in 10 formats; 82 questions on 10 documents never used
for tuning):

| | Value |
|---|---|
| Correct answers | 95.3% for the agent (94.2-96.2% over 3 runs; 96.8% after the prompt-injection guard) vs 91.0% for plain retrieval |
| Never-tuned documents | 97.0% correct (90.9% plain retrieval); Excel, PowerPoint and image formats, seen only there, all 100% |
| Grounding | 99.4% of answers supported by their passages; all 11 unanswerable questions answered "not found" |
| Latency | median 2.6-9.4 s per checked answer depending on Gemini load (plain mode ~1 s); answers stream as written |
| Cost | ~$0.001 per question; $0 for cached answers; ~$0.005 per scanned page at upload |
| Safety | prompt-injection guard (tested: an embedded "ignore all instructions" was ignored), CSRF and browser hardening |
| Budget | hard monthly cap (`MONTHLY_BUDGET_USD`, default $5), per-client rate limits, optional password |

| Piece | Tool | File |
|---|---|---|
| Any document → structured chunks (tables kept intact, OCR for scans) | Docling `DocumentConverter` + `HybridChunker` | `ingest.py` |
| Dense + sparse retrieval, RRF fusion | Qdrant (embedded) via `langchain-qdrant`, Vertex `text-embedding-005` + BM25 | `ingest.py`, `agent.py` |
| Agent loop | LangGraph | `agent.py` |
| LLMs | Gemini (or Claude) on Vertex AI: separate generator, critic and judge models | `agent.py:llm()` |
| Quality scores | One LLM-judge call per question against reference answers | `eval.py`, `testset.json` |
| Tracing | Langfuse | `agent.py:run()` |
| Web app: upload, library, chat | Python `http.server` + HTML/CSS/JS | `server.py`, `web/` |

Runs locally with no Docker: Qdrant is a local folder (`qdrant_data/`), Langfuse uses the cloud tier. A
`Dockerfile` + `docker-compose.yml` package it for a single VM (see Deploy).

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
the generator is skipped for `PRIMARY_COOLDOWN` seconds (30), so an outage costs one slow answer, not every
answer.

## Cost controls

- **Monthly cap:** `MONTHLY_BUDGET_USD` (default $5). Every model call's token usage is priced and added to
  `spend.json`; once the month's total reaches the cap, chat returns "monthly AI budget reached" (cached
  answers still work) and scanned pages fall back to free local OCR. Unknown models are priced high so the
  cap never undercounts. Set a GCP budget alert as well: the app only sees its own calls.
- **Per question:** about $0.0005-0.001 (generator `gemini-2.5-flash`, critic `gemini-2.5-flash-lite`,
  thinking off, 5 passages). Repeated questions are answered from cache for $0.
- **Uploads:** digital pages parse locally for $0. Scanned pages cost about $0.005 each, at most
  `SCAN_MAX_PAGES` (10) per document; the rest use Docling OCR.
- **Evals:** `eval.py` uses one `gemini-2.5-flash-lite` judge call per question and stops at
  `EVAL_BUDGET_USD` (default $1); a full 53-question run costs about $0.06-0.07 (measured). Use `--type` / `--limit`
  to test a subset. (The earlier Ragas + `gemini-2.5-pro` judge made 10-20 billed thinking calls per
  question and cost several dollars a run.)

## Production features

- **Cost:** monthly spend cap with a ledger, cheap models with thinking off, answer + query-embedding caches,
  scan-page cap, budget-capped evals (above).
- **Latency:** streamed answers (SSE), parallel dense + sparse search, parallel rewritten queries, local
  reranker (no API call), fallback model with a cooldown circuit breaker, pages indexed in batches during upload.
- **Accuracy:** hybrid search + reranking, table-aware chunking, vision transcription for scanned pages,
  targeted retry for partial answers, grounding check by a second model, "not found" instead of guessing.
- **Safety:** prompt-injection guard (rules in a system message; document text wrapped in `<passage>` tags as
  untrusted data; instruction-like passages flagged at ingest and to the model; the grounding check rejects
  answers that obey document instructions), CSRF origin check, CSP / no-framing / nosniff headers, per-client
  rate limits (20 questions/min, 10 uploads/hour), upload type whitelist and 50 MB limit, optional password,
  model and document text never rendered as HTML, secrets only in `.env`. Gemini's default safety filters stay on.
- **Operations:** `/api/health`, `/api/stats` (spend vs cap), Langfuse traces with per-question scores,
  offline unit checks, a reproducible eval with a before/after table.

## Tradeoffs made

| Choice | Gain | Cost |
|---|---|---|
| `gemini-2.5-flash` writes, `flash-lite` checks | ~half the price of 3.x Flash, stable | 3.7 Flash got 2 more of 30 right |
| Thinking off everywhere | seconds faster, no billed reasoning tokens | weaker multi-step table reasoning |
| Self-correcting agent vs plain retrieve-generate | +7 points correct (94.5% vs 87.5%), 100% grounded | median 2.6-9.4 s vs ~1 s per question |
| Vision model for scanned pages | correct cells on noisy scans | ~$0.005/page, capped at 10 pages/doc |
| Embedded Qdrant, in-process cache and rate limits | zero infrastructure | one server process; Qdrant server + Redis to scale out |
| One cheap judge call per eval question | ~$0.04 per full eval (was several dollars) | coarser metrics than Ragas |

## Deploy (single VM)

```bash
cp .env.example .env                  # fill in project, models, budget; set APP_PASSWORD for a public site
export GCP_KEY_FILE=/path/to/key.json # service account with the Vertex AI User role
export DOMAIN=veritas.example.com     # DNS A record pointing at the VM
docker compose up -d --build          # app + Qdrant server + Caddy (automatic HTTPS) -> https://$DOMAIN
```

The compose file runs the index as a Qdrant server (`QDRANT_URL`) instead of the embedded folder, so the spend
ledger (file-locked) and index can be shared; Caddy terminates TLS and streams answers unbuffered. With
`APP_PASSWORD` set, every page and API asks for that password (HTTP Basic auth); `/api/health` stays open for
uptime checks. Give the VM at least 4 GB RAM (parsing peaks around 1.5 GB). The Docker files have not been run
yet: the dev machine has no Docker.

## Scaling

Two modes, same code:

| | Single process (default) | Scaled out (`docker compose`) |
|---|---|---|
| App | one `python server.py` | N stateless app copies behind Caddy (`--scale app=N`) |
| Uploads | background thread | separate `python worker.py` process (`RUN_WORKER=0` on app copies) |
| Cache, rate limits, job queue | in-process | Redis (`REDIS_URL`) |
| Index | embedded Qdrant folder | Qdrant server (`QDRANT_URL`) |
| Spend ledger | `spend.json` | same file, shared volume, OS file lock |

What sets capacity:
- **Chat** spends most of its time waiting on Gemini (local search + rerank take ~0.5-1 s of a 3-9 s answer), so one 2-vCPU copy can handle
  dozens of concurrent questions (threads, streamed responses). The local reranker is the main CPU cost per
  question.
- **Vertex AI quota** (requests and tokens per minute) is the real ceiling under load; the app retries 429s with
  backoff and tells the user the model is busy instead of hanging. Raise the quota in the GCP console.
- **Uploads** are CPU-bound (~5 s per page); add worker processes for more parallel uploads (each ~1.5 GB RAM).
- **Cost** scales linearly with questions (~$0.001 each, $0 when cached) and is capped per month.

Not load-tested: these figures come from per-request measurements, not a concurrent-user test.

## Known limits

- One upload parses at a time; a 300-page book takes ~25 minutes on a laptop CPU (searchable as it goes).
- One shared password, not per-user accounts: everyone who signs in sees the same library.
- The table clean-up rules (section labels, OCR fixes) were developed on one IRS scan; the held-out Kestrel
  scan and ResNet tables test them, but other layouts may need new rules.
- Embedding cost is counted at a conservative per-character estimate, not the exact bill.

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

**Demo:** upload `data/kestrel_annual_report_2025.pdf`, a fictional 20-page annual report with KPI and
financial tables, a multi-page product table, range-lookup tables and one image-only scanned page, and ask
the questions in [`DEMO_QUESTIONS.md`](DEMO_QUESTIONS.md) (expected answers included). Rebuild it with
`python scripts/make_demo_pdf.py` (`pip install reportlab`).

**Session report:** *End session* opens a side panel listing every question of the session with its time,
cost, tokens, grounding verdict, rewrites and sources, plus totals (average and slowest time, total cost,
grounded share, cache hits), downloadable as CSV. It is kept per browser tab; no database is needed.

```bash
python ingest.py data/                       # add a folder from the command line (already-indexed files skipped)
python ingest.py --reset data/               # wipe the index and rebuild from the folder (stop the server first)
python agent.py "What dropout does the big Transformer use?"
python eval.py --baseline                    # before: plain retrieve -> generate
python eval.py                               # after: self-correcting agent; rewrites the table below
python eval.py --split heldout --repeat 3    # only documents never used for tuning, 3 runs averaged
python test_routing.py                       # routing, rerank, spend cap and follow-up self-check
python test_eval.py                          # eval scoring, splits and repeats self-check
python test_ingest.py                        # table labels, OCR cleanup, scanned-page split self-check
```

With Langfuse keys set, every run is a trace (retrieve, rewrite, generate, check spans plus each LLM
call), and `eval.py` attaches its scores to each trace. Langfuse US-region accounts use
`LANGFUSE_HOST=https://us.cloud.langfuse.com`.

**Rate limits:** each question makes 2-8 model calls. When throttled (HTTP 429) each call is capped at 45 s
and retried, and the web app says the model is busy instead of hanging.

## Results: before / after

Baseline = hybrid retrieve then generate. Agent = the full graph above. Same test set, same models,
same judge. Generator `gemini-2.5-flash`, critic `gemini-2.5-flash-lite`. The table below was scored with
the earlier Ragas + `gemini-2.5-pro` judge; `eval.py` now uses a single `gemini-2.5-flash-lite` call per
question (correct / supported / retrieved), so the next run regenerates it with those metrics.

<!-- scores:start -->
| Metric | Baseline (retrieve → generate) | Self-correcting agent | Δ |
|---|---|---|---|
| correct | 0.91 | 0.968 | +0.058 |
| supported | 0.981 | 0.994 | +0.013 |
| retrieved | 0.949 | 0.981 | +0.032 |
| correct [direct] | 0.879 | 0.97 | +0.091 |
| correct [vague] | 0.75 | 0.833 | +0.083 |
| correct [messy] | 0.962 | 0.987 | +0.025 |
| correct [csv] | 1 | 1 | +0 |
| correct [docx] | 1 | 1 | +0 |
| correct [html] | 1 | 1 | +0 |
| correct [image] | 1 | 1 | +0 |
| correct [markdown] | 1 | 1 | +0 |
| correct [pdf] | 0.835 | 0.941 | +0.106 |
| correct [pptx] | 1 | 1 | +0 |
| correct [scanned pdf] | 1 | 1 | +0 |
| correct [xlsx] | 1 | 1 | +0 |
| correct [dev] | 0.911 | 0.962 | +0.051 |
| correct [heldout] | 0.909 | 0.974 | +0.065 |
| checker_agreement | – | 1 | |
| checker_passed_unsupported | – | 0 | |
| abstention_rate [unanswerable] | 1 | 1 | +0 |
| rewrite_rate | 0 | 0.084 | +0.084 |
| mean_seconds | 1.1 | 3.5 | +2.4 |
| mean_cost_usd | 0.00081 | 0.001 | +0.00019 |

_167 questions from `testset.json` (direct, vague, messy scanned tables, unanswerable; `heldout` = documents never used for tuning), mean of 1 runs, graded by `gemini-2.5-flash-lite` against reference answers; unanswerable questions are scored on abstention instead._
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
