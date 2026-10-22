"""Serve the web/ frontend and a JSON API over the agent and the document library.
Run: python server.py -> http://localhost:8000

POST   /api/ask              {"question", "self_correct"} -> answer, sources, rewrites, trace link, usage
POST   /api/ask/stream       same request; server-sent events: steps, answer tokens, then the result
POST   /api/upload?name=...  raw file bytes -> queued for parsing + indexing in the background
GET    /api/library          indexed documents + in-progress uploads
GET    /api/stats            AI spend this month and the monthly cap
GET    /api/health           liveness check for deploys
DELETE /api/documents/<id>   remove a document's chunks
"""
import base64
import hashlib
import hmac
import json
import os
import queue
import re
import threading
import time
import traceback
from collections import defaultdict, deque
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from agent import (MONTHLY_BUDGET_USD, BudgetExceeded, build, check_budget, flush, run, spent, standalone, store,
                   stream, tracing)
from cache import ANSWER_TTL, cache, library_version
from ingest import SUPPORTED, add, chunk_batches, delete, doc_id, exists, list_documents

MAX_UPLOAD = 50 * 1024 * 1024
# Optional shared password (HTTP Basic auth, any username) for a public deployment; serve it behind TLS.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
UPLOADS = Path("uploads")
UPLOADS.mkdir(exist_ok=True)
# per-client limits, so a public demo can't be scripted into the monthly budget: (requests, window seconds)
LIMITS = {"ask": (20, 60), "upload": (10, 3600)}
_hits: dict[tuple[str, str], deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def allow(client: str, kind: str) -> bool:
    """Sliding-window rate limit. ponytail: in-process; move to Redis with several server processes."""
    limit, window = LIMITS[kind]
    now = time.monotonic()
    with _hits_lock:
        q = _hits[(client, kind)]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(now)
        return True

vs = store()  # embedded Qdrant allows one client per process; everything shares it under agent.DB_LOCK
GRAPHS = {True: build(vs=vs), False: build(max_attempts=0, vs=vs)}
JOBS: dict[str, dict] = {}
QUEUE: queue.Queue = queue.Queue()


def worker():
    """Parse and index uploads one at a time (Docling is CPU-heavy and not built for parallel use)."""
    while True:
        job, data = QUEUE.get()
        try:
            job.update(status="parsing", started=time.time())
            for docs, done, total in chunk_batches(job["name"], data):  # pages become searchable batch by batch
                if docs:
                    add(vs, docs)
                    job["chunks"] += len(docs)
                job.update(pages_done=done, pages_total=total)
            if not job["chunks"]:
                raise ValueError("no text could be extracted from this file")
            job["status"] = "done"
        except ValueError as e:
            job.update(status="error", error=str(e))
        except Exception as e:
            traceback.print_exc()
            job.update(status="error", error=cloud_problem(e) or "could not process this file; see server log")


threading.Thread(target=worker, daemon=True).start()


def safe_name(raw: str) -> str:
    name = Path(unquote(raw).replace("\\", "/")).name
    return re.sub(r"[^\w.\- ()]", "_", name).strip()[:120]


class Handler(SimpleHTTPRequestHandler):
    def authorized(self) -> bool:
        """With APP_PASSWORD set, every route except /api/health needs it; replies 401 otherwise."""
        if not APP_PASSWORD or urlsplit(self.path).path == "/api/health":
            return True
        header = self.headers.get("Authorization", "")
        try:
            password = base64.b64decode(header.removeprefix("Basic ")).decode().partition(":")[2]
        except ValueError:
            password = ""
        if header.startswith("Basic ") and hmac.compare_digest(password.encode(), APP_PASSWORD.encode()):
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Veritas"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def send_json(self, status: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.authorized():
            return
        path = urlsplit(self.path).path
        if path in ("/veritas", "/veritas/"):  # about page
            self.path = "/veritas.html"
            return super().do_GET()
        if path == "/api/health":
            return self.send_json(200, {"ok": True})
        if path == "/api/stats":
            return self.send_json(200, {"spent_usd": round(spent(), 4), "budget_usd": MONTHLY_BUDGET_USD})
        if path == "/api/library":
            active = [j for j in JOBS.values() if j["status"] != "done"]
            return self.send_json(200, {"documents": list_documents(vs), "jobs": active})
        if self.path.startswith("/api/"):
            return self.send_json(404, {"error": "not found"})
        super().do_GET()

    def do_DELETE(self):
        if not self.authorized():
            return
        m = re.fullmatch(r"/api/documents/([0-9a-f]{16})", urlsplit(self.path).path)
        if not m:
            return self.send_json(404, {"error": "not found"})
        did = m[1]
        if not delete(vs, did):
            return self.send_json(404, {"error": "document not found"})
        for f in UPLOADS.glob(f"{did}_*"):
            f.unlink(missing_ok=True)
        JOBS.pop(did, None)
        self.send_json(200, {"deleted": did})

    def do_POST(self):
        if not self.authorized():
            return
        path = urlsplit(self.path).path
        kind = {"/api/ask": "ask", "/api/ask/stream": "ask", "/api/upload": "upload"}.get(path)
        if kind and not allow(self.client_address[0], kind):
            self.close_connection = True
            return self.send_json(429, {"error": "too many requests; wait a minute and try again"})
        if path == "/api/ask":
            return self.ask()
        if path == "/api/ask/stream":
            return self.ask_stream()
        if path == "/api/upload":
            return self.upload()
        self.send_json(404, {"error": "not found"})

    def upload(self):
        name = safe_name(parse_qs(urlsplit(self.path).query).get("name", [""])[0])
        if not name or Path(name).suffix.lower() not in SUPPORTED:
            self.close_connection = True
            return self.send_json(415, {"error": f"unsupported file type; use one of {', '.join(sorted(SUPPORTED))}"})
        size = int(self.headers.get("Content-Length") or 0)
        if size <= 0:
            return self.send_json(400, {"error": "empty file"})
        if size > MAX_UPLOAD:
            self.close_connection = True  # don't read the oversized body
            return self.send_json(413, {"error": f"file too large (max {MAX_UPLOAD // 2**20} MB)"})
        data = self.rfile.read(size)
        did = doc_id(data)
        pending = JOBS.get(did, {}).get("status") in ("queued", "parsing", "indexing")
        if pending or exists(vs, did):
            return self.send_json(200, {"id": did, "doc_id": did, "name": name, "status": "duplicate"})
        (UPLOADS / f"{did}_{name}").write_bytes(data)
        job = JOBS[did] = {"id": did, "doc_id": did, "name": name, "status": "queued", "chunks": 0, "error": None}
        QUEUE.put((job, data))
        self.send_json(202, job)

    def read_question(self):
        """Parse and validate an ask request; returns (question, self_correct, history) or None after replying.
        history: up to 3 earlier {"q", "a"} turns from this chat, for follow-up questions."""
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self.send_json(400, {"error": "invalid JSON"})
            return None
        question = str(body.get("question", "")).strip()
        if not question or len(question) > 2000:
            self.send_json(400, {"error": "question is required (max 2000 characters)"})
            return None
        raw = body.get("history") or []
        history = [{"q": str(t.get("q", ""))[:2000], "a": str(t.get("a", ""))[:4000]}
                   for t in raw[-3:] if isinstance(t, dict)] if isinstance(raw, list) else []
        return question, bool(body.get("self_correct", True)), history

    def resolve(self, parsed) -> str:
        """The question to answer: a follow-up rewritten to stand alone (after the budget check)."""
        check_budget()
        return standalone(parsed[0], parsed[2])

    def ask(self):
        """Blocking JSON answer (scripts, eval tooling). The web app uses /api/ask/stream."""
        parsed = self.read_question()
        if not parsed:
            return
        start = time.perf_counter()
        try:
            question = self.resolve(parsed)
            key = answer_key(question, parsed[1])
            if cached := cache.get(key):
                return self.send_json(200, {**cached, "cached": True, "seconds": 0, "question": question})
            out = run(GRAPHS[parsed[1]], question)
            flush()
        except Exception as e:
            traceback.print_exc()
            return self.send_json(*failure(e))
        result = payload(out, time.perf_counter() - start)
        remember(key, result)
        self.send_json(200, result)

    def ask_stream(self):
        """Server-sent events: step progress, answer tokens as they are written, then the full result."""
        parsed = self.read_question()
        if not parsed:
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(event: dict):
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()

        try:
            start = time.perf_counter()
            question = self.resolve(parsed)
            if question != parsed[0]:
                emit({"type": "step", "node": "condense", "query": question})
            key = answer_key(question, parsed[1])
            if cached := cache.get(key):
                return emit({"type": "done", "result": {**cached, "cached": True, "seconds": 0, "question": question}})
            for event in stream(GRAPHS[parsed[1]], question):
                if event["type"] == "done":
                    flush()
                    result = payload(event["result"], time.perf_counter() - start)
                    remember(key, result)
                    emit({"type": "done", "result": result})
                else:
                    emit(event)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # browser went away mid-answer
        except Exception as e:
            traceback.print_exc()
            try:
                emit({"type": "error", "error": failure(e)[1]["error"]})
            except OSError:
                pass


def answer_key(question: str, self_correct: bool) -> str:
    """Same question (case/spacing/trailing punctuation ignored), same mode, same models, same library."""
    norm = " ".join(question.lower().split()).rstrip("?.! ")
    raw = json.dumps([norm, self_correct, os.getenv("LLM_MODEL"), os.getenv("CRITIC_MODEL"), library_version()])
    return "answer:" + hashlib.sha256(raw.encode()).hexdigest()


def remember(key: str, result: dict):
    if result["grounded"] is not False:  # don't pin an answer the agent itself flagged as unsupported
        cache.set(key, result, ANSWER_TTL)


def cloud_problem(e: Exception) -> str | None:
    """Plain-language message for Google Cloud setup errors (API disabled, missing role, bad key)."""
    msg = str(e)
    if "SERVICE_DISABLED" in msg or "has not been used in project" in msg:
        return "the Vertex AI API is disabled in the Google Cloud project; enable it, then retry"
    if "PERMISSION_DENIED" in msg or "403" in msg:
        return "the service account lacks permission; give it the Vertex AI User role"
    if "DefaultCredentialsError" in type(e).__name__ or "credentials" in msg.lower():
        return "Google Cloud credentials not found; check GOOGLE_APPLICATION_CREDENTIALS"
    return None


def failure(e: Exception) -> tuple[int, dict]:
    if problem := cloud_problem(e):
        return 503, {"error": problem}
    if isinstance(e, BudgetExceeded):
        return 503, {"error": f"{e}; answers resume next month (cached answers still work)"}
    if "429" in str(e) or "timed out" in str(e).lower() or "Timeout" in type(e).__name__:
        return 503, {"error": "the model service is busy (rate limit or timeout); try again in a minute"}
    return 500, {"error": "the agent failed; see server log"}


def payload(out: dict, seconds: float) -> dict:
    trace_url = None
    if out.get("trace_id") and tracing():
        from langfuse import get_client
        trace_url = get_client().get_trace_url(trace_id=out["trace_id"])
    return {
        "question": out["question"],
        "answer": out["answer"],
        "attempts": out["attempts"],
        "grounded": out.get("grounded"),
        "queries": out.get("queries", []),
        "trace_url": trace_url,
        "usage": {k: v for k, v in out["usage"].items() if k != "by_model"},
        "cached": False,
        "seconds": round(seconds, 1),
        "sources": [{"source": d.metadata.get("source"), "page": d.metadata.get("page"),
                     "headings": d.metadata.get("headings"), "text": d.page_content} for d in out["docs"]],
    }


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")  # 0.0.0.0 inside a container; put TLS (e.g. Caddy) in front publicly
    print(f"Veritas running at http://localhost:8000 (listening on {host})")
    ThreadingHTTPServer((host, 8000), partial(Handler, directory="web")).serve_forever()
