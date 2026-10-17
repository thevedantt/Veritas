"""Serve the web/ frontend and a JSON API over the agent and the document library.
Run: python server.py -> http://localhost:8000

POST   /api/ask              {"question", "self_correct"} -> answer, sources, rewrites, trace link, usage
POST   /api/ask/stream       same request; server-sent events: steps, answer tokens, then the result
POST   /api/upload?name=...  raw file bytes -> queued for parsing + indexing in the background
GET    /api/library          indexed documents + in-progress uploads
DELETE /api/documents/<id>   remove a document's chunks
"""
import hashlib
import json
import os
import queue
import re
import threading
import traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from agent import BudgetExceeded, build, flush, run, store, stream, tracing
from cache import ANSWER_TTL, cache, library_version
from ingest import SUPPORTED, add, chunk_batches, delete, doc_id, exists, list_documents

MAX_UPLOAD = 50 * 1024 * 1024
UPLOADS = Path("uploads")
UPLOADS.mkdir(exist_ok=True)

vs = store()  # embedded Qdrant allows one client per process; everything shares it under agent.DB_LOCK
GRAPHS = {True: build(vs=vs), False: build(max_attempts=0, vs=vs)}
JOBS: dict[str, dict] = {}
QUEUE: queue.Queue = queue.Queue()


def worker():
    """Parse and index uploads one at a time (Docling is CPU-heavy and not built for parallel use)."""
    while True:
        job, data = QUEUE.get()
        try:
            job["status"] = "parsing"
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
        except Exception:
            traceback.print_exc()
            job.update(status="error", error="could not process this file; see server log")


threading.Thread(target=worker, daemon=True).start()


def safe_name(raw: str) -> str:
    name = Path(unquote(raw).replace("\\", "/")).name
    return re.sub(r"[^\w.\- ()]", "_", name).strip()[:120]


class Handler(SimpleHTTPRequestHandler):
    def send_json(self, status: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if urlsplit(self.path).path == "/api/library":
            active = [j for j in JOBS.values() if j["status"] != "done"]
            return self.send_json(200, {"documents": list_documents(vs), "jobs": active})
        if self.path.startswith("/api/"):
            return self.send_json(404, {"error": "not found"})
        super().do_GET()

    def do_DELETE(self):
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
        path = urlsplit(self.path).path
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
        """Parse and validate an ask request; returns (question, self_correct) or None after replying."""
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self.send_json(400, {"error": "invalid JSON"})
            return None
        question = str(body.get("question", "")).strip()
        if not question or len(question) > 2000:
            self.send_json(400, {"error": "question is required (max 2000 characters)"})
            return None
        return question, bool(body.get("self_correct", True))

    def ask(self):
        """Blocking JSON answer (scripts, eval tooling). The web app uses /api/ask/stream."""
        parsed = self.read_question()
        if not parsed:
            return
        key = answer_key(*parsed)
        if cached := cache.get(key):
            return self.send_json(200, {**cached, "cached": True})
        try:
            out = run(GRAPHS[parsed[1]], parsed[0])
            flush()
        except Exception as e:
            traceback.print_exc()
            return self.send_json(*failure(e))
        result = payload(out)
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

        key = answer_key(*parsed)
        try:
            if cached := cache.get(key):
                return emit({"type": "done", "result": {**cached, "cached": True}})
            for event in stream(GRAPHS[parsed[1]], parsed[0]):
                if event["type"] == "done":
                    flush()
                    result = payload(event["result"])
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


def failure(e: Exception) -> tuple[int, dict]:
    if isinstance(e, BudgetExceeded):
        return 503, {"error": f"{e}; answers resume next month (cached answers still work)"}
    if "429" in str(e) or "timed out" in str(e).lower() or "Timeout" in type(e).__name__:
        return 503, {"error": "the model service is busy (rate limit or timeout); try again in a minute"}
    return 500, {"error": "the agent failed; see server log"}


def payload(out: dict) -> dict:
    trace_url = None
    if out.get("trace_id") and tracing():
        from langfuse import get_client
        trace_url = get_client().get_trace_url(trace_id=out["trace_id"])
    return {
        "answer": out["answer"],
        "attempts": out["attempts"],
        "grounded": out.get("grounded"),
        "queries": out.get("queries", []),
        "trace_url": trace_url,
        "usage": {k: v for k, v in out["usage"].items() if k != "by_model"},
        "cached": False,
        "sources": [{"source": d.metadata.get("source"), "page": d.metadata.get("page"),
                     "headings": d.metadata.get("headings"), "text": d.page_content} for d in out["docs"]],
    }


if __name__ == "__main__":
    print("Veritas running at http://localhost:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), partial(Handler, directory="web")).serve_forever()
