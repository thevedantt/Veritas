"""Serve the web/ frontend and a JSON API over the agent and the document library.
Run: python server.py -> http://localhost:8000

POST   /api/ask              {"question", "self_correct"} -> answer, sources, rewrites, trace link
POST   /api/upload?name=...  raw file bytes -> queued for parsing + indexing in the background
GET    /api/library          indexed documents + in-progress uploads
DELETE /api/documents/<id>   remove a document's chunks
"""
import json
import queue
import re
import threading
import traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from agent import build, flush, run, store, tracing
from ingest import SUPPORTED, add, chunk, delete, doc_id, exists, list_documents

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
            docs = chunk(job["name"], data)
            if not docs:
                raise ValueError("no text could be extracted from this file")
            job["status"] = "indexing"
            add(vs, docs)
            job.update(status="done", chunks=len(docs))
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

    def ask(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self.send_json(400, {"error": "invalid JSON"})
        question = str(body.get("question", "")).strip()
        if not question or len(question) > 2000:
            return self.send_json(400, {"error": "question is required (max 2000 characters)"})
        try:
            out = run(GRAPHS[bool(body.get("self_correct", True))], question)
            flush()
        except Exception as e:
            traceback.print_exc()
            if "429" in str(e) or "timed out" in str(e).lower() or "Timeout" in type(e).__name__:
                return self.send_json(503, {"error": "the model service is busy (rate limit or timeout); try again in a minute"})
            return self.send_json(500, {"error": "the agent failed; see server log"})

        trace_url = None
        if out["trace_id"] and tracing():
            from langfuse import get_client
            trace_url = get_client().get_trace_url(trace_id=out["trace_id"])
        self.send_json(200, {
            "answer": out["answer"],
            "attempts": out["attempts"],
            "grounded": out.get("grounded"),
            "queries": out.get("queries", []),
            "trace_url": trace_url,
            "sources": [{"source": d.metadata.get("source"), "page": d.metadata.get("page"),
                         "headings": d.metadata.get("headings"), "text": d.page_content} for d in out["docs"]],
        })


if __name__ == "__main__":
    print("Veritas running at http://localhost:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), partial(Handler, directory="web")).serve_forever()
