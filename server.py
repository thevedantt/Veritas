"""Serve the web/ frontend and a JSON API over the agent. Run: python server.py -> http://localhost:8000"""
import json
import traceback
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

from agent import build, flush, run, store, tracing

vs = store()  # embedded Qdrant allows one client per process, so both graphs share it
GRAPHS = {True: build(vs=vs), False: build(max_attempts=0, vs=vs)}


class Handler(SimpleHTTPRequestHandler):
    def send_json(self, status: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/api/ask":
            return self.send_json(404, {"error": "not found"})
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
        except Exception:
            traceback.print_exc()
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
    # ponytail: single-threaded server, one question at a time; fine for a local demo
    print("Veritas running at http://localhost:8000")
    HTTPServer(("127.0.0.1", 8000), partial(Handler, directory="web")).serve_forever()
