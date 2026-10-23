"""Upload worker: parses and indexes queued uploads one at a time (Docling is CPU-heavy and not built for parallel
use). Single process: server.py runs work() on a background thread. Scaled out (REDIS_URL + QDRANT_URL set): run
`python worker.py` as its own process, so parsing never competes with chat, and set RUN_WORKER=0 on the app
processes. Upload bytes are read from uploads/ (a shared volume), job status lives in the shared cache."""
import time
import traceback
from pathlib import Path

from cache import cache
from ingest import add, chunk_batches

UPLOADS = Path("uploads")


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


def process(vs, job_id: str):
    job = cache.get_job(job_id)
    if job is None:  # removed before it started
        return
    try:
        data = (UPLOADS / f"{job['doc_id']}_{job['name']}").read_bytes()
        cache.update_job(job_id, status="parsing", started=time.time())
        chunks = 0
        for docs, done, total in chunk_batches(job["name"], data):  # pages become searchable batch by batch
            if docs:
                add(vs, docs)
                chunks += len(docs)
            cache.update_job(job_id, chunks=chunks, pages_done=done, pages_total=total)
        if not chunks:
            raise ValueError("no text could be extracted from this file")
        cache.update_job(job_id, status="done")
    except ValueError as e:
        cache.update_job(job_id, status="error", error=str(e))
    except Exception as e:
        traceback.print_exc()
        cache.update_job(job_id, status="error", error=cloud_problem(e) or "could not process this file; see the log")


def work(vs):
    while True:
        if job_id := cache.next_job():
            process(vs, job_id)


if __name__ == "__main__":
    from agent import store
    print("Veritas upload worker waiting for jobs")
    work(store())
