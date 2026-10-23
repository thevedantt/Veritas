"""Shared state behind one small interface: answer/embedding cache (get / set with TTL), counters (incr), rate
limits (hit), and the upload job queue (put_job / update_job / jobs / next_job).

Default is in-process (one server process). With REDIS_URL set, everything lives in Redis instead, so several app
processes and a separate upload worker share it: a delete in one process invalidates cached answers in all of
them, rate limits count across processes, and queued uploads survive a restart."""
import json
import os
import queue
import threading
import time
from collections import OrderedDict, deque

ANSWER_TTL = 24 * 3600
EMBED_TTL = 7 * 24 * 3600
JOB_TTL = 7 * 24 * 3600


class Cache:
    """In-process backend; LRU eviction caps memory."""

    def __init__(self, max_items: int = 10_000):
        self._items: OrderedDict = OrderedDict()
        self._counters: dict[str, int] = {}  # never evicted: losing a version counter would revive stale answers
        self._hits: dict[str, deque] = {}
        self._jobs: dict[str, dict] = {}
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self.max_items = max_items

    def get(self, key: str):
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            value, expires = item
            if expires < time.monotonic():
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value, ttl: float):
        with self._lock:
            self._items[key] = (value, time.monotonic() + ttl)
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def incr(self, key: str) -> int:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1
            return self._counters[key]

    def counter(self, key: str) -> int:
        with self._lock:
            return self._counters.get(key, 0)

    def hit(self, key: str, limit: int, window: float) -> bool:
        """Sliding-window rate limit: record a hit, False if `limit` hits already happened within `window` s."""
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def put_job(self, job: dict):
        with self._lock:
            self._jobs[job["id"]] = dict(job)
        self._queue.put(job["id"])

    def update_job(self, job_id: str, **fields):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def drop_job(self, job_id: str):
        with self._lock:
            self._jobs.pop(job_id, None)

    def jobs(self) -> list[dict]:
        with self._lock:
            return [dict(j) for j in self._jobs.values()]

    def next_job(self, timeout: float = 5) -> str | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


class RedisCache(Cache):
    """Same interface on Redis. Values are stored as JSON (answers and vectors are plain data)."""

    def __init__(self, url: str):
        import redis
        self.r = redis.Redis.from_url(url, decode_responses=True)

    def get(self, key: str):
        raw = self.r.get(key)
        return None if raw is None else json.loads(raw)

    def set(self, key: str, value, ttl: float):
        self.r.set(key, json.dumps(value), ex=int(ttl))

    def incr(self, key: str) -> int:
        return self.r.incr(f"counter:{key}")

    def counter(self, key: str) -> int:
        return int(self.r.get(f"counter:{key}") or 0)

    def hit(self, key: str, limit: int, window: float) -> bool:
        # sorted set of hit timestamps: drop old ones, count, add (atomic in one transaction)
        now = time.time()
        k = f"hits:{key}"
        pipe = self.r.pipeline()
        pipe.zremrangebyscore(k, 0, now - window)
        pipe.zcard(k)
        count = pipe.execute()[1]
        if count >= limit:
            return False
        pipe = self.r.pipeline()
        pipe.zadd(k, {f"{now}:{os.urandom(4).hex()}": now})
        pipe.expire(k, int(window) + 1)
        pipe.execute()
        return True

    def put_job(self, job: dict):
        self.r.set(f"job:{job['id']}", json.dumps(job), ex=JOB_TTL)
        self.r.sadd("jobs", job["id"])
        self.r.lpush("job-queue", job["id"])

    def update_job(self, job_id: str, **fields):
        if (job := self.get_job(job_id)) is not None:
            job.update(fields)
            self.r.set(f"job:{job_id}", json.dumps(job), ex=JOB_TTL)

    def get_job(self, job_id: str) -> dict | None:
        raw = self.r.get(f"job:{job_id}")
        return None if raw is None else json.loads(raw)

    def drop_job(self, job_id: str):
        self.r.delete(f"job:{job_id}")
        self.r.srem("jobs", job_id)

    def jobs(self) -> list[dict]:
        ids = self.r.smembers("jobs")
        out = [self.get_job(i) for i in ids]
        for i, j in zip(ids, out):
            if j is None:  # expired
                self.r.srem("jobs", i)
        return [j for j in out if j]

    def next_job(self, timeout: float = 5) -> str | None:
        item = self.r.brpop("job-queue", timeout=int(timeout))
        return item[1] if item else None


cache = RedisCache(os.environ["REDIS_URL"]) if os.getenv("REDIS_URL") else Cache()


def library_version() -> int:
    """Bumped on every upload or delete; part of every answer-cache key, so a library change
    makes all cached answers unreachable instead of serving answers from old documents."""
    return cache.counter("library_version")


def library_changed():
    cache.incr("library_version")
