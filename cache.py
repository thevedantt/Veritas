"""In-process cache with a Redis-shaped interface (get / set with TTL / incr), so Phase 2 can swap in Redis
without touching callers. Values are kept as Python objects; LRU eviction caps memory."""
import threading
import time
from collections import OrderedDict

ANSWER_TTL = 24 * 3600
EMBED_TTL = 7 * 24 * 3600


class Cache:
    def __init__(self, max_items: int = 10_000):
        self._items: OrderedDict = OrderedDict()
        self._counters: dict[str, int] = {}  # never evicted: losing a version counter would revive stale answers
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


cache = Cache()


def library_version() -> int:
    """Bumped on every upload or delete; part of every answer-cache key, so a library change
    makes all cached answers unreachable instead of serving answers from old documents."""
    return cache.counter("library_version")


def library_changed():
    cache.incr("library_version")
