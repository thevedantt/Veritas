"""Run: python test_cache.py   (also against Redis: REDIS_URL=redis://localhost:6379/15 python test_cache.py)"""
import os
import time

from agent import usage_summary
from cache import Cache, RedisCache


def check(c, name):
    """Behaviour every backend must share (the server and worker only use this interface)."""
    c.set(f"{name}:x", {"answer": "a", "n": [1.5, 2]}, ttl=60)
    assert c.get(f"{name}:x") == {"answer": "a", "n": [1.5, 2]} and c.get(f"{name}:missing") is None
    base = c.counter(f"{name}:v")
    assert c.incr(f"{name}:v") == base + 1 and c.counter(f"{name}:v") == base + 1
    assert [c.hit(f"{name}:ip", 2, 60) for _ in range(3)] == [True, True, False]  # 3rd hit in the window refused
    c.put_job({"id": f"{name}-j1", "name": "a.pdf", "status": "queued", "chunks": 0})
    assert c.next_job(timeout=1) == f"{name}-j1"
    c.update_job(f"{name}-j1", status="parsing", chunks=4)
    assert c.get_job(f"{name}-j1")["status"] == "parsing" and c.get_job(f"{name}-j1")["chunks"] == 4
    assert any(j["id"] == f"{name}-j1" for j in c.jobs())
    c.drop_job(f"{name}-j1")
    assert c.get_job(f"{name}-j1") is None and c.next_job(timeout=1) is None


check(Cache(), "mem")
if url := os.getenv("REDIS_URL"):
    check(RedisCache(url), f"t{int(time.time())}")
    print("redis backend ok")

# in-process specifics: LRU eviction, expiry, counters never evicted
c = Cache(max_items=2)
c.set("a", 1, ttl=60)
c.set("b", 2, ttl=60)
c.get("a")                      # touch a, so b is least recently used
c.set("c", 3, ttl=60)
assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3  # LRU eviction
c.set("short", 1, ttl=0.01)
time.sleep(0.02)
assert c.get("short") is None   # expiry
assert c.counter("v") == 0 and c.incr("v") == 1 and c.counter("v") == 1
for i in range(5):
    c.set(f"k{i}", i, ttl=60)
assert c.counter("v") == 1      # counters survive eviction

u = usage_summary({"gemini-2.5-flash": {"input_tokens": 1_000_000, "total_tokens": 1_500_000}})
assert u["output_tokens"] == 500_000 and abs(u["cost_usd"] - (0.30 + 0.5 * 2.50)) < 1e-9
assert usage_summary({"unknown-model": {"input_tokens": 5, "total_tokens": 9}})["cost_usd"] is None
print("ok")
