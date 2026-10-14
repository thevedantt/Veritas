import time

from agent import usage_summary
from cache import Cache

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
