"""Run: python test_eval.py  (offline: scoring, splits, repeats and judge-reply parsing; no model calls)"""
import eval as e

rows = [dict(type="direct", split="dev", correct=True, supported=True, retrieved=True, grounded=True, attempts=0,
             seconds=1, cost_usd=.001),
        dict(type="messy", split="heldout", correct=False, supported=False, retrieved=True, grounded=True, attempts=1,
             seconds=3, cost_usd=.002),
        dict(type="unanswerable", split="heldout", abstained=True, grounded=None, attempts=2, seconds=2, cost_usd=.001)]
s = e.summarize(rows, 0.01)
assert s["correct [dev]"] == 1 and s["correct [heldout]"] == 0 and s["abstention_rate [unanswerable]"] == 1
assert s["checker_agreement"] == 0.5 and s["checker_passed_unsupported"] == 0.5  # grounded=True on an unsupported answer
c = e.combine([s, {**s, "correct": 0.0}])
assert c["correct"] == 0.25 and c["correct min-max"] == "0.000-0.500" and c["runs"] == 2
tests = [{"split": "heldout"}, {}, {"split": "heldout", "type": "messy"}]
assert len(e.pick(tests, ["--split", "heldout"])) == 2 and len(e.pick(tests, ["--split", "dev"])) == 1
assert e.verdict('```json\n{"correct": true, "supported": false}\n```') == {"correct": True, "supported": False,
                                                                           "retrieved": False}
assert e.verdict("not json") == {"correct": False, "supported": False, "retrieved": False}
print("ok")
