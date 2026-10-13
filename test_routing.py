from langgraph.graph import END

from agent import after_check, after_grade, parse_indices

assert parse_indices("0, 3, 9", 6) == [0, 3]
assert parse_indices("none", 6) == []
assert after_grade({"docs": [1], "attempts": 0}, 3) == "generate"
assert after_grade({"docs": [], "attempts": 0}, 3) == "rewrite"
assert after_grade({"docs": [], "attempts": 3}, 3) == "generate"  # out of retries: answer anyway
assert after_check({"grounded": False, "attempts": 1}, 3) == "rewrite"
assert after_check({"grounded": False, "attempts": 3}, 3) == END
assert after_check({"grounded": True, "attempts": 0}, 3) == END
print("ok")
