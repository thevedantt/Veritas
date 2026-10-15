from langgraph.graph import END

from agent import after_check, after_generate, could_be_marker

assert after_generate({"insufficient": False, "attempts": 0}, 3) == "check"
assert after_generate({"insufficient": True, "attempts": 0}, 3) == "rewrite"
assert after_generate({"insufficient": True, "attempts": 3}, 3) == END  # out of retries: "not found" is final
assert after_check({"grounded": False, "attempts": 1}, 3) == "rewrite"
assert after_check({"grounded": False, "attempts": 3}, 3) == END
assert after_check({"grounded": True, "attempts": 0}, 3) == END

# the streaming hold-back logic: the marker must never be released, real answers must be
for chunks, released in ((["NOT", "_FOUND"], False), (["NOT_FO", "UND"], False), (["The", " answer"], True),
                         (["No", "t in the docs"], True), (["  NOT_FOUND"], False)):
    pending, out = "", False
    for t in chunks:
        pending += t
        if not out and not could_be_marker(pending):
            out = True
    assert out == released, chunks
print("ok")
