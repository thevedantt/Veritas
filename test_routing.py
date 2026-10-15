from langgraph.graph import END

from agent import after_check, after_generate, could_be_marker, held_tail, split_missing

assert after_generate({"insufficient": False, "attempts": 0}, 3) == "check"
assert after_generate({"insufficient": False, "missing": "number of heads", "attempts": 0}, 3) == "rewrite"
assert after_generate({"insufficient": False, "missing": "number of heads", "attempts": 3}, 3) == "check"

# partial answers: the MISSING line is split off, and never streamed
assert split_missing("d_model is 512.\nHeads not given.\nMISSING: number of attention heads h") == \
    ("d_model is 512.\nHeads not given.", "number of attention heads h")
assert split_missing("Full answer (a.pdf p.2).") == ("Full answer (a.pdf p.2).", None)
streamed, pending = "", ""
for t in ["d_model is 512.", "\nMIS", "SING: heads", " h"]:
    pending += t
    cut = held_tail(pending)
    streamed, pending = streamed + pending[:cut], pending[cut:]
assert streamed == "d_model is 512." and "MISSING" not in streamed
assert held_tail("line one\nThe rest") == len("line one\nThe rest")  # ordinary lines aren't held
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

# rerank fuses with search order: hybrid's top hit survives even when the cross-encoder ranks it last
import agent  # noqa: E402
from langchain_core.documents import Document  # noqa: E402


class FakeReranker:
    def rerank(self, q, texts):
        return [0.1, 0.2, 0.9, 0.3, 0.4, 0.5, 0.6]


agent.RERANKER, agent._reranker = "fake", FakeReranker()
assert [d.page_content for d in agent.rerank("q", [Document(page_content=str(i)) for i in range(7)], 3)] == ["2", "0", "1"]
print("ok")
