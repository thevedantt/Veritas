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

# spend ledger: unknown models are charged high, and the cap trips once the month's spend reaches it
import tempfile  # noqa: E402
from pathlib import Path as _P  # noqa: E402

agent.SPEND_FILE = _P(tempfile.mkdtemp()) / "spend.json"
agent.MONTHLY_BUDGET_USD = 0.01
assert agent.cost_of({"gemini-2.5-flash": {"input_tokens": 1_000_000, "total_tokens": 1_000_000}}) == 0.30
assert agent.cost_of({"mystery-model": {"input_tokens": 0, "total_tokens": 1_000_000}}) == agent.UNKNOWN_PRICE[1]
agent.check_budget()  # nothing spent yet
agent.charge(0.004)
agent.charge(0.007)
assert abs(agent.spent() - 0.011) < 1e-9
try:
    agent.check_budget()
    raise AssertionError("budget cap did not trip")
except agent.BudgetExceeded:
    pass
print("ok")

# follow-ups: only questions that lean on earlier turns trigger the (billed) rewrite
from agent import needs_context, standalone  # noqa: E402

assert needs_context("what about 250 robots?") and needs_context("and its price?") and needs_context("why?")
assert not needs_context("What discount applies to an order of 60 robots?")
assert standalone("What discount applies to an order of 60 robots?", [{"q": "x", "a": "y"}]) == \
    "What discount applies to an order of 60 robots?"  # standalone question: no model call
assert standalone("and for 250?", []) == "and for 250?"  # no history: no model call
print("ok")
