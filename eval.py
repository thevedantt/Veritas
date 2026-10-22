"""Score the agent on testset.json with one cheap LLM-judge call per question; attach scores to Langfuse traces.

python eval.py                       # self-correcting agent  -> scores_agentic.json
python eval.py --baseline            # plain retrieve+generate -> scores_baseline.json
python eval.py --type messy          # only one question type (direct, vague, messy, unanswerable)
python eval.py --limit 5             # only the first 5 questions (after --type)
python eval.py --split heldout       # only questions on documents never used for tuning (or --split dev)
python eval.py --repeat 3            # run 3 times; report the mean and min-max of each score
Once both full-run files exist the before/after table in README.md is regenerated.

Cost: the whole command stops before it spends more than EVAL_BUDGET_USD (default $1). The judge is flash-lite with
thinking off; a full 33-question agent run costs about $0.03-0.05 (answers ~$0.0007 each, judge ~$0.0004 each).
The earlier Ragas + gemini-2.5-pro judge made 10-20 thinking calls per question and cost several dollars a run.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from langchain_core.callbacks import UsageMetadataCallbackHandler

from agent import build, charge, cost_of, flush, label, llm, run, store, text, tracing

EVAL_BUDGET_USD = float(os.getenv("EVAL_BUDGET_USD", 1.0))
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-flash-lite")

JUDGE = """You grade an answer produced by a document question-answering system.

Question: {q}
Reference answer: {ref}

Passages the system was given:
{ctx}

System answer:
{a}

Reply with JSON only, no prose: {{"correct": true|false, "supported": true|false, "retrieved": true|false}}
- correct: the answer correctly answers what the question asks, agreeing with the reference on every value it
  states. The reference often adds detail the question did not ask about (ranges' upper bounds, conditions,
  related settings); leaving that out is fine. Wrong values, a missing part of what was asked, or "not found"
  are not correct. PDF text loses superscripts: "n2" means n^2 and "2.3 · 1019" means 2.3 x 10^19.
- supported: every factual claim in the answer is stated in the passages. Citations such as (file.pdf p.3)
  name a passage's label and are not claims.
- retrieved: the passages contain the facts needed for the reference answer."""

ABSTAIN = """Question: {q}
Answer: {a}

Does the answer decline to answer or say the information is not available, WITHOUT asserting a specific
answer to the question? Reply "yes" or "no" only."""


def verdict(raw: str) -> dict:
    """Parse the judge's JSON reply; a malformed reply counts as all-false rather than crashing the run."""
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        v = json.loads(m[0]) if m else {}
    except ValueError:
        v = {}
    return {k: bool(v.get(k, False)) for k in ("correct", "supported", "retrieved")}


def pick(tests: list[dict], argv: list[str]) -> list[dict]:
    if "--split" in argv:
        tests = [t for t in tests if t.get("split", "dev") == argv[argv.index("--split") + 1]]
    if "--type" in argv:
        tests = [t for t in tests if t.get("type", "direct") == argv[argv.index("--type") + 1]]
    if "--limit" in argv:
        tests = tests[:int(argv[argv.index("--limit") + 1])]
    return tests


def evaluate_mode(baseline: bool, tests: list[dict], budget: float = EVAL_BUDGET_USD,
                  vs=None) -> tuple[dict, list[dict]]:
    graph = build(max_attempts=0, vs=vs) if baseline else build(vs=vs)
    judge = llm(JUDGE_MODEL, 0).with_retry(stop_after_attempt=3)
    judge_usage = UsageMetadataCallbackHandler()
    spent, rows = 0.0, []
    for i, t in enumerate(tests, 1):
        if spent >= budget:
            print(f"stopping: eval budget reached after {i - 1} questions")
            break
        for attempt in range(3):  # a throttled run (Vertex 429) waits and retries instead of killing the eval
            try:
                start = time.perf_counter()
                out = run(graph, t["question"])
                out["seconds"] = time.perf_counter() - start
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  run failed ({type(e).__name__}: {str(e)[:80]}); retrying in 60s")
                time.sleep(60)
        kind = t.get("type", "direct")
        before = cost_of(judge_usage.usage_metadata)
        cfg = {"callbacks": [judge_usage]}
        if kind == "unanswerable":
            reply = text(judge.invoke(ABSTAIN.format(q=t["question"], a=out["answer"]), config=cfg))
            row = {"abstained": reply.strip().lower().startswith("yes")}
        else:
            # passages exactly as the generator saw them, source label included
            ctx = "\n\n".join(f"({label(d)})\n{d.page_content}" for d in out["docs"]) or "(none)"
            row = verdict(text(judge.invoke(JUDGE.format(q=t["question"], ref=t["ground_truth"], ctx=ctx,
                                                         a=out["answer"]), config=cfg)))
        judge_cost = cost_of(judge_usage.usage_metadata) - before
        charge(judge_cost)  # the agent run charged itself; the judge goes on the same monthly ledger
        run_cost = cost_of(out["usage"]["by_model"])
        spent += run_cost + judge_cost
        row.update(type=kind, split=t.get("split", "dev"), grounded=out.get("grounded"),
                   question=t["question"], answer=out["answer"], attempts=out["attempts"],
                   seconds=out["seconds"], cost_usd=round(run_cost, 6), trace_id=out.get("trace_id"),
                   contexts=[d.page_content for d in out["docs"]])
        print(f"[{i}/{len(tests)}] {kind:<12} attempts={out['attempts']} "
              f"{ {k: row[k] for k in ('correct', 'supported', 'retrieved', 'abstained') if k in row} } "
              f"{t['question'][:50]}")
        rows.append(row)

    if tracing():
        from langfuse import get_client
        lf = get_client()
        for r in rows:
            for k in ("correct", "supported", "retrieved", "abstained"):
                if k in r and r["trace_id"]:
                    lf.create_score(trace_id=r["trace_id"], name=k, value=float(r[k]))
        flush()
    return summarize(rows, spent), rows


def summarize(rows: list[dict], spent: float) -> dict:
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    summary = {k: mean([r[k] for r in answerable]) for k in ("correct", "supported", "retrieved")}
    for kind in ("direct", "vague", "messy"):
        part = [r for r in answerable if r["type"] == kind]
        if part:
            summary[f"correct [{kind}]"] = mean([r["correct"] for r in part])
    for split in ("dev", "heldout"):
        part = [r for r in answerable if r["split"] == split]
        if part:
            summary[f"correct [{split}]"] = mean([r["correct"] for r in part])
    # how far to trust the app's own "Grounded" badge: its verdict vs the judge's "supported"
    checked = [r for r in answerable if r.get("grounded") is not None]
    summary["checker_agreement"] = mean([r["grounded"] == r["supported"] for r in checked])
    summary["checker_passed_unsupported"] = mean([r["grounded"] and not r["supported"] for r in checked])
    summary["abstention_rate [unanswerable]"] = mean([r["abstained"] for r in rows if "abstained" in r])
    summary["rewrite_rate"] = mean([r["attempts"] > 0 for r in rows])
    summary["mean_seconds"] = round(sum(r["seconds"] for r in rows) / len(rows), 1) if rows else None
    summary["mean_cost_usd"] = round(sum(r["cost_usd"] for r in rows) / len(rows), 5) if rows else None
    summary["eval_cost_usd"] = round(spent, 4)
    summary["n"] = len(rows)
    return summary


def readme_table(before: dict, after: dict) -> str:
    lines = ["| Metric | Baseline (retrieve → generate) | Self-correcting agent | Δ |", "|---|---|---|---|"]
    for k in after:
        if k in ("n", "eval_cost_usd", "runs") or "min-max" in k:
            continue
        a, b = after[k], before.get(k)
        if None in (a, b):  # metric missing from an older run
            lines.append(f"| {k} | {'–' if b is None else f'{b:.4g}'} | {'–' if a is None else f'{a:.4g}'} | |")
        else:
            lines.append(f"| {k} | {b:.4g} | {a:.4g} | {a - b:+.4g} |")
    spread = "; ".join(f"{k}: baseline {before.get(k, '–')}, agent {after[k]}" for k in after if "min-max" in k)
    runs = f"mean of {after.get('runs', 1)} runs" + (f" (spread {spread})" if spread else "")
    lines.append(f"\n_{after['n']} questions from `testset.json` (direct, vague, messy scanned tables, unanswerable; "
                 f"`heldout` = documents never used for tuning), {runs}, "
                 f"graded by `{JUDGE_MODEL}` against reference answers; unanswerable questions are scored on "
                 "abstention instead._")
    return "\n".join(lines)


def combine(runs: list[dict]) -> dict:
    """Mean of each score over repeated runs, plus the min-max spread of the headline scores."""
    out = {}
    for k in runs[0]:
        vals = [r[k] for r in runs if r.get(k) is not None]
        if k == "eval_cost_usd":
            out[k] = round(sum(vals), 4)
        elif vals and k != "n":
            out[k] = round(sum(vals) / len(vals), 3 if k != "mean_cost_usd" else 5)
    for k in ("correct", "correct [heldout]", "supported"):
        vals = [r[k] for r in runs if r.get(k) is not None]
        if len(vals) > 1:
            out[f"{k} min-max"] = f"{min(vals):.3f}-{max(vals):.3f}"
    out["n"], out["runs"] = runs[0]["n"], len(runs)
    return out


def main():
    baseline = "--baseline" in sys.argv
    all_tests = json.loads(Path("testset.json").read_text(encoding="utf-8"))
    tests = pick(all_tests, sys.argv)
    repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1
    summaries, rows, spent = [], [], 0.0
    vs = store()  # one index connection for all repeats (the embedded index allows only one)
    for r in range(repeat):
        summary, run_rows = evaluate_mode(baseline, tests, EVAL_BUDGET_USD - spent, vs)
        spent += summary["eval_cost_usd"]
        summaries.append(summary)
        rows += [{**row, "run": r + 1} for row in run_rows]
        print(f"run {r + 1}/{repeat}: {summary}")
        if len(run_rows) < len(tests):
            break  # budget ran out mid-run
    summary = combine(summaries) if len(summaries) > 1 else summaries[0]
    mode = "baseline" if baseline else "agentic"
    Path(f"eval_details_{mode}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{mode}: {summary}")
    if summary["n"] < len(all_tests):
        print("partial run: scores_*.json and README left unchanged")
        return
    out_file = Path(f"scores_{mode}.json")
    out_file.write_text(json.dumps(summary, indent=2))
    a, b = Path("scores_agentic.json"), Path("scores_baseline.json")
    readme = Path("README.md")
    if a.exists() and b.exists() and readme.exists():
        table = readme_table(json.loads(b.read_text()), json.loads(a.read_text()))
        text_ = re.sub(r"(<!-- scores:start -->).*?(<!-- scores:end -->)",
                       lambda m: f"{m[1]}\n{table}\n{m[2]}", readme.read_text(encoding="utf-8"), flags=re.S)
        readme.write_text(text_, encoding="utf-8")
        print("README.md score table updated")


if __name__ == "__main__":
    main()
