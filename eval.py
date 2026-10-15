"""Score the agent on testset.json with Ragas and attach scores to the Langfuse traces.

python eval.py              # self-correcting agent  -> scores_agentic.json
python eval.py --baseline   # plain retrieve+generate -> scores_baseline.json
Once both files exist the before/after table in README.md is regenerated.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

from agent import build, embeddings, flush, llm, run, text, tracing

METRICS = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]


ABSTAIN = """Question: {q}
Answer: {a}

Does the answer decline to answer or say the information is not available, WITHOUT asserting a specific
answer to the question? Reply "yes" or "no" only."""


def evaluate_mode(baseline: bool) -> dict:
    """Ragas scores answerable questions; unanswerable ones are scored on whether the system abstains
    (there is no reference answer to compare against). Also breaks scores down by question type."""
    graph = build(max_attempts=0) if baseline else build()
    tests = json.loads(Path("testset.json").read_text(encoding="utf-8"))
    runs = []
    for i, t in enumerate(tests, 1):
        start = time.perf_counter()
        out = run(graph, t["question"])
        out["seconds"] = time.perf_counter() - start
        out["type"] = t.get("type", "direct")
        print(f"[{i}/{len(tests)}] {out['type']:<12} attempts={out['attempts']} {t['question'][:60]}")
        runs.append(out)

    judge_llm = llm(os.getenv("JUDGE_MODEL", "gemini-2.5-pro"))
    answerable = [(t, o) for t, o in zip(tests, runs) if o["type"] != "unanswerable"]
    rows = [{"user_input": t["question"], "response": o["answer"],
             "retrieved_contexts": [d.page_content for d in o["docs"]] or [""],
             "reference": t["ground_truth"]} for t, o in answerable]
    df = evaluate(EvaluationDataset.from_list(rows), metrics=METRICS, llm=LangchainLLMWrapper(judge_llm),
                  embeddings=LangchainEmbeddingsWrapper(embeddings()),
                  run_config=RunConfig(timeout=600, max_workers=4)).to_pandas()  # 2.5 Pro is slow; avoid NaN timeouts
    df["type"] = [o["type"] for _, o in answerable]
    names = [m.name for m in METRICS]

    abstained = {}
    for t, o in zip(tests, runs):
        if o["type"] == "unanswerable":
            verdict = text(judge_llm.with_retry(stop_after_attempt=3).invoke(ABSTAIN.format(q=t["question"], a=o["answer"])))
            abstained[id(o)] = verdict.strip().lower().startswith("yes")

    if tracing():
        from langfuse import get_client
        lf = get_client()
        for (_, out), (_, row) in zip(answerable, df.iterrows()):
            for n in names:
                if out["trace_id"] and row[n] == row[n]:  # skip NaN
                    lf.create_score(trace_id=out["trace_id"], name=n, value=float(row[n]))
        for out in runs:
            if id(out) in abstained and out["trace_id"]:
                lf.create_score(trace_id=out["trace_id"], name="abstained", value=float(abstained[id(out)]))
        flush()

    print(df[["type", "user_input", *names]].to_string(max_colwidth=45))
    summary = {n: round(float(df[n].mean()), 3) for n in names}
    for kind in ("direct", "vague", "messy"):
        part = df[df["type"] == kind]
        for n in ("faithfulness", "context_recall"):
            if len(part):
                summary[f"{n} [{kind}]"] = round(float(part[n].mean()), 3)
    summary["abstention_rate [unanswerable]"] = round(sum(abstained.values()) / len(abstained), 3) if abstained else None
    summary["rewrite_rate"] = round(sum(o["attempts"] > 0 for o in runs) / len(runs), 3)
    summary["mean_attempts"] = round(sum(o["attempts"] for o in runs) / len(runs), 2)
    summary["mean_seconds"] = round(sum(o["seconds"] for o in runs) / len(runs), 1)
    costs = [o["usage"]["cost_usd"] for o in runs]
    summary["mean_cost_usd"] = round(sum(costs) / len(costs), 5) if None not in costs else None
    summary["n"] = len(runs)
    return summary


def readme_table(before: dict, after: dict) -> str:
    lines = ["| Metric | Baseline (retrieve → generate) | Self-correcting agent | Δ |", "|---|---|---|---|"]
    for k in after:
        if k == "n":
            continue
        a, b = after[k], before.get(k)
        if None in (a, b):  # metric missing from an older run
            lines.append(f"| {k} | {'–' if b is None else f'{b:.4g}'} | {'–' if a is None else f'{a:.4g}'} | |")
        else:
            lines.append(f"| {k} | {b:.4g} | {a:.4g} | {a - b:+.4g} |")
    lines.append(f"\n_{after['n']} questions from `testset.json` (direct, vague, messy scanned tables, unanswerable), "
                 "judged by Ragas; unanswerable questions are scored on abstention instead._")
    return "\n".join(lines)


def main():
    baseline = "--baseline" in sys.argv
    summary = evaluate_mode(baseline)
    out_file = Path("scores_baseline.json" if baseline else "scores_agentic.json")
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nmeans -> {out_file}: {summary}")

    a, b = Path("scores_agentic.json"), Path("scores_baseline.json")
    readme = Path("README.md")
    if a.exists() and b.exists() and readme.exists():
        table = readme_table(json.loads(b.read_text()), json.loads(a.read_text()))
        text = re.sub(r"(<!-- scores:start -->).*?(<!-- scores:end -->)",
                      lambda m: f"{m[1]}\n{table}\n{m[2]}", readme.read_text(encoding="utf-8"), flags=re.S)
        readme.write_text(text, encoding="utf-8")
        print("README.md score table updated")


if __name__ == "__main__":
    main()
