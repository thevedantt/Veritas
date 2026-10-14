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

from agent import build, embeddings, flush, llm, run, tracing

METRICS = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]


def evaluate_mode(baseline: bool) -> dict:
    graph = build(max_attempts=0) if baseline else build()
    tests = json.loads(Path("testset.json").read_text(encoding="utf-8"))
    rows, runs = [], []
    for i, t in enumerate(tests, 1):
        start = time.perf_counter()
        out = run(graph, t["question"])
        out["seconds"] = time.perf_counter() - start
        print(f"[{i}/{len(tests)}] attempts={out['attempts']} {t['question'][:70]}")
        runs.append(out)
        rows.append({"user_input": t["question"], "response": out["answer"],
                     "retrieved_contexts": [d.page_content for d in out["docs"]] or [""],
                     "reference": t["ground_truth"]})

    judge = LangchainLLMWrapper(llm(os.getenv("JUDGE_MODEL", "gemini-2.5-pro")))
    df = evaluate(EvaluationDataset.from_list(rows), metrics=METRICS, llm=judge,
                  embeddings=LangchainEmbeddingsWrapper(embeddings()),
                  run_config=RunConfig(timeout=600, max_workers=4)).to_pandas()  # 2.5 Pro is slow; avoid NaN timeouts
    names = [m.name for m in METRICS]

    if tracing():
        from langfuse import get_client
        lf = get_client()
        for out, (_, row) in zip(runs, df.iterrows()):
            for n in names:
                if out["trace_id"] and row[n] == row[n]:  # skip NaN
                    lf.create_score(trace_id=out["trace_id"], name=n, value=float(row[n]))
        flush()

    print(df[["user_input", *names]].to_string(max_colwidth=50))
    summary = {n: round(float(df[n].mean()), 3) for n in names}
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
    lines.append(f"\n_{after['n']} questions from `testset.json`, judged by Ragas._")
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
