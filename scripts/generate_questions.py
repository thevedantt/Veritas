"""Draft benchmark questions for text documents with a cheap model, keeping only verifiable ones.

For each document, 5 evenly spaced text windows each yield 2 question/answer pairs with a verbatim evidence quote.
A pair is kept only if the quote occurs in the document text (whitespace/case-insensitive) and every number in the
answer occurs in the quote, so a drafted answer can't be invented. Output: bench_generated.json.
python scripts/generate_questions.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from agent import charge, cost_of, llm, text  # noqa: E402
from scripts.make_benchmark import visible_text  # noqa: E402

BENCH = Path("data/bench")
DOCS = {  # file -> split (held-out documents are never looked at while tuning)
    "adam.pdf": "dev", "fw4.pdf": "dev", "eiffel.html": "dev", "fastapi_readme.md": "dev",
    "batchnorm.pdf": "heldout", "layernorm.pdf": "heldout", "nist_csf2.pdf": "heldout", "fomc.html": "heldout",
    "mars.html": "heldout",
}
MODEL = "gemini-2.5-flash-lite"
PROMPT = """You write test questions for a document question-answering system. From the excerpt below, write
exactly 2 questions a user might ask, each answerable from this excerpt alone.
- One "direct" question about a specific fact, and one that is either "messy" (a number, date or value from a
  table, list or specification) or "vague" (a loosely worded, everyday phrasing of something the excerpt explains).
- The answer must be short and complete. Copy numbers exactly as written.
- evidence: copy, word for word, one sentence or table row from the excerpt (under 250 characters) that contains the
  answer.
Reply with JSON only: [{{"question": "...", "answer": "...", "evidence": "...", "type": "direct|messy|vague"}}, ...]

Document: {name}
Excerpt:
{excerpt}"""


def doc_text(path: Path) -> str:
    if path.suffix == ".pdf":
        import pypdfium2
        pdf = pypdfium2.PdfDocument(path.read_bytes())
        return "\n".join(pdf[i].get_textpage().get_text_range() for i in range(len(pdf)))
    raw = path.read_text(encoding="utf-8")
    return visible_text(raw) if path.suffix == ".html" else raw


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("￾", "-").replace("­", "")).strip().lower()


def verified(pair: dict, full: str) -> bool:
    ev, ans = norm(pair.get("evidence", "")), pair.get("answer", "")
    if len(ev) < 15 or ev not in full or not pair.get("question"):
        return False
    return all(n in ev for n in re.findall(r"\d[\d,.]*\d|\d", ans.replace(",", "")))  # numbers come from the quote


def windows(t: str, n: int = 5, size: int = 3500) -> list[str]:
    t = re.sub(r"[ \t]+", " ", t)
    if len(t) <= size * n:
        step = max(1, len(t) // n)
        return [t[i:i + size] for i in range(0, len(t), step)][:n]
    starts = [int(i * (len(t) - size) / (n - 1)) for i in range(n)]
    return [t[s:s + size] for s in starts]


def main():
    model = llm(MODEL, 0, timeout=90).with_retry(stop_after_attempt=3)
    out = []
    for name, split in DOCS.items():
        full_raw = doc_text(BENCH / name)
        full = norm(full_raw)
        kept = []
        for w in windows(full_raw):
            for attempt in range(2):  # one retry if the window yields nothing verifiable
                msg = model.invoke(PROMPT.format(name=name, excerpt=w))
                charge(cost_of({MODEL: msg.usage_metadata or {}}))
                try:
                    pairs = json.loads(re.search(r"\[.*\]", text(msg), re.S)[0])
                except (TypeError, ValueError):
                    pairs = []
                good = [p for p in pairs if isinstance(p, dict) and verified(p, full)][:2]
                if good:
                    kept += good
                    break
        for p in kept[:10]:
            out.append({"file": name, "split": split, "type": p.get("type", "direct") if p.get("type") in
                        ("direct", "messy", "vague") else "direct", "question": p["question"].strip(),
                        "ground_truth": p["answer"].strip(), "evidence": p["evidence"].strip()})
        print(f"{name}: {len(kept[:10])} verified questions")
    Path("bench_generated.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} questions -> bench_generated.json")


if __name__ == "__main__":
    main()
