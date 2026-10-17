"""Parse documents with Docling (layout, tables, OCR), chunk by document structure, and add them to the
Qdrant hybrid collection (Vertex dense vectors + BM25 sparse vectors). Also the document-library
operations the web server uses. A document's id is the hash of its bytes, so re-adding a file is a no-op.

python ingest.py data/            # add new files from a folder; already-indexed files are skipped
python ingest.py --reset data/    # wipe the collection and rebuild it from the folder
"""
import hashlib
import os
import re
import shutil
import sys
from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document
from qdrant_client import models

from agent import (COLLECTION, DB_LOCK, LLM_ATTEMPTS, MONTHLY_BUDGET_USD, QDRANT_PATH, charge, cost_of, llm,
                   spent, store, text)
from cache import library_changed

HEADER_PART = 30  # max characters per header level of a table column, repeated in every chunk of the table
CHUNK_TOKENS = 512  # room for a table header plus several rows; text-embedding-005 accepts up to 2048
PAGE_BATCH = 4  # PDFs are parsed and indexed a few pages at a time, so early pages are searchable within seconds
# Scanned PDF pages (no text layer) are transcribed by a vision model instead of Docling OCR: on a noisy scanned
# table Docling's table recognizer shifted values between rows, while Gemini 2.5 Flash read every checked cell
# right, at about $0.005 and 10 s per page. "off" falls back to Docling OCR.
SCAN_MODEL = os.getenv("SCAN_MODEL", "gemini-2.5-flash")
SCAN_MAX_PAGES = int(os.getenv("SCAN_MAX_PAGES", 10))  # per document; later scanned pages use Docling OCR (free)
SCAN_PROMPT = """Transcribe this scanned document page to Markdown. Reproduce every table as a Markdown table with
one table row per line and every cell in its own column, exactly as printed (do not merge rows, do not
compute or skip anything). Use a single header row with exactly one cell per column. Make each header short
(under 30 characters per part) and start it with its column group when columns are grouped or tables sit
side by side, e.g. "STANDARD - At least", "STANDARD - But less than", "Checkbox - At least",
"Head of Household - Standard". Put long notes printed above a table in a paragraph before it. Use GitHub
Markdown table syntax: every row starts and ends with "|", and a "|---|" separator row follows the header
row. Keep headings and other text as plain Markdown. Output only the Markdown."""
SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".csv",
             ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_tools = None


def tools():
    """Docling converter + chunker, loaded once (model loading takes a few seconds)."""
    global _tools
    if _tools is None:
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.transforms.chunker.hierarchical_chunker import (ChunkingDocSerializer,
                                                                          ChunkingSerializerProvider)
        from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

        class ShortHeaderTables(MarkdownTableSerializer):
            # The chunker repeats a table's header row in every chunk of that table. Scanned tables can have
            # sentence-long header cells, and Markdown pads the separator to match, so the header outgrew the
            # chunk and later rows lost their column names. Trim cells and padding so it always fits.
            def get_header_and_body_lines(self, *, table_text, **kwargs):
                head, body = super().get_header_and_body_lines(table_text=table_text, **kwargs)
                def short(cell):  # stacked header levels are joined with " - "; trim each level, keep them all
                    return " - ".join(trim(part.strip()) for part in cell.split(" - "))
                head = ["|" + "|".join(f" {short(c)} " for c in line.strip().strip("|").split("|")) + "|\n"
                        for line in head]
                return [compact(line) for line in head], with_sections(body)

        class MarkdownTables(ChunkingSerializerProvider):
            # Markdown rows keep each table row on one line with its column order; the default
            # "header = value" triplets scrambled rows of OCR'd tables across many tiny chunks
            def get_serializer(self, doc):
                return ChunkingDocSerializer(doc=doc, table_serializer=ShortHeaderTables())

        # GPU when PyTorch has CUDA (layout, tables and OCR models), all CPU cores otherwise. OCR runs only on
        # image regions, so digital PDFs skip it. TABLE_MODE=fast trades some table accuracy for speed.
        pdf = PdfPipelineOptions(accelerator_options=AcceleratorOptions(device=AcceleratorDevice.AUTO,
                                                                        num_threads=os.cpu_count() or 4))
        if os.getenv("TABLE_MODE", "accurate") == "fast":
            pdf.table_structure_options.mode = TableFormerMode.FAST
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf)})
        tokenizer = HuggingFaceTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2",
                                                         max_tokens=CHUNK_TOKENS)  # Docling's default counter
        _tools = converter, HybridChunker(tokenizer=tokenizer, serializer_provider=MarkdownTables())
    return _tools


def trim(part: str, n: int = HEADER_PART) -> str:
    """Shorten a header part keeping both ends: the distinguishing words ("... is: At least") are often last."""
    return part if len(part) <= n else part[:n // 2 - 1] + "…" + part[-(n - n // 2):]


def with_sections(rows: list[str]) -> list[str]:
    """A row whose non-empty cells all hold the same text label ("Head of Household") starts a section of the table. Prefix
    each later row with that label, so a row keeps its section after the table is split across chunks."""
    section, out = "", []
    for row in rows:
        cells = {c.strip().strip("*").strip() for c in row.strip().strip("|").split("|")} - {""}
        if len(cells) == 1 and not re.search(r"\d", label := cells.pop()):
            section = label
            out.append(row)
        else:
            out.append(f"| {section}: " + row.lstrip()[1:].lstrip() if section and row.lstrip().startswith("|") else row)
    return out


def compact(text: str) -> str:
    """Drop Markdown cell padding (runs of spaces / dashes), which otherwise costs embedding and LLM tokens."""
    text = re.sub(r"-{4,}", "---", re.sub(r"[ \t]{2,}", " ", text))
    # common OCR misreads in table cells of scanned money tables: S755 -> $755, $O -> $0, %0 -> 0%
    text = re.sub(r"(?<=\| )S(?=\d)", "$", text)
    text = re.sub(r"(?<=\| )\$O(?= \|)", "$0", text)
    return re.sub(r"(?<=\| )%0(?= \|)", "0%", text)


def doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def page_count(name: str, data: bytes) -> int | None:
    if not name.lower().endswith(".pdf"):
        return None
    import pypdfium2
    return len(pypdfium2.PdfDocument(data))


def scanned_pages(data: bytes) -> set[int]:
    """1-based numbers of PDF pages with no text layer (images only)."""
    import pypdfium2
    pdf = pypdfium2.PdfDocument(data)
    return {i + 1 for i in range(len(pdf)) if len(pdf[i].get_textpage().get_text_range().strip()) < 20}


def segments(first: int, last: int, scanned: set[int]):
    """Split pages first..last into runs of text pages (parsed together) and single scanned pages."""
    run = None
    for n in range(first, last + 1):
        if n in scanned:
            if run:
                yield run, False
            run = None
            yield (n, n), True
        else:
            run = (run[0], n) if run else (n, n)
    if run:
        yield run, False


def chunk_batches(name: str, data: bytes):
    """Yield (chunks, pages_done, pages_total) as a document is parsed: PDFs PAGE_BATCH pages at a time,
    other formats in one pass (pages_total None)."""
    total = page_count(name, data)
    if not total:
        yield chunk(name, data), None, None
        return
    scanned = set()
    if SCAN_MODEL != "off" and spent() < MONTHLY_BUDGET_USD:  # over budget: Docling OCR only
        scanned = set(sorted(scanned_pages(data))[:SCAN_MAX_PAGES])
    for first in range(1, total + 1, PAGE_BATCH):
        last = min(first + PAGE_BATCH - 1, total)
        docs = []
        for pages, is_scan in segments(first, last, scanned):
            docs += chunk_scan(name, data, pages[0]) if is_scan else chunk(name, data, pages)
        yield docs, last, total


def chunk(name: str, data: bytes, pages: tuple[int, int] | None = None) -> list[Document]:
    from docling.datamodel.base_models import DocumentStream
    converter, _ = tools()
    kwargs = {"page_range": pages} if pages else {}
    return to_docs(converter.convert(DocumentStream(name=name, stream=BytesIO(data)), **kwargs).document,
                   name, doc_id(data))


def chunk_scan(name: str, data: bytes, page: int) -> list[Document]:
    """Transcribe one scanned PDF page to Markdown with a vision model, then chunk it like any document."""
    import base64

    import pypdfium2
    from docling.datamodel.base_models import DocumentStream
    from langchain_core.messages import HumanMessage
    buf = BytesIO()
    pypdfium2.PdfDocument(data)[page - 1].render(scale=2).to_pil().save(buf, format="PNG")
    image = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    model = llm(SCAN_MODEL, 0, timeout=180).with_retry(stop_after_attempt=LLM_ATTEMPTS + 1,
                                                       wait_exponential_jitter=True)  # 429s happen
    msg = model.invoke([HumanMessage(content=[{"type": "text", "text": SCAN_PROMPT},
                                              {"type": "image_url", "image_url": image}])])
    charge(cost_of({SCAN_MODEL: msg.usage_metadata or {}}))
    md = text(msg)
    md = square_tables(md.strip().removeprefix("```markdown").removeprefix("```").removesuffix("```"))
    converter, _ = tools()
    dl = converter.convert(DocumentStream(name=f"{Path(name).stem}_p{page}.md", stream=BytesIO(md.encode()))).document
    return to_docs(dl, name, doc_id(data), page)


def square_tables(md: str) -> str:
    """Pad every row of each Markdown table to the widest row and rebuild its separator: a model's table can
    have a header shorter than its rows, and the Markdown parser then drops the extra columns."""
    out, table = [], []

    def flush():
        if table:
            rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in table]
            rows = [r for r in rows if not all(re.fullmatch(r":?-{3,}:?", c) for c in r if c)]  # drop separators
            width = max(map(len, rows))
            rows = [r + [""] * (width - len(r)) for r in rows]
            out.extend(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
                       + ["| " + " | ".join(r) + " |" for r in rows[1:]])
            table.clear()

    for line in md.splitlines():
        if line.strip().startswith("|"):
            table.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def to_docs(dl, name: str, did: str, page: int | None = None) -> list[Document]:
    """Chunk a parsed document; page overrides the page number (a transcribed page has none of its own)."""
    _, chunker = tools()
    docs = []
    for c in chunker.chunk(dl_doc=dl):
        prov = c.meta.doc_items[0].prov if c.meta.doc_items else []
        docs.append(Document(
            page_content=compact(chunker.contextualize(c)),  # prepends section headings; tables as Markdown rows
            metadata={"doc_id": did, "source": name, "page": page or (prov[0].page_no if prov else None),
                      "headings": " > ".join(c.meta.headings or []),
                      "has_table": any(i.label == "table" for i in c.meta.doc_items)}))
    return docs


def _of(did: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=did))])


def exists(vs, did: str) -> bool:
    with DB_LOCK:
        return vs.client.count(COLLECTION, count_filter=_of(did), exact=True).count > 0


def add(vs, docs: list[Document]):
    with DB_LOCK:
        vs.add_documents(docs)
    library_changed()


def delete(vs, did: str) -> bool:
    with DB_LOCK:
        if not vs.client.count(COLLECTION, count_filter=_of(did), exact=True).count:
            return False
        vs.client.delete(COLLECTION, points_selector=models.FilterSelector(filter=_of(did)))
    library_changed()
    return True


def list_documents(vs) -> list[dict]:
    # ponytail: scans every chunk's metadata; keep a separate documents table past ~100k chunks
    docs, offset = {}, None
    with DB_LOCK:
        while True:
            points, offset = vs.client.scroll(COLLECTION, limit=1000, offset=offset,
                                              with_payload=["metadata"], with_vectors=False)
            for p in points:
                m = p.payload.get("metadata", {})
                d = docs.setdefault(m.get("doc_id"), {"doc_id": m.get("doc_id"), "name": m.get("source"),
                                                      "chunks": 0, "tables": 0})
                d["chunks"] += 1
                d["tables"] += bool(m.get("has_table"))
            if offset is None:
                break
    return sorted(docs.values(), key=lambda d: str(d["name"]).lower())


def main(args: list[str]):
    folder = Path(next((a for a in args if not a.startswith("--")), "data"))
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        sys.exit(f"no supported files in {folder} ({', '.join(sorted(SUPPORTED))})")
    if "--reset" in args:
        # delete the folder, not the collection: on Windows local Qdrant can't remove the locked
        # storage file, and a recreated collection silently reloads the old points
        from qdrant_client import QdrantClient
        try:  # check the lock first so a running server never loses half its files
            QdrantClient(path=QDRANT_PATH).close()
        except RuntimeError:
            sys.exit(f"{QDRANT_PATH} is in use; stop the server before --reset")
        shutil.rmtree(QDRANT_PATH, ignore_errors=True)
    vs = store()
    for f in files:
        data = f.read_bytes()
        if exists(vs, doc_id(data)):
            print(f"{f.name}: already indexed, skipped")
            continue
        n = tables = 0
        for docs, done, total in chunk_batches(f.name, data):
            if docs:
                add(vs, docs)
                n, tables = n + len(docs), tables + sum(d.metadata["has_table"] for d in docs)
            if total:
                print(f"  {f.name}: pages {done}/{total}", end="\r")
        print(f"{f.name}: {n} chunks, {tables} with tables" if n else f"{f.name}: no text found, skipped")
    vs.client.close()  # release the embedded Qdrant lock cleanly before exit


if __name__ == "__main__":
    main(sys.argv[1:])
