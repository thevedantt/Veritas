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

from agent import COLLECTION, DB_LOCK, QDRANT_PATH, store
from cache import library_changed

PAGE_BATCH = 4  # PDFs are parsed and indexed a few pages at a time, so early pages are searchable within seconds
SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".csv",
             ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_tools = None


def tools():
    """Docling converter + chunker, loaded once (model loading takes a few seconds)."""
    global _tools
    if _tools is None:
        from docling.chunking import HybridChunker
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.transforms.chunker.hierarchical_chunker import (ChunkingDocSerializer,
                                                                          ChunkingSerializerProvider)
        from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

        class MarkdownTables(ChunkingSerializerProvider):
            # Markdown rows keep each table row on one line with its column order; the default
            # "header = value" triplets scrambled rows of OCR'd tables across many tiny chunks
            def get_serializer(self, doc):
                return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer())

        # GPU when PyTorch has CUDA (layout, tables and OCR models), all CPU cores otherwise. OCR runs only on
        # image regions, so digital PDFs skip it. TABLE_MODE=fast trades some table accuracy for speed.
        pdf = PdfPipelineOptions(accelerator_options=AcceleratorOptions(device=AcceleratorDevice.AUTO,
                                                                        num_threads=os.cpu_count() or 4))
        if os.getenv("TABLE_MODE", "accurate") == "fast":
            pdf.table_structure_options.mode = TableFormerMode.FAST
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf)})
        _tools = converter, HybridChunker(serializer_provider=MarkdownTables())
    return _tools


def compact(text: str) -> str:
    """Drop Markdown cell padding (runs of spaces / dashes), which otherwise costs embedding and LLM tokens."""
    return re.sub(r"-{4,}", "---", re.sub(r"[ \t]{2,}", " ", text))


def doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def page_count(name: str, data: bytes) -> int | None:
    if not name.lower().endswith(".pdf"):
        return None
    import pypdfium2
    return len(pypdfium2.PdfDocument(data))


def chunk_batches(name: str, data: bytes):
    """Yield (chunks, pages_done, pages_total) as a document is parsed: PDFs PAGE_BATCH pages at a time,
    other formats in one pass (pages_total None)."""
    total = page_count(name, data)
    if not total:
        yield chunk(name, data), None, None
        return
    for first in range(1, total + 1, PAGE_BATCH):
        last = min(first + PAGE_BATCH - 1, total)
        yield chunk(name, data, (first, last)), last, total


def chunk(name: str, data: bytes, pages: tuple[int, int] | None = None) -> list[Document]:
    from docling.datamodel.base_models import DocumentStream
    converter, chunker = tools()
    kwargs = {"page_range": pages} if pages else {}
    dl = converter.convert(DocumentStream(name=name, stream=BytesIO(data)), **kwargs).document
    did, docs = doc_id(data), []
    for c in chunker.chunk(dl_doc=dl):
        prov = c.meta.doc_items[0].prov if c.meta.doc_items else []
        docs.append(Document(
            page_content=compact(chunker.contextualize(c)),  # prepends section headings; tables as Markdown rows
            metadata={"doc_id": did, "source": name, "page": prov[0].page_no if prov else None,
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
