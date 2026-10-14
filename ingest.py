"""Parse documents with Docling (layout, tables, OCR), chunk by document structure, and add them to the
Qdrant hybrid collection (Vertex dense vectors + BM25 sparse vectors). Also the document-library
operations the web server uses. A document's id is the hash of its bytes, so re-adding a file is a no-op.

python ingest.py data/            # add new files from a folder; already-indexed files are skipped
python ingest.py --reset data/    # wipe the collection and rebuild it from the folder
"""
import hashlib
import shutil
import sys
from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document
from qdrant_client import models

from agent import COLLECTION, DB_LOCK, QDRANT_PATH, store

SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".csv",
             ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_tools = None


def tools():
    """Docling converter + chunker, loaded once (model loading takes a few seconds)."""
    global _tools
    if _tools is None:
        from docling.chunking import HybridChunker
        from docling.document_converter import DocumentConverter
        _tools = DocumentConverter(), HybridChunker()
    return _tools


def doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def chunk(name: str, data: bytes) -> list[Document]:
    from docling.datamodel.base_models import DocumentStream
    converter, chunker = tools()
    dl = converter.convert(DocumentStream(name=name, stream=BytesIO(data))).document
    did, docs = doc_id(data), []
    for c in chunker.chunk(dl_doc=dl):
        prov = c.meta.doc_items[0].prov if c.meta.doc_items else []
        docs.append(Document(
            page_content=chunker.contextualize(c),  # prepends section headings; tables serialized row by row
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


def delete(vs, did: str) -> bool:
    with DB_LOCK:
        if not vs.client.count(COLLECTION, count_filter=_of(did), exact=True).count:
            return False
        vs.client.delete(COLLECTION, points_selector=models.FilterSelector(filter=_of(did)))
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
        docs = chunk(f.name, data)
        if not docs:
            print(f"{f.name}: no text found, skipped")
            continue
        add(vs, docs)
        print(f"{f.name}: {len(docs)} chunks, {sum(d.metadata['has_table'] for d in docs)} with tables")
    vs.client.close()  # release the embedded Qdrant lock cleanly before exit


if __name__ == "__main__":
    main(sys.argv[1:])
