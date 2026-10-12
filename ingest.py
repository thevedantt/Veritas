"""Parse PDFs with Docling (layout + table structure), chunk by document structure, index into Qdrant
as a hybrid collection (Vertex dense vectors + BM25 sparse vectors).

Usage: python ingest.py data/
"""
import sys
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from agent import COLLECTION, QDRANT_PATH, embeddings, sparse_embeddings


def chunk_pdf(converter, chunker, pdf: Path) -> list[Document]:
    dl = converter.convert(pdf).document
    docs = []
    for c in chunker.chunk(dl_doc=dl):
        prov = c.meta.doc_items[0].prov if c.meta.doc_items else []
        docs.append(Document(
            page_content=chunker.contextualize(c),  # prepends section headings; tables serialized row by row
            metadata={"source": pdf.name, "page": prov[0].page_no if prov else None,
                      "headings": " > ".join(c.meta.headings or []),
                      "has_table": any(i.label == "table" for i in c.meta.doc_items)}))
    return docs


def main(folder: str):
    converter, chunker = DocumentConverter(), HybridChunker()
    docs = []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        chunks = chunk_pdf(converter, chunker, pdf)
        print(f"{pdf.name}: {len(chunks)} chunks, {sum(d.metadata['has_table'] for d in chunks)} with tables")
        docs += chunks
    if not docs:
        sys.exit(f"no PDFs found in {folder}")
    QdrantVectorStore.from_documents(
        docs, embedding=embeddings(), sparse_embedding=sparse_embeddings(), retrieval_mode=RetrievalMode.HYBRID,
        collection_name=COLLECTION, path=QDRANT_PATH, force_recreate=True)
    print(f"indexed {len(docs)} chunks into {QDRANT_PATH}/{COLLECTION}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
