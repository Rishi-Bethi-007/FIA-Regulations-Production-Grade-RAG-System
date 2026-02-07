# index/build_index.py
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

from config import (
    DATASET_NAME,
    CHUNKER,
    CHUNK_SIZE,
    OVERLAP,
    OVERLAP_SENTENCES,
    PDF_DIR,
    DOCSTORE_PATH,
    PINECONE_NAMESPACE,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    PINECONE_HOST,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBED_DIM,
    METRIC,
)

from index.pdf_loader import load_pdf_pages
from index.metadata_infer import infer_metadata
from embeddings.embedder import embed_texts

from index.docstore_sqlite import SQLiteDocStore
from index.pinecone_store import PineconeStore

from chunking.sentence_aware import chunk as sentence_chunk
from chunking.overlap import chunk as overlap_chunk


ARTICLE_EXPLICIT_RE = re.compile(
    r"\b(?:article|art\.?)\s*(\d{1,3}(?:\.\d{1,3})?)\b",
    re.IGNORECASE
)
ARTICLE_DOTTED_RE = re.compile(r"\b(\d{1,3}\.\d{1,3})\b")


def extract_article_refs(text: str) -> List[str]:
    if not text:
        return []

    refs: List[str] = []

    # explicit first
    for m in ARTICLE_EXPLICIT_RE.finditer(text):
        r = m.group(1)
        if r not in refs:
            refs.append(r)

    # dotted next
    for m in ARTICLE_DOTTED_RE.finditer(text):
        r = m.group(1)
        if r not in refs:
            refs.append(r)

    return refs[:20]


def stable_doc_id(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def file_version_hash(pdf_path: Path) -> str:
    st = pdf_path.stat()
    key = f"{pdf_path.name}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def chunk_text(page_text: str) -> List[str]:
    if CHUNKER == "sentence":
        return sentence_chunk(page_text, chunk_size=CHUNK_SIZE, overlap_sentences=OVERLAP_SENTENCES)
    elif CHUNKER == "overlap":
        return overlap_chunk(page_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    else:
        raise ValueError(f"Unknown CHUNKER={CHUNKER}. Use 'sentence' or 'overlap'.")


def build_index_from_pdfs(pdf_dir: str):
    pdf_dir_path = Path(pdf_dir)
    pages = load_pdf_pages(pdf_dir)

    docstore = SQLiteDocStore(DOCSTORE_PATH)

    store = PineconeStore(
        api_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX,
        dimension=EMBED_DIM,
        metric=METRIC,
        cloud=PINECONE_CLOUD,
        region=PINECONE_REGION,
        host=PINECONE_HOST,
    )
    store.ensure_index()

    all_ids: List[str] = []
    all_texts: List[str] = []
    all_metas: List[Dict[str, Any]] = []
    doc_rows: List[Tuple[str, str, Dict[str, Any]]] = []

    docs_seen = set()

    for p in pages:
        source = p["source"]
        doc_id = stable_doc_id(source)

        pdf_path = pdf_dir_path / source
        if not pdf_path.exists():
            continue

        docs_seen.add(source)

        # doc-level metadata (now includes series + doc_type + regulation_type etc.)
        doc_meta = infer_metadata(pdf_path, dataset_name=DATASET_NAME)
        doc_meta = dict(doc_meta)
        doc_meta["doc_title"] = pdf_path.name
        doc_meta["version_hash"] = file_version_hash(pdf_path)

        chunks = chunk_text(p["text"])

        for ci, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}-p{p['page']}-c{ci}"

            base_meta = {
                "doc_id": doc_id,
                "source": source,
                "page": p["page"],
                "chunk_index": ci,
                "chunker": CHUNKER,
                "chunk_size": CHUNK_SIZE,
                "overlap": OVERLAP if CHUNKER == "overlap" else None,
                "overlap_sentences": OVERLAP_SENTENCES if CHUNKER == "sentence" else None,
                "chunk_id": chunk_id,
            }

            meta = {**doc_meta, **base_meta}

            # add article refs (Pinecone-safe: list of strings)
            refs = extract_article_refs(chunk)
            if refs:
                meta["article_refs"] = refs
                meta["article_primary"] = refs[0]

            meta = drop_none(meta)  # remove nulls (Pinecone rejects)

            # DocStore gets the text
            doc_rows.append((chunk_id, chunk, meta))

            # Pinecone gets vectors + metadata (no text)
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metas.append(meta)

    # write docstore first
    docstore.put_many(doc_rows)

    # embed + upsert
    batch_size = 96
    for i in range(0, len(all_texts), batch_size):
        texts_b = all_texts[i:i + batch_size]
        ids_b = all_ids[i:i + batch_size]
        metas_b = all_metas[i:i + batch_size]

        embeds_b = embed_texts(texts_b)

        vectors = [
            {"id": cid, "values": vec, "metadata": meta}
            for cid, vec, meta in zip(ids_b, embeds_b, metas_b)
        ]
        store.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)

    print(f"Indexed {len(all_texts)} chunks from {len(docs_seen)} PDFs")
    print(f"namespace={PINECONE_NAMESPACE} | docstore={DOCSTORE_PATH}")
    if store.host:
        print(f"PINECONE_HOST={store.host}")


if __name__ == "__main__":
    build_index_from_pdfs(PDF_DIR)
