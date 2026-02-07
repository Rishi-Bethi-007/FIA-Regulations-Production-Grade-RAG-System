'''
Docstring for rerank.reranker

A cross-encoder reranker reads the query and each candidate chunk together in a transformer model and outputs a relevance score, 
giving much higher accuracy than embeddings but at higher computational cost, 
which is why it is used only after vector retrieval on a small candidate set.

'''

# rerank/cross_encoder_reranker.py
from __future__ import annotations

from typing import List, Any
from dataclasses import replace

from sentence_transformers import CrossEncoder

from config import CROSS_ENCODER_MODEL, CROSS_ENCODER_BATCH_SIZE, RERANK_MAX_CHARS


_ce: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _ce
    if _ce is None:
        # device is auto-selected by sentence-transformers/torch
        _ce = CrossEncoder(CROSS_ENCODER_MODEL)
    return _ce


def _snippet(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    return t[:max_chars]


def rerank_chunks_cross_encoder(
    *,
    query: str,
    chunks: List[Any],  # retriever_interface.Chunk
    top_k: int,
) -> List[Any]:
    """
    Cross-encoder reranker. Returns chunks sorted by CE score (desc), truncated to top_k.
    """
    if not chunks:
        return []
    if len(chunks) <= top_k:
        return chunks

    model = _get_model()

    pairs = [(query, _snippet(c.text, RERANK_MAX_CHARS)) for c in chunks]
    scores = model.predict(pairs, batch_size=CROSS_ENCODER_BATCH_SIZE)

    # Sort by score descending
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: float(x[1]), reverse=True)

    # Optional: if you want to keep the CE score, you can attach it in metadata
    out = []
    for c, s in scored[:top_k]:
        md = dict(getattr(c, "metadata", {}) or {})
        md["rerank_score"] = float(s)
        # Chunk is a dataclass in your interface, so replace is safe if it's that object
        try:
            out.append(replace(c, metadata=md))
        except Exception:
            # fallback if Chunk isn't dataclass for any reason
            c.metadata = md
            out.append(c)

    return out
