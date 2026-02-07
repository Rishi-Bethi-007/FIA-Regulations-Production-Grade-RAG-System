# embeddings/embedder.py
from __future__ import annotations

from typing import List, Sequence
from openai import OpenAI

from config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBED_DIM


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """
    Embed a batch of texts using OpenAI embeddings.

    Returns:
        list of vectors (list[float]) in the same order as inputs.
    """
    if not texts:
        return []

    client = _get_client()
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=list(texts),
    )
    vecs = [d.embedding for d in resp.data]

    # Optional: sanity check (dimension mismatch is a common bug)
    for i, v in enumerate(vecs):
        if len(v) != EMBED_DIM:
            raise RuntimeError(
                f"Embedding dim mismatch at i={i}: got {len(v)} expected {EMBED_DIM}. "
                f"Check EMBEDDING_MODEL/EMBED_DIM and Pinecone index dimension."
            )

    return vecs


def embed_query(query: str) -> List[float]:
    """
    Convenience wrapper: embed a single query string.
    """
    vecs = embed_texts([query])
    return vecs[0]
