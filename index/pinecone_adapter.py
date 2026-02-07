# index/pinecone_adapter.py
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List

from retriever_interface import Retriever, Chunk
from embeddings.embedder import embed_query
from index.pinecone_store import PineconeStore
from index.docstore_sqlite import SQLiteDocStore

from cache.client import get_redis
from cache.keys import embedding_key, retrieval_key

from config import (
    CACHE_ENABLED,
    CACHE_EMBEDDINGS,
    CACHE_RETRIEVAL,
    EMBEDDING_MODEL,
    PINECONE_NAMESPACE,
    DOCSTORE_PATH,
)


def _hash_embedding(vec: List[float]) -> str:
    """
    Compact stable representation for cache key usage.
    """
    b = ",".join(f"{v:.6f}" for v in vec).encode("utf-8")
    return hashlib.sha1(b).hexdigest()


def _to_jsonable(obj: Any) -> Any:
    """
    Convert Pinecone SDK objects (e.g., QueryResponse) into plain dict/list primitives.
    Works across SDK versions.
    """
    if obj is None:
        return None

    # SDK helpers (varies by version)
    for attr in ("to_dict", "dict", "model_dump"):
        if hasattr(obj, attr) and callable(getattr(obj, attr)):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass

    # JSON export helpers
    for attr in ("to_json", "json"):
        if hasattr(obj, attr) and callable(getattr(obj, attr)):
            try:
                return json.loads(getattr(obj, attr)())
            except Exception:
                pass

    # Already JSONable primitives/containers
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}

    # Try vars()
    try:
        return _to_jsonable(vars(obj))
    except Exception:
        # Last resort
        return str(obj)


class PineconeRetriever(Retriever):
    """
    Retriever adapter that:
      - embeds query (optional Redis cache)
      - queries Pinecone (optional Redis cache)
      - hydrates text from SQLite DocStore
      - exposes per-call cache metrics via self.last_debug
    """

    def __init__(self, *, pinecone_store: PineconeStore):
        self.store = pinecone_store
        self.docstore = SQLiteDocStore(DOCSTORE_PATH)

        self.cache_enabled = CACHE_ENABLED
        self.cache_embeddings = CACHE_EMBEDDINGS
        self.cache_retrieval = CACHE_RETRIEVAL

        self.redis = get_redis() if self.cache_enabled else None

        # Last call metrics
        self._last_embed_cache_hit = False
        self._last_retrieval_cache_hit = False
        self.last_debug: Dict[str, Any] = {}

    # -------------------------
    # Internal helpers
    # -------------------------

    def _embed_with_cache(self, query: str) -> List[float]:
        self._last_embed_cache_hit = False

        if not self.cache_enabled or not self.cache_embeddings:
            return embed_query(query)

        key = embedding_key(query, EMBEDDING_MODEL)
        cached = self.redis.get(key)
        if cached:
            self._last_embed_cache_hit = True
            return json.loads(cached)

        vec = embed_query(query)
        self.redis.set(key, json.dumps(vec).encode("utf-8"), ex=7 * 24 * 3600)
        return vec

    def _retrieve_with_cache(
        self,
        *,
        embedding: List[float],
        recall_k: int,
        filters: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._last_retrieval_cache_hit = False

        if not self.cache_enabled or not self.cache_retrieval:
            # Convert to dict for downstream consistency
            res = self.store.query(
                vector=embedding,
                top_k=recall_k,
                namespace=PINECONE_NAMESPACE,
                flt=filters,
            )
            return _to_jsonable(res)

        emb_hash = _hash_embedding(embedding)
        key = retrieval_key(
            embedding=emb_hash,          # if your retrieval_key expects list, swap to `embedding=embedding`
            namespace=PINECONE_NAMESPACE,
            filters=filters,
            recall_k=recall_k,
        )

        cached = self.redis.get(key)
        if cached:
            self._last_retrieval_cache_hit = True
            return json.loads(cached)

        res = self.store.query(
            vector=embedding,
            top_k=recall_k,
            namespace=PINECONE_NAMESPACE,
            flt=filters,
        )

        res_json = _to_jsonable(res)
        self.redis.set(key, json.dumps(res_json).encode("utf-8"), ex=30 * 60)
        return res_json

    # -------------------------
    # Public API
    # -------------------------

    def retrieve(
        self,
        query: str,
        *,
        recall_k: int,
        filters: Dict[str, Any],
    ) -> List[Chunk]:
        t0 = time.time()

        embedding = self._embed_with_cache(query)
        res = self._retrieve_with_cache(
            embedding=embedding,
            recall_k=recall_k,
            filters=filters,
        )

        matches = res.get("matches", []) if isinstance(res, dict) else []
        chunk_ids = [m.get("id") for m in matches if isinstance(m, dict) and m.get("id")]

        texts = self.docstore.get_many(chunk_ids)

        chunks: List[Chunk] = []
        for m in matches:
            cid = m.get("id")
            if not cid:
                continue
            text = texts.get(cid)
            if not text:
                continue

            chunks.append(
                Chunk(
                    id=cid,
                    text=text,
                    metadata=m.get("metadata", {}) or {},
                    score=float(m.get("score", 0.0) or 0.0),
                )
            )

        self.last_debug = {
            "embed_cache_hit": bool(self._last_embed_cache_hit),
            "retrieval_cache_hit": bool(self._last_retrieval_cache_hit),
            "retrieval_ms": (time.time() - t0) * 1000.0,
            "returned": len(chunks),
        }

        return chunks
