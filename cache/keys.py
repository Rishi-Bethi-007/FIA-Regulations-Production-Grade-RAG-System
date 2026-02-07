# cache/keys.py
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

def _stable_json(x: Any) -> str:
    # deterministic JSON: sorted keys, no whitespace
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def embedding_key(query: str, model: str) -> str:
    h = hashlib.sha1(query.strip().encode("utf-8")).hexdigest()
    return f"emb:{model}:{h}"

def retrieval_key(*, embedding: str, namespace: str, filters: Dict[str, Any], recall_k: int) -> str:
    """
    embedding: should be a short stable string (e.g., sha1 hash), NOT raw floats.
    """
    f = _stable_json(filters or {})
    base = f"{namespace}|k={recall_k}|emb={embedding}|flt={f}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()
    return f"ret:{h}"
