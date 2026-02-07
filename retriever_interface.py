# retriever_interface.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float  # similarity score (higher = better)


class Retriever:
    def retrieve(
        self,
        query: str,
        *,
        recall_k: int,
        filters: Dict[str, Any],
    ) -> List[Chunk]:
        raise NotImplementedError
