# index/pinecone_store.py
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec


class PineconeStore:
    """
    Thin wrapper around Pinecone operations:
    - ensure_index
    - get_host
    - upsert
    - query

    Note: Use Index(host=...) (recommended in production).
    If host is not provided, we fallback to describe_index() to find it.
    """

    def __init__(
        self,
        *,
        api_key: str,
        index_name: str,
        dimension: int,
        metric: str,
        cloud: str,
        region: str,
        host: Optional[str] = None,
    ):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self._host = host

    def ensure_index(self) -> None:
        existing = [i["name"] for i in self.pc.list_indexes().get("indexes", [])]
        if self.index_name in existing:
            # Optional sanity check: dimension mismatch is a common mistake
            desc = self.pc.describe_index(self.index_name)
            idx_dim = desc.get("dimension")
            if idx_dim is not None and int(idx_dim) != int(self.dimension):
                raise RuntimeError(
                    f"Index '{self.index_name}' dimension={idx_dim} but code expects {self.dimension}. "
                    f"Use a different index name or recreate the index with correct dimension."
                )
            return

        self.pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            spec=ServerlessSpec(cloud=self.cloud, region=self.region),
        )

        # Wait for readiness
        while True:
            desc = self.pc.describe_index(self.index_name)
            if desc.get("status", {}).get("state") == "Ready":
                break
            time.sleep(2)

    def get_host(self) -> str:
        desc = self.pc.describe_index(self.index_name)
        host = desc.get("host")
        if not host:
            raise RuntimeError("describe_index returned no host.")
        return host

    def index(self):
        if not self._host:
            self._host = self.get_host()
        return self.pc.Index(host=self._host)

    @property
    def host(self) -> Optional[str]:
        return self._host

    def upsert(self, *, vectors: List[Dict[str, Any]], namespace: str) -> None:
        self.index().upsert(vectors=vectors, namespace=namespace)

    def query(
        self,
        *,
        vector: List[float],
        top_k: int,
        namespace: str,
        flt: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        return self.index().query(
            namespace=namespace,
            vector=vector,
            top_k=top_k,
            include_metadata=include_metadata,
            filter=flt or {},
        )
