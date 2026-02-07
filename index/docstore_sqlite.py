# index/docstore_sqlite.py
from __future__ import annotations

import sqlite3
import json
from typing import Dict, Iterable, List, Optional, Tuple


class SQLiteDocStore:
    """
    Production-style doc store:
    - Vector DB (Pinecone) stores vectors + metadata pointers
    - DocStore stores chunk text keyed by chunk_id

    This implementation is:
    - persistent
    - fast enough for dev + single-node prod
    - easy to swap to Postgres later
    """

    def __init__(self, path: str):
        self.path = path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        # check_same_thread=False lets you reuse in threaded apps (FastAPI)
        return sqlite3.connect(self.path, check_same_thread=False)

    def _init_db(self) -> None:
        with self._conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    meta_json TEXT
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id)")
            con.commit()

    def put_many(self, rows: Iterable[Tuple[str, str, Optional[dict]]]) -> None:
        """
        rows: iterable of (chunk_id, text, meta_dict_or_none)
        """
        payload = [
            (cid, txt, json.dumps(meta or {}, ensure_ascii=False))
            for cid, txt, meta in rows
        ]
        with self._conn() as con:
            con.executemany(
                "INSERT OR REPLACE INTO chunks(chunk_id, text, meta_json) VALUES (?, ?, ?)",
                payload,
            )
            con.commit()

    def get_many(self, chunk_ids: List[str]) -> Dict[str, str]:
        """
        Returns a dict {chunk_id: text} for the requested ids.
        Missing ids are simply absent in the returned dict.
        """
        if not chunk_ids:
            return {}

        placeholders = ",".join(["?"] * len(chunk_ids))
        query = f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({placeholders})"

        with self._conn() as con:
            cur = con.execute(query, chunk_ids)
            return {cid: txt for cid, txt in cur.fetchall()}

    def get_one(self, chunk_id: str) -> Optional[str]:
        res = self.get_many([chunk_id])
        return res.get(chunk_id)
