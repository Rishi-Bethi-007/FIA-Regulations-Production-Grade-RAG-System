# eval/run_eval.py
from __future__ import annotations

import json
import time
from statistics import mean
from typing import Any, Dict, List, Optional

from rag.rag_pipeline import run_rag
from eval.faithfulness_judge import judge_faithfulness


def _percentile(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    vals = sorted(vals)
    idx = int(round((len(vals) - 1) * p))
    return vals[idx]


def run_eval(*, dataset_path: str, out_path: str, retriever):
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    rows: List[Dict[str, Any]] = []
    lat: List[float] = []

    embed_hits = 0
    retr_hits = 0
    cache_rows = 0

    faithful_flags: List[int] = []
    judged = 0

    for item in items:
        q = item["query"]

        t0 = time.time()
        out = run_rag(query=q, retriever=retriever, tenant="fia")
        ms = (time.time() - t0) * 1000
        lat.append(ms)

        debug = out.get("debug", {}) or {}

        # cache metrics
        cache = debug.get("cache")
        if isinstance(cache, dict):
            cache_rows += 1
            if cache.get("embed_cache_hit"):
                embed_hits += 1
            if cache.get("retrieval_cache_hit"):
                retr_hits += 1

        judge = {"faithful": None, "issues": [], "confidence": None}
        if not debug.get("refusal"):
            evidence = debug.get("judge_evidence", [])
            if evidence:
                judge = judge_faithfulness(answer=out["answer"], cited_chunks=evidence)
                judged += 1
                if judge.get("faithful") is True:
                    faithful_flags.append(1)
                elif judge.get("faithful") is False:
                    faithful_flags.append(0)

        row = {
            "query": q,
            "latency_ms": ms,
            "answer": out["answer"],
            "citations": out.get("citations", []),
            "debug": debug,
            "judge": judge,
        }
        rows.append(row)

    report = {
        "n": len(rows),
        "latency_ms": {
            "mean": mean(lat) if lat else None,
            "p50": _percentile(lat, 0.50),
            "p95": _percentile(lat, 0.95),
        },
        "cache": {
            "scored": cache_rows,
            "embed_hit_rate": (embed_hits / cache_rows) if cache_rows else None,
            "retrieval_hit_rate": (retr_hits / cache_rows) if cache_rows else None,
        },
        "faithfulness": {
            "judged": judged,
            "rate": (sum(faithful_flags) / len(faithful_flags)) if faithful_flags else None,
        },
        "rows": rows,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("✅ wrote", out_path)
    print("Latency mean:", report["latency_ms"]["mean"])
    print("Latency p50:", report["latency_ms"]["p50"])
    print("Latency p95:", report["latency_ms"]["p95"])
    print("Cache embed hit rate:", report["cache"]["embed_hit_rate"])
    print("Cache retrieval hit rate:", report["cache"]["retrieval_hit_rate"])
    print("Faithfulness rate:", report["faithfulness"]["rate"])
