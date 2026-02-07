# index/retrieval_executor.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from retriever_interface import Chunk
from index.query_planner import QueryPlan
from index.filters import build_filters


def _has_season_filter(flt: Dict[str, Any]) -> bool:
    """
    Detect if a Pinecone filter already contains a season constraint.
    Handles nested {"$and": [...]} structures.
    """
    if not flt:
        return False
    if "season" in flt:
        return True
    if "$and" in flt and isinstance(flt["$and"], list):
        return any(_has_season_filter(x) for x in flt["$and"])
    return False


def _force_season_filter(flt: Dict[str, Any], season: int) -> Dict[str, Any]:
    """
    Add season filter only if not already present.
    """
    if _has_season_filter(flt):
        return flt
    if not flt:
        return {"season": {"$eq": season}}
    return {"$and": [flt, {"season": {"$eq": season}}]}


def _merge_balanced(per_season: Dict[int, List[Chunk]], top_k: int) -> List[Chunk]:
    """
    Round-robin merge across seasons, preserving per-season rank.
    Dedupes by chunk id.
    """
    out: List[Chunk] = []
    seen = set()

    seasons = list(per_season.keys())
    pointers = {s: 0 for s in seasons}

    while len(out) < top_k:
        progressed = False
        for s in seasons:
            i = pointers[s]
            lst = per_season[s]

            # skip seen ids
            while i < len(lst) and lst[i].id in seen:
                i += 1

            if i < len(lst):
                out.append(lst[i])
                seen.add(lst[i].id)
                pointers[s] = i + 1
                progressed = True
                if len(out) >= top_k:
                    break
            else:
                pointers[s] = i

        if not progressed:
            break

    return out


def execute_plan(
    *,
    retriever,
    plan: QueryPlan,
    base_query: str,
    recall_k: int,
    top_k: int,
    tenant: str = "fia",
) -> Tuple[List[Chunk], Dict[str, Any]]:
    """
    Execute a QueryPlan and return (final_chunks, debug_info).

    Notes:
    - SINGLE with no seasons: one retrieval call
    - SINGLE with 1 season: one retrieval call, season enforced
    - COMPARE with N seasons: retrieval per season, balanced merge
    """
    debug: Dict[str, Any] = {
        "mode": plan.mode,
        "seasons": plan.seasons,
        "per_season_counts": {},
    }

    # -----------------------
    # SINGLE (no seasons)
    # -----------------------
    if plan.mode == "single" and not plan.seasons:
        flt = build_filters(base_query, tenant=tenant)
        chunks = retriever.retrieve(base_query, recall_k=recall_k, filters=flt)
        debug["filters"] = flt
        debug["total"] = len(chunks)
        return chunks[:top_k], debug

    # -----------------------
    # SINGLE-SEASON
    # -----------------------
    if plan.mode == "single" and plan.subqueries:
        sq = plan.subqueries[0]
        flt = build_filters(sq.query, tenant=tenant)
        flt = _force_season_filter(flt, sq.season)

        chunks = retriever.retrieve(sq.query, recall_k=recall_k, filters=flt)
        debug["filters"] = flt
        debug["total"] = len(chunks)
        return chunks[:top_k], debug

    # -----------------------
    # COMPARE (N seasons)
    # -----------------------
    seasons = plan.seasons or []
    if not seasons:
        flt = build_filters(base_query, tenant=tenant)
        chunks = retriever.retrieve(base_query, recall_k=recall_k, filters=flt)
        debug["filters"] = flt
        debug["total"] = len(chunks)
        return chunks[:top_k], debug

    # Split recall budget across seasons
    per_season_recall = max(6, recall_k // max(1, len(seasons)))
    debug["per_season_recall"] = per_season_recall

    per_season_chunks: Dict[int, List[Chunk]] = {}

    for sq in plan.subqueries:
        # Build base filters (series, doc_type, regulation_type, article_refs, tenant, etc.)
        flt = build_filters(sq.query, tenant=tenant)

        # Ensure season filter exists exactly once
        flt = _force_season_filter(flt, sq.season)

        chunks = retriever.retrieve(sq.query, recall_k=per_season_recall, filters=flt)
        per_season_chunks[sq.season] = chunks
        debug["per_season_counts"][sq.season] = len(chunks)

    merged = _merge_balanced(per_season_chunks, top_k=top_k)
    debug["total"] = len(merged)
    return merged, debug
