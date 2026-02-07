# index/filters.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

MIN_SEASON = 2018
MAX_SEASON = 2026

YEAR_RE = re.compile(r"\b(20(?:1[8-9]|2[0-6]))\b")

REG_TYPE_KEYWORDS = {
    "sporting": ["sporting", "sporting regulations", "sporting regs", "sport regs"],
    "technical": ["technical", "technical regulations", "technical regs", "tech regs"],
    "operational": ["operational", "operational regulations", "operational regs", "operations"],
}

SERIES_KEYWORDS = {
    "f1": ["f1", "formula 1", "formula_1"],
    "f2": ["f2", "formula 2", "formula_2"],
    "f3": ["f3", "formula 3", "formula_3"],
}

# ✅ Explicit-only article mention in the USER QUERY
# This prevents accidental capture of years/page numbers/etc.
ARTICLE_EXPLICIT_RE = re.compile(
    r"\b(?:article|art\.?)\s*(\d{1,3}(?:\.\d{1,3})?)\b",
    re.IGNORECASE
)


def _detect_season(q: str) -> Optional[int]:
    m = YEAR_RE.search(q)
    if not m:
        return None
    yr = int(m.group(1))
    return yr if MIN_SEASON <= yr <= MAX_SEASON else None


def _detect_regulation_type(q: str) -> Optional[str]:
    ql = q.lower()
    for canon, kws in REG_TYPE_KEYWORDS.items():
        if any(kw in ql for kw in kws):
            return canon
    return None


def _detect_series(q: str) -> Optional[str]:
    ql = q.lower()
    found = []
    for series, kws in SERIES_KEYWORDS.items():
        if any(kw in ql for kw in kws):
            found.append(series)
    if len(found) == 1:
        return found[0]
    return None


def _detect_article_explicit(q: str) -> Optional[str]:
    m = ARTICLE_EXPLICIT_RE.search(q)
    return m.group(1) if m else None


def build_filters(query: str, *, tenant: str = "fia") -> Dict[str, Any]:
    clauses: List[Dict[str, Any]] = []

    # Always isolate tenant
    clauses.append({"tenant": {"$eq": tenant}})

    # Series scope
    series = _detect_series(query)
    if series:
        clauses.append({"doc_type": {"$eq": f"fia_{series}_regulations"}})
    else:
        clauses.append(
            {"doc_type": {"$in": ["fia_f1_regulations", "fia_f2_regulations", "fia_f3_regulations"]}}
        )

    season = _detect_season(query)
    if season is not None:
        clauses.append({"season": {"$eq": season}})

    reg_type = _detect_regulation_type(query)
    if reg_type is not None:
        clauses.append({"regulation_type": {"$eq": reg_type}})

    # ✅ Only apply article filter when explicitly mentioned
    article = _detect_article_explicit(query)
    if article:
        clauses.append({"article_refs": {"$in": [article]}})

    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
