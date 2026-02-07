# index/query_planner.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

MIN_SEASON = 2018
MAX_SEASON = 2026

YEAR_RE = re.compile(r"\b(20(?:1[8-9]|2[0-6]))\b")
RANGE_RE = re.compile(r"\bfrom\s+(20(?:1[8-9]|2[0-6]))\s+to\s+(20(?:1[8-9]|2[0-6]))\b", re.IGNORECASE)

COMPARE_HINTS = [
    "compare", "difference", "changes", "changed", "vs", "versus",
    "between", "from", "to", "across", "over the years"
]


@dataclass(frozen=True)
class SubQuery:
    season: int
    query: str


@dataclass(frozen=True)
class QueryPlan:
    mode: str  # "single" | "compare"
    seasons: List[int]
    subqueries: List[SubQuery]


def _unique_preserve_order(nums: List[int]) -> List[int]:
    out = []
    seen = set()
    for n in nums:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def extract_seasons(query: str) -> List[int]:
    q = query

    seasons: List[int] = []

    # ranges: "from 2021 to 2026"
    m = RANGE_RE.search(q)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        lo, hi = min(a, b), max(a, b)
        lo = max(lo, MIN_SEASON)
        hi = min(hi, MAX_SEASON)
        seasons.extend(list(range(lo, hi + 1)))

    # explicit years
    years = [int(y) for y in YEAR_RE.findall(q)]
    years = [y for y in years if MIN_SEASON <= y <= MAX_SEASON]
    seasons.extend(years)

    return _unique_preserve_order(seasons)


def is_compare_query(query: str, seasons: List[int]) -> bool:
    ql = query.lower()
    if len(seasons) >= 2:
        return True
    return any(h in ql for h in COMPARE_HINTS)


def make_subquery_text(base_query: str, season: int) -> str:
    """
    Force season into the subquery text so the embedding also reflects the season.
    We still also apply metadata filters for season.
    """
    return f"{base_query.strip()} (season {season})"


def plan_query(query: str) -> QueryPlan:
    seasons = extract_seasons(query)

    if not seasons:
        # no season detected -> single plan
        return QueryPlan(mode="single", seasons=[], subqueries=[])

    if is_compare_query(query, seasons) and len(seasons) >= 2:
        subs = [SubQuery(season=s, query=make_subquery_text(query, s)) for s in seasons]
        return QueryPlan(mode="compare", seasons=seasons, subqueries=subs)

    # one season only -> treat as single-season plan
    return QueryPlan(mode="single", seasons=seasons[:1], subqueries=[SubQuery(season=seasons[0], query=make_subquery_text(query, seasons[0]))])
