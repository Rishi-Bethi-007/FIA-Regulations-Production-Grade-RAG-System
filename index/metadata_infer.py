# index/metadata_infer.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, Optional

_RE_YEAR = re.compile(r"(20\d{2})")
_RE_ISSUE = re.compile(r"\bissue[\s_-]*(\d{1,2})\b", re.IGNORECASE)
_RE_PUBLISHED_ISO = re.compile(r"\b(20\d{2})[-_/](\d{1,2})[-_/](\d{1,2})\b")


def _infer_series(name_lower: str, parts_lower: list[str]) -> str:
    """
    Infer series from filename/folders.
    Defaults to f1 if unknown.
    """
    hay = " ".join([name_lower] + parts_lower)

    # Common patterns in FIA filenames
    if "formula 2" in hay or "formula_2" in hay or "f2" in hay:
        return "f2"
    if "formula 3" in hay or "formula_3" in hay or "f3" in hay:
        return "f3"

    # Default
    return "f1"


def infer_metadata(pdf_path: Path, dataset_name: str = "fia") -> Dict[str, Any]:
    """
    Infer doc-level metadata from filename + folder structure.

    Returns Pinecone-friendly metadata (no nested dicts, no None).
    """
    name = pdf_path.name
    low = name.lower()
    parts = [p.lower() for p in pdf_path.parts]

    meta: Dict[str, Any] = {}

    meta["dataset"] = dataset_name
    meta["tenant"] = "fia"

    # Series + doc_type
    series = _infer_series(low, parts)
    meta["series"] = series
    meta["doc_type"] = f"fia_{series}_regulations"  # fia_f1_regulations / fia_f2_regulations / fia_f3_regulations

    # Season = max year in filename
    years = [int(y) for y in _RE_YEAR.findall(name)]
    if years:
        meta["season"] = max(years)

    # Regulation type
    if "sporting" in low or "sporting" in parts:
        meta["regulation_type"] = "sporting"
    elif "technical" in low or "technical" in parts:
        meta["regulation_type"] = "technical"
    elif "operational" in low or "operational" in parts:
        meta["regulation_type"] = "operational"

    # Issue
    m = _RE_ISSUE.search(name)
    if m:
        meta["issue"] = int(m.group(1))

    # Published date
    m = _RE_PUBLISHED_ISO.search(name)
    if m:
        yyyy, mm, dd = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
        meta["published"] = f"{yyyy}-{mm}-{dd}"

    return meta
