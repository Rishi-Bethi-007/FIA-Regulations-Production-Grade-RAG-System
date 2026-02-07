# index/pdf_loader.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple
import re
from collections import Counter

from pypdf import PdfReader

from config import (
    CLEAN_HEADERS_FOOTERS,
    HF_MIN_PAGE_FRACTION,
    HF_MIN_LINE_LEN,
    HF_MAX_LINE_LEN,
    HF_MAX_REMOVE_PER_PAGE,
)

# Common footer/header patterns (helpful for FIA-style docs)
_RE_PAGE_X_OF_Y = re.compile(r"\bpage\s*\d+\s*(of|/)\s*\d+\b", re.IGNORECASE)
_RE_JUST_NUMBERS = re.compile(r"^\d{1,4}\s*/\s*\d{1,4}$")  # e.g. "31/120"
_RE_DATE = re.compile(r"\b\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}\b")  # e.g. "13 February 2024"
_RE_ISSUE = re.compile(r"\bissue\s*\d{1,2}\b", re.IGNORECASE)


def _norm_line(line: str) -> str:
    """Normalize a line for matching (strip + collapse spaces)."""
    line = line.replace("\u00a0", " ")
    line = re.sub(r"[ \t]+", " ", line).strip()
    return line


def _extract_lines(page_text: str) -> List[str]:
    """Split page text into normalized lines."""
    if not page_text:
        return []
    # Keep line structure: pypdf extract_text uses \n for line breaks
    raw_lines = page_text.replace("\r\n", "\n").split("\n")
    lines = []
    for ln in raw_lines:
        n = _norm_line(ln)
        if n:
            lines.append(n)
    return lines


def _is_boilerplate_candidate(line: str) -> bool:
    """Heuristics to avoid removing meaningful content."""
    if not (HF_MIN_LINE_LEN <= len(line) <= HF_MAX_LINE_LEN):
        return False

    # Obvious boilerplate patterns (very common in PDFs)
    if _RE_PAGE_X_OF_Y.search(line):
        return True
    if _RE_JUST_NUMBERS.match(line):
        return True
    if _RE_ISSUE.search(line) and len(line) <= 40:
        return True

    # Copyright / FIA lines often repeat; allow those to be removed if frequent
    if "fia" in line.lower() and len(line) <= 80:
        return True
    if "copyright" in line.lower() or "©" in line:
        return True

    # Otherwise, only remove if frequency-based detector flags it
    return False


def _find_repeated_lines(pages_lines: List[List[str]], min_fraction: float) -> set[str]:
    """
    Identify lines that appear on many pages of the same PDF.
    We count line presence per page (not total occurrences).
    """
    if not pages_lines:
        return set()

    page_count = len(pages_lines)
    min_pages = max(2, int(page_count * min_fraction))

    presence = Counter()
    for lines in pages_lines:
        for ln in set(lines):
            presence[ln] += 1

    repeated = {ln for ln, c in presence.items() if c >= min_pages}

    # Safety: don't remove extremely long repeated lines (rare, but could be headings)
    repeated = {ln for ln in repeated if HF_MIN_LINE_LEN <= len(ln) <= HF_MAX_LINE_LEN}
    return repeated


def _remove_boilerplate_from_lines(lines: List[str], repeated: set[str]) -> Tuple[str, int]:
    """
    Remove detected boilerplate lines.
    Returns cleaned text and number of removed lines.
    Safety cap prevents deleting too much from a page.
    """
    removed = 0
    kept: List[str] = []

    for ln in lines:
        # Remove if:
        # - it is in the repeated set OR matches boilerplate regex patterns
        should_remove = (ln in repeated) or _is_boilerplate_candidate(ln)

        if should_remove and removed < HF_MAX_REMOVE_PER_PAGE:
            removed += 1
            continue

        kept.append(ln)

    cleaned = " ".join(kept).strip()
    # Normalize whitespace across the page text
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, removed


def load_pdf_pages(pdf_dir: str) -> List[Dict]:
    """
    Returns list of dicts: {"text": str, "source": filename, "page": int}

    If CLEAN_HEADERS_FOOTERS is enabled:
      - detects repeated header/footer lines per PDF and removes them.
    """
    pdf_dir_path = Path(pdf_dir)
    out_pages: List[Dict] = []

    for pdf_path in sorted(pdf_dir_path.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))

        # Pass 1: extract per-page lines
        pages_lines: List[List[str]] = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            lines = _extract_lines(txt)
            pages_lines.append(lines)

        repeated: set[str] = set()
        if CLEAN_HEADERS_FOOTERS:
            repeated = _find_repeated_lines(pages_lines, min_fraction=HF_MIN_PAGE_FRACTION)

        # Pass 2: build final cleaned page records
        for i, lines in enumerate(pages_lines):
            if CLEAN_HEADERS_FOOTERS:
                cleaned_text, _removed = _remove_boilerplate_from_lines(lines, repeated)
            else:
                cleaned_text = re.sub(r"\s+", " ", " ".join(lines)).strip()

            if cleaned_text:
                out_pages.append(
                    {
                        "text": cleaned_text,
                        "source": pdf_path.name,
                        "page": i + 1,  # 1-indexed
                    }
                )

    return out_pages
