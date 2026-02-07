from typing import List
import re

# Split into "paragraph-like" blocks first (PDF text often loses real newlines,
# but sometimes you still get them; we handle both cases).
_PAR_SPLIT = re.compile(r"\n{2,}|\r\n{2,}")

# More careful sentence boundary:
# - split after .!? when followed by whitespace + a capital/quote/number
# - avoids splitting inside "Art. 42.3" and some abbreviations better than naive
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

# Clause markers common in regulations: "1)", "1.", "(a)", "a)", etc.
# We will try to split on these when they appear as standalone boundaries.
_CLAUSE_SPLIT = re.compile(
    r"(?:(?<=\s)|^)(?:\(?[a-zA-Z]\)|[0-9]{1,3}\.|[0-9]{1,3}\))\s+"
)

def _normalize_whitespace(text: str) -> str:
    # Keep single newlines if present; collapse other whitespace
    text = text.replace("\r\n", "\n")
    # Collapse spaces/tabs but keep newlines
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def _split_into_units(text: str) -> List[str]:
    """
    Try to split text into "meaning units" (best effort):
    1) Paragraph-like blocks (if double newlines exist)
    2) Clause-like splits (numbered/bulleted) inside each block
    3) Sentence split fallback
    """
    text = _normalize_whitespace(text)
    if not text:
        return []

    # If we have paragraph boundaries, use them
    blocks = _PAR_SPLIT.split(text)
    units: List[str] = []

    for b in blocks:
        b = b.strip()
        if not b:
            continue

        # If clause markers exist, split by them while keeping content intact.
        # _CLAUSE_SPLIT finds the marker positions; we reconstruct chunks.
        parts = _CLAUSE_SPLIT.split(b)
        parts = [p.strip() for p in parts if p and p.strip()]

        if len(parts) >= 2:
            # Clause split worked; treat each clause as a unit
            units.extend(parts)
        else:
            # Fallback: sentence split
            sents = _SENT_SPLIT.split(b)
            sents = [s.strip() for s in sents if s and s.strip()]
            units.extend(sents)

    return units

def chunk(text: str, chunk_size: int = 900, overlap_sentences: int = 1) -> List[str]:
    """
    Regulations-friendly, sentence/clause-aware chunking.

    Strategy:
    - Split text into meaning units (paragraphs -> clauses -> sentences)
    - Pack units into chunks up to chunk_size characters
    - Optional small overlap measured in *units* (sentences/clauses), not characters

    Why this is better for regs:
    - Keeps rule clauses intact more often
    - Reduces repetition compared to heavy char overlap
    - Produces chunks that are easier to cite and compare
    """
    units = _split_into_units(text)
    if not units:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append(" ".join(current).strip())
        current = []
        current_len = 0

    for u in units:
        u_len = len(u)
        # If one unit is huge, hard-split it
        if u_len > chunk_size:
            flush()
            for i in range(0, u_len, chunk_size):
                part = u[i:i + chunk_size].strip()
                if part:
                    chunks.append(part)
            continue

        # If it fits, add to current chunk
        if current_len + u_len + (1 if current else 0) <= chunk_size:
            current.append(u)
            current_len += u_len + (1 if current_len > 0 else 0)
        else:
            # flush current chunk
            flush()
            # start new chunk with this unit
            current.append(u)
            current_len = u_len

    flush()

    # Optional overlap: add last N units from previous chunk into next chunk
    if overlap_sentences > 0 and len(chunks) > 1:
        overlapped: List[str] = []
        prev_units: List[str] = []

        # Rebuild using units again to apply overlap cleanly.
        # We'll do a simpler approach: overlap by tail text of previous chunk split by SENT_SPLIT.
        # (Good enough for now; keeps code simple.)
        overlapped.append(chunks[0])
        for i in range(1, len(chunks)):
            tail_units = _SENT_SPLIT.split(chunks[i - 1])
            tail_units = [t.strip() for t in tail_units if t.strip()]
            tail = " ".join(tail_units[-overlap_sentences:]) if tail_units else ""
            combined = (tail + " " + chunks[i]).strip() if tail else chunks[i]
            overlapped.append(combined)

        return overlapped

    return chunks
