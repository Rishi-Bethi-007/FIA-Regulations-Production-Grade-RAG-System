'''
A small security + quality filter that runs before retrieval, after retrieval, and after generation.

It protects against:
        Prompt injection
        Cross-tenant data leaks
        Garbage chunks
        Empty / citation-less answers

Big Picture:
        User Query
            ↓ input_guard
        Retriever → chunks
            ↓ context_guard
        LLM → answer + citations
            ↓ output_guard
        Return to user

'''


# guardrails/guards.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from retriever_interface import Chunk

# Lightweight prompt-injection heuristics
_INJECTION_PATTERNS = [
    r"\bignore\b.*\b(previous|prior|above)\b.*\b(instructions|rules)\b",
    r"\byou are now\b",
    r"\b(system|developer)\s+prompt\b",
    r"\breveal\b.*\b(prompt|instructions|policy|rules)\b",
    r"\bjailbreak\b",
    r"\bdo anything now\b",
]


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str = ""


def input_guard(user_query: str) -> GuardResult:
    q = (user_query or "").strip()
    if not q:
        return GuardResult(False, "Empty query.")
    ql = q.lower()
    for pat in _INJECTION_PATTERNS:
        if re.search(pat, ql):
            return GuardResult(False, "Potential prompt injection attempt.")
    return GuardResult(True)


def context_guard(chunks: List[Chunk], *, tenant: str) -> List[Chunk]:
    """
    Enforce tenant isolation + drop empty/garbage chunks.
    """
    out: List[Chunk] = []
    for c in chunks:
        if not c.text or len(c.text.strip()) < 30:
            continue
        md = c.metadata or {}
        # enforce tenant
        if md.get("tenant") is not None and md.get("tenant") != tenant:
            continue
        out.append(c)
    return out


def output_guard(answer: str, citations: List[Dict[str, Any]]) -> GuardResult:
    """
    Minimal production rule:
    - must return an answer string
    - must include citations (we’ll do deeper faithfulness later)
    """
    a = (answer or "").strip()
    if not a:
        return GuardResult(False, "Empty answer.")
    if not citations:
        return GuardResult(False, "No citations produced.")
    return GuardResult(True)
