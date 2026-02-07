# eval/faithfulness_judge.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from openai import OpenAI

from config import OPENAI_API_KEY
import os

JUDGE_MODEL = os.getenv("JUDGE_MODEL", os.getenv("GEN_MODEL", "gpt-4.1-mini"))

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def judge_faithfulness(*, answer: str, cited_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Judge whether the answer is fully supported by the evidence text.
    Returns:
      { faithful: bool, issues: [..], confidence: float }
    """
    evidence = []
    for c in cited_chunks:
        evidence.append({
            "ref": c.get("ref"),
            "source": c.get("source"),
            "page": c.get("page"),
            "text": c.get("text", "")[:1200],
        })

    prompt = {
        "task": (
            "Decide if the answer is fully supported by the evidence.\n"
            "If any claim is not supported, mark faithful=false and list the unsupported claims.\n"
            "Be strict: if evidence does not directly support it, it's unsupported."
        ),
        "answer": answer,
        "evidence": evidence,
        "output_schema": {"faithful": "bool", "issues": "array of strings", "confidence": "0..1"},
    }

    client = _get_client()
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a strict evaluator. Output JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    )

    txt = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return {"faithful": False, "issues": ["Judge returned non-JSON output."], "confidence": 0.0}

    try:
        data = json.loads(m.group(0))
        return {
            "faithful": bool(data.get("faithful", False)),
            "issues": list(data.get("issues", []))[:10],
            "confidence": float(data.get("confidence", 0.0)),
        }
    except Exception:
        return {"faithful": False, "issues": ["Judge JSON parse failed."], "confidence": 0.0}
