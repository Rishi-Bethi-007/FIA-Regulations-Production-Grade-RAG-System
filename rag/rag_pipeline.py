# rag/rag_pipeline.py
from __future__ import annotations

from typing import Any, Dict, List

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    GEN_MODEL,
    RECALL_K,
    TOP_K,
    RERANK_ENABLED,
    RERANK_STRATEGY,
)

from retriever_interface import Chunk
from index.query_planner import plan_query
from index.retrieval_executor import execute_plan
from guardrails.guards import input_guard, context_guard, output_guard

from rerank.cross_encoder_reranker import rerank_chunks_cross_encoder


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _build_context(chunks: List[Chunk]) -> str:
    parts: List[str] = []
    for i, c in enumerate(chunks, start=1):
        md = c.metadata or {}
        header = (
            f"[{i}] source={md.get('source')} page={md.get('page')} "
            f"season={md.get('season')} series={md.get('series')} "
            f"type={md.get('regulation_type')} article={md.get('article_primary')} "
            f"chunk_id={c.id}"
        )
        parts.append(header)
        parts.append(c.text.strip())
        parts.append("")
    return "\n".join(parts).strip()


def _format_citations(chunks: List[Chunk]) -> List[Dict[str, Any]]:
    cites: List[Dict[str, Any]] = []
    for i, c in enumerate(chunks, start=1):
        md = c.metadata or {}
        cites.append(
            {
                "ref": i,
                "chunk_id": c.id,
                "source": md.get("source"),
                "doc_title": md.get("doc_title"),
                "season": md.get("season"),
                "series": md.get("series"),
                "regulation_type": md.get("regulation_type"),
                "page": md.get("page"),
                "article_primary": md.get("article_primary"),
            }
        )
    return cites


def _judge_evidence(chunks: List[Chunk], max_chars: int = 1200) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(chunks, start=1):
        md = c.metadata or {}
        out.append(
            {
                "ref": i,
                "source": md.get("source"),
                "page": md.get("page"),
                "season": md.get("season"),
                "series": md.get("series"),
                "regulation_type": md.get("regulation_type"),
                "article_primary": md.get("article_primary"),
                "text": (c.text or "")[:max_chars],
            }
        )
    return out


def run_rag(*, query: str, retriever, tenant: str = "fia") -> Dict[str, Any]:
    g = input_guard(query)
    if not g.ok:
        return {"answer": f"Refused: {g.reason}", "citations": [], "debug": {"refusal": True, "reason": g.reason}}

    plan = plan_query(query)

    # retrieve more than TOP_K before reranking
    pre_rerank_k = min(24, max(TOP_K * 4, 16))  # usually 24


    chunks, dbg = execute_plan(
        retriever=retriever,
        plan=plan,
        base_query=query,
        recall_k=RECALL_K,
        top_k=pre_rerank_k,
        tenant=tenant,
    )

    # attach cache metrics if retriever exposes them
    retr_dbg = getattr(retriever, "last_debug", {})
    dbg = dict(dbg)
    if retr_dbg:
        dbg["cache"] = retr_dbg

    chunks = context_guard(chunks, tenant=tenant)

    if RERANK_ENABLED and RERANK_STRATEGY == "cross_encoder":
        chunks = rerank_chunks_cross_encoder(query=query, chunks=chunks, top_k=TOP_K)
    else:
        chunks = chunks[:TOP_K]

    citations = _format_citations(chunks)
    context = _build_context(chunks)

    client = _get_client()
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "You are a regulations assistant.\n"
                "Use ONLY the provided context.\n"
                "If a detail is not explicitly supported, say: \"I don't know based on the provided documents.\"\n"
                "Write the answer as bullet points.\n"
                "Every bullet MUST end with at least one citation like [1] or [2].\n"
                "Do not include any uncited claims.\n"
                )},
            {"role": "user", "content": f"Question:\n{query}\n\nContext:\n{context}\n\nAnswer:"},
        ],
    )
    answer = (resp.choices[0].message.content or "").strip()

    g2 = output_guard(answer, citations)
    if not g2.ok:
        dbg["refusal"] = True
        dbg["reason"] = g2.reason
        return {"answer": f"Refused: {g2.reason}", "citations": citations, "debug": dbg}

    # include evidence text for judging in eval
    dbg["judge_evidence"] = _judge_evidence(chunks)

    return {"answer": answer, "citations": citations, "debug": dbg}
