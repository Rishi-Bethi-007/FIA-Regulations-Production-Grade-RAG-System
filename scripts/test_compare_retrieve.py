# scripts/test_compare_retrieve.py
import time

from config import RECALL_K, TOP_K
from index.pinecone_store import PineconeStore
from index.pinecone_adapter import PineconeRetriever
from config import (
    PINECONE_API_KEY, PINECONE_INDEX, EMBED_DIM, METRIC,
    PINECONE_CLOUD, PINECONE_REGION, PINECONE_HOST
)

from index.query_planner import plan_query
from index.retrieval_executor import execute_plan


def main():
    store = PineconeStore(
        api_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX,
        dimension=EMBED_DIM,
        metric=METRIC,
        cloud=PINECONE_CLOUD,
        region=PINECONE_REGION,
        host=PINECONE_HOST,
    )
    store.ensure_index()

    retriever = PineconeRetriever(pinecone_store=store)

    q = "What does Article 12.3 say?"
    plan = plan_query(q)
    print("Plan:", plan.mode, plan.seasons)

    for run in range(2):
        t0 = time.time()
        chunks, dbg = execute_plan(
            retriever=retriever,
            plan=plan,
            base_query=q,
            recall_k=RECALL_K,
            top_k=TOP_K,
            tenant="fia",
        )
        ms = (time.time() - t0) * 1000
        print(f"\nRun {run+1}: {ms:.1f} ms | final_chunks={len(chunks)} | debug={dbg}")

        for i, c in enumerate(chunks[:TOP_K], start=1):
            md = c.metadata or {}
            print(
                f"{i}. score={c.score:.3f} "
                f"season={md.get('season')} series={md.get('series')} "
                f"type={md.get('regulation_type')} article={md.get('article_primary')} "
                f"source={md.get('source')}"
            )
        print("-" * 80)


if __name__ == "__main__":
    main()
