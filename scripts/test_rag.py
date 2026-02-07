# scripts/test_rag.py
from index.pinecone_store import PineconeStore
from index.pinecone_adapter import PineconeRetriever
from rag.rag_pipeline import run_rag

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX,
    EMBED_DIM,
    METRIC,
    PINECONE_CLOUD,
    PINECONE_REGION,
    PINECONE_HOST,
)

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

    q = "What are the formation lap rules in 2024 Formula 1 sporting regulations?"
    out = run_rag(query=q, retriever=retriever, tenant="fia")
    print("\nANSWER:\n", out["answer"])
    print("\nCITATIONS:")
    for c in out["citations"]:
        print(c)
    print("\nDEBUG(cache):", out.get("debug", {}).get("cache"))
    print("\nDEBUG(plan):", {k: out.get("debug", {}).get(k) for k in ["mode", "seasons", "total"]})

if __name__ == "__main__":
    main()
