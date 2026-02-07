import time
from config import RECALL_K
from index.pinecone_store import PineconeStore
from index.pinecone_adapter import PineconeRetriever
from index.filters import build_filters
from config import (
    PINECONE_API_KEY, PINECONE_INDEX, EMBED_DIM, METRIC,
    PINECONE_CLOUD, PINECONE_REGION, PINECONE_HOST
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

    r = PineconeRetriever(pinecone_store=store)

    q = "What does Article 12.3 say about parc fermé in 2026?"
    filters = build_filters(q, tenant="fia") # next step: build_filters(q)

    for i in range(2):
        t0 = time.time()
        chunks = r.retrieve(q, recall_k=RECALL_K, filters=filters)
        ms = (time.time() - t0) * 1000
        print(f"Run {i+1}: {ms:.1f} ms | chunks={len(chunks)}")
        if chunks:
            print("Top chunk id:", chunks[0].id)
            print("Top chunk score:", chunks[0].score)
            print("Top chunk text preview:", chunks[0].text[:200].replace("\n"," "))
        print("-"*60)

if __name__ == "__main__":
    main()

