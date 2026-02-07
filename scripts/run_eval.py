# scripts/run_eval.py
from index.pinecone_store import PineconeStore
from index.pinecone_adapter import PineconeRetriever
from eval.run_eval import run_eval

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX,
    EMBED_DIM,
    METRIC,
    PINECONE_CLOUD,
    PINECONE_REGION,
    PINECONE_HOST,
)

def make_retriever():
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
    return PineconeRetriever(pinecone_store=store)

if __name__ == "__main__":
    retriever = make_retriever()
    run_eval(
        dataset_path="gold_rag_eval.json",
        out_path="eval_report.json",
        retriever=retriever,
    )
