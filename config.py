# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------
# Env loading
# -----------------------------
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# -----------------------------
# OpenAI
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4.1-mini")

# Vector dimension (matches text-embedding-3-small)
# If you ever change embedding model, you MUST update dimension + index.
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))
METRIC = os.getenv("METRIC", "cosine")  # Pinecone index metric

# Dataset name (useful for multi-dataset projects later)
DATASET_NAME = os.getenv("DATASET_NAME", "fia")

# -----------------------------
# Inputs / storage
# -----------------------------
PDF_DIR = os.getenv("PDF_DIR", str(ROOT / "data" / "fia_pdfs"))

# DocStore (chunk text store)
DOCSTORE_PATH = os.getenv("DOCSTORE_PATH", str(ROOT / "docstore.sqlite"))

# -----------------------------
# PDF cleaning (header/footer removal)
# -----------------------------
CLEAN_HEADERS_FOOTERS = os.getenv("CLEAN_HEADERS_FOOTERS", "1") == "1"
HF_MIN_PAGE_FRACTION = float(os.getenv("HF_MIN_PAGE_FRACTION", "0.6"))  # >=60% pages
HF_MIN_LINE_LEN = int(os.getenv("HF_MIN_LINE_LEN", "8"))
HF_MAX_LINE_LEN = int(os.getenv("HF_MAX_LINE_LEN", "180"))
HF_MAX_REMOVE_PER_PAGE = int(os.getenv("HF_MAX_REMOVE_PER_PAGE", "6"))  # safety cap

# -----------------------------
# Chunking
# -----------------------------
CHUNKER = os.getenv("CHUNKER", "sentence")  # "sentence" or "overlap"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))

# overlap chunker (chars)
OVERLAP = int(os.getenv("OVERLAP", "100"))

# sentence-aware chunker (units)
OVERLAP_SENTENCES = int(os.getenv("OVERLAP_SENTENCES", "1"))

# -----------------------------
# Retrieval + reranking knobs
# -----------------------------
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") == "1"
RECALL_K = int(os.getenv("RECALL_K", "24"))
TOP_K = int(os.getenv("TOP_K", "6"))

RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4.1-mini")
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "450"))
# Reranking (cross-encoder)
RERANK_STRATEGY = os.getenv("RERANK_STRATEGY", "cross_encoder")  # "cross_encoder" or "none"
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CROSS_ENCODER_BATCH_SIZE = int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "16"))


# -----------------------------
# Pinecone (Vector DB)
# -----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "fia-rag-1536")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", f"{DATASET_NAME}_{CHUNKER}")

# Recommended: use host-based Index(host=...) in prod for speed/stability.
# If empty, your code can fallback to describe_index() to fetch host once.
PINECONE_HOST = os.getenv("PINECONE_HOST", "").strip() or None

# Serverless location (EU close to Sweden)
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# -----------------------------
# Redis (Cache)
# -----------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# -----------------------------
# Cache controls
# -----------------------------
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "1") == "1"
CACHE_EMBEDDINGS = os.getenv("CACHE_EMBEDDINGS", "1") == "1"
CACHE_RETRIEVAL = os.getenv("CACHE_RETRIEVAL", "1") == "1"


# -----------------------------
# Optional: Keep old Chroma settings for fallback / A-B testing
# -----------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
_default_collection = os.getenv("CHROMA_COLLECTION", f"{DATASET_NAME}_{CHUNKER}")
CHROMA_COLLECTION = _default_collection
