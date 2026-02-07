📘 FIA Regulations Production-Grade RAG System

An industry-grade Retrieval-Augmented Generation (RAG) system built over FIA Formula 1 / Formula 2 / Formula 3 regulations (2018–2026), designed with scalability, latency, and evaluation in mind.

This project goes beyond a demo RAG:

uses Pinecone for scalable vector search

Redis caching for real performance gains

cross-encoder reranking (not LLM rerank)

metadata-aware retrieval (season, article, series, regulation type)

guardrails + evaluation harness (latency & faithfulness)

🔍 What problem does this solve?

Regulatory documents (like FIA rules) are:

long, dense, and frequently updated

spread across seasons, series, and revisions

difficult to search precisely (e.g. “Article 12.3 in 2026 F1 Sporting Regulations”)

This system allows users to:

ask natural language questions

retrieve exact regulation clauses

get grounded answers with citations

while maintaining low latency at scale

🧠 High-Level Architecture
PDFs → Chunking → Embeddings → Pinecone (Vector DB)
                           ↘ SQLite DocStore
User Query
  → Planner (filters, compare logic)
  → Pinecone Retrieval (cached)
  → Cross-Encoder Reranker
  → Guardrails
  → LLM Answer Generation
  → Evaluation (latency + faithfulness)


Key design goals:

Fast (Redis caching, rerank limits)

Accurate (metadata filters + reranker)

Measurable (evaluation suite)

Production-ready (clear abstractions)

✨ Key Features
🔹 Retrieval & Search

Pinecone vector database (scales to 10k+ documents)

Metadata filtering:

season (2018–2026)

series (F1 / F2 / F3)

regulation type (sporting / technical / operational)

article references (e.g. 12.3)

Namespace strategy for safe re-indexing

🔹 Performance

Redis caching:

embedding cache

retrieval cache

Measured cache hit rates

Cross-encoder reranking (ms-marco-MiniLM) for precision

🔹 Safety & Quality

Input guardrails (prompt injection detection)

Context guardrails (tenant isolation, empty chunks)

Output guardrails (citation enforcement)

Faithfulness evaluation using an LLM judge

🔹 Evaluation

Latency metrics: mean / p50 / p95

Cache hit-rate reporting

Faithfulness scoring against retrieved evidence

📂 Repository Structure
chunking/          → sentence-aware & overlap chunkers
data/              → FIA PDFs (2018–2026)
embeddings/        → OpenAI embedding wrapper
index/             → ingestion, metadata, Pinecone adapter
rag/               → end-to-end RAG pipeline
rerank/            → cross-encoder reranker
guardrails/        → input / context / output guards
eval/              → latency + faithfulness evaluation
scripts/           → runnable entry points
cache/             → Redis helpers (keys, client)
config.py          → centralized configuration
retriever_interface.py → clean DB-agnostic retriever interface


Each component is intentionally decoupled so that:

vector DBs can be swapped

rerankers can be changed

evaluation is independent of retrieval logic

🚀 Getting Started (Run Locally)
1️⃣ Clone & install dependencies
git clone <your-repo-url>
cd FIA_PROD_RAG

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -r requirements.txt

2️⃣ Set environment variables

Create a .env file:

OPENAI_API_KEY=your_openai_key

PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=fia-regulations
PINECONE_NAMESPACE=fia_prod

CACHE_ENABLED=1
CACHE_EMBEDDINGS=1
CACHE_RETRIEVAL=1
REDIS_HOST=localhost
REDIS_PORT=6379

EMBEDDING_MODEL=text-embedding-3-small
GEN_MODEL=gpt-4.1-mini


Make sure Redis is running:

redis-server

3️⃣ Index the documents
python -m scripts.build_index


This will:

load & clean PDFs

chunk documents

infer metadata

store text in SQLite

store vectors + metadata in Pinecone

4️⃣ Ask a question
python -m scripts.test_rag


Example query:

“What does Article 12.3 say about parc fermé in 2026?”

Output includes:

answer

citations (document, page, article)

debug info (cache hits, planner mode)

5️⃣ Run evaluation
python -m scripts.run_eval


Generates:

eval_report.json

latency stats (mean / p50 / p95)

cache hit rates

faithfulness score

📊 Sample Performance (Local)

Latency (p50): ~2.3s

Latency (p95): ~6–7s

Embedding cache hit rate: ~1.0

Retrieval cache hit rate: ~1.0

Faithfulness score: ~0.8–0.9 (strict judge)

🧩 Design Decisions & Trade-offs

Pinecone vs FAISS → managed scaling, metadata filters

SQLite DocStore → simple, fast text hydration

Cross-encoder rerank → deterministic, cheaper than LLM rerank

Redis caching → largest latency reduction lever

Evaluation first → changes are measured, not guessed

🔮 Future Extensions (Optional)

UI (Streamlit / React)

Auth & multi-tenant access

Streaming responses

Feedback-driven retrieval tuning

Production deployment (AWS/GCP)

👤 About This Project

This project was built to reflect real AI Engineer work, not a demo:

system thinking

performance trade-offs

evaluation & iteration

production-style abstractions

It is suitable as a flagship portfolio project for AI / ML / Applied LLM Engineer roles.
