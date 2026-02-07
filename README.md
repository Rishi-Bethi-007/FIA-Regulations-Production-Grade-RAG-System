
### Design goals
- ⚡ **Fast** — Redis caching, limited reranking
- 🎯 **Accurate** — metadata filtering + reranker
- 📊 **Measurable** — evaluation first
- 🏗️ **Production-ready** — clean abstractions

---

## ✨ Key Features

### 🔹 Retrieval & Search
- Pinecone vector database (scales to **10k+ documents**)
- Metadata-aware retrieval:
  - Season (`2018–2026`)
  - Series (`F1 / F2 / F3`)
  - Regulation type (sporting / technical / operational)
  - Article references (e.g. `12.3`)
- Namespace strategy for safe re-indexing

### 🔹 Performance
- Redis caching:
  - Embedding cache
  - Retrieval cache
- Measured cache hit rates
- Cross-encoder reranking  
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

### 🔹 Safety & Quality
- Input guardrails (prompt injection detection)
- Context guardrails (tenant isolation, empty chunk filtering)
- Output guardrails (citation enforcement)
- Faithfulness evaluation using an LLM judge

### 🔹 Evaluation
- Latency metrics: **mean / p50 / p95**
- Cache hit-rate reporting
- Faithfulness scoring against retrieved evidence

---

## 📂 Repository Structure

chunking/ Sentence-aware & overlap chunkers
data/ FIA PDFs (2018–2026)
embeddings/ OpenAI embedding wrapper
index/ Ingestion, metadata, Pinecone adapter
rag/ End-to-end RAG pipeline
rerank/ Cross-encoder reranker
guardrails/ Input / context / output guards
eval/ Latency & faithfulness evaluation
scripts/ Runnable entry points
cache/ Redis helpers (keys, client)
config.py Centralized configuration
retriever_interface.py DB-agnostic retriever interface


Each component is **intentionally decoupled**, so:
- vector DBs can be swapped
- rerankers can be replaced
- evaluation is independent of retrieval logic

---

## 🚀 Getting Started (Run Locally)

### 1️⃣ Clone & install dependencies

````
git clone <your-repo-url>
cd FIA_PROD_RAG

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
````
### 2️⃣ Set environment variables
Create a .env file in the project root:
````
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
````
Make sure Redis is running:

redis-server
### 3️⃣ Index the documents
python -m scripts.build_index
This step:

Loads & cleans PDFs

Chunks documents

Infers metadata

Stores text in SQLite

Stores vectors + metadata in Pinecone

### 4️⃣ Ask a question
python -m scripts.test_rag
Example query:

“What does Article 12.3 say about parc fermé in 2026?”

Output includes:

Answer

Citations (document, page, article)

Debug info (cache hits, planner mode)

### 5️⃣ Run evaluation
python -m scripts.run_eval
Generates:

eval_report.json

Latency statistics

Cache hit rates

Faithfulness score

### 📊 Sample Performance (Local)
Latency (p50): ~2.3s

Latency (p95): ~6–7s

Embedding cache hit rate: ~1.0

Retrieval cache hit rate: ~1.0

Faithfulness score: ~0.8–0.9 (strict judge)

### 🧩 Design Decisions & Trade-offs
Pinecone vs FAISS → managed scaling & metadata filters

SQLite DocStore → simple, fast text hydration

Cross-encoder rerank → deterministic, cheaper than LLM rerank

Redis caching → biggest latency reduction lever

Evaluation-first → improvements are measured, not guessed

### 🔮 Future Extensions (Optional)
UI (Streamlit / React)

Authentication & multi-tenant access

Streaming responses

Feedback-driven retrieval tuning

Production deployment (AWS / GCP)

