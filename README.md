# 📘 FIA Regulations – Production-Grade RAG System

An **industry-grade Retrieval-Augmented Generation (RAG) system** built over **FIA Formula 1 / Formula 2 / Formula 3 regulations (2018–2026)**, designed with **scalability, low latency, and evaluation** in mind.

This project goes **far beyond a demo RAG** and mirrors how real production AI systems are built.

---

## 🚀 Why this project is different

Most RAG projects stop at *“it works”*.  
This system focuses on:

- **Performance** (Redis caching, rerank limits)
- **Accuracy** (metadata filters + cross-encoder reranking)
- **Safety** (guardrails)
- **Measurement** (latency & faithfulness evaluation)

**Technologies used**
- Pinecone (vector database)
- Redis (caching)
- OpenAI embeddings & generation
- Cross-encoder reranker (no LLM rerank)
- SQLite DocStore
- Evaluation harness

---

## 🔍 What problem does this solve?

Regulatory documents like FIA rules are:

- Long, dense, and frequently updated  
- Spread across **seasons, series, and revisions**  
- Hard to search precisely  
  > e.g. *“Article 12.3 in 2026 F1 Sporting Regulations”*

This system allows users to:

- Ask **natural language questions**
- Retrieve **exact regulation clauses**
- Get **grounded answers with citations**
- Maintain **low latency at scale**

---

## 🧠 High-Level Architecture

