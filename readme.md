# Polars RAG CodeMinds Hackathon 🚀

This repository contains a specialized **RAG (Retrieval-Augmented Generation) Pipeline** designed to generate high-quality synthetic training data for Polars code generation. It leverages the **ALBERT API** (sovereign infrastructure by Etalab) and grounds the **Ministral-3-8B** model in real Polars documentation.

## 📌 Project Overview

The goal of this project is to bridge the gap between natural language requests and efficient Polars code. By feeding scraped documentation into a two-stage RAG system, we ensure the generated code is syntactically correct, performant, and adheres to the **Eager API** standards.

### Core Tech Stack:
* **SLM**: `mistralai/Ministral-3-8B-Instruct-2512` (via Albert API)
* **Large LLM**: `openweight-large` (120B) for complex synthesis
* **Vector DB**: Local [ChromaDB](https://www.trychroma.com/) for persistence
* **Retriever**: BGE-based semantic search + `openweight-rerank`
* **Infrastructure**: ALBERT Cloud (Etalab)

---

## 🛠️ Pipeline Architecture

### 1. Data Migration & Ingestion
The pipeline starts by extracting indexed documentation from a local `chroma.sqlite3` instance and ingesting it into a sovereign ALBERT collection.
* **Metadata Cleaning**: Filters out empty values to ensure clean indexing.
* **Chunking Control**: Uses `disable_chunking: true` to maintain the integrity of pre-processed documentation snippets.

### 2. Two-Stage Retrieval (RAG)
To maximize accuracy, we use a reranking pattern:
1.  **Stage 1 (Semantic Search)**: Retrieves the top 40 candidate chunks from the collection using vector similarity.
2.  **Stage 2 (Reranking)**: Uses the `openweight-rerank` model to score those 40 candidates against the query, returning only the top `n` most relevant snippets to the model's context window.

### 3. Synthetic Data Generation
The system iterates through key Polars categories (joins, window functions, aggregations, etc.) to build a fine-tuning dataset.
* **Constraint Enforcement**: The system prompt strictly forbids `.lazy()`, `.collect()`, and Pandas `.apply()`.
* **Robust Extraction**: A regex-based JSON extractor handles OSS model "preamble" text to ensure valid data collection.

---

## 🚀 Setup & Usage

### Prerequisites
* Google Colab environment.
* `ALBERT_API_KEY` stored in