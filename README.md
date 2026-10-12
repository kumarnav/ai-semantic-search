# 🤖 AI Semantic Search & Vector Embedding Engine

A lightweight, local Natural Language Processing (NLP) utility that performs semantic vector search using transformer models and cosine similarity. Designed to demonstrate foundational concepts used in modern RAG (Retrieval-Augmented Generation) pipelines.

## 🚀 Key Features
- **Vector Embeddings:** Transforms raw text into dense numerical vector representations using Hugging Face transformers (`all-MiniLM-L6-v2`).
- **Semantic Understanding:** Matches user queries to relevant text chunks based on contextual meaning rather than rigid keyword filtering.
- **Local Execution:** Runs entirely offline with zero external API dependencies or costs.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Libraries:** `sentence-transformers`, `torch`, `numpy`

## 📦 Setup & Execution

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt