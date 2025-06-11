# The Batch – Multimodal RAG System

## Project Overview

**The Batch RAG system** is a retrieval-augmented generation application that lets users query *The Batch* newsletter (published by deeplearning.ai) using both text and image content. It scrapes past newsletter issues (HTML content and images) and builds a knowledge base so that a question-answering chat interface can provide answers with citations and relevant images. The system is designed to offer conversational, source-grounded responses with contextual illustrations using the original newsletter's content.

## Architecture and Processing Stages

### Stage 1: HTML & Image Scraping

* Downloads all issues from The Batch website (`src/batch_scraper/fetch_all.py`)
* Saves HTML files under `data/raw/issues/`, and all image assets under `data/raw/images/`
* Polite crawler with custom User-Agent and 1.5s delay between requests
* Idempotent: skips issues and images already downloaded
* Supports filtering by date (`--since`) or range of issue numbers
* Uses deterministic image filenames via MD5 hash for stable references

### Stage 2: HTML to JSON Normalization

* Parses each HTML file and extracts structured records
* Differentiates between "Intro" and "News" sections using regex and structural heuristics
* Extracts text, titles, dates (via meta tag, `<time>` or regex), and local image references with alt text
* Ignores irrelevant sections (e.g. Sponsors) using skip-patterns
* Outputs one JSON per article or intro letter into `data/processed/batch_articles.jsonl`

### Stage 3: Embedding & Vector Indexing

* Converts article text and image content into embeddings
* Uses SentenceTransformers (MiniLM-L6-v2) for text
* Uses OpenCLIP ViT-B/32 for image embeddings (optional)
* Stores FAISS indexes to `data/index/` alongside NumPy and Pickle metadata
* Two text indexes (article-level and chunk-level) allow for fast coarse/fine retrieval

### Runtime: Hybrid Retrieval and Generation

* `src/rag/retrieval.py` retrieves top articles using hybrid score (cosine + recency)
* Applies iterative Gemini preview loop to refine relevance selection
* `src/rag/generator.py` generates the answer with inline citations, based on retrieved context
* `src/rag/images.py` selects relevant images by comparing query with image alt-text (via CLIP)
* Streamlit UI (`ui_app.py`) enables interactive chat with embedded images and clickable citations
* Incorporates recent turns of chat history to support follow-up questions and contextual continuity

## Technology and Model Justification

* **SentenceTransformer (MiniLM)**: Fast, compact 384-dim embeddings ideal for short-form articles
* **OpenCLIP (ViT-B/32)**: Efficient multimodal vector space, used for alt-text-to-query similarity
* **FAISS**: Optimized vector search with index + ID mapping; fast and scalable
* **Google Gemini API**: Handles long context, flexible response generation with citation patterns
* **Streamlit**: Easy to deploy UI, session state supports conversational memory
* **Prompt strategies**: Structured prompt with citation instructions ensures fact-grounded output
* **Ethical scraping**: Custom User-Agent and 1.5s delay; skips files already on disk

## Repository Structure

```
project_root/
├── data/
│   ├── raw/
│   │   ├── issues/        # HTML files
│   │   └── images/        # Downloaded assets
│   ├── processed/
│   │   └── batch_articles.jsonl
│   └── index/             # FAISS indexes + .npy + .pkl
├── src/
│   ├── batch_scraper/     # Stage 1
│   ├── preprocessing/     # Stage 2
│   ├── indexing/          # Stage 3
│   └── rag/               # Runtime: retrieval, LLM, image matching
├── ui_app.py              # Streamlit UI
├── requirements.txt       # Dependencies
└── Makefile               # Automation commands
```

## Setup Instructions

### 1. Prerequisites

* Python 3.11+
* Google Gemini API key

### 2. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
```

### 3. Run Pipeline Manually

```bash
# Stage 1: Scrape raw content
python -m src.batch_scraper.fetch_all

# Stage 2: Normalize into structured JSONL
python -m src.preprocessing.build_json

# Stage 3: Build vector indexes
python -m src.indexing.build_index

# Launch Streamlit UI
streamlit run ui_app.py
```

### 4. Or Use Makefile

```bash
make scrape-now
make parse
make index
make ui
```

## Usage

* Ask a question in the chat (e.g. "What did The Batch say about GPT-4?")
* System retrieves, cites articles (e.g. \[issue-123\_news\_1])
* Gemini generates a narrative with links and images inline
* Session history is included in the prompt for improved continuity on follow-ups

## Notes and Limitations

* **LLM errors**: Handled gracefully (e.g. Gemini content filter fallback)
* **Recency bias**: Retrieval favors recent articles via a tunable alpha weight
* **Prompt grounding**: Answers cite sources using inline `[issue-xxx_xx]` tags that resolve to URLs
* **Extensible**: Easy to adapt to other newsletters or corpora with similar structure

---

Built by Erik Savchyn
