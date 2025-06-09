"""Stage4 – RAG powered by **GoogleGemini2.5Pro (experimental)**
=================================================================
Cloud‑only retrieval‑augmented answering that calls Google’s newest
**Gemini2.5Pro** endpoint via the official `google‑generativeai` SDK.
No OpenAI / OpenRouter / local LLM downloads.

Free quota (as of June2025)
---------------------------
* ~1million input characters per day
* ~20 requests per minute
Perfect for testing. When you upgrade to a paid GoogleCloud project you
just swap the API key.

Quick start
-----------
```bash
pip install google-generativeai faiss-cpu sentence-transformers numpy tqdm pillow

# 1– create a key at https://aistudio.google.com/app/apikey
export GEMINI_API_KEY="AIza..."

# 2– run a query
python -m src.rag.qa "What did Andrew Ng announce in the first issue of The Batch?"
```

Optional env‑vars
-----------------
| Variable        | Default                        | Purpose                               |
|-----------------|--------------------------------|---------------------------------------|
| `GEMINI_API_KEY`| —                              | *Required* AIStudio key               |
| `GEMINI_MODEL`  | `gemini-1.5-pro-latest`        | Gemini variant (2.5Pro uses this tag) |
| `RAG_TOP_K`     | `6`                            | Retrieval depth                       |
| `RAG_TEMP`      | `0.2`                          | Sampling temperature                  |
| `RAG_MAX_OUT`   | `512`                          | Max output tokens                     |
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer  # type: ignore

try:
    import faiss  # type: ignore
except ImportError as err:  # pragma: no cover
    raise SystemExit("faiss-cpu missing – `pip install faiss-cpu`") from err

try:
    import google.generativeai as genai  # type: ignore
except ImportError as err:  # pragma: no cover
    raise SystemExit("google-generativeai missing – `pip install google-generativeai`") from err

# ---------------------------------------------------------------------------
# Paths & constants ----------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]  # project root
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
TEXT_INDEX_PATH = INDEX_DIR / "text.index"
META_PATH = INDEX_DIR / "text_meta.pkl"
ARTICLES_JSONL = DATA_DIR / "processed" / "batch_articles.jsonl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("Set GEMINI_API_KEY env var. Create one at https://aistudio.google.com/app/apikey")

genai.configure(api_key=API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest")

TOP_K = int(os.getenv("RAG_TOP_K", 6))
TEMP = float(os.getenv("RAG_TEMP", 0.2))
MAX_OUT = int(os.getenv("RAG_MAX_OUT", 512))
SNIP = 350  # characters per snippet

PROMPT_TMPL = (
    "You are a precise AI assistant. Use only the numbered CONTEXT to answer the QUESTION. "
    "Quote phrases and cite the source number like [3]. If the answer is not in context, say 'I don't know'.\n\n"
    "CONTEXT:\n{ctx}\n\nQUESTION: {q}\nANSWER:"
)

# ---------------------------------------------------------------------------
# Lazy singletons ------------------------------------------------------------

_text_index = _meta = _articles = None
_embedder: SentenceTransformer | None = None
_gemini = None

# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------

def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_index():
    global _text_index, _meta  # noqa: PLW0603
    if _text_index is None:
        _text_index = faiss.read_index(str(TEXT_INDEX_PATH))
        _meta = pickle.load(META_PATH.open("rb"))


def _load_articles():
    global _articles  # noqa: PLW0603
    if _articles is None:
        _articles = {}
        with ARTICLES_JSONL.open() as fh:
            for line in fh:
                rec = json.loads(line)
                _articles[rec["id"]] = rec


def _load_embedder(dev: str):
    global _embedder  # noqa: PLW0603
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL, device=dev)
    return _embedder


def _gemini():
    global _gemini  # noqa: PLW0603
    if _gemini is None:
        _gemini = genai.GenerativeModel(GEMINI_MODEL)
    return _gemini

# ---------------------------------------------------------------------------
# Core API -------------------------------------------------------------------

def answer(question: str, *, k: int = TOP_K) -> Dict:
    """Return answer + sources + first image (if any)."""
    dev = _device()
    _load_index(); _load_articles()
    embedder = _load_embedder(dev)

    qv = embedder.encode(question, normalize_embeddings=True).astype("float32")
    _, idxs = _text_index.search(qv.reshape(1, -1), k)

    ctx_lines: List[str] = []
    cites: List[dict] = []
    images: List[dict] = []

    for rank, idx in enumerate(idxs[0], 1):
        m = _meta[idx]
        art = _articles[m["parent_id"]]
        snippet = m.get("text") or art["text"][:SNIP]
        ctx_lines.append(f"[{rank}] {snippet}")
        cites.append({"title": m["title"], "date": m["date"], "chunk": m["chunk"], "rank": rank})
        if art.get("images"):
            images.append(art["images"][0])

    prompt = PROMPT_TMPL.format(ctx="\n".join(ctx_lines), q=question)

    gem = _gemini()
    resp = gem.generate_content(prompt, generation_config={
        "temperature": TEMP,
        "top_p": 0.95,
        "max_output_tokens": MAX_OUT,
    })
    answer_text = resp.text.strip()

    return {"answer": answer_text, "sources": cites, "images": images}


if __name__ == "__main__":
    import sys, json
    q = sys.argv[1] if len(sys.argv) > 1 else "What did Andrew Ng announce in the very first issue of The Batch?"
    print(json.dumps(answer(q), indent=2, ensure_ascii=False))
