"""
Stage 4 — Retrieval-Augmented QA with Gemini 2.5 Pro
===================================================

• Loads FAISS indexes if present; else falls back to NumPy matrices saved
  by Stage 3 — zero crashes on machines without faiss-cpu wheels.
• Returns answer, per-snippet citations, and first image of each article.

Environment
-----------
GEMINI_API_KEY  (required) — get one free at https://aistudio.google.com/app/apikey
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- optional FAISS --------------------------------------------------
try:
    import faiss
except ImportError:
    faiss = None

# ---------- Google Gemini client -------------------------------------------
try:
    import google.generativeai as genai
except ImportError:
    raise SystemExit("`pip install google-generativeai` first")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("Set GEMINI_API_KEY environment variable.")

genai.configure(api_key=API_KEY)
GEMINI  = genai.GenerativeModel("gemini-2.0-flash")

# ---------- paths & constants ----------------------------------------------
ROOT   = Path(__file__).resolve().parents[2]
IDXDIR = ROOT / "data" / "index"

TXT_META_PKL = IDXDIR / "text_meta.pkl"
TXT_VEC_NPY  = IDXDIR / "text_vecs.npy"
TXT_FAISS    = IDXDIR / "text.index"

ARTICLES_JSONL = ROOT / "data" / "processed" / "batch_articles.jsonl"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K   = 200
SNIP    = 5000         # chars in prompt
TEMP    = 0.2
MAX_TOK = 50000

PROMPT = (
    "You are a precise AI assistant.  Use only the numbered CONTEXT lines "
    "to answer the QUESTION.  Quote and cite by number like [2]. "
    "If the answer is not in context, say \"I don't know.\".\n\n"
    "CONTEXT:\n{ctx}\n\nQUESTION: {q}\nANSWER:"
)

# ---------- lazy globals ----------------------------------------------------
_txt_idx   = None     # faiss index
_txt_vecs  = None     # np.ndarray
_txt_meta  = None
_articles  = None
_st_model  = None

# ---------- loaders ---------------------------------------------------------
def _load_articles():
    global _articles
    if _articles is None:
        _articles = {}
        with ARTICLES_JSONL.open() as fh:
            for line in fh:
                rec = json.loads(line)
                _articles[rec["id"]] = rec


def _load_text_backend():
    """Guarantee either _txt_idx OR _txt_vecs + _txt_meta are ready."""
    global _txt_idx, _txt_vecs, _txt_meta

    if _txt_meta is None:
        _txt_meta = pickle.load(TXT_META_PKL.open("rb"))

    if faiss is not None and TXT_FAISS.exists():
        if _txt_idx is None:
            _txt_idx = faiss.read_index(str(TXT_FAISS))
        return

    # fall back to numpy matrix
    if _txt_vecs is None:
        _txt_vecs = np.load(TXT_VEC_NPY)
    return


def _get_embedder():
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _st_model

# ---------- core ------------------------------------------------------------
def answer(question: str, top_k: int = TOP_K) -> Dict:
    _load_articles()
    _load_text_backend()
    embedder = _get_embedder()

    qv = embedder.encode(question, normalize_embeddings=True).astype("float32")

    # --- retrieve -----------------------------------------------------------
    if _txt_idx is not None:                                # faiss path
        _, idxs = _txt_idx.search(qv.reshape(1, -1), top_k)
        hit_idx = idxs[0]
    else:                                                   # NumPy path
        sims = _txt_vecs @ qv
        hit_idx = np.argsort(-sims)[:top_k]

    ctx_lines: List[str] = []
    citations: List[dict] = []
    images: List[dict] = []

    for rank, idx in enumerate(hit_idx, 1):
        m = _txt_meta[idx]
        art = _articles[m["parent_id"]]
        snippet = m.get("text") or art["text"][:SNIP]
        ctx_lines.append(f"[{rank}] {snippet}")
        citations.append({"rank": rank, "title": art["title"], "date": art["date"]})
        if art.get("images"):
            images.append(art["images"][0])

    prompt = PROMPT.format(ctx="\n".join(ctx_lines), q=question)

    resp = GEMINI.generate_content(
        prompt,
        generation_config={
            "temperature": TEMP,
            "top_p": 0.95,
            "max_output_tokens": MAX_TOK,
        },
    )
    return {"answer": resp.text.strip(), "sources": citations, "images": images}

# ---------- CLI -------------------------------------------------------------
if __name__ == "__main__":
    import sys, json
    q = "What did Andrew Ng announce in the first issue?" if len(sys.argv) == 1 else " ".join(sys.argv[1:])
    print(json.dumps(answer(q), indent=2, ensure_ascii=False))
