"""
retrieval.py – hybrid similarity-plus-recency retriever with preview loop.

* Scores = cosine × (1 – α) + recency × α
* Iteratively previews article snippets to Gemini, asking for IDs or MORE.
* Falls back to top-K similarity if model fails to choose.
"""
from __future__ import annotations

import json
import pickle
import re
from typing import List, Tuple

import numpy as np
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError

from .config import (
    DATA,
    FULL_K,
    ID_RE,
    LATEST_WORDS,
    MAX_PREVIEW_ROUNDS,
    PREVIEW_BATCH,
    REC_ALPHA,
    STOPWORDS,
    URL_TMPL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)
from .embedder import embed

# ── model client ---------------------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)
LLM = genai.GenerativeModel(GEMINI_MODEL)


def _llm_safe(prompt: str, **cfg) -> str:
    """Return model text or '' on safety/filter/transport errors."""
    try:
        resp = LLM.generate_content(prompt, generation_config=cfg)
        return resp.text.strip()
    except (ValueError, GoogleAPICallError, Exception):
        return ""


# ── index & corpus -------------------------------------------------------
ART_META = pickle.load((DATA / "index" / "article_meta.pkl").open("rb"))
ART_VEC = np.load(DATA / "index" / "article_vecs.npy")
ARTICLES = {
    rec["id"]: rec
    for rec in map(json.loads, (DATA / "processed" / "batch_articles.jsonl").open())
}

_dates = np.array([m["date"] for m in ART_META], dtype="datetime64[D]")
NEWEST, OLDEST = _dates.max(), _dates.min()
SPAN_D = (NEWEST - OLDEST).astype(int) or 1


def _recency(date_str: str) -> float:
    return 1 - (NEWEST - np.datetime64(date_str)).astype(int) / SPAN_D


def _score_matrix(qv: np.ndarray) -> np.ndarray:
    sims = ART_VEC @ qv
    recs = np.array([_recency(m["date"]) for m in ART_META])
    return sims * (1 - REC_ALPHA) + recs * REC_ALPHA


def _preview(i: int) -> str:
    m = ART_META[i]
    return f"[{m['id']}] ({m['date']}) {m['preview']}"


def _keyword_idx(text: str) -> set[int]:
    toks = {t.lower() for t in re.findall(r"[A-Za-z]{4,}", text)} - STOPWORDS
    if not toks:
        return set()
    pat = re.compile("|".join(map(re.escape, toks)), re.I)
    return {
        i
        for i, m in enumerate(ART_META)
        if pat.search(m["title"]) or pat.search(m["preview"])
    }


def choose_articles(question: str) -> Tuple[List[str], str]:
    """Return article IDs and concatenated full-text context."""
    qv = embed(question)
    ranked = np.argsort(-_score_matrix(qv))
    pool = list(ranked) + list(_keyword_idx(question))

    sent, chosen, round_ = set(), [], 0
    while round_ < MAX_PREVIEW_ROUNDS:
        batch = [i for i in pool if ART_META[i]["id"] not in sent][:PREVIEW_BATCH]
        if not batch:
            break
        sent.update(ART_META[i]["id"] for i in batch)
        previews = "\n".join(_preview(i) for i in batch)
        prompt = (
            "Pick relevant article IDs (JSON array) or reply MORE.\n\n"
            f"{previews}\n\nQUESTION:\n{question}"
        )
        rep = _llm_safe(prompt, temperature=0, max_output_tokens=128)
        if rep.upper() == "MORE":
            round_ += 1
            continue
        try:
            chosen = json.loads(rep)
            if isinstance(chosen, list):
                break
        except Exception:
            break

    if not chosen:
        chosen = [ART_META[i]["id"] for i in ranked[:FULL_K]]

    context = "\n\n".join(f"[{aid}] {ARTICLES[aid]['text']}" for aid in chosen)
    return chosen, context


def source_url(aid: str) -> str:
    return URL_TMPL.format(num=int(ID_RE.match(aid).group(1)))
