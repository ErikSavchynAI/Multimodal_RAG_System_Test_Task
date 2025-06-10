"""
Stage 4 – Two-Pass Multimodal RAG for Gemini-Flash
==================================================

• Works with gemini-2.0-flash or gemini-2.5-flash.
• Optional vision: set  `GEMINI_VISION=1`  to attach ≤2 compressed JPEG images.
• Handles date-specific queries (±3-day filter) and computes The Batch’s
  publication frequency when asked “how often”.
• Keyword booster makes sure rare nouns in the question are not lost.
• Images are resized to IMG_MAXDIM (default 768 px) and JPEG-compressed
  (quality 70) before base-64 upload.

Environment variables
---------------------
GEMINI_API_KEY     – required
GEMINI_MODEL       – default ‘gemini-2.0-flash’
GEMINI_VISION      – ‘1’ to enable image parts (default ‘0’)
GEMINI_IMG_MAXDIM  – max longer side in px (default 768)
GEMINI_IMG_QUALITY – JPEG quality 1-95 (default 70)
"""
from __future__ import annotations

import base64
import datetime as _dt
import json
import mimetypes
import os
import pickle
import re
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import numpy as np
import google.generativeai as genai
from dateutil import parser as _dateparse
from sentence_transformers import SentenceTransformer

# ── paths & static metadata ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
IDX  = ROOT / "data/index"

ART_META = pickle.load((IDX / "article_meta.pkl").open("rb"))
ART_VEC  = np.load(IDX / "article_vecs.npy")
ARTICLES = {r["id"]: r for r in map(json.loads,
                                   (ROOT / "data/processed/batch_articles.jsonl").open())}
IMG_META = pickle.load((IDX / "image_meta.pkl").open("rb")) if (IDX / "image_meta.pkl").exists() else []

# ── models & global constants ─────────────────────────────────────────────
EMBED           = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
MODEL           = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
VISUAL_ENABLED  = os.getenv("GEMINI_VISION", "0") == "1"
TEMP            = 0.2
OUT_TOK         = 32_768

PREVIEW_K       = 120
FULL_K          = 40
IMG_SELECT_K    = 2
MAX_CTX_TOK     = 150_000

IMG_MAXDIM      = int(os.getenv("GEMINI_IMG_MAXDIM", "768"))
IMG_QUALITY     = int(os.getenv("GEMINI_IMG_QUALITY", "70"))

URL_TMPL        = "https://www.deeplearning.ai/the-batch/issue-{num}/"
ID_RE           = re.compile(r"issue-(\d+)")
_DATE_PAT       = re.compile(
    r"\b(?:\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})\b",
    re.I,
)
_STOPWORDS      = set("the of on in a an how what why where when which is are does do did".split())

genai.configure(api_key=os.getenv("GEMINI_API_KEY") or exit("Set GEMINI_API_KEY"))
LLM = genai.GenerativeModel(MODEL)

# ── utility helpers ───────────────────────────────────────────────────────


def _embed(text: str) -> np.ndarray:
    return EMBED.encode(text, normalize_embeddings=True).astype("float32")


def _topk(qv: np.ndarray, k: int) -> List[int]:
    return np.argsort(-(ART_VEC @ qv))[:k]


def _preview_line(i: int) -> str:
    m = ART_META[i]
    return f"[{m['id']}] ({m['date']}) {m['preview']}"


def _url(aid: str) -> str:
    return URL_TMPL.format(num=ID_RE.match(aid).group(1))


# ── query-analysis helpers ────────────────────────────────────────────────
def _extract_query_date(text: str) -> str | None:
    m = _DATE_PAT.search(text)
    if not m:
        return None
    try:
        d = _dateparse.parse(m.group(0), dayfirst=False, fuzzy=True).date()
        return d.isoformat()
    except Exception:
        return None


_FREQ_MEMO: str | None = None


def _batch_frequency() -> str:
    global _FREQ_MEMO
    if _FREQ_MEMO:
        return _FREQ_MEMO
    dates = [_dt.date.fromisoformat(m["date"]) for m in ART_META]
    dates.sort()
    deltas = [
        int((b - a).days)
        for a, b in zip(dates, dates[1:])
        if 1 <= (b - a).days <= 14
    ]
    mode = Counter(deltas).most_common(1)[0][0] if deltas else 7
    wording = "weekly" if 6 <= mode <= 8 else f"every ~{mode} days"
    _FREQ_MEMO = f"The Batch is a {wording} newsletter (modal interval ≈ {mode} days)."
    return _FREQ_MEMO


def _keyword_candidates(question: str) -> set[int]:
    tokens = {t.lower() for t in re.findall(r"[A-Za-z]{4,}", question)}
    keywords = tokens - _STOPWORDS
    if not keywords:
        return set()
    kw_re = re.compile("|".join(map(re.escape, keywords)), re.I)
    return {
        i
        for i, m in enumerate(ART_META)
        if kw_re.search(m["preview"]) or kw_re.search(m["title"])
    }


# ── image selection & compression ─────────────────────────────────────────
def _select_images(question: str, cand_imgs: List[dict]) -> List[dict]:
    if not cand_imgs:
        return []
    qv_txt = _embed(question)
    scored = []
    for im in cand_imgs:
        alt = im.get("alt", "") or im.get("title", "")
        score = float(_embed(alt) @ qv_txt) if alt else -1.0
        scored.append((score, im))
    scored.sort(reverse=True, key=lambda t: t[0])
    top = [im for _, im in scored[:IMG_SELECT_K]]

    # vision re-rank with CLIP
    try:
        import open_clip
        import torch
        from PIL import Image

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        clip, _, preproc = open_clip.create_model_and_transforms(
            "ViT-B-32", "openai", device=dev
        )
        clip.eval()

        with torch.no_grad():
            q_clip = clip.encode_text(open_clip.tokenize([question]).to(dev))
            q_clip /= q_clip.norm(dim=-1, keepdim=True)

            clip_scored = []
            for im in top:
                p = ROOT / im["path"]
                pil = Image.open(p).convert("RGB")
                vec = clip.encode_image(preproc(pil).unsqueeze(0).to(dev))
                vec /= vec.norm(dim=-1, keepdim=True)
                sim = float((vec @ q_clip.T).item())
                clip_scored.append((sim, im))
        clip_scored.sort(reverse=True, key=lambda t: t[0])
        return [im for _, im in clip_scored]
    except Exception:
        return top


def _image_part(img_path: Path) -> dict:
    from PIL import Image

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im.thumbnail((IMG_MAXDIM, IMG_MAXDIM), Image.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=IMG_QUALITY, optimize=True)
        data_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": "image/jpeg", "data": data_b64}}


# ── main answer function ──────────────────────────────────────────────────
def answer(question: str) -> Dict:
    q_date_str = _extract_query_date(question)
    keyword_idx = _keyword_candidates(question)
    qv = _embed(question)

    # ── candidate pool (date filter + keywords) ───────────────────────────
    if q_date_str:
        target = _dt.date.fromisoformat(q_date_str)
        pool = [
            i
            for i, m in enumerate(ART_META)
            if abs((_dt.date.fromisoformat(m["date"]) - target).days) <= 3
        ]
        if not pool:
            pool = list(range(len(ART_META)))
    else:
        pool = list(range(len(ART_META)))

    pool = sorted(set(pool) | keyword_idx)

    # similarity ranking in the pool
    sims = ART_VEC[pool] @ qv
    pv_ids = [pool[i] for i in np.argsort(-sims)[:PREVIEW_K]]

    # ── pass 1 prompt ─────────────────────────────────────────────────────
    previews = "\n".join(_preview_line(i) for i in pv_ids)
    prompt1 = (
        "Below is a list of article previews from *The Batch* "
        "(date in parentheses).\n"
        "Return a **JSON array** of the article IDs you need to read fully. "
        "If the question names a date, prefer matching dates.\n\n"
        f"PREVIEWS:\n{previews}\n\nQUESTION:\n{question}"
    )

    try:
        raw = LLM.generate_content(
            prompt1, generation_config={"temperature": 0, "max_output_tokens": 128}
        )
        need = json.loads(raw.text)
    except Exception:
        need = []

    need = [aid for aid in need if isinstance(aid, str)]
    need += [ART_META[i]["id"] for i in pv_ids if ART_META[i]["id"] not in need]
    need = need[:FULL_K]

    # ── full context & images ─────────────────────────────────────────────
    ctx_lines: List[str] = []
    sources, cand_imgs = [], []
    for aid in need:
        art = ARTICLES[aid]
        ctx_lines.append(f"[{aid}] {art['text']}")
        sources.append(
            {
                "id": aid,
                "title": art["title"],
                "date": art["date"],
                "url": _url(aid),
            }
        )
        cand_imgs.extend([im for im in IMG_META if im["parent"] == aid])

    # synthetic fact for “how often”
    if re.search(r"\bhow (often|frequent)\b", question, re.I):
        ctx_lines.insert(0, f"[about-frequency] {_batch_frequency()}")

    images = _select_images(question, cand_imgs)
    ctx = "\n\n".join(ctx_lines)

    base_prompt = (
        "Using the CONTEXT, answer the QUESTION as accurately as possible. "
        "Cite article IDs inline like [issue-123_news_4].\n\n"
        f"CONTEXT:\n{ctx}\n\nQUESTION:\n{question}"
    )

    # ── send to Gemini ────────────────────────────────────────────────────
    if VISUAL_ENABLED and images:
        parts = [{"text": base_prompt}] + [
            _image_part(ROOT / im["path"]) for im in images
        ]
        resp = LLM.generate_content(
            contents=parts,
            generation_config={"temperature": TEMP, "max_output_tokens": OUT_TOK},
        )
    else:
        resp = LLM.generate_content(
            base_prompt,
            generation_config={"temperature": TEMP, "max_output_tokens": OUT_TOK},
        )

    # ── prune to actually cited sources/images ────────────────────────────
    cited = set(re.findall(r"\[(issue-[^\]]+|about-frequency)\]", resp.text))
    if cited:
        sources = [s for s in sources if s["id"] in cited]
        images = [im for im in images if im["parent"] in cited]

    return {"answer": resp.text, "sources": sources, "images": images}


# ── CLI helper ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint
    import sys

    q = (
        " ".join(sys.argv[1:])
        or "What is the article about on May 27, 2020? Write a short version"
    )
    pprint.pprint(answer(q))
