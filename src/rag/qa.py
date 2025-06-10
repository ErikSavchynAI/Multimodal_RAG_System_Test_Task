"""
Stage-4 Multimodal RAG · rev 6.1
────────────────────────────────
Same as rev 6 but fixes a TypeError in _select_images()
by sorting tuples via `key=lambda t: t[0]`.
"""
from __future__ import annotations
import base64, datetime as dt, json, os, pickle, re
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import numpy as np
import google.generativeai as genai
from dateutil import parser as dparse
from sentence_transformers import SentenceTransformer

# ── project paths ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
IDX  = ROOT / "data/index"

# ── runtime parameters ----------------------------------------------------
MODEL      = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
API_KEY    = os.getenv("GEMINI_API_KEY") or exit("Set GEMINI_API_KEY")

PREVIEW_BATCH       = 40
MAX_PREVIEW_ROUNDS  = 4
FULL_K              = 40

REC_ALPHA_DEFAULT   = float(os.getenv("REC_ALPHA", "0.25"))
REC_ALPHA_LATEST    = 0.60              # extra freshness weight for “latest”

IMG_SELECT_K        = 4
MIN_IMG_SIM         = 0.28
IMG_MAXDIM          = int(os.getenv("GEMINI_IMG_MAXDIM", "768"))
IMG_QUALITY         = int(os.getenv("GEMINI_IMG_QUALITY", "70"))
VISUAL              = os.getenv("GEMINI_VISION", "0") == "1"

TEMP, OUT_TOK       = 0.2, 32_768

URL_TMPL = "https://www.deeplearning.ai/the-batch/issue-{num}/"
ID_RE    = re.compile(r"issue-(\d+)")
STOPW    = set("the of on in a an how what why where when which is are does do did".split())
LATEST_WORDS = {"latest", "newest", "most recent", "last"}
IMG_WORDS    = {"image", "picture", "photo", "figure", "illustration"}

# ── load corpus -----------------------------------------------------------
ART_META = pickle.load((IDX / "article_meta.pkl").open("rb"))
ART_VEC  = np.load(IDX / "article_vecs.npy")
ARTICLES = {r["id"]: r for r in map(json.loads,
                                    (ROOT / "data/processed/batch_articles.jsonl").open())}
IMG_META = pickle.load((IDX / "image_meta.pkl").open("rb")) if (IDX / "image_meta.pkl").exists() else []

EMBED = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
genai.configure(api_key=API_KEY)
LLM = genai.GenerativeModel(MODEL)

# ── helpers ---------------------------------------------------------------
_dates = [dt.date.fromisoformat(m["date"]) for m in ART_META]
DATE_NEWEST, DATE_OLDEST = max(_dates), min(_dates)
DATE_SPAN_D = max(1, (DATE_NEWEST - DATE_OLDEST).days)

def _embed(t: str) -> np.ndarray:
    return EMBED.encode(t, normalize_embeddings=True).astype("float32")

def _rec_score(date_str: str) -> float:
    d = dt.date.fromisoformat(date_str)
    return 1.0 - (DATE_NEWEST - d).days / DATE_SPAN_D

def _combined_scores(qv: np.ndarray, alpha: float) -> np.ndarray:
    sims = ART_VEC @ qv
    rec  = np.array([_rec_score(m["date"]) for m in ART_META], dtype="float32")
    return sims * (1 - alpha) + rec * alpha

def _preview_line(i: int) -> str:
    m = ART_META[i]
    return f"[{m['id']}] ({m['date']}) {m['preview']}"

def _keyword_hits(q: str) -> set[int]:
    toks = {t.lower() for t in re.findall(r"[A-Za-z]{4,}", q)} - STOPW
    if not toks: return set()
    pat = re.compile("|".join(map(re.escape, toks)), re.I)
    return {i for i, m in enumerate(ART_META)
            if pat.search(m["title"]) or pat.search(m["preview"])}

def _image_part(p: Path) -> dict:
    from PIL import Image
    with Image.open(p) as im:
        im = im.convert("RGB")
        im.thumbnail((IMG_MAXDIM, IMG_MAXDIM))
        buf = BytesIO(); im.save(buf, "JPEG", quality=IMG_QUALITY, optimize=True)
    return {"inline_data": {"mime_type": "image/jpeg",
                            "data": base64.b64encode(buf.getvalue()).decode()}}

# ── image selector --------------------------------------------------------
def _select_images(question: str, cand: List[dict]) -> List[dict]:
    if not cand: return []
    qv = _embed(question)
    thresh = MIN_IMG_SIM * (0.5 if any(w in question.lower() for w in IMG_WORDS) else 1.0)

    by_parent, picked = {}, []
    for im in cand:
        caption = im.get("alt") or im.get("title", "")
        sim = float(_embed(caption) @ qv) if caption else 0.0
        if sim >= thresh:
            by_parent.setdefault(im["parent"], []).append((sim, im))

    # choose best per parent
    for lst in by_parent.values():
        lst.sort(key=lambda t: t[0], reverse=True)   # ← fixed line
        picked.append(lst[0][1])

    # fallback first-image if parent still lacks one
    for im in cand:
        if im["parent"] not in by_parent:
            picked.append(im); by_parent[im["parent"]] = [(0, im)]
        if len(picked) >= IMG_SELECT_K:
            break

    return picked[:IMG_SELECT_K]

# ── main RAG --------------------------------------------------------------
def answer(question: str) -> Dict:
    q_low = question.lower()
    alpha = REC_ALPHA_LATEST if any(w in q_low for w in LATEST_WORDS) else REC_ALPHA_DEFAULT

    qv = _embed(question)
    scores = _combined_scores(qv, alpha)
    ranked = np.argsort(-scores)

    pool = list(ranked) + list(_keyword_hits(question))

    sent, need_ids, round_no = set(), [], 0
    while round_no < MAX_PREVIEW_ROUNDS:
        batch = [i for i in pool if ART_META[i]["id"] not in sent][:PREVIEW_BATCH]
        if not batch: break
        previews = "\n".join(_preview_line(i) for i in batch)
        sent.update(ART_META[i]["id"] for i in batch)
        prompt = (
            "You are selecting articles from *The Batch*. "
            "Newer previews appear first. Prefer newer ones if relevance ties.\n"
            "Reply with ONLY a JSON array of IDs or the single word MORE.\n\n"
            f"{previews}\n\nQUESTION:\n{question}"
        )
        reply = LLM.generate_content(prompt,
                                     generation_config={"temperature":0,"max_output_tokens":128}).text.strip()
        if reply.upper() == "MORE":
            round_no += 1; continue
        try:
            need_ids = json.loads(reply)
            if isinstance(need_ids, list): break
        except Exception:
            need_ids = []; break

    if not need_ids:
        need_ids = [ART_META[i]["id"] for i in ranked[:FULL_K]]
    need_ids = need_ids[:FULL_K]

    # context
    ctx_lines, sources, cand_imgs = [], [], []
    for aid in need_ids:
        art = ARTICLES[aid]
        ctx_lines.append(f"[{aid}] {art['text']}")
        sources.append({"id": aid, "title": art["title"],
                        "date": art["date"],
                        "url": URL_TMPL.format(num=ID_RE.match(aid).group(1))})
        cand_imgs.extend([im for im in IMG_META if im["parent"] == aid])

    # frequency fact
    if re.search(r"\bhow (often|frequent)\b", question, re.I):
        ds = sorted(dt.date.fromisoformat(m["date"]) for m in ART_META)
        mode = Counter((b - a).days for a, b in zip(ds, ds[1:])).most_common(1)[0][0]
        ctx_lines.insert(0, f"[about-frequency] The Batch is published every ~{mode} days (weekly).")

    context = "\n\n".join(ctx_lines)
    images = _select_images(question, cand_imgs)

    prompt2 = (
        "Use only the CONTEXT to answer the QUESTION. "
        "Cite facts with IDs like [issue-123_news_1]. "
        "Prefer newer sources when possible.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    )
    parts = [{"text": prompt2}] + [_image_part(ROOT / im["path"]) for im in images] \
            if (VISUAL and images) else prompt2

    ans = LLM.generate_content(parts,
                               generation_config={"temperature":TEMP,"max_output_tokens":OUT_TOK}).text.strip()

    cited = set(re.findall(r"\[(issue-[^\]]+)\]", ans))
    sources = [s for s in sources if s["id"] in cited]
    images  = [im for im in images if im["parent"] in cited]

    id2url = {s["id"]: s["url"] for s in sources}
    ans = re.sub(r"\[(issue-[^\]]+)\]",
                 lambda m: f"[{m.group(1)}]({id2url.get(m.group(1),'#')})",
                 ans)

    return {"answer": ans, "sources": sources, "images": images}

# CLI test
if __name__ == "__main__":
    import pprint, sys
    pprint.pprint(answer("What was the very first article on The Batch about and when was it published?"))
