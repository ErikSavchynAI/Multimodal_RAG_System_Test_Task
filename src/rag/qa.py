# """
# Stage-4 Multimodal RAG · rev 9
# ──────────────────────────────
# Features
# • Recency-aware retrieval (semantic × recency blend).
# • Iterative “MORE” preview loop (≤ 160 previews).
# • Conversation prefix (last chat turns) forwarded to Gemini but *not* used
#   in vector search — avoids retrieval pollution.
# • Image selector returns only caption-relevant pictures (no fallback noise).
# • Final prompt asks for ≤ 4 articles, 3-5 sentences each (engaging style).
# • Links use zero-stripped IDs (issue-70, not issue-070).
#
# Environment
# GEMINI_API_KEY   – (required)
# GEMINI_MODEL     – gemini-2.0-flash (default)
# GEMINI_VISION    – '1' to attach images
# REC_ALPHA        – 0.25 (freshness weight 0–1)
# IMG_MAXDIM       – 768
# IMG_QUALITY      – 70
# """
#
# from __future__ import annotations
#
# import base64
# import datetime as dt
# import json
# import os
# import pickle
# import re
# from collections import Counter
# from io import BytesIO
# from pathlib import Path
# from typing import Dict, List
#
# import numpy as np
# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer
#
# # ── paths -----------------------------------------------------------------
# ROOT = Path(__file__).resolve().parents[2]
# IDX_DIR = ROOT / "data/index"
#
# # ── load corpus -----------------------------------------------------------
# ART_META: list[dict] = pickle.load((IDX_DIR / "article_meta.pkl").open("rb"))
# ART_VEC: np.ndarray = np.load(IDX_DIR / "article_vecs.npy")
# ARTICLES: dict = {j["id"]: j for j in map(json.loads, (ROOT / "data/processed/batch_articles.jsonl").open())}
# IMG_META: list[dict] = (
#     pickle.load((IDX_DIR / "image_meta.pkl").open("rb"))
#     if (IDX_DIR / "image_meta.pkl").exists()
#     else []
# )
#
# # ── model initialisation --------------------------------------------------
# genai.configure(api_key=os.getenv("GEMINI_API_KEY") or exit("Set GEMINI_API_KEY"))
# MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
# LLM = genai.GenerativeModel(MODEL_NAME)
# EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#
# # ── global constants ------------------------------------------------------
# PREVIEW_BATCH = 40
# MAX_PREVIEW_ROUNDS = 4
# FULL_K = 40
#
# REC_ALPHA_DEF = float(os.getenv("REC_ALPHA", "0.25"))
# REC_ALPHA_LATEST = 0.60
#
# IMG_SELECT_K = 4
# MIN_IMG_SIM = 0.28
# IMG_MAXDIM = int(os.getenv("IMG_MAXDIM", "768"))
# IMG_QUALITY = int(os.getenv("IMG_QUALITY", "70"))
# VISION = os.getenv("GEMINI_VISION", "0") == "1"
#
# TEMP = 0.35
# OUT_TOK = 32_768
#
# URL_TMPL = "https://www.deeplearning.ai/the-batch/issue-{num}/"
# ID_RE = re.compile(r"issue-(\d+)")
# STOPWORDS = set("the a an is are was were of on in and to for how what why which".split())
# LATEST_WORDS = {"latest", "newest", "most recent", "last"}
# IMG_QUERY_WORDS = {"image", "picture", "photo", "figure", "illustration"}
#
# # ── date helpers ----------------------------------------------------------
# _dates = [dt.date.fromisoformat(m["date"]) for m in ART_META]
# DATE_NEWEST = max(_dates)
# DATE_OLDEST = min(_dates)
# DATE_SPAN_D = max(1, (DATE_NEWEST - DATE_OLDEST).days)
#
# # -------------------------------------------------------------------------
# # Helper functions
# # -------------------------------------------------------------------------
#
#
# def embed(text: str) -> np.ndarray:
#     """Unit-normalised embedding."""
#     return EMBEDDER.encode(text, normalize_embeddings=True).astype("float32")
#
#
# def recency_score(date_str: str) -> float:
#     """0 (oldest) … 1 (newest)."""
#     return 1.0 - (DATE_NEWEST - dt.date.fromisoformat(date_str)).days / DATE_SPAN_D
#
#
# def combined_scores(qv: np.ndarray, alpha: float) -> np.ndarray:
#     """Blend semantic similarity and freshness."""
#     sims = ART_VEC @ qv
#     rec = np.array([recency_score(m["date"]) for m in ART_META], dtype="float32")
#     return sims * (1 - alpha) + rec * alpha
#
#
# def preview_line(idx: int) -> str:
#     m = ART_META[idx]
#     return f"[{m['id']}] ({m['date']}) {m['preview']}"
#
#
# def keyword_indices(text: str) -> set[int]:
#     toks = {t.lower() for t in re.findall(r"[A-Za-z]{4,}", text)} - STOPWORDS
#     if not toks:
#         return set()
#     pat = re.compile("|".join(map(re.escape, toks)), re.I)
#     return {i for i, m in enumerate(ART_META) if pat.search(m["title"]) or pat.search(m["preview"])}
#
#
# def strip_leading_zeros(aid: str) -> str:
#     """issue-070 → 70"""
#     return str(int(ID_RE.match(aid).group(1)))
#
#
# def issue_url(aid: str) -> str:
#     return URL_TMPL.format(num=strip_leading_zeros(aid))
#
#
# # -------------------------------------------------------------------------
# # Image selection helpers
# # -------------------------------------------------------------------------
#
#
# def img_data_uri(path: Path) -> str:
#     from PIL import Image
#
#     with Image.open(path) as im:
#         im = im.convert("RGB")
#         im.thumbnail((IMG_MAXDIM, IMG_MAXDIM))
#         buf = BytesIO()
#         im.save(buf, "JPEG", quality=IMG_QUALITY, optimize=True)
#     return base64.b64encode(buf.getvalue()).decode()
#
#
# def select_images(question: str, candidates: List[dict]) -> List[dict]:
#     """Return ≤ IMG_SELECT_K images that match the question context."""
#     if not candidates:
#         return []
#     qv = embed(question)
#     threshold = MIN_IMG_SIM * (0.5 if any(w in question.lower() for w in IMG_QUERY_WORDS) else 1.0)
#
#     # score by caption similarity
#     scored: list[tuple[float, dict]] = []
#     for im in candidates:
#         caption = im.get("alt") or im.get("title", "")
#         sim = float(embed(caption) @ qv) if caption else 0.0
#         if sim >= threshold:
#             scored.append((sim, im))
#
#     scored.sort(key=lambda t: -t[0])
#     selected = []
#     seen_parent = set()
#     for sim, im in scored:
#         if im["parent"] in seen_parent:
#             continue
#         selected.append(im)
#         seen_parent.add(im["parent"])
#         if len(selected) >= IMG_SELECT_K:
#             break
#
#     return selected
#
#
# # -------------------------------------------------------------------------
# # Core RAG function
# # -------------------------------------------------------------------------
#
#
# def answer(prompt: str) -> Dict:
#     """
#     Parameters
#     ----------
#     prompt
#         Either a standalone user question or a multi-turn context ending with
#         “NEW USER QUESTION:” delimiter.
#
#     Returns
#     -------
#     dict with keys:
#         answer   – markdown text
#         sources  – list[dict]  (title, date, url)
#         images   – list[dict]  (parent id, path, alt/title)
#     """
#
#     # -------- split conversation prefix ----------------------------------
#     if "NEW USER QUESTION:" in prompt:
#         conv, question = prompt.rsplit("NEW USER QUESTION:", 1)
#         conv = conv.strip()
#         question = question.strip()
#     else:
#         conv, question = "", prompt.strip()
#
#     # -------- initial ranking --------------------------------------------
#     alpha = REC_ALPHA_LATEST if any(w in question.lower() for w in LATEST_WORDS) else REC_ALPHA_DEF
#     qv = embed(question)
#     ranked = np.argsort(-combined_scores(qv, alpha))
#     pool = list(ranked) + list(keyword_indices(question))
#
#     # -------- iterative preview loop -------------------------------------
#     sent_ids: set[str] = set()
#     need_ids: list[str] = []
#     round_no = 0
#
#     while round_no < MAX_PREVIEW_ROUNDS:
#         batch = [i for i in pool if ART_META[i]["id"] not in sent_ids][:PREVIEW_BATCH]
#         if not batch:
#             break
#         sent_ids.update(ART_META[i]["id"] for i in batch)
#
#         previews = "\n".join(preview_line(i) for i in batch)
#         prompt1 = (
#             "Select relevant article IDs from the batch below.\n"
#             "Newer previews appear first — choose newer when relevance ties.\n"
#             "Reply with a JSON array of IDs or the single word MORE.\n\n"
#             f"{previews}\n\nQUESTION:\n{question}"
#         )
#         reply = LLM.generate_content(
#             prompt1, generation_config={"temperature": 0, "max_output_tokens": 128}
#         ).text.strip()
#
#         if reply.upper() == "MORE":
#             round_no += 1
#             continue
#
#         try:
#             need_ids = json.loads(reply)
#             if isinstance(need_ids, list):
#                 break
#         except Exception:
#             need_ids = []
#             break
#
#     if not need_ids:
#         need_ids = [ART_META[i]["id"] for i in ranked[:FULL_K]]
#     need_ids = need_ids[:FULL_K]
#
#     # -------- build full context -----------------------------------------
#     ctx_lines: list[str] = []
#     sources: list[dict] = []
#     img_candidates: list[dict] = []
#
#     for aid in need_ids:
#         art = ARTICLES[aid]
#         ctx_lines.append(f"[{aid}] {art['text']}")
#         sources.append(
#             {"id": aid, "title": art["title"], "date": art["date"], "url": issue_url(aid)}
#         )
#         img_candidates.extend([im for im in IMG_META if im["parent"] == aid])
#
#     # frequency fact if asked
#     if re.search(r"\bhow (often|frequent)\b", question, re.I):
#         ds = sorted(dt.date.fromisoformat(m["date"]) for m in ART_META)
#         mode = Counter((b - a).days for a, b in zip(ds, ds[1:])).most_common(1)[0][0]
#         ctx_lines.insert(
#             0, f"[about-frequency] The Batch is published every ~{mode} days (weekly)."
#         )
#
#     context = "\n\n".join(ctx_lines)
#     images = select_images(question, img_candidates)
#
#     # -------- final prompt -----------------------------------------------
#     narrative_instr = (
#         "Write an **engaging narrative**: pick **at most four** distinct articles "
#         "and devote *three to five complete sentences* to each, explaining what it "
#         "says and why it matters. Integrate the points naturally."
#     )
#     prompt2 = (
#         (conv + "\n\n" if conv else "")
#         + narrative_instr
#         + "\n\nCite facts inline like [issue-123_news_1].\n\n"
#         f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
#     )
#
#     # assemble multi-part if vision
#     parts = [{"text": prompt2}]
#     if VISION and images:
#         parts.extend(
#             {
#                 "inline_data": {
#                     "mime_type": "image/jpeg",
#                     "data": img_data_uri(ROOT / im["path"]),
#                 }
#             }
#             for im in images
#         )
#
#     response = LLM.generate_content(
#         parts if (VISION and images) else prompt2,
#         generation_config={"temperature": TEMP, "max_output_tokens": OUT_TOK},
#     ).text.strip()
#
#     # -------- post-process citations & filter assets ---------------------
#     cited_ids = set(re.findall(r"\[(issue-[^\]]+)\]", response))
#     sources = [s for s in sources if s["id"] in cited_ids]
#     images = [im for im in images if im["parent"] in cited_ids]
#
#     id2url = {s["id"]: s["url"] for s in sources}
#     response = re.sub(
#         r"\[(issue-[^\]]+)\]", lambda m: f"[{m.group(1)}]({id2url.get(m.group(1), '#')})", response
#     )
#
#     return {"answer": response, "sources": sources, "images": images}
#
#
# # -------------------------------------------------------------------------
# # CLI test
# # -------------------------------------------------------------------------
# if __name__ == "__main__":
#     import pprint, sys
#
#     user_q = (
#         "What were Andrew Ng's essays on AI and policy? Include pictures."
#         if len(sys.argv) == 1
#         else " ".join(sys.argv[1:])
#     )
#     pprint.pprint(answer(user_q))
