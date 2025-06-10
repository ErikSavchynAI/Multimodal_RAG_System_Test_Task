"""
Stage 3 – Rich Index Builder (2025-06-12 rev 2)
==============================================

• Builds THREE vector spaces
    ① Article-level (first 150 words)
    ② Chunk-level  (400-word windows, stride 200)
    ③ Image vision vectors (optional)

• Saves previews (1 200 chars) + issue number → enables URL lookup.
• Always writes both FAISS indexes **and** `.npy` matrices.

Directory layout
----------------
data/index/
    article.index / article_vecs.npy / article_meta.pkl
    chunk.index   / chunk_vecs.npy   / chunk_meta.pkl
    image.index   / image_vecs.npy   / image_meta.pkl
"""
from __future__ import annotations
import argparse, json, pickle, platform, re
from pathlib import Path
from typing import List

import numpy as np, torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import faiss
except ImportError:
    faiss = None                           # NumPy fallback

# ── config ────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[2]
RAW    = ROOT / "data/processed/batch_articles.jsonl"
OUT    = ROOT / "data/index"; OUT.mkdir(parents=True, exist_ok=True)

TXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IMG_MODEL = ("ViT-B-32", "openai")         # (arch, weights tag)

CHUNK_W, STRIDE = 600, 300
SUMMARY_W       = 200                      # article-level summary window
PREVIEW_CHARS   = 1_600                    # stored in article_meta

ID_RE = re.compile(r"issue-(\d+)")

# ── helpers ───────────────────────────────────────────────────────────────
def device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def windows(words: List[str], size: int, step: int):
    if len(words) <= size:
        yield " ".join(words)
    else:
        for i in range(0, len(words) - size + 1, step):
            yield " ".join(words[i : i + size])

def dump(tag: str, vecs: np.ndarray, meta: list[dict]):
    np.save(OUT / f"{tag}_vecs.npy", vecs)
    with (OUT / f"{tag}_meta.pkl").open("wb") as f:
        pickle.dump(meta, f)
    if faiss:
        idx = faiss.IndexIDMap(faiss.IndexFlatIP(vecs.shape[1]))
        idx.add_with_ids(vecs, np.arange(len(vecs), dtype="int64"))
        faiss.write_index(idx, str(OUT / f"{tag}.index"))
        print(f"[info] {tag:<8} {len(meta)} vectors  +faiss")
    else:
        print(f"[warn] {tag:<8} {len(meta)} vectors  (.npy only)")

# ── main build routine ────────────────────────────────────────────────────
def build(skip_images: bool = False) -> None:
    dev = device()
    print(f"[info] device: {dev} ({platform.system()})")

    txt_encoder = SentenceTransformer(TXT_MODEL, device=dev)

    import open_clip
    clip, _, preproc = open_clip.create_model_and_transforms(*IMG_MODEL, device=dev)
    clip.eval()

    art_v, art_m, chk_v, chk_m, img_v, img_m = [], [], [], [], [], []

    for line in tqdm(RAW.open(), desc="Articles"):
        art = json.loads(line)
        aid, title, date, text = art["id"], art["title"], art["date"], art["text"]
        issue_num = ID_RE.match(aid).group(1) if ID_RE.match(aid) else "0"

        # ── article-level vector (summary) ────────────────────────────────
        summary = " ".join(text.split()[:SUMMARY_W])
        art_v.append(
            txt_encoder.encode(f"{title} – {summary}", normalize_embeddings=True)
        )
        art_m.append(
            {
                "id": aid,
                "title": title,
                "date": date,
                "issue": int(issue_num),
                "preview": text[:PREVIEW_CHARS],
            }
        )

        # ── chunk-level vectors ───────────────────────────────────────────
        words = text.split()
        for c_id, chunk_txt in enumerate(windows(words, CHUNK_W, STRIDE)):
            chk_v.append(txt_encoder.encode(chunk_txt, normalize_embeddings=True))
            chk_m.append({"parent": aid, "chunk": c_id})

        # ── images (optional) ────────────────────────────────────────────
        if not skip_images:
            for img in art.get("images", []):
                p = ROOT / img["path"]
                try:
                    pil = Image.open(p).convert("RGB")
                except Exception:  # corrupt / missing image
                    continue
                with torch.no_grad():
                    vec = clip.encode_image(preproc(pil).unsqueeze(0).to(dev))
                    vec /= vec.norm(dim=-1, keepdim=True)
                img_v.append(vec.cpu().numpy())
                img_m.append(
                    {
                        "parent": aid,
                        "title": title,
                        "path": str(p.relative_to(ROOT)),
                        "alt": img.get("alt", ""),
                    }
                )

    # ── persist ──────────────────────────────────────────────────────────
    dump("article", np.vstack(art_v).astype("float32"), art_m)
    dump("chunk", np.vstack(chk_v).astype("float32"), chk_m)
    if not skip_images and img_v:
        dump("image", np.vstack(img_v).astype("float32"), img_m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-images", action="store_true")
    build(skip_images=parser.parse_args().no_images)
