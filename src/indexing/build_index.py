"""Rich Index Builder for *The Batch* newsletters (2025‑06‑12 rev2).

Builds three vector spaces:
    1. Article‑level (first 200words, title‑prefixed)
    2. Chunk‑level  (600‑word windows, stride300)
    3. Image embeddings (CLIP, optional)

Always writes both FAISS *.index* files and raw *.npy* matrices + *.pkl*
metadata under *data/index/*.

Usage:
    python -m src.batch_scraper.build_index [--no-images]
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None  # NumPy fallback only

# ── configuration ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/processed/batch_articles.jsonl"
OUT = ROOT / "data/index"
OUT.mkdir(parents=True, exist_ok=True)

TXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IMG_MODEL = ("ViT-B-32", "openai")

CHUNK_W, STRIDE = 600, 300
SUMMARY_W = 200
PREVIEW_CHARS = 1_600

ID_RE = re.compile(r"issue-(\d+)")

# ── helpers ─────────────────────────────────────────────────────────────

def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _windows(words: List[str], size: int, step: int):
    if len(words) <= size:
        yield " ".join(words)
    else:
        for i in range(0, len(words) - size + 1, step):
            yield " ".join(words[i : i + size])


def _dump(tag: str, vecs: np.ndarray, meta: list[dict]):
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


# ── build routine ───────────────────────────────────────────────────────

def build(*, skip_images: bool = False) -> None:
    dev = _device()
    print(f"[info] device: {dev} ({platform.system()})")

    txt_enc = SentenceTransformer(TXT_MODEL, device=dev)

    import open_clip

    clip, _, preprocess = open_clip.create_model_and_transforms(*IMG_MODEL, device=dev)
    clip.eval()

    art_v, art_m, chk_v, chk_m, img_v, img_m = [], [], [], [], [], []

    for line in tqdm(RAW.open(), desc="Articles"):
        art = json.loads(line)
        aid, title, date, text = art["id"], art["title"], art["date"], art["text"]
        issue_num_match = ID_RE.match(aid)
        issue_num = int(issue_num_match.group(1)) if issue_num_match else 0

        summary = " ".join(text.split()[:SUMMARY_W])
        art_v.append(
            txt_enc.encode(f"{title} – {summary}", normalize_embeddings=True)
        )
        art_m.append(
            {
                "id": aid,
                "title": title,
                "date": date,
                "issue": issue_num,
                "preview": text[:PREVIEW_CHARS],
            }
        )

        words = text.split()
        for c_id, chunk_txt in enumerate(_windows(words, CHUNK_W, STRIDE)):
            chk_v.append(txt_enc.encode(chunk_txt, normalize_embeddings=True))
            chk_m.append({"parent": aid, "chunk": c_id})

        if not skip_images:
            for img in art.get("images", []):
                path = ROOT / img["path"]
                try:
                    pil = Image.open(path).convert("RGB")
                except Exception:  # pragma: no cover
                    continue
                with torch.no_grad():
                    vec = clip.encode_image(preprocess(pil).unsqueeze(0).to(dev))
                    vec /= vec.norm(dim=-1, keepdim=True)
                img_v.append(vec.cpu().numpy())
                img_m.append(
                    {
                        "parent": aid,
                        "title": title,
                        "path": str(path.relative_to(ROOT)),
                        "alt": img.get("alt", ""),
                    }
                )

    _dump("article", np.vstack(art_v).astype("float32"), art_m)
    _dump("chunk", np.vstack(chk_v).astype("float32"), chk_m)
    if img_v and not skip_images:
        _dump("image", np.vstack(img_v).astype("float32"), img_m)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-images", action="store_true")
    build(skip_images=p.parse_args().no_images)
