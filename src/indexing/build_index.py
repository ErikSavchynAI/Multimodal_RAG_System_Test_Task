"""
Stage 3 — Multimodal Embedding & Index Builder
==============================================

• Embeds text chunks (Sentence-Transformers MiniLM) and images (OpenCLIP ViT-B/32).
• Always writes *both* FAISS indexes **and** `.npy` fallback matrices so
  Stage 4 works on any machine, with or without faiss-cpu wheels.
• Cross-platform: CUDA ▶︎ Apple Silicon (MPS) ▶︎ CPU.

Output layout
-------------
data/index/
├── text.index            # if faiss available
├── text_meta.pkl
├── text_vecs.npy         # always
├── image.index           # if faiss available
├── image_meta.pkl
└── image_vecs.npy        # always
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------- optional FAISS --------------------------------------------------
try:
    import faiss       # type: ignore
except ImportError:    # pragma: no cover
    faiss = None

# ---------- paths & models --------------------------------------------------
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_JSON = ROOT / "data" / "processed" / "batch_articles.jsonl"
OUT_DIR        = ROOT / "data" / "index"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d
CLIP_MODEL = "ViT-B-32"
CLIP_WEIGHTS = "openai"                                 # OpenCLIP tag
CHUNK_WORDS  = 200                                      # default

# ---------- helpers ---------------------------------------------------------
def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def chunk(text: str, max_words: int) -> List[str]:
    w = text.split()
    return [" ".join(w[i : i + max_words]) for i in range(0, len(w), max_words)] or [""]


# ---------- build routine ---------------------------------------------------
def build(jsonl: Path, chunk_words: int, with_images: bool) -> None:
    dev = device()
    print(f"[info] embedding on {dev} ({platform.system()})")

    txt_model = SentenceTransformer(TEXT_MODEL, device=dev)

    import open_clip                          # local import keeps dependency optional
    clip_model, _, clip_pp = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_WEIGHTS, device=dev
    )
    clip_model.eval()

    text_vecs, text_meta = [], []
    img_vecs,  img_meta  = [], []

    with jsonl.open(encoding="utf-8") as fh:
        for line in tqdm(fh, desc="Embedding articles"):
            art = json.loads(line)
            aid = art["id"]

            # ---- text ----
            for c_id, chunk_txt in enumerate(chunk(art["text"], chunk_words)):
                if not chunk_txt.strip():
                    continue
                vec = txt_model.encode(chunk_txt, normalize_embeddings=True).astype("float32")
                text_vecs.append(vec)
                text_meta.append({
                    "parent_id": aid, "chunk": c_id,
                    "title": art["title"], "date": art["date"],
                    "text": chunk_txt[:300]               # preview for Stage 4 prompt
                })

            # ---- images ----
            if with_images:
                for img in art.get("images", []):
                    p = ROOT / img["path"]
                    if not p.exists():
                        continue
                    try:
                        pil = Image.open(p).convert("RGB")
                    except Exception:                      # corrupt file
                        continue
                    inp = clip_pp(pil).unsqueeze(0).to(dev)
                    with torch.no_grad():
                        vec = clip_model.encode_image(inp)
                        vec = vec / vec.norm(dim=-1, keepdim=True)
                    img_vecs.append(vec.cpu().numpy().astype("float32"))
                    img_meta.append({
                        "parent_id": aid,
                        "title": art["title"],
                        "alt": img.get("alt", ""),
                        "path": str(p.relative_to(ROOT)),
                    })

    _write("text", np.vstack(text_vecs), text_meta)
    if with_images:
        _write("image", np.vstack(img_vecs), img_meta)
    print("[done] Indexes & vectors saved to", OUT_DIR.relative_to(ROOT))


def _write(tag: str, vecs: np.ndarray, meta: list[dict]) -> None:
    """Dump both .npy and FAISS (if available)."""
    np.save(OUT_DIR / f"{tag}_vecs.npy", vecs)
    with (OUT_DIR / f"{tag}_meta.pkl").open("wb") as f:
        pickle.dump(meta, f)

    if faiss is None:
        print(f"[warn] faiss-cpu missing → saved {tag}_vecs.npy only ({len(meta)} items)")
        return

    idx = faiss.IndexIDMap(faiss.IndexFlatIP(vecs.shape[1]))
    idx.add_with_ids(vecs, np.arange(vecs.shape[0], dtype="int64"))
    faiss.write_index(idx, str(OUT_DIR / f"{tag}.index"))
    print(f"[info] {tag:<5} index built   ({len(meta)} items)")


# ---------- CLI -------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",  default=PROCESSED_JSON, type=Path)
    ap.add_argument("--chunk",  default=CHUNK_WORDS,    type=int, help="words per text chunk")
    ap.add_argument("--no-images", action="store_true", help="skip image embeddings")
    args = ap.parse_args()

    build(args.jsonl, args.chunk, with_images=not args.no_images)
