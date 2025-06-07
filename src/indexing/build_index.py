"""Stage3–Cross‑platformEmbedding & FAISSIndexBuilder
======================================================
Builds text+image vector indexes from the `batch_articles.jsonl` file.
Works out‑of‑the‑box on **macOS M‑series (mps)**, **Windows**, and **Linux
CUDA** machines.

Output layout
-------------
```
data/index/
├── text.index        # FAISS cosine index (Sentence‑Transformers)
├── text_meta.pkl
├── image.index       # FAISS cosine index (OpenCLIP image encoder)
└── image_meta.pkl
```
If FAISS is *not* available for your platform, raw vectors are stashed to
`text_vecs.npy` / `image_vecs.npy` so you can `pip install faiss-cpu`
later and rebuild instantly.

Install tips
------------
### macOS(AppleSilicon)
```bash
# 1– Install torch with MPS backend
pip install "torch==2.3.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# 2– Rest of deps (faiss‑cpu wheel exists for arm64 Py>=3.10)
pip install sentence-transformers openclip-torch faiss-cpu pillow tqdm
```
### Windows(CPUorCUDA)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121    # or cpu
pip install sentence-transformers openclip-torch faiss-cpu pillow tqdm
```

Run
---
```bash
python -m src.indexing.build_index                 # text+images, 200‑word chunks
python -m src.indexing.build_index --no-images     # text only (faster)
python -m src.indexing.build_index --chunk 300     # 300‑word chunks
```
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Attempt FAISS import (optional) -------------------------------------------

try:
    import faiss
except ImportError:  # pragma: no cover – let code run without FAISS
    faiss = None  # noqa: N816 – lowercase for clarity

# ---------------------------------------------------------------------------
# Config --------------------------------------------------------------------

DEFAULT_JSONL = Path("data/processed/batch_articles.jsonl")
OUT_DIR = Path("data/index")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384‑dim
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAIN   = "openai"

# ---------------------------------------------------------------------------
# Utility functions ---------------------------------------------------------

def pick_device() -> str:
    """CUDA ▶︎ MPS ▶︎ CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple‑Silicon GPU
        return "mps"
    return "cpu"


def chunk_text(text: str, max_words: int) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


# ---------------------------------------------------------------------------
# Model loader --------------------------------------------------------------

def load_models(device: str):
    txt_model = SentenceTransformer(TEXT_MODEL_NAME, device=device)

    import open_clip  # local import keeps dep optional

    clip_model, _, clip_pp = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAIN, device=device
    )
    clip_model.eval()
    return txt_model, clip_model, clip_pp


# ---------------------------------------------------------------------------
# Vector persistence --------------------------------------------------------

def save_index(name: str, vecs: List[np.ndarray], meta: List[dict]) -> None:
    arr = np.vstack(vecs).astype("float32")
    if faiss is None:
        np.save(OUT_DIR / f"{name}_vecs.npy", arr)
        with (OUT_DIR / f"{name}_meta.pkl").open("wb") as f:
            pickle.dump(meta, f)
        print(f"[warn] FAISS missing → dumped {name}_vecs.npy ({len(meta)} items)")
        return

    index = faiss.IndexIDMap(faiss.IndexFlatIP(arr.shape[1]))
    index.add_with_ids(arr, np.arange(arr.shape[0]).astype("int64"))
    faiss.write_index(index, str(OUT_DIR / f"{name}.index"))
    with (OUT_DIR / f"{name}_meta.pkl").open("wb") as f:
        pickle.dump(meta, f)
    print(f"[{name:<5}] {len(meta):>6} items → {name}.index")


# ---------------------------------------------------------------------------
# Main builder --------------------------------------------------------------


def build_indexes(jsonl: Path, chunk: int = 200, with_images: bool = True) -> None:  # noqa: D401
    device = pick_device()
    print(f"[info] Using device: {device} ({platform.system()})")

    txt_model, clip_model, clip_pp = load_models(device)

    text_vecs, text_meta = [], []
    img_vecs, img_meta = [], []

    with jsonl.open("r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc="Embedding records"):
            rec = json.loads(line)
            pid = rec["id"]

            # -- text ---------------------------------------------------
            for idx, chunk_txt in enumerate(chunk_text(rec["text"], chunk)):
                if chunk_txt.strip():
                    vec = txt_model.encode(chunk_txt, normalize_embeddings=True)
                    text_vecs.append(vec.astype("float32"))
                    text_meta.append({"parent_id": pid, "chunk": idx, "title": rec["title"], "date": rec["date"]})

            # -- images -------------------------------------------------
            if with_images:
                for img in rec.get("images", []):
                    p = Path(img["path"])
                    if not p.exists():
                        continue
                    try:
                        pil = Image.open(p).convert("RGB")
                    except Exception:
                        continue
                    inp = clip_pp(pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        vec = clip_model.encode_image(inp)
                        vec = vec / vec.norm(dim=-1, keepdim=True)
                    img_vecs.append(vec.cpu().numpy().astype("float32"))
                    img_meta.append({"parent_id": pid, "title": rec["title"], "alt": img.get("alt", ""), "path": p.as_posix()})

    save_index("text", text_vecs, text_meta)
    if with_images:
        save_index("image", img_vecs, img_meta)

    print("[done] Artifacts in", OUT_DIR.as_posix())


# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------

def main() -> None:  # noqa: D401
    ap = argparse.ArgumentParser(description="Build (or stash) FAISS indexes for TheBatch RAG")
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--chunk", type=int, default=200, help="words per text chunk (default 200)")
    ap.add_argument("--no-images", action="store_true", help="skip image embeddings")
    args = ap.parse_args()

    build_indexes(args.jsonl, chunk=args.chunk, with_images=not args.no_images)


if __name__ == "__main__":
    main()
