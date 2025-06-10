"""
images.py – caption-based image picker (guaranteed >=1 if available)
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from .config import IMG_MAXDIM, IMG_QUALITY, IMG_SELECT_K, MIN_IMG_SIM
from .embedder import embed


def _encode_jpeg(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((IMG_MAXDIM, IMG_MAXDIM))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=IMG_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _score(caption: str, qv) -> float:
    return float(embed(caption) @ qv) if caption else 0.0


def pick(question: str, candidates: List[Dict]) -> List[Dict]:
    """
    Return up to IMG_SELECT_K images.
    • Primary: caption similarity ≥ MIN_IMG_SIM.
    • Fallback: first image of first candidate article if none matched.
    Each image dict gains 'data_uri' key (base-64 JPEG).
    """
    if not candidates:
        return []

    qv = embed(question)
    ranked: List[Tuple[float, Dict]] = []

    for im in candidates:
        caption = im.get("alt") or im.get("title", "")
        sim = _score(caption, qv)
        if sim >= MIN_IMG_SIM:
            ranked.append((sim, im))

    # sort by similarity desc, enforce one per parent article
    ranked.sort(key=lambda t: -t[0])
    selected, parents = [], set()
    for sim, im in ranked:
        if im["parent"] in parents:
            continue
        im["data_uri"] = _encode_jpeg(Path(im["path"]))
        selected.append(im)
        parents.add(im["parent"])
        if len(selected) >= IMG_SELECT_K:
            break

    # fallback: at least one picture if any exist
    if not selected and candidates:
        first = candidates[0]
        first["data_uri"] = _encode_jpeg(Path(first["path"]))
        selected = [first]

    return selected
