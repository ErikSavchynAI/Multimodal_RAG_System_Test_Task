"""
images.py – caption-based image selector (text-only)

Every returned dict has:
    parent   – article ID
    path     – original path on disk
    file     – filename only (for UI convenience)
    data_uri – base-64 JPEG (inline display)
"""
from __future__ import annotations

import base64
import copy
from io import BytesIO
from pathlib import Path
from typing import Dict, List

from PIL import Image

from .config import IMG_MAXDIM, IMG_QUALITY, IMG_SELECT_K, MIN_IMG_SIM
from .embedder import embed


def _jpeg_b64(path: Path) -> str:
    """Return base-64 JPEG (RGB, resized)."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((IMG_MAXDIM, IMG_MAXDIM))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=IMG_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def pick(question: str, candidates: List[Dict]) -> List[Dict]:
    """
    Select ≤ IMG_SELECT_K images whose captions match *question*.

    Guarantees *at least one* image if candidates list is non-empty.
    """
    if not candidates:
        return []

    qv = embed(question)
    scored: list[tuple[float, Dict]] = []

    # work on copies so we never mutate the original data
    for im in candidates:
        iw = copy.deepcopy(im)
        iw["parent"] = iw.get("parent") or Path(iw["path"]).stem.split("_")[0]
        caption = iw.get("alt") or iw.get("title", "")
        sim = float(embed(caption) @ qv) if caption else 0.0
        scored.append((sim, iw))

    # rank by similarity
    scored.sort(key=lambda t: -t[0])

    selected, seen = [], set()
    for sim, im in scored:
        if sim < MIN_IMG_SIM:
            break
        if im["parent"] in seen:
            continue
        im["file"] = Path(im["path"]).name
        im["data_uri"] = _jpeg_b64(Path(im["path"]))
        selected.append(im)
        seen.add(im["parent"])
        if len(selected) >= IMG_SELECT_K:
            break

    # fallback: first candidate if nothing passed threshold
    if not selected:
        sim, im = scored[0]
        im["file"] = Path(im["path"]).name
        im["data_uri"] = _jpeg_b64(Path(im["path"]))
        selected = [im]

    return selected
