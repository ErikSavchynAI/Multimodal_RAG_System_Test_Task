"""
images.py – caption-based image selector.

Returns ≤ IMG_SELECT_K images (always ≥ 1 when candidates exist), each
dict containing:
    parent   – article ID
    path     – absolute path on disk
    file     – filename for UI
    data_uri – base-64-encoded, resized JPEG preview
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
    img = Image.open(path).convert("RGB")
    img.thumbnail((IMG_MAXDIM, IMG_MAXDIM))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=IMG_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def pick(question: str, candidates: List[Dict]) -> List[Dict]:
    if not candidates:
        return []

    qv = embed(question)
    scored: list[tuple[float, Dict]] = []

    for im in candidates:
        item = copy.deepcopy(im)
        item["parent"] = item.get("parent") or Path(item["path"]).stem.split("_")[0]
        caption = item.get("alt") or item.get("title", "")
        sim = float(embed(caption) @ qv) if caption else 0.0
        scored.append((sim, item))

    scored.sort(key=lambda t: -t[0])

    selected, parents = [], set()
    for sim, im in scored:
        if sim < MIN_IMG_SIM:
            break
        if im["parent"] in parents:
            continue
        im["file"] = Path(im["path"]).name
        im["data_uri"] = _jpeg_b64(Path(im["path"]))
        selected.append(im)
        parents.add(im["parent"])
        if len(selected) >= IMG_SELECT_K:
            break

    if not selected:
        sim, im = scored[0]
        im["file"] = Path(im["path"]).name
        im["data_uri"] = _jpeg_b64(Path(im["path"]))
        selected = [im]

    return selected
