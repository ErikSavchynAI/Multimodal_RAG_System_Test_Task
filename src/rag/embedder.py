"""
embedder.py – cached wrapper around Sentence-Transformers.

* Loads the model only once (LRU-cached singleton).
* `embed()` returns L2-normalised vectors.
    - str  → 1-D  float32 array
    - list → 2-D  float32 array
"""
from functools import lru_cache
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMB_MODEL


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMB_MODEL)


def embed(text: Union[str, List[str]]) -> np.ndarray:
    vec = _model().encode(text, normalize_embeddings=True)
    return vec if isinstance(text, list) else vec.astype("float32")
