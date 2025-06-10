"""
Tiny wrapper around Sentence-Transformers with singleton initialisation.
"""
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMB_MODEL


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMB_MODEL)


def embed(text: str | List[str]) -> np.ndarray:
    """
    Return L2-normalised embedding(s).
    For a single string → 1-D array, for list[str] → 2-D array.
    """
    vec = _model().encode(text, normalize_embeddings=True)
    return vec if isinstance(text, list) else vec.astype("float32")
