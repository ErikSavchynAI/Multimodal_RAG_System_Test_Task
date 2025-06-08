"""Stage4–Retriever+LLMAnswerGenerator
========================================
Expose a single function `answer(query: str, *, k: int = 4) -> dict` that:

1.Embeds the **query** with the same models used for indexing.
2.Searches the *text* FAISS index (and optionally *image*) for top‑k hits.
3.Builds a prompt with numbered context snippets + minimal instructions.
4.Invokes a local **Mistral‑7B‑Instruct** (quantised) via
   `transformers.AutoModelForCausalLM`.
5.Returns a dict suitable for the Streamlit UI:
   ```python
   {
       "answer": "…string…",
       "sources": [ {"title": str, "date": str, "chunk": int} , … ],
       "images":  [ {"path": str, "alt": str} , … ]
   }
   ```

Dependencies
------------
```bash
pip install transformers accelerate xformers  # for vLLM‑style speed‑ups
pip install bitsandbytes                    # 4‑bit quant loader
```

Environment variables (or defaults) control model path & generation
params. We load in 4‑bit to fit on a single 16GB GPU *or* run on CPU if
no GPU.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import faiss  # type: ignore
import numpy as np
import torch
from sentence_transformers import SentenceTransformer  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

# ---------------------------------------------------------------------------
# Paths & global objects -----------------------------------------------------

INDEX_DIR = Path("data/index")
TEXT_INDEX_PATH = INDEX_DIR / "text.index"
TEXT_META_PATH = INDEX_DIR / "text_meta.pkl"
IMAGE_INDEX_PATH = INDEX_DIR / "image.index"
IMAGE_META_PATH = INDEX_DIR / "image_meta.pkl"

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = os.getenv("RAG_LLM", "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")

# generation hyperparams – can be tuned via env vars
GEN_KWARGS = dict(
    max_new_tokens=int(os.getenv("RAG_MAX_NEW", 320)),
    temperature=float(os.getenv("RAG_TEMP", 0.2)),
    top_p=float(os.getenv("RAG_TOP_P", 0.9)),
)

# ---------------------------------------------------------------------------
# Lazy singletons ------------------------------------------------------------

_text_index = _text_meta = None  # populated at first call
_embedder = None
_llm_pipe: TextGenerationPipeline | None = None


def _load_text_index():
    global _text_index, _text_meta  # noqa: WPS420
    if _text_index is None:
        _text_index = faiss.read_index(str(TEXT_INDEX_PATH))
        _text_meta = pickle.load(TEXT_META_PATH.open("rb"))


def _load_embedder(device: str):
    global _embedder  # noqa: WPS420
    if _embedder is None:
        _embedder = SentenceTransformer(EMB_MODEL_NAME, device=device)
    return _embedder


def _load_llm(device: str):
    global _llm_pipe  # noqa: WPS420
    if _llm_pipe is not None:
        return _llm_pipe

    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        device_map="auto" if device != "cpu" else None,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(LLM_NAME, use_fast=True)
    _llm_pipe = TextGenerationPipeline(model=model, tokenizer=tok, device=0 if device != "cpu" else -1)
    return _llm_pipe

# ---------------------------------------------------------------------------
# Core API -------------------------------------------------------------------

PROMPT_TEMPLATE = """You are a helpful AI assistant. Use only the numbered context below to answer the question. Cite sources by number, e.g. [1]. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
Answer: """


def answer(query: str, *, k: int = 4) -> Dict:
    """Return answer + citations + images for *query*."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. embed query & retrieve ------------------------------------------------
    _load_text_index()
    embedder = _load_embedder(device)
    qvec = embedder.encode(query, normalize_embeddings=True).astype("float32")
    D, I = _text_index.search(qvec.reshape(1, -1), k)  # I shape (1,k)

    contexts: List[str] = []
    citations: List[dict] = []
    for rank, idx in enumerate(I[0]):
        meta = _text_meta[idx]
        snippet = meta["title"] + " – " + meta["date"] + ": " + meta.get("text", "")[:280]
        contexts.append(f"[{rank+1}] {snippet}")
        citations.append({"title": meta["title"], "date": meta["date"], "chunk": meta["chunk"]})

    prompt = PROMPT_TEMPLATE.format(context="\n".join(contexts), question=query)

    # 2. generate -------------------------------------------------------------
    llm = _load_llm(device)
    resp = llm(prompt, **GEN_KWARGS)[0]["generated_text"][len(prompt):].strip()

    return {
        "answer": resp,
        "sources": citations,
        "images": [],  # images retrieval could be added here
    }


if __name__ == "__main__":  # quick CLI demo
    import sys, json
    print(json.dumps(answer("What did Andrew Ng announce in the very first issue?"), indent=2))
