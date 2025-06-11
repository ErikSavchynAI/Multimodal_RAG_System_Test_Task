"""
Central configuration — adjust here, not in multiple files.
"""
from pathlib import Path
import os
import re

# ── project directories --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"                 # raw / processed / index live here

# ── models ---------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or exit("Set GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
EMB_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"

# ── retrieval parameters -------------------------------------------------
PREVIEW_BATCH      = 40
MAX_PREVIEW_ROUNDS = 4
FULL_K             = 40
REC_ALPHA          = 0.30           # 0: ignore date · 1: date only
LATEST_WORDS       = {"latest", "newest", "most recent", "last"}
STOPWORDS          = set("the a an is are was were of on in and to for how what why which".split())

# ── image parameters -----------------------------------------------------
IMG_SELECT_K = 3
MIN_IMG_SIM  = 0.15
IMG_MAXDIM   = 768
IMG_QUALITY  = 70

# ── generation -----------------------------------------------------------
TEMP    = 0.35
OUT_TOK = 32_768

# ── misc -----------------------------------------------------------------
URL_TMPL = "https://www.deeplearning.ai/the-batch/issue-{num}/"
ID_RE    = re.compile(r"issue-(\d+)")
