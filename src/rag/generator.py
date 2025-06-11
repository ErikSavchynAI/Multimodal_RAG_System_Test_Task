"""
generator.py – high-level `answer()` entry point.

Workflow:
1. Split prior chat history from the new user question.
2. Retrieve relevant articles (text) + candidate images.
3. Compose a narrative prompt and query Gemini.
4. Linkify inline citations and attach matching images.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError

from .config import GEMINI_API_KEY, GEMINI_MODEL, OUT_TOK, TEMP
from .images import pick as pick_images, _jpeg_b64
from .retrieval import ARTICLES, choose_articles, source_url

genai.configure(api_key=GEMINI_API_KEY)
LLM = genai.GenerativeModel(GEMINI_MODEL)


def _llm_safe(prompt: str, **cfg) -> str:
    try:
        resp = LLM.generate_content(prompt, generation_config=cfg)
        return resp.text
    except (ValueError, GoogleAPICallError, Exception):
        return "Sorry, I couldn’t generate an answer just now."


def answer(prompt: str) -> Dict:
    # --- chat split -----------------------------------------------------
    if "NEW USER QUESTION:" in prompt:
        conv, question = map(str.strip, prompt.rsplit("NEW USER QUESTION:", 1))
    else:
        conv, question = "", prompt.strip()

    # --- retrieve -------------------------------------------------------
    aids, context = choose_articles(question)
    cand_imgs = [
        {**im, "parent": aid}
        for aid in aids
        for im in ARTICLES[aid].get("images", [])
    ]
    images = pick_images(question, cand_imgs)

    # --- compose & generate --------------------------------------------
    narrative = (
        "Write an engaging narrative: discuss **≤4** articles in depth "
        "(3–5 sentences each). Cite inline like [issue-123_news_1]."
    )
    p_final = (
        (f"{conv}\n\n" if conv else "")
        + narrative
        + "\n\nCONTEXT:\n"
        + context
        + "\n\nQUESTION:\n"
        + question
    )

    reply = _llm_safe(
        p_final,
        temperature=TEMP,
        max_output_tokens=OUT_TOK,
    )

    # --- linkify citations ---------------------------------------------
    id2url = {aid: source_url(aid) for aid in aids}
    reply = re.sub(
        r"\[(issue-[^\]]+)\]",
        lambda m: f"[{m.group(1)}]({id2url.get(m.group(1), '#')})",
        reply,
    )

    # --- image filtering / fallback ------------------------------------
    cited = set(re.findall(r"\[(issue-[^\]]+)\]", reply))
    images = [im for im in images if im["parent"] in cited]

    if not images and cand_imgs:
        for im in cand_imgs:
            if im["parent"] in cited:
                break
        else:
            im = cand_imgs[0]
        im = im.copy()
        im["file"] = Path(im["path"]).name
        im["data_uri"] = _jpeg_b64(Path(im["path"]))
        images = [im]

    return {"answer": reply, "sources": aids, "images": images}
