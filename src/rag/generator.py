"""
generator.py – high level `answer()`
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import google.generativeai as genai

from .config import GEMINI_API_KEY, GEMINI_MODEL, OUT_TOK, TEMP
from .images import pick as pick_images, _jpeg_b64
from .retrieval import ARTICLES, choose_articles, source_url

genai.configure(api_key=GEMINI_API_KEY)
LLM = genai.GenerativeModel(GEMINI_MODEL)


def answer(prompt: str) -> Dict:
    # --- split chat prefix --------------------------------------------
    if "NEW USER QUESTION:" in prompt:
        conv, question = prompt.rsplit("NEW USER QUESTION:", 1)
        conv, question = conv.strip(), question.strip()
    else:
        conv, question = "", prompt.strip()

    # --- retrieve articles/context ------------------------------------
    aids, context = choose_articles(question)

    # all images from chosen articles
    cand = [
        {**im, "parent": aid} for aid in aids for im in ARTICLES[aid].get("images", [])
    ]
    images = pick_images(question, cand)

    # --- build prompt --------------------------------------------------
    narrative = (
        "Write an engaging narrative: discuss **≤4** articles in depth "
        "(3–5 sentences each). Cite inline like [issue-123_news_1]."
    )
    p_final = (
        (conv + "\n\n" if conv else "")
        + narrative
        + "\n\nCONTEXT:\n"
        + context
        + "\n\nQUESTION:\n"
        + question
    )

    reply = LLM.generate_content(
        p_final, generation_config={"temperature": TEMP, "max_output_tokens": OUT_TOK}
    ).text

    # --- linkify citations -------------------------------------------
    id2url = {aid: source_url(aid) for aid in aids}
    reply = re.sub(
        r"\[(issue-[^\]]+)\]", lambda m: f"[{m.group(1)}]({id2url.get(m.group(1), '#')})", reply
    )

    # --- filter images / fallback ------------------------------------
    cited = set(re.findall(r"\[(issue-[^\]]+)\]", reply))
    images = [im for im in images if im["parent"] in cited]

    if not images and cand:
        # first image whose parent was cited, else first overall
        for im in cand:
            if im["parent"] in cited:
                break
        else:
            im = cand[0]
        im = im.copy()
        im["file"] = Path(im["path"]).name
        im["data_uri"] = _jpeg_b64(Path(im["path"]))
        images = [im]

    return {"answer": reply, "sources": aids, "images": images}
