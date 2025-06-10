"""
generator.py – high-level `answer()` for the chat UI.
"""
from __future__ import annotations

import re
from typing import Dict

import google.generativeai as genai

from .config import GEMINI_API_KEY, GEMINI_MODEL, OUT_TOK, TEMP
from .images import pick as pick_images
from .retrieval import ARTICLES, choose_articles, source_url

genai.configure(api_key=GEMINI_API_KEY)
LLM = genai.GenerativeModel(GEMINI_MODEL)


def answer(prompt: str) -> Dict:
    # ── split chat prefix ------------------------------------------------
    if "NEW USER QUESTION:" in prompt:
        conv, question = prompt.rsplit("NEW USER QUESTION:", 1)
        conv, question = conv.strip(), question.strip()
    else:
        conv, question = "", prompt.strip()

    # ── choose relevant articles & context ------------------------------
    aids, context = choose_articles(question)

    # ── collect candidate images (attach parent id) ---------------------
    cand_imgs = []
    for aid in aids:
        for im in ARTICLES[aid].get("images", []):
            im2 = dict(im)           # shallow copy
            im2["parent"] = aid
            cand_imgs.append(im2)

    images = pick_images(question, cand_imgs)

    # ── build final prompt ----------------------------------------------
    narrative = (
        "Write an engaging narrative: discuss **≤4** articles in depth "
        "(3–5 sentences each). Cite sources inline like [issue-123_news_1]."
    )
    final_prompt = (
        (conv + "\n\n" if conv else "")
        + narrative
        + "\n\nCONTEXT:\n"
        + context
        + "\n\nQUESTION:\n"
        + question
    )

    reply = LLM.generate_content(
        final_prompt,
        generation_config={"temperature": TEMP, "max_output_tokens": OUT_TOK},
    ).text

    # ── linkify citations ----------------------------------------------
    id2url = {aid: source_url(aid) for aid in aids}
    reply = re.sub(
        r"\[(issue-[^\]]+)\]",
        lambda m: f"[{m.group(1)}]({id2url.get(m.group(1), '#')})",
        reply,
    )

    # ── keep only images whose parent was cited -------------------------
    cited = set(re.findall(r"\[(issue-[^\]]+)\]", reply))
    images = [im for im in images if im["parent"] in cited]

    return {"answer": reply, "sources": aids, "images": images}
