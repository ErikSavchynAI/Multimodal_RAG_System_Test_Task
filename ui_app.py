"""
ui_app.py – minimal Streamlit chat (release)
Run with:
    streamlit run ui_app.py
Requires environment variable GEMINI_API_KEY.
"""
from __future__ import annotations
import base64, mimetypes, re
from pathlib import Path

import streamlit as st

from src.rag.generator import answer as rag_answer

ROOT = Path(__file__).resolve().parent
IMG_ROOT = ROOT / "data" / "raw" / "images"
IMG_WIDTH = 50  # percent

st.set_page_config("📰 The Batch Chat", layout="wide")
st.title("📰 Ask *The Batch*")

if "history" not in st.session_state:
    st.session_state.history: list[dict] = []


def img_tag(path: Path, caption: str) -> str:
    mime, _ = mimetypes.guess_type(path.name); mime = mime or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'\n\n<img src="data:{mime};base64,{b64}" alt="{caption}" style="width:{IMG_WIDTH}%;border-radius:8px;">\n\n'


# replay chat
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        for im in msg.get("images", []):
            st.markdown(img_tag(IMG_ROOT / im["file"], im.get("alt") or im.get("title")), unsafe_allow_html=True)

# new question
if q := st.chat_input("Ask The Batch…"):
    st.chat_message("user").markdown(q)
    st.session_state.history.append({"role": "user", "content": q})

    # build simple 1-turn context (last 10 msg)
    ctx = "\n\n".join(f"{m['role'].upper()}: {m['content']}"
                      for m in st.session_state.history[-10:]
                      if m["role"] in {"user", "assistant"})
    backend_prompt = q if not ctx else f"{ctx}\n\nNEW USER QUESTION:\n{q}"

    with st.spinner("Retrieving…"):
        res = rag_answer(backend_prompt)

    answer_md, imgs = res["answer"], res["images"]

    # inline images at first citation
    id2img = {im["parent"]: im for im in imgs}
    parts, last = [], 0
    for m in re.finditer(r"\[(issue-[^\]]+)\]", answer_md):
        parts.append(answer_md[last : m.end()])
        aid = m.group(1)
        if aid in id2img:
            im = id2img.pop(aid)
            parts.append(img_tag(IMG_ROOT / im["file"], im.get("alt") or im.get("title")))
        last = m.end()
    parts.append(answer_md[last:])

    with st.chat_message("assistant"):
        st.markdown("".join(parts), unsafe_allow_html=True)
        for im in id2img.values():
            st.markdown(img_tag(IMG_ROOT / im["file"], im.get("alt") or im.get("title")), unsafe_allow_html=True)

    st.session_state.history.append({"role": "assistant", "content": answer_md, "images": imgs})
