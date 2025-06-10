"""
Streamlit chat UI for Multimodal RAG · rev 4
────────────────────────────────────────────
• Embeds images as Base-64 in markdown so they render without HTTP requests.
• Keeps conversation context (last 3 turns) for every backend call.
• No deprecated Streamlit args.
"""
from __future__ import annotations
import base64, mimetypes, os, re, textwrap
from pathlib import Path
import streamlit as st
from PIL import Image

from src.rag.qa import answer as rag_answer

ROOT = Path(__file__).resolve().parents[0]  # repo root
IMG_ROOT = ROOT.resolve()

st.set_page_config("📰 The Batch · Chat", layout="wide")
st.title("📰 Ask *The Batch*")

# -------- session state ---------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list[dict(role,content,images)]

# -------- sidebar ---------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    enable_img = st.checkbox(
        "Enable image reasoning", value=os.getenv("GEMINI_VISION", "0") == "1"
    )
    os.environ["GEMINI_VISION"] = "1" if enable_img else "0"
    st.markdown(
        textwrap.dedent(
            """
            **Examples**
            • What is the latest issue about?  
            • Summarise May 27 2020 issue with an image.  
            • How often does The Batch come out?
            """
        )
    )

# -------- helper: Base-64 inline markdown ---------------------------------
def _img_md(path: Path, caption: str) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        mime = "image/jpeg"
    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'\n\n![{caption}](data:{mime};base64,{b64})\n\n'

# -------- replay history --------------------------------------------------
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        for im in msg.get("images", []):
            md = _img_md(IMG_ROOT / im["path"], im.get("alt") or im.get("title"))
            st.markdown(md, unsafe_allow_html=True)

# -------- new user input --------------------------------------------------
if user_q := st.chat_input("Ask The Batch…"):
    st.chat_message("user").markdown(user_q)
    st.session_state.history.append({"role": "user", "content": user_q})

    # build short conversational context
    ctx_pairs = st.session_state.history[-6:]  # last 3 Q&A
    convo = "\n\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in ctx_pairs
        if m["role"] in {"user", "assistant"}
    )
    backend_prompt = user_q if not convo else f"{convo}\n\nNEW USER QUESTION:\n{user_q}"

    with st.spinner("Retrieving…"):
        res = rag_answer(backend_prompt)

    answer, imgs = res["answer"], res["images"]

    # inject inline images
    id2im = {im["parent"]: im for im in imgs}
    parts, last = [], 0
    for m in re.finditer(r"\[(issue-[^\]]+)\]", answer):
        parts.append(answer[last:m.end()])
        iid = m.group(1)
        if iid in id2im:
            im = id2im.pop(iid)
            parts.append(_img_md(IMG_ROOT / im["path"], im.get("alt") or im.get("title")))
        last = m.end()
    parts.append(answer[last:])

    with st.chat_message("assistant"):
        st.markdown("".join(parts), unsafe_allow_html=True)
        # leftover images
        for im in id2im.values():
            st.markdown(_img_md(IMG_ROOT / im["path"], im.get("alt") or im.get("title")),
                        unsafe_allow_html=True)

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "images": imgs}
    )
