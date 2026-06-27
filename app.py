"""Streamlit web UI for PDF Q&A — a polished, app-like chat experience.

    streamlit run app.py

Upload a PDF, then chat with it. Answers stream live and cite their pages.
"""

import os
from pathlib import Path

import streamlit as st

from pdfqa import Config, PdfQAEngine, pages_label

st.set_page_config(page_title="PDF Document Q&A", page_icon="📄", layout="centered")

# On Streamlit Community Cloud the API key is provided via the Secrets UI
# (st.secrets). Mirror it into the environment so Config.from_env() — which is
# framework-agnostic and reads os.environ — works both locally (.env) and in the
# cloud without any code changes.
try:
    for _key in ("GEMINI_API_KEY", "PDFQA_CHAT_MODEL", "PDFQA_EMBED_MODEL"):
        if _key not in os.environ and _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:  # no secrets.toml present (e.g. local dev) — that's fine
    pass

EXAMPLES = [
    "📝 Summarize this document",
    "🔑 What are the key points?",
    "❓ What is this about?",
]
USER_AVATAR = "🧑"
AI_AVATAR = "🤖"


def load_css() -> None:
    css = Path(__file__).parent / "assets" / "style.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>📄 PDF Document Q&A</h1>
            <p>Upload a PDF and chat with it — every answer is grounded in your
            document and cites the exact pages it came from.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, icon: str) -> str:
    return (
        f'<div class="stat-card"><div class="stat-icon">{icon}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-label">{label}</div></div>'
    )


def reset_chat() -> None:
    st.session_state.messages = []


# --------------------------------------------------------------------------- UI

load_css()
hero()
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    uploaded = st.file_uploader("Upload a PDF", type="pdf", label_visibility="visible")
    st.divider()
    model = st.text_input("Chat model", value="gemini-2.5-flash")
    top_k = st.slider("Passages to retrieve", min_value=2, max_value=12, value=5)

# ---- Empty state -----------------------------------------------------------
if uploaded is None:
    st.markdown(
        """
        <div class="empty">
            <div class="empty-icon">📂</div>
            <div class="empty-title">Drop a PDF to get started</div>
            <div class="empty-sub">Use the uploader in the sidebar, then ask anything about it.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ---- Build / rebuild the index when the file or settings change ------------
file_bytes = uploaded.getvalue()
doc_sig = (uploaded.name, len(file_bytes), model, top_k)
if st.session_state.get("doc_sig") != doc_sig:
    try:
        with st.spinner(f"Indexing {uploaded.name} …"):
            cfg = Config.from_env(chat_model=model, top_k=top_k)
            engine = PdfQAEngine(cfg)
            engine.build(data=file_bytes, name=uploaded.name)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not index this PDF: {exc}")
        st.stop()
    st.session_state.engine = engine
    st.session_state.doc_sig = doc_sig
    reset_chat()

engine: PdfQAEngine = st.session_state.engine

# ---- Document stat cards ---------------------------------------------------
max_page = max((c.page_end for c in engine.chunks), default=0)
c1, c2, c3 = st.columns(3)
c1.markdown(stat_card("Pages", str(max_page), "📄"), unsafe_allow_html=True)
c2.markdown(stat_card("Chunks", str(len(engine.chunks)), "🧩"), unsafe_allow_html=True)
c3.markdown(stat_card("Model", model.replace("gemini-", ""), "🤖"), unsafe_allow_html=True)
st.markdown(f"<p class='hint'>Ready · <b>{uploaded.name}</b></p>", unsafe_allow_html=True)

# ---- Conversation history --------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=USER_AVATAR if message["role"] == "user" else AI_AVATAR):
        st.markdown(message["content"])
        if message.get("passages"):
            with st.expander(f"📑 Sources · {message['sources']}"):
                for p in message["passages"]:
                    st.markdown(f"<span class='src-pill'>{p['label']}</span> {p['text']}…", unsafe_allow_html=True)

# ---- Example questions (only before the chat starts) -----------------------
if not st.session_state.messages:
    st.markdown("<p class='hint'>Try asking…</p>", unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, key=f"ex_{example}", use_container_width=True):
            st.session_state.pending = example.split(" ", 1)[1]  # drop the emoji

# ---- Handle input (typed or example chip) ----------------------------------
prompt = st.chat_input("Ask a question about the document")
if not prompt and st.session_state.pending:
    prompt = st.session_state.pending
st.session_state.pending = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    history = [
        {"q": u["content"], "a": a["content"]}
        for u, a in zip(st.session_state.messages[0:-1:2], st.session_state.messages[1::2])
    ]

    with st.chat_message("assistant", avatar=AI_AVATAR):
        try:
            chunks = engine.retrieve(prompt)
            answer = st.write_stream(engine.answer_stream(prompt, chunks, history))
            sources = pages_label(chunks)
            passages = [
                {
                    "label": f"p.{c.page_start}" if c.page_start == c.page_end else f"p.{c.page_start}-{c.page_end}",
                    "text": " ".join(c.text.split())[:240],
                }
                for c in chunks
            ]
            with st.expander(f"📑 Sources · {sources}"):
                for p in passages:
                    st.markdown(f"<span class='src-pill'>{p['label']}</span> {p['text']}…", unsafe_allow_html=True)
        except Exception as exc:  # noqa: BLE001
            answer, sources, passages = f"⚠️ {exc}", "", []
            st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "passages": passages}
    )

# ---- Sidebar utilities for the active conversation -------------------------
if st.session_state.messages:
    with st.sidebar:
        st.divider()
        transcript = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages)
        st.download_button("⬇️ Download conversation", transcript, file_name="conversation.txt", use_container_width=True)
        st.button("🗑️ Clear chat", on_click=reset_chat, use_container_width=True)
