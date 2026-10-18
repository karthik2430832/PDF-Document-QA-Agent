"""DocuMind AI — Conversational Document Intelligence Agent.

    streamlit run app.py

A full conversational agent UI for querying PDFs with citations and live streaming.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types

from pdfqa import Config, PdfQAEngine, pages_label

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI · Conversational PDF Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Visual Settings ────────────────────────────────────────────────────────────
USER_AVATAR = "🧑"
AI_AVATAR   = "🤖"

WELCOME_QUOTES = [
    ("“The more that you read, the more things you will know.”", "Dr. Seuss"),
    ("“Knowledge has to be improved, challenged, and increased constantly, or it vanishes.”", "Peter Drucker"),
    ("“Information is not knowledge. The only source of knowledge is experience and inquiry.”", "Albert Einstein"),
    ("“In the digital age, speed of insight is the ultimate competitive advantage.”", "DocuMind"),
]

EXAMPLE_QUESTIONS = [
    ("📋 Executive Summary", "Provide a comprehensive executive summary of this document."),
    ("💡 Key Insights", "What are the most critical takeaways or novel ideas discussed?"),
    ("🔍 Methodology & Findings", "What methodology was used and what were the main findings?"),
    ("⚠️ Risks & Challenges", "What potential risks, limitations, or challenges are highlighted?"),
]

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
]


# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_css() -> None:
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending  = None
    st.session_state.engine   = None
    st.session_state.doc_sig  = None
    st.session_state.uploaded_doc_name = None


def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key.strip()

    for fname in [".env", ".env.local"]:
        p = Path(__file__).parent / fname
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and not val.startswith("your-"):
                        return val
    return ""


# ─── Initialise session state ──────────────────────────────────────────────────
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending",  None)
st.session_state.setdefault("engine",   None)
st.session_state.setdefault("doc_sig",  None)
st.session_state.setdefault("uploaded_doc_name", None)

load_css()
api_key = get_api_key()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand-box">
            <div class="brand-logo-glow">🧠</div>
            <div>
                <div class="brand-text">DocuMind</div>
                <div style="font-size: 0.72rem; color: #64748b; font-weight: 600;">AGENTIC PDF INTELLIGENCE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-sec-title">Agent Configuration</div>', unsafe_allow_html=True)
    model = st.selectbox("Reasoning Engine", MODELS, index=0)
    top_k = st.slider(
        "Context Memory Depth (Passages)",
        min_value=2, max_value=12, value=5,
        help="Controls how many relevant document sections the agent references for each answer.",
    )
    temperature = st.slider(
        "Agent Creativity",
        min_value=0.0, max_value=1.0, value=0.2, step=0.05,
        help="Lower values yield strictly factual responses directly from text.",
    )

    st.markdown("---")
    if st.session_state.messages:
        transcript = "\n\n".join(
            f"{'User' if m['role'] == 'user' else 'DocuMind Agent'}: {m['content']}"
            for m in st.session_state.messages
        )
        st.download_button(
            "Export Chat Log",
            transcript,
            file_name="documind_chat.txt",
            use_container_width=True,
        )
        st.button("Clear Conversation", on_click=reset_chat, use_container_width=True)

    st.markdown(
        """
        <div style="text-align: center; font-size: 0.75rem; color: #475569; margin-top: 2rem;">
            Powered by <b>Google Gemini 2.5</b><br>
            Secure, Instant & Document-Grounded
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Agent Top Bar ─────────────────────────────────────────────────────────────
doc_badge = (
    f'<span class="doc-pill">📄 {st.session_state.uploaded_doc_name}</span>'
    if st.session_state.uploaded_doc_name
    else '<span class="doc-pill unattached">📎 No Document Attached Yet</span>'
)

st.markdown(
    f"""
    <div class="agent-header">
        <div class="agent-profile">
            <div class="agent-avatar-wrap">
                🧠
                <div class="agent-status-dot"></div>
            </div>
            <div class="agent-info">
                <h2>DocuMind Assistant</h2>
                <div class="agent-subtitle">
                    <span>Model: <b>{model}</b></span>
                    <span>•</span>
                    {doc_badge}
                </div>
            </div>
        </div>
        <div style="font-size: 0.8rem; color: #10b981; font-weight: 600; display: flex; align-items: center; gap: 0.4rem;">
            <span style="font-size: 0.6rem;">🟢</span> Online & Ready
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Center Screen Welcome & Greeting (Shown Before Conversation Starts) ───────
if not st.session_state.messages:
    quote_text, quote_author = WELCOME_QUOTES[0]
    st.markdown(
        f"""
        <div class="welcome-hero">
            <div class="welcome-badge">✨ Conversational Document Agent</div>
            <div class="welcome-title">Where Inquiry Meets Clarity.</div>
            <div class="welcome-quote">
                {quote_text}<br>
                — <b>{quote_author}</b>
            </div>
            <div class="feature-cards">
                <div class="feature-card">
                    <div class="feature-icon">💬</div>
                    <div class="feature-title">Ask Directly or Say Hi</div>
                    <div class="feature-text">Start a dialogue anytime. Ask broad questions, brainstorm ideas, or query concepts.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📎</div>
                    <div class="feature-title">Attach in Chat Box</div>
                    <div class="feature-text">Click the paperclip in the bottom chat bar to attach any PDF directly with your message.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📑</div>
                    <div class="feature-title">Exact Page Citations</div>
                    <div class="feature-text">Answers automatically cross-reference specific page excerpts for 100% verification.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Metadata Strip (When Document is Loaded) ──────────────────────────────────
engine: PdfQAEngine | None = st.session_state.engine
if engine and engine.chunks:
    max_page = max((c.page_end for c in engine.chunks), default=0)
    n_chunks = len(engine.chunks)
    st.markdown(
        f"""
        <div class="stats-strip">
            <div class="stat-chip">📄 Document: <b>{st.session_state.uploaded_doc_name}</b></div>
            <div class="stat-chip">📖 Pages: <b>{max_page}</b></div>
            <div class="stat-chip">🧩 Sections: <b>{n_chunks}</b></div>
            <div class="stat-chip">🎯 Depth: <b>{top_k} passages</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Render Message History ────────────────────────────────────────────────────
for message in st.session_state.messages:
    is_user = message["role"] == "user"
    avatar = USER_AVATAR if is_user else AI_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("passages"):
            with st.expander(f"📑 Referencing Sources ({message.get('sources', '')})"):
                for p in message["passages"]:
                    st.markdown(
                        f'<span class="src-pill">{p["label"]}</span> {p["text"]}…',
                        unsafe_allow_html=True,
                    )

# ─── Suggested Starters (When document is attached) ────────────────────────────
if engine and engine.chunks and len(st.session_state.messages) <= 2:
    st.markdown('<div class="prompt-chips-label">✨ Quick Questions For This Document</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for i, (label, question) in enumerate(EXAMPLE_QUESTIONS):
        col = c1 if i % 2 == 0 else c2
        if col.button(f"{label}", key=f"starter_{i}", use_container_width=True):
            st.session_state.pending = question

# ─── Conversational Chat Input with Integrated Attachment ─────────────────────
# accept_file=True embeds the native paperclip upload button right inside the chat bar!
chat_submission = st.chat_input(
    "Ask a question, say hello, or attach a PDF with the paperclip...",
    accept_file=True,
    file_type=["pdf"],
)

user_text = None
attached_file = None

if chat_submission is not None:
    if isinstance(chat_submission, str):
        user_text = chat_submission
    else:
        user_text = chat_submission.text
        if hasattr(chat_submission, "files") and chat_submission.files:
            attached_file = chat_submission.files[0]

# Handle button-clicked quick prompts
if not user_text and st.session_state.pending:
    user_text = st.session_state.pending
st.session_state.pending = None

# If user attached a PDF via the paperclip button, index it immediately
if attached_file is not None:
    file_bytes = attached_file.getvalue()
    doc_sig = (attached_file.name, len(file_bytes), model, top_k, temperature)

    if st.session_state.doc_sig != doc_sig:
        with st.spinner(f"⚡ Ingesting & indexing **{attached_file.name}**..."):
            try:
                cfg = Config(
                    api_key=api_key,
                    chat_model=model,
                    top_k=top_k,
                    temperature=temperature,
                )
                engine = PdfQAEngine(cfg)
                engine.build(data=file_bytes, name=attached_file.name)
                st.session_state.engine = engine
                st.session_state.doc_sig = doc_sig
                st.session_state.uploaded_doc_name = attached_file.name

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"📎 **{attached_file.name}** has been attached and indexed successfully "
                            f"({len(engine.chunks)} sections parsed, {max((c.page_end for c in engine.chunks), default=0)} pages). "
                            "Ask any question about it below!"
                        ),
                    }
                )
                st.toast(f"Attached {attached_file.name}!", icon="📄")
            except Exception as exc:
                st.error(f"Failed to process document: {exc}")

# If user provided a message prompt
if user_text:
    prompt = user_text.strip()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    history = [
        {"q": u["content"], "a": a["content"]}
        for u, a in zip(
            st.session_state.messages[0:-1:2],
            st.session_state.messages[1::2],
        )
    ]

    engine = st.session_state.engine
    with st.chat_message("assistant", avatar=AI_AVATAR):
        try:
            # Case 1: PDF is attached -> Document-grounded RAG retrieval
            if engine is not None and engine.chunks:
                chunks  = engine.retrieve(prompt)
                answer  = st.write_stream(engine.answer_stream(prompt, chunks, history))
                sources = pages_label(chunks)
                passages = [
                    {
                        "label": (
                            f"p.{c.page_start}"
                            if c.page_start == c.page_end
                            else f"pp.{c.page_start}–{c.page_end}"
                        ),
                        "text": " ".join(c.text.split())[:280],
                    }
                    for c in chunks
                ]
                if sources:
                    with st.expander(f"📑 Referencing Sources ({sources})"):
                        for p in passages:
                            st.markdown(
                                f'<span class="src-pill">{p["label"]}</span> {p["text"]}…',
                                unsafe_allow_html=True,
                            )
            # Case 2: No document yet -> Friendly conversational assistant mode
            else:
                client = genai.Client(api_key=api_key)
                system_prompt = (
                    "You are DocuMind, an intelligent, helpful, and eloquent conversational AI assistant. "
                    "The user has not attached a PDF document yet. Chat with them engagingly and helpfully. "
                    "If they ask specific questions about a document, report, or file, politely remind them "
                    "that they can attach a PDF anytime using the paperclip icon in this chat input box."
                )
                contents = []
                for turn in history or []:
                    contents.append(types.Content(role="user", parts=[types.Part(text=turn["q"])]))
                    contents.append(types.Content(role="model", parts=[types.Part(text=turn["a"])]))
                contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

                def stream_general():
                    resp = client.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=temperature,
                        ),
                    )
                    for chunk in resp:
                        if chunk.text:
                            yield chunk.text

                answer = st.write_stream(stream_general())
                sources = ""
                passages = []

        except Exception as exc:
            answer   = f"⚠️ Agent encountered an issue: {exc}"
            sources  = ""
            passages = []
            st.error(answer)

    st.session_state.messages.append(
        {
            "role":     "assistant",
            "content":  answer,
            "sources":  sources,
            "passages": passages,
        }
    )
    st.rerun()
