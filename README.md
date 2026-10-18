# DocuMind AI — Conversational PDF Intelligence Agent 🧠

A production-grade conversational agent that lets you chat naturally and ask questions about any PDF document with **live streaming answers** and **exact page citations**.

Powered by **Google Gemini** for generation and a high-performance **TF-IDF + Cosine similarity retrieval pipeline** running in pure Python/NumPy (100% portable, zero C++ runtime dependencies).

---

## 🌟 Key Features

- 💬 **Natural Conversational Agent**: Start conversing immediately (say "hi", brainstorm, ask general questions). No mandatory upfront PDF upload.
- 📎 **Integrated Chat Bar Attachment**: Upload your PDF directly using the native paperclip icon inside the chat bar.
- 📌 **Verifiable Page Citations**: Every document answer cross-references exact page numbers (e.g. `(p. 3)` or `(pp. 4–5)`), complete with expandable source passage previews.
- ⚡ **Real-Time Token Streaming**: Watch responses stream dynamically as they are synthesized.
- 🚀 **100% Native & Portable RAG Pipeline**: Uses `pypdfium2` and NumPy for local indexing and vector search. No PyTorch, no ONNX Runtime, and no Visual C++ Redistributable requirement—works out of the box on Windows (x64/ARM64), macOS, and Linux.
- 🎨 **Modern Cyber-Glassmorphism UI**: Beautiful dark theme with glowing agent status indicator, curated welcome quotes, quick prompt starters, and clean chat bubbles.
- 💾 **Smart Disk Caching**: Previously parsed PDFs are hashed and cached under `.pdfqa_cache/` for instant subsequent loads.
- 🖥️ **Dual Interface**: Includes both a web UI and an interactive terminal REPL.

---

## 🚀 Quick Start

### 1. Clone & Set Up Virtual Environment

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Your API Key

Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your-gemini-api-key-here
```
*(Optional: you can also set `PDFQA_CHAT_MODEL=gemini-2.5-flash`)*

---

## 💻 Running the Application

### Modern Conversational Web UI (Streamlit)

```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

- **Start chatting right away**: Say hello, ask questions, or explore capabilities.
- **Attach a PDF**: Click the **paperclip** icon in the chat input bar, pick any PDF, and immediately ask questions about it.
- **Sidebar tools**: Switch Gemini reasoning models, adjust passage retrieval depth, export chat logs, or clear history.

### Terminal REPL

```powershell
python main.py path/to/document.pdf
python main.py document.pdf --top-k 8 --model gemini-2.5-pro
```

Slash commands available in the terminal:
- `/help` — Display command guide
- `/sources` — View citations from the last response
- `/clear` — Reset conversation memory
- `/quit` — Exit the agent

---

## ⚙️ Architecture & Pipeline

```text
[PDF Document]
      │
      ▼
Page-Aware Extraction (pypdfium2)
      │
      ▼
Overlapping Chunking (preserving 1-based page numbers)
      │
      ▼
TF-IDF & Cosine Similarity Indexing (NumPy / Scipy-free)
      │
      ▼ [On-disk cache: .pdfqa_cache/]
Query ──► Vectorize ──► Cosine Top-K ──► Grounded Excerpts ──► Google Gemini
                                                                    │
                                                                    ▼
                                                    Live Streamed Cited Answer
```

| Module | Purpose |
|---|---|
| [`app.py`](app.py) | Full-featured conversational agent interface with integrated chat attachment. |
| [`main.py`](main.py) | Terminal REPL with rich formatted live output. |
| [`pdfqa/engine.py`](pdfqa/engine.py) | Indexing, cosine retrieval, caching, and Gemini streaming integration. |
| [`pdfqa/extract.py`](pdfqa/extract.py) | High-fidelity page-aware PDF text extraction via `pypdfium2`. |
| [`pdfqa/chunking.py`](pdfqa/chunking.py) | Sliding-window text chunker maintaining page coordinate metadata. |
| [`pdfqa/config.py`](pdfqa/config.py) | Runtime settings, environment variables, and model defaults. |
| [`assets/style.css`](assets/style.css) | Custom glassmorphism dark theme styling and responsive layout rules. |

---

## 🔧 Supported Models

- `gemini-2.5-flash` *(Default — blazing fast, highly capable)*
- `gemini-2.5-pro` *(Complex reasoning & deep synthesis)*
- `gemini-2.0-flash`

---

## 🛡️ License

MIT License. Feel free to use, modify, and distribute.
