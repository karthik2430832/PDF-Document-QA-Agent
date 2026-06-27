# PDF Document Q&A Agent

Ask natural-language questions about any PDF and get answers grounded in the
document, with page-number citations. Works from the terminal or a small web UI.

Built on Google **Gemini** with a lightweight **retrieval-augmented generation
(RAG)** pipeline, so it stays fast and accurate even on large documents.

## Features

- 📄 **Any PDF** — pass a path on the CLI or drag-and-drop in the web UI.
- 🔎 **RAG retrieval** — the PDF is chunked and embedded; only the most relevant
  passages are sent to the model, instead of the whole document.
- 📌 **Page citations** — answers cite the pages they came from (e.g. `(p. 3)`).
- ⚡ **Streaming answers** — tokens appear as they're generated.
- 💾 **Embedding cache** — re-asking about the same PDF skips re-embedding.
- 🧱 **Robust** — automatic retry with backoff when the model is busy.
- 💬 **Two interfaces** — a terminal REPL and a Streamlit chat app.

## Setup

```bash
# 1. Install dependencies (a virtualenv is recommended)
pip install -r requirements.txt

# 2. Add your Gemini API key
cp .env.example .env        # then edit .env and paste your key
```

Get a free key at <https://aistudio.google.com/apikey>.

## Usage

### Command line

```bash
python main.py "Karthik Resume GenAI.pdf"     # any PDF path
python main.py report.pdf --top-k 8           # retrieve more passages
python main.py report.pdf --model gemini-2.5-pro
```

Then ask away. The answer renders live as formatted markdown in a panel, with a
`↳ sources` line. Slash commands: `/help`, `/sources`, `/clear`, `/quit`.

```text
You › What are the candidate's core skills?
╭─ AI ─────────────────────────────────────────────╮
│ The core skills include … (p. 1)                  │
╰───────────────────────────────────────────────────╯
↳ sources: p.1, p.2
```

| Flag | Description |
|------|-------------|
| `pdf` | Path to the PDF (defaults to the bundled resume). |
| `--model` | Chat model (default `gemini-2.5-flash`). |
| `--embed-model` | Embedding model (default `gemini-embedding-001`). |
| `--top-k` | Number of passages to retrieve (default 5). |
| `--no-cache` | Ignore the on-disk embedding cache. |
| `-v` / `--verbose` | Show indexing/retry progress. |

### Web UI

```bash
streamlit run app.py
```

Upload a PDF in the sidebar, then chat. You can adjust the model and retrieval
depth, download the conversation, and clear the chat.

## How it works

```text
PDF ──► extract text per page ──► chunk (with overlap, keeping page numbers)
    ──► embed chunks (Gemini, cached) ──► store vectors in memory
question ──► embed ──► cosine top-k ──► prompt Gemini with those excerpts
         ──► streamed, page-cited answer
```

| Module | Responsibility |
|--------|----------------|
| [pdfqa/extract.py](pdfqa/extract.py) | Page-aware text extraction (PyMuPDF). |
| [pdfqa/chunking.py](pdfqa/chunking.py) | Overlapping chunks that remember their pages. |
| [pdfqa/engine.py](pdfqa/engine.py) | Embeddings, retrieval, caching, streaming answers. |
| [pdfqa/config.py](pdfqa/config.py) | Settings + environment loading. |
| [main.py](main.py) | Terminal REPL. |
| [app.py](app.py) | Streamlit web UI. |

## Configuration

Settings live in [pdfqa/config.py](pdfqa/config.py) and can be overridden per run
via CLI flags, the Streamlit sidebar, or these environment variables:

- `GEMINI_API_KEY` (required)
- `PDFQA_CHAT_MODEL`, `PDFQA_EMBED_MODEL` (optional)

## Notes

- Scanned/image-only PDFs have no extractable text and will report an error;
  they'd need an OCR step first.
- Embeddings are cached under `.pdfqa_cache/` (gitignored).
