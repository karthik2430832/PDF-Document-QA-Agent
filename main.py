"""Command-line PDF Q&A with a polished terminal UI.

    python main.py path/to/file.pdf
    python main.py report.pdf --top-k 8 --model gemini-2.5-pro

Ask a question and watch a page-cited answer render live. Slash commands:
  /help  /sources  /clear  /quit
"""

import argparse
import logging
import sys

# Ensure UTF-8 output so panels/emoji render on Windows consoles (and pipes)
# instead of crashing under the legacy cp1252 codepage.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from pdfqa import Config, PdfQAEngine, pages_label

DEFAULT_PDF = "Karthik Resume GenAI.pdf"
EXIT_WORDS = {"quit", "exit", "q"}

SUGGESTIONS = [
    "Summarize this document in 3 bullet points",
    "What are the key points?",
    "What is this document about?",
]

THEME = Theme(
    {
        "brand": "bold magenta",
        "user": "bold cyan",
        "ai": "bold green",
        "muted": "grey62",
        "src": "italic cyan",
        "err": "bold red",
    }
)
console = Console(theme=THEME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions about a PDF using Gemini + retrieval (RAG).",
    )
    parser.add_argument("pdf", nargs="?", default=DEFAULT_PDF, help=f"PDF to ask about (default: {DEFAULT_PDF!r}).")
    parser.add_argument("--model", help="Override the chat model.")
    parser.add_argument("--embed-model", help="Override the embedding model.")
    parser.add_argument("--top-k", type=int, help="How many passages to retrieve.")
    parser.add_argument("--no-cache", action="store_true", help="Skip the on-disk embedding cache.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show progress/retry messages.")
    return parser.parse_args()


def print_banner() -> None:
    console.print()
    console.print(
        Panel(
            Text.assemble(
                ("📄  PDF Q&A\n", "brand"),
                ("Ask anything about your document — answers cite their pages.", "muted"),
            ),
            border_style="brand",
            padding=(1, 4),
        )
    )


def print_ready(engine: PdfQAEngine) -> None:
    max_page = max((c.page_end for c in engine.chunks), default=0)
    info = Table.grid(padding=(0, 2))
    info.add_column(style="muted", justify="right")
    info.add_column(style="bold")
    info.add_row("document", engine.doc_name or "—")
    info.add_row("pages", str(max_page))
    info.add_row("chunks indexed", str(len(engine.chunks)))
    console.print(Panel(info, title="[ai]Ready[/]", border_style="ai", expand=False))

    tips = "  ".join(f"[muted]·[/] {s}" for s in SUGGESTIONS)
    console.print(f"\n[muted]Try:[/] {tips}")
    console.print("[muted]Type a question, or /help for commands.[/]\n")


def print_help() -> None:
    table = Table(title="Commands", title_style="brand", border_style="muted", expand=False)
    table.add_column("Command", style="user", no_wrap=True)
    table.add_column("What it does", style="muted")
    table.add_row("/help", "show this help")
    table.add_row("/sources", "show the pages cited by the last answer")
    table.add_row("/clear", "forget the conversation so far")
    table.add_row("/quit", "exit (also: quit, exit, q, Ctrl-C)")
    console.print(table)


def stream_answer(engine: PdfQAEngine, question: str, chunks, history: list[dict]) -> str:
    """Render the streamed answer live as markdown inside a panel."""
    answer = ""
    thinking = Panel(Spinner("dots", text=" thinking…", style="muted"), title="[ai]AI[/]", border_style="ai")
    with Live(thinking, console=console, refresh_per_second=12, vertical_overflow="visible") as live:
        for token in engine.answer_stream(question, chunks, history):
            answer += token
            live.update(Panel(Markdown(answer), title="[ai]AI[/]", border_style="ai"))
        if not answer:
            live.update(Panel("[muted](no response)[/]", title="[ai]AI[/]", border_style="ai"))
    return answer


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    overrides = {}
    if args.model:
        overrides["chat_model"] = args.model
    if args.embed_model:
        overrides["embed_model"] = args.embed_model
    if args.top_k:
        overrides["top_k"] = args.top_k
    if args.no_cache:
        overrides["use_cache"] = False

    print_banner()
    cfg = Config.from_env(**overrides)
    engine = PdfQAEngine(cfg)

    try:
        with console.status(f"[muted]Indexing {args.pdf} …[/]", spinner="dots"):
            engine.build(args.pdf)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[err]Error:[/] {exc}")
        return 1

    print_ready(engine)

    history: list[dict] = []
    last_chunks = None
    while True:
        try:
            question = console.input("[user]You[/] [muted]›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[brand]Goodbye![/]")
            return 0

        if not question:
            continue

        lowered = question.lower()
        if lowered in EXIT_WORDS or lowered == "/quit":
            console.print("[brand]Goodbye![/]")
            return 0
        if lowered == "/help":
            print_help()
            continue
        if lowered == "/clear":
            history.clear()
            console.print("[muted]Conversation cleared.[/]")
            continue
        if lowered == "/sources":
            if last_chunks:
                console.print(f"[src]↳ sources: {pages_label(last_chunks)}[/]")
            else:
                console.print("[muted]No question asked yet.[/]")
            continue

        try:
            last_chunks = engine.retrieve(question)
            answer = stream_answer(engine, question, last_chunks, history)
            console.print(f"[src]↳ sources: {pages_label(last_chunks)}[/]\n")
            history.append({"q": question, "a": answer})
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive
            console.print(f"[err]Error:[/] {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
