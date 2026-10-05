"""
Chat CLI (online half of the RAG pipeline): interactive question -> retrieve
-> prompt -> generate -> answer + citations (guideline v4, Sections 1, 6, 7).

Run from the project root:   python -m hypeonoath.serving.chat   (or: hypeonoath-chat)
(package installed: `pip install -e .`).

Never re-embeds the corpus: it only reads the index written by
build_index.py. The raw `sources` list is always printed (with line
ranges for code), so citations are visible even if the model forgets to
cite inline.
"""
from __future__ import annotations

import argparse
import logging
import sys

from hypeonoath.core import config
from hypeonoath.core.records import RetrievedItem
from hypeonoath.serving import llm_client
from hypeonoath.serving.prompt import build_prompt
from hypeonoath.serving.retriever import Retriever


def format_sources(items: list[RetrievedItem]) -> list[str]:
    lines = []
    for n, item in enumerate(items, start=1):
        chunks = item.ordered_chunks
        first, last = chunks[0], chunks[-1]
        where = first.source_document
        if first.line_start is not None:
            where += f":{first.line_start}-{last.line_end if last.line_end is not None else first.line_start}"
        extra = f", {first.chunk_method}" if first.chunk_method else ""
        if len(chunks) > 1:
            extra += f", parts {chunks[0].part}-{chunks[-1].part}/{first.total_parts}"
        lines.append(f"  [{n}] {where} ({first.content_type}{extra}) score={item.score:.3f} id={item.chunk.chunk_id}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with the indexed corpus (answers with citations)")
    parser.add_argument("-k", type=int, default=config.DEFAULT_TOP_K, help="top-k chunks to retrieve")
    parser.add_argument("--show-prompt", action="store_true", help="print the assembled prompt")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    try:
        retriever = Retriever()
    except Exception as exc:  # noqa: BLE001 -- missing index is the common case
        print(f"Could not open the index at {config.CHROMA_DIR} ({exc}). "
              "Run: python -m hypeonoath.indexing.build_index", file=sys.stderr)
        return 1

    print("Ask a question (empty line or 'exit' to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        items = retriever.retrieve(question, k=args.k)
        prompt = build_prompt(question, items)
        if args.show_prompt:
            print(prompt.as_text())
        try:
            print("\n" + llm_client.generate(prompt))
        except llm_client.LLMUnavailableError as exc:
            print(f"\n[LLM unavailable] {exc}")
        print("\nSources:")
        print("\n".join(format_sources(items)) or "  (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
