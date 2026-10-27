"""Ask one question and see the search query, the sources used, and the answer.

    python -m scripts.ask "What is the iMac's idle power consumption with the display on at 115V?"
"""

from __future__ import annotations

import argparse

from app.answer import answer, load_retriever
from app.config import get_settings
from app.llm import Gemini


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--rerank", action="store_true", help="turn the reranker on for this question")
    args = parser.parse_args()

    settings = get_settings()
    gemini = Gemini(settings)
    retriever = load_retriever(settings, gemini)
    result = answer(args.question, retriever, gemini, settings, rerank=args.rerank or settings.use_reranker)

    print(f"Search query: {result.query}")
    print("Sources used:")
    for n, c in enumerate(result.sources, 1):
        print(f"  [S{n}] {c.kind} | page {c.page} | {(c.section or '')[:50]}")
    print()
    print(result.text)


if __name__ == "__main__":
    main()
