"""Build the iMac index: text layer + vision pass, one chunk per element, footnotes linked, then embed.

    python -m scripts.ingest
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pymupdf

from app.chunker import build_chunks, to_dict
from app.config import get_settings
from app.extract import parse_pages
from app.embedder import Embedder
from app.llm import Gemini
from app.vision import check_against_text, extract_visuals


def main() -> None:
    s = get_settings()
    pages, endnotes = parse_pages(str(s.pdf_path))
    page_count = pymupdf.open(str(s.pdf_path)).page_count

    gemini = Gemini(s)
    visuals = extract_visuals(s, gemini, page_count)
    chunks = build_chunks(pages, visuals, endnotes)

    kinds = Counter(c.kind for c in chunks)
    linked = [c for c in chunks if c.footnote_numbers]
    print(f"endnotes parsed: {len(endnotes)}")
    print(f"chunks: {len(chunks)}  by kind: {dict(kinds)}")
    print(f"chunks carrying footnotes: {len(linked)}")

    page_texts = {p.page: " ".join(b.text for b in p.blocks) for p in pages}
    for page_no in range(1, page_count + 1):
        page_texts.setdefault(page_no, "")
    text_layer = {}
    doc = pymupdf.open(str(s.pdf_path))
    for i, page in enumerate(doc, start=1):
        text_layer[i] = page.get_text()
    issues = check_against_text(visuals, text_layer)
    print(f"vision numbers not found in text layer: {len(issues)}")
    for issue in issues[:15]:
        print("  -", issue)

    vectors = Embedder(s).embed([c.embed_text for c in chunks], task="RETRIEVAL_DOCUMENT")
    s.index_dir.mkdir(parents=True, exist_ok=True)
    (s.index_dir / "index.json").write_text(
        json.dumps({"chunks": [to_dict(c) for c in chunks]}, ensure_ascii=False), encoding="utf-8"
    )
    np.save(s.index_dir / "vectors.npy", vectors)
    print(f"saved {len(chunks)} vectors ({vectors.shape[1]} dims) to {s.index_dir}")


if __name__ == "__main__":
    main()
