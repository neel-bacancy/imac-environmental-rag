"""Text-layer extraction for the iMac report: prose blocks, footnote markers, and endnotes.

The report mixes three kinds of content, and each needs its own handling:

  * Prose blocks:   read from the text layer, in reading order, with superscript footnote markers kept
                    as "[^n]" so a claim and its footnote stay linked.
  * Endnotes:       the numbered notes at the back ("10 Greenhouse gas emissions were calculated ...").
                    They are parsed as a sequence (1, 2, 3, ...), because the numbers alone are ambiguous.
  * Tables/charts:  NOT read here. A text-layer table comes out as loose labels and values in separate
                    runs, and the chart labels are scattered. Those go to the vision pass (vision.py).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pymupdf

PAGE_HEADER_BAND = 50  # points from the top: running header
PAGE_FOOTER_BAND = 45  # points from the bottom: running footer and page number
SUPERSCRIPT = 1  # PyMuPDF span flag bit for superscript text


@dataclass
class ProseBlock:
    page: int
    text: str  # with footnote markers as [^n]
    markers: list[int]
    size: float  # dominant font size, used to detect headings


@dataclass
class Endnote:
    number: int
    text: str
    page: int


@dataclass
class PageText:
    page: int
    title: str | None  # the largest heading on the page, e.g. "Progress toward carbon neutral"
    blocks: list[ProseBlock] = field(default_factory=list)
    superscript_markers: set[int] = field(default_factory=set)


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)  # ligatures and compatibility forms
    t = t.replace(" ", " ").replace("­", "")
    t = t.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", t).strip()


def _reading_order(blocks: list[dict], width: float) -> list[dict]:
    """Column-aware order: full-width blocks in y order, narrow blocks left-to-right by column, then down."""
    ordered: list[dict] = []
    band: list[dict] = []

    def flush() -> None:
        mid = width / 2 - 10
        left = sorted((b for b in band if b["bbox"][0] < mid), key=lambda b: b["bbox"][1])
        right = sorted((b for b in band if b["bbox"][0] >= mid), key=lambda b: b["bbox"][1])
        ordered.extend(left + right)
        band.clear()

    for block in sorted(blocks, key=lambda b: b["bbox"][1]):
        if block["bbox"][2] - block["bbox"][0] > 0.55 * width:
            flush()
            ordered.append(block)
        else:
            band.append(block)
    flush()
    return ordered


def _block_text(block: dict) -> tuple[str, list[int], float]:
    """Join the block's spans. Superscript digits become [^n] markers placed right after their word."""
    parts: list[str] = []
    markers: list[int] = []
    sizes: list[tuple[int, float]] = []
    for line in block["lines"]:
        for span in line["spans"]:
            raw = span["text"]
            if not raw.strip():
                continue
            sizes.append((len(raw), span["size"]))
            if span["flags"] & SUPERSCRIPT and raw.strip().isdigit():
                n = int(raw.strip())
                markers.append(n)
                parts.append(f"[^{n}]")
            else:
                parts.append(raw)
        parts.append(" ")
    text = normalise("".join(parts))
    text = re.sub(r"\s+\[\^", "[^", text)  # "chemistry [^5]" -> "chemistry[^5]"
    # The text layer often drops the space after a sentence end or a closing bracket: "products.That's".
    text = re.sub(r"([.!?])(?=[A-Z][a-z])", r"\1 ", text)
    text = re.sub(r"(\[\^\d+\])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"([)])(?=[a-z])", r"\1 ", text)
    dominant = max(sizes, key=lambda s: s[0])[1] if sizes else 0.0
    return text, markers, dominant


def parse_pages(path: str) -> tuple[list[PageText], list[Endnote]]:
    doc = pymupdf.open(path)
    pages: list[PageText] = []
    endnote_source: list[tuple[int, str]] = []

    for page_no, page in enumerate(doc, start=1):
        width, height = page.rect.width, page.rect.height
        raw_blocks = []
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            y0, y1 = block["bbox"][1], block["bbox"][3]
            if y0 < PAGE_HEADER_BAND or y1 > height - PAGE_FOOTER_BAND:
                continue
            raw_blocks.append(block)

        ordered = _reading_order(raw_blocks, width)
        texts = [(_block_text(b), b) for b in ordered]
        headings = [t for (t, _, size), _ in texts if size >= 18 and t]
        is_endnote_page = any(t.startswith("Endnotes") for (t, _, _), _ in texts)

        if is_endnote_page:
            for (t, _, _), _ in texts:
                if t and t != "Endnotes":
                    endnote_source.append((page_no, re.sub(r"\[\^(\d+)\]", r"\1", t)))
            continue

        page_text = PageText(page=page_no, title=headings[0] if headings else None)
        for (text, markers, size), _ in texts:
            if not text or size >= 18:
                continue
            page_text.blocks.append(ProseBlock(page=page_no, text=text, markers=markers, size=size))
            page_text.superscript_markers.update(markers)
        pages.append(page_text)

    return pages, parse_endnotes(endnote_source)


def parse_endnotes(source: list[tuple[int, str]]) -> list[Endnote]:
    """Find notes 1, 2, 3, ... in order. A number starts a note only when it is the next expected number
    and is followed by a capital letter, bullet or quote, which avoids matches inside ISO numbers and dates."""
    text = ""
    page_at: list[tuple[int, int]] = []  # (char offset, page) so each note records the page it starts on
    for page, chunk in source:
        page_at.append((len(text), page))
        text += chunk + " "

    def page_of(offset: int) -> int:
        return max((p for o, p in page_at if o <= offset), default=source[0][0] if source else 0)

    starts: list[tuple[int, int, int]] = []  # (number, match start, text start)
    pos, expected = 0, 1
    while True:
        match = re.compile(rf"(?:^|\s){expected}\s*(?=[A-Za-z•\"(])").search(text, pos)
        if not match:
            break
        starts.append((expected, match.start(), match.end()))
        pos = match.end()
        expected += 1

    notes = []
    for i, (number, _, text_start) in enumerate(starts):
        text_end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        notes.append(Endnote(number=number, text=normalise(text[text_start:text_end]), page=page_of(text_start)))
    return notes


MARKER_IN_TEXT = re.compile(r"(?<=[A-Za-z\)\%])(\d{1,2})\b")


def markers_in(text: str, candidates: set[int]) -> list[int]:
    """Markers for a visual element, whose text is retyped from the image. A number glued to a word
    ("iMac emissions10") counts as a marker only if the page has that superscript."""
    found = []
    for m in MARKER_IN_TEXT.finditer(text):
        n = int(m.group(1))
        if n in candidates and n not in found:
            found.append(n)
    return found
