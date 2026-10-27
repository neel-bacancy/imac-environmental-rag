"""One chunk per element: a prose paragraph, a table, a chart, a callout, or an endnote.

Fixed-size windows would cut tables and charts in half, and a chart's labels are meaningless apart from its
values. So each element is its own chunk. What makes a chunk useful for a question is also what makes it
correct: its footnotes. Every footnote marker in the element ([^10]) is resolved to its endnote text and
appended to the chunk, so "346 kg" and the note that says how it was calculated are retrieved together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.extract import Endnote, PageText, markers_in
from app.vision import VisualElement

MIN_PROSE_WORDS = 6  # shorter text blocks are labels, already covered by the vision pass


@dataclass
class Chunk:
    id: str
    kind: str  # "text", "table", "chart", "callout" or "footnote"
    page: int
    section: str | None
    text: str  # what is shown and cited, with its footnotes appended
    footnote_numbers: list[int]
    embed_text: str  # heading, page and kind, then text: what is embedded and matched by BM25


def _clean(text: str) -> str:
    return text.replace("�", "").replace("  ", " ").strip()


def _footnote_block(numbers: list[int], notes: dict[int, Endnote]) -> str:
    lines = [f"[^{n}] {notes[n].text}" for n in numbers if n in notes]
    return ("\nFootnotes:\n" + "\n".join(lines)) if lines else ""


def build_chunks(pages: list[PageText], visuals: list[VisualElement], endnotes: list[Endnote]) -> list[Chunk]:
    notes = {n.number: n for n in endnotes}
    chunks: list[Chunk] = []

    def add(kind: str, page: int, section: str | None, text: str, footnotes: list[int]) -> None:
        text = _clean(text)
        body = text + _footnote_block(footnotes, notes)
        label = f"{section or 'iMac report'} | page {page} | {kind}"
        chunks.append(
            Chunk(
                id=f"C{len(chunks):03d}",
                kind=kind,
                page=page,
                section=section,
                text=body,
                footnote_numbers=footnotes,
                embed_text=f"{label}\n{body}",
            )
        )

    for page in pages:
        pending_heading = ""
        for block in page.blocks:
            words = block.text.split()
            if len(words) < MIN_PROSE_WORDS:
                pending_heading = f"{pending_heading} {block.text}".strip()
                continue
            text = f"{pending_heading} {block.text}".strip() if pending_heading else block.text
            pending_heading = ""
            footnotes = sorted(set(block.markers))
            add("text", page.page, page.title, text, footnotes)

    for el in visuals:
        page_markers = {n for p in pages if p.page == el.page for n in p.superscript_markers}
        footnotes = markers_in(el.text, page_markers)
        page_title = next((p.title for p in pages if p.page == el.page), None)
        add(el.kind, el.page, page_title, el.text, footnotes)

    for note in endnotes:
        add("footnote", note.page, "Endnotes", f"Footnote {note.number}: {note.text}", [])

    return chunks


def to_dict(chunk: Chunk) -> dict:
    return asdict(chunk)
