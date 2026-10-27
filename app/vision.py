"""Vision extraction for tables and charts.

Why a vision pass: the text layer of this report cannot rebuild its tables. PyMuPDF's table finder splits
a table into header and body fragments, and the text layer returns a table's labels and its values as
separate runs. Chart labels come out in no particular order. So each page is rendered to an image and
read by the Gemini vision model into structured JSON: tables as rows, charts as labelled values.

Safeguards:
  * One call per page, cached to data/cache/vision/page_NN.json, so a re-run costs nothing.
  * The model is told to copy numbers exactly and to say when a value is not printed, rather than estimate.
  * Numbers are cross-checked against the text layer (see check_against_text) and the mismatches are reported.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from app.config import Settings
from app.llm import Gemini

VISION_PROMPT = """This is page {page} of Apple's iMac Product Environmental Report. Extract every TABLE and every
CHART on this page so the content can be searched. Do not include body paragraphs.

Return JSON in this shape:
{{"elements": [
  {{"type": "table", "title": "...", "headers": ["..."], "rows": [["label", "value", "..."]], "description": "..."}},
  {{"type": "chart", "title": "...", "series": [{{"label": "...", "value": "..."}}], "description": "..."}},
  {{"type": "callout", "title": "...", "description": "..."}}
]}}

Rules:
- Copy every number, unit and label exactly as printed (for example "0.27W", "91.1%", "346 kg CO2e", "58%").
- Keep superscript footnote markers that follow a label, such as "iMac emissions10", exactly as printed.
- For a chart, list each labelled bar or segment with its printed value. If a segment has no printed value,
  use "not printed" for its value. Never estimate a value from the bar's length.
- For a table, put the printed column headers in "headers" and one list per printed row in "rows".
  Keep multi-level headers in the title or description.
- "description": one sentence stating what the element shows.
- If the page has no table, chart or callout, return {{"elements": []}}."""


@dataclass
class VisualElement:
    page: int
    kind: str  # "table", "chart" or "callout"
    title: str
    text: str  # the searchable rendering of the element
    raw: dict = field(default_factory=dict)


def _render_page(pdf: str, page_no: int) -> bytes:
    doc = pymupdf.open(pdf)
    return doc[page_no - 1].get_pixmap(dpi=150).tobytes("png")


def _render_text(item: dict) -> str:
    kind = item.get("type", "")
    title = item.get("title", "").strip()
    description = item.get("description", "").strip()
    if kind == "table":
        headers = item.get("headers") or []
        rows = item.get("rows") or []
        lines = [f"Table: {title}"]
        if headers:
            lines.append(" | ".join(str(h) for h in headers))
            lines.append(" | ".join("---" for _ in headers))
        for row in rows:
            lines.append(" | ".join("" if c is None else str(c) for c in row))
        if description:
            lines.append(description)
        return "\n".join(lines)
    if kind == "chart":
        series = item.get("series") or []
        values = "; ".join(f"{s.get('label', '')} = {s.get('value', '')}" for s in series)
        return "\n".join(part for part in [f"Chart: {title}", description, f"Values: {values}" if values else ""] if part)
    return "\n".join(part for part in [f"Note: {title}", description] if part)


def _elements(page_no: int, data: dict) -> list[VisualElement]:
    return [
        VisualElement(
            page=page_no,
            kind=item.get("type", "callout"),
            title=item.get("title", ""),
            text=_render_text(item),
            raw=item,
        )
        for item in data.get("elements", [])
    ]


def read_cached_visuals(settings: Settings, page_count: int) -> list[VisualElement]:
    """What the last ingest extracted, read from the vision cache. Makes no API calls."""
    elements: list[VisualElement] = []
    for page_no in range(1, page_count + 1):
        cache = settings.vision_cache_dir / f"page_{page_no:02d}.json"
        if cache.exists():
            elements.extend(_elements(page_no, json.loads(cache.read_text(encoding="utf-8"))))
    return elements


def extract_visuals(settings: Settings, gemini: Gemini, page_count: int) -> list[VisualElement]:
    settings.vision_cache_dir.mkdir(parents=True, exist_ok=True)
    elements: list[VisualElement] = []
    for page_no in range(1, page_count + 1):
        cache = settings.vision_cache_dir / f"page_{page_no:02d}.json"
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            raw = gemini.generate(
                VISION_PROMPT.format(page=page_no),
                model=settings.llm_model,
                json_output=True,
                image_png=_render_page(str(settings.pdf_path), page_no),
            )
            data = json.loads(raw)
            cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        elements.extend(_elements(page_no, data))
    return elements


NUMBER = re.compile(r"\d+(?:\.\d+)?")


def check_against_text(elements: list[VisualElement], page_texts: dict[int, str]) -> list[str]:
    """Flag numbers the vision pass read that never appear in the page's text layer. A mismatch is not always
    an error (a value can be drawn, not text), so these are reported for review, not corrected silently."""
    issues = []
    for el in elements:
        if el.kind != "table":
            continue
        layer = page_texts.get(el.page, "")
        for value in NUMBER.findall(el.text):
            if len(value) >= 2 and value not in layer:
                issues.append(f"p{el.page} table '{el.title[:40]}': {value} not in text layer")
    return issues
