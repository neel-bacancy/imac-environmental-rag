"""Load the iMac index and answer from numbered element sources, with footnotes kept attached to their figures."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from app.chunker import Chunk
from app.config import Settings
from app.embedder import Embedder
from app.llm import Gemini
from app.retriever import Hit, Retriever, Trace, elapsed_ms

REFUSAL = "The report does not contain this information."

SYSTEM_PROMPT = f"""You answer questions about Apple's iMac Product Environmental Report (October 2024).

Rules:
- Use only the numbered sources. Do not use outside knowledge.
- Give figures exactly as printed, with their unit, and say what they refer to (product, configuration, year).
- Say where a figure comes from when it matters: a table, a chart, body text, or a footnote.
- A footnote attached to a source is part of that source. If a footnote limits or explains a figure (its
  configuration, definition, rounding or scope), state that limit next to the figure.
- If sources conflict, or a footnote contradicts a table's label, say so plainly. Do not pick one silently.
- End each factual sentence with its source tag, for example [S1].
- If the sources do not answer the question, reply exactly: "{REFUSAL}"
- Be concise."""


@dataclass
class Answer:
    question: str
    text: str
    query: str
    sources: list[Chunk]
    hits: list[Hit]
    trace: Trace | None = None  # every retrieval stage, for the step-by-step view
    prompt: str = ""  # the user message sent to the answer model (the system prompt is SYSTEM_PROMPT)
    timings_ms: dict[str, int] = field(default_factory=dict)

    @property
    def refused(self) -> bool:
        return REFUSAL.lower() in self.text.lower()


def load_retriever(settings: Settings, gemini: Gemini) -> Retriever:
    payload = json.loads((settings.index_dir / "index.json").read_text(encoding="utf-8"))
    chunks = [Chunk(**c) for c in payload["chunks"]]
    vectors = np.load(settings.index_dir / "vectors.npy")
    return Retriever(settings, gemini, Embedder(settings), chunks, vectors)


def build_prompt(question: str, sources: list[Chunk]) -> str:
    blocks = [f"[S{n}] {c.section or 'report'} | page {c.page} | {c.kind}\n{c.text}" for n, c in enumerate(sources, 1)]
    return "Sources:\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nQuestion: {question}"


def answer(question: str, retriever: Retriever, gemini: Gemini, settings: Settings, rerank: bool | None = None) -> Answer:
    trace = retriever.search(question, rerank=rerank)
    sources = [h.chunk for h in trace.hits]
    prompt = build_prompt(question, sources)
    t = time.perf_counter()
    text = gemini.generate(prompt, model=settings.llm_model, system=SYSTEM_PROMPT)
    timings = {**trace.timings_ms, "generation_ms": elapsed_ms(t)}
    return Answer(
        question=question,
        text=text,
        query=trace.query,
        sources=sources,
        hits=trace.hits,
        trace=trace,
        prompt=prompt,
        timings_ms=timings,
    )
