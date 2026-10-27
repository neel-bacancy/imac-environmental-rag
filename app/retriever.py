"""Hybrid retrieval over element chunks: dense (bge-small) and BM25, fused with Reciprocal Rank Fusion.

Pipeline for one question:
  1. Rewrite: a small Gemini model turns the question into a search query (key terms, units, the product
     name). It only rewrites; it never answers.
  2. Search:  dense cosine and BM25 over the element chunks. Each element is one chunk, so a question
              about "power at 115V" matches the power table itself, not a paragraph near it.
  3. Fuse:    RRF, score(d) = sum over lists of 1 / (k + rank), k = 60.
  4. Rerank:  optional. Off by default; see README for the measured effect on this report.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from app.chunker import Chunk
from app.config import Settings
from app.embedder import Embedder
from app.llm import Gemini

STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was what which who will with "
    "does do say says about main much many".split()
)
# Words that are on nearly every page of this report. They match everything, so they carry no signal.
DOCUMENT_TERMS = frozenset("apple imac product environmental report".split())
TOKEN = re.compile(r"[a-z0-9]+")

REWRITE_PROMPT = """Rewrite this question about Apple's iMac Product Environmental Report as a short search query.
Keep the key terms, the units and any numbers in the question. Do not answer the question.
Return JSON: {{"query": "..."}}

Question: {question}"""

RERANK_PROMPT = """Question: {question}

Passages from the report:

{passages}

Rank every passage id from most to least useful for answering the question. A table or footnote that gives the
exact figure asked for ranks above a passage that only mentions the topic. Return JSON: {{"ranking": ["C1", "C2", ...]}}"""


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS and t not in DOCUMENT_TERMS]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.dl = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if len(docs) else 1.0
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.dl / self.avgdl)
        for term in set(query):
            if term not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term)
                if f:
                    out[i] += self.idf[term] * f * (self.k1 + 1) / (f + norm[i])
        return out


@dataclass
class Hit:
    chunk: Chunk
    score: float


def elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


@dataclass
class Trace:
    """What each retrieval stage produced for one question. Chunks are referred to by their index in the index."""

    query: str  # the rewritten search query
    dense: list[tuple[int, float]] = field(default_factory=list)  # cosine, best first (top dense_k)
    bm25: list[tuple[int, float]] = field(default_factory=list)  # BM25 score, best first (score > 0)
    fused: list[tuple[int, float]] = field(default_factory=list)  # RRF score, the reranker's candidates
    ranked: list[int] = field(default_factory=list)  # candidate order after reranking (= fused order if off)
    hits: list[Hit] = field(default_factory=list)  # the final_k chunks passed to the answer model
    reranked: bool = False
    timings_ms: dict[str, int] = field(default_factory=dict)


class Retriever:
    def __init__(self, settings: Settings, gemini: Gemini, embedder: Embedder, chunks: list[Chunk], vectors: np.ndarray):
        self.s = settings
        self.gem = gemini
        self.embedder = embedder
        self.chunks = chunks
        self.vectors = vectors
        self.bm25 = BM25([tokens(c.embed_text) for c in chunks])

    def rewrite(self, question: str) -> str:
        try:
            data = json.loads(self.gem.generate(REWRITE_PROMPT.format(question=question), model=self.s.router_model, json_output=True))
            return (data.get("query") or question).strip()
        except (json.JSONDecodeError, RuntimeError, AttributeError):
            return question

    def search(self, question: str, rerank: bool | None = None) -> Trace:
        rerank = self.s.use_reranker if rerank is None else rerank
        timings: dict[str, int] = {}

        t = time.perf_counter()
        query = self.rewrite(question)
        timings["rewrite_ms"] = elapsed_ms(t)

        t = time.perf_counter()
        query_vec = self.embedder.embed([query], task="RETRIEVAL_QUERY")[0]
        timings["embed_ms"] = elapsed_ms(t)

        t = time.perf_counter()
        dense = self.vectors @ query_vec
        dense_top = np.argsort(-dense)[: self.s.dense_k]

        bm25 = self.bm25.scores(tokens(f"{question} {query}"))
        order = np.argsort(-bm25)[: self.s.bm25_k]
        bm25_top = [int(i) for i in order if bm25[i] > 0]

        fused: defaultdict[int, float] = defaultdict(float)
        for rank, i in enumerate(dense_top):
            fused[int(i)] += 1 / (self.s.rrf_k + rank + 1)
        for rank, i in enumerate(bm25_top):
            fused[i] += 1 / (self.s.rrf_k + rank + 1)
        ranked = sorted(fused, key=lambda i: -fused[i])[: self.s.candidates]
        timings["search_ms"] = elapsed_ms(t)

        t = time.perf_counter()
        reranked = bool(rerank and len(ranked) > 1)
        final_order = self._rerank(question, ranked) if reranked else ranked
        timings["rerank_ms"] = elapsed_ms(t) if reranked else 0

        return Trace(
            query=query,
            dense=[(int(i), float(dense[i])) for i in dense_top],
            bm25=[(i, float(bm25[i])) for i in bm25_top],
            fused=[(i, fused[i]) for i in ranked],
            ranked=final_order,
            hits=[Hit(self.chunks[i], fused[i]) for i in final_order[: self.s.final_k]],
            reranked=reranked,
            timings_ms=timings,
        )

    def _rerank(self, question: str, indices: list[int]) -> list[int]:
        passages = "\n\n".join(f"[C{n}] {self.chunks[i].embed_text[:1200]}" for n, i in enumerate(indices, 1))
        try:
            data = json.loads(self.gem.generate(RERANK_PROMPT.format(question=question, passages=passages), model=self.s.router_model, json_output=True))
            order = []
            for tag in data["ranking"]:
                n = int(str(tag).lstrip("C"))
                if 1 <= n <= len(indices) and indices[n - 1] not in order:
                    order.append(indices[n - 1])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError):
            return indices
        return order + [i for i in indices if i not in order]
