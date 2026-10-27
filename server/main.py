"""FastAPI backend for the step-by-step RAG viewer. Run from the project root:

    uvicorn server.main:app --reload --port 8000

Ingest endpoints read what `python -m scripts.ingest` stored (index, vision cache) and the PDF itself.
The query endpoint runs the live pipeline and returns the output of every stage, not just the answer.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from contextlib import asynccontextmanager

import pymupdf
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.answer import SYSTEM_PROMPT, answer, load_retriever
from app.config import ROOT, get_settings
from app.extract import parse_pages
from app.llm import Gemini
from app.retriever import REWRITE_PROMPT, Retriever
from app.vision import check_against_text, read_cached_visuals

settings = get_settings()
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["gemini"] = Gemini(settings)
    state["retriever"] = load_retriever(settings, state["gemini"])
    state["overview"] = build_overview(state["retriever"])
    yield


app = FastAPI(title="iMac Environmental RAG", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])


def build_overview(retriever: Retriever) -> dict:
    """The ingest stages as data: what extraction read, what vision returned, how chunks were built."""
    pages, endnotes = parse_pages(str(settings.pdf_path))
    doc = pymupdf.open(str(settings.pdf_path))
    text_layer = {i: page.get_text() for i, page in enumerate(doc, start=1)}
    page_count = doc.page_count

    visuals = read_cached_visuals(settings, page_count)
    issues = check_against_text(visuals, text_layer)
    by_page = {p.page: p for p in pages}

    page_rows = []
    for n in range(1, page_count + 1):
        p = by_page.get(n)
        page_rows.append(
            {
                "page": n,
                "role": "body" if p else "endnotes",
                "title": p.title if p else None,
                "prose_blocks": len(p.blocks) if p else 0,
                "words": sum(len(b.text.split()) for b in p.blocks) if p else 0,
                "footnote_markers": sorted(p.superscript_markers) if p else [],
                "visuals": [{"kind": v.kind, "title": v.title, "text": v.text} for v in visuals if v.page == n],
            }
        )

    chunks = retriever.chunks
    sample = json.loads((ROOT / "eval" / "sample_questions.json").read_text(encoding="utf-8"))
    return {
        "pdf": {"name": settings.pdf_path.name, "pages": page_count},
        "pages": page_rows,
        "endnotes": [{"number": e.number, "page": e.page, "text": e.text} for e in endnotes],
        "vision": {"elements": len(visuals), "issues": issues},
        "chunks": {
            "total": len(chunks),
            "by_kind": dict(Counter(c.kind for c in chunks)),
            "footnote_linked": sum(1 for c in chunks if c.footnote_numbers),
        },
        "index": {
            "vectors": list(retriever.vectors.shape),
            "embedding_model": settings.embedding_model,
            "dense_k": settings.dense_k,
            "bm25_k": settings.bm25_k,
            "rrf_k": settings.rrf_k,
            "candidates": settings.candidates,
            "final_k": settings.final_k,
            "use_reranker": settings.use_reranker,
        },
        "models": {"answer": settings.llm_model, "router": settings.router_model},
        "sample_questions": [q["question"] for q in sample],
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "chunks_indexed": len(state["retriever"].chunks), "llm_model": settings.llm_model}


@app.get("/api/overview")
def overview() -> dict:
    return state["overview"]


@app.get("/api/chunks")
def chunks() -> list[dict]:
    return [
        {
            "id": c.id,
            "kind": c.kind,
            "page": c.page,
            "section": c.section,
            "text": c.text,
            "footnote_numbers": c.footnote_numbers,
        }
        for c in state["retriever"].chunks
    ]


@app.get("/api/pages/{page}/image")
def page_image(page: int) -> Response:
    doc = pymupdf.open(str(settings.pdf_path))
    if not 1 <= page <= doc.page_count:
        raise HTTPException(404, f"The report has {doc.page_count} pages.")
    return Response(doc[page - 1].get_pixmap(dpi=110).tobytes("png"), media_type="image/png")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    rerank: bool | None = None  # None = the setting in app/config.py


@app.post("/api/query")
def query(req: QueryRequest) -> dict:
    retriever: Retriever = state["retriever"]
    try:
        res = answer(req.question, retriever, state["gemini"], settings, rerank=req.rerank)
    except RuntimeError as e:
        raise HTTPException(502, f"Model call failed: {e}") from e

    trace = res.trace
    cid = {i: c.id for i, c in enumerate(retriever.chunks)}
    dense_rank = {i: r for r, (i, _) in enumerate(trace.dense, 1)}
    bm25_rank = {i: r for r, (i, _) in enumerate(trace.bm25, 1)}
    # Citations look like [S1] or [S1, S2]: collect every S-number inside a bracket group.
    cited = {int(n) for group in re.findall(r"\[((?:S\d+[,\s]*)+)\]", res.text) for n in re.findall(r"S(\d+)", group)}

    return {
        "question": req.question,
        "rewrite_prompt": REWRITE_PROMPT.format(question=req.question),
        "rewritten_query": trace.query,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": res.prompt,
        "answer": res.text,
        "found": not res.refused,
        "model": settings.llm_model,
        "timings_ms": res.timings_ms,
        "reranked": trace.reranked,
        "dense": [{"chunk_id": cid[i], "rank": r, "score": round(s, 4)} for r, (i, s) in enumerate(trace.dense, 1)],
        "bm25": [{"chunk_id": cid[i], "rank": r, "score": round(s, 3)} for r, (i, s) in enumerate(trace.bm25, 1)],
        "fused": [
            {
                "chunk_id": cid[i],
                "rank": r,
                "score": round(s, 5),
                "dense_rank": dense_rank.get(i),
                "bm25_rank": bm25_rank.get(i),
            }
            for r, (i, s) in enumerate(trace.fused, 1)
        ],
        "rerank_order": [cid[i] for i in trace.ranked],
        "sources": [
            {
                "index": n,
                "chunk_id": h.chunk.id,
                "kind": h.chunk.kind,
                "page": h.chunk.page,
                "section": h.chunk.section,
                "text": h.chunk.text,
                "score": round(h.score, 5),
                "cited": n in cited,
            }
            for n, h in enumerate(res.hits, 1)
        ],
    }
