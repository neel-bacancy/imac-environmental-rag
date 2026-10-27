# iMac Product Environmental Report: Multimodal RAG

Question answering over Apple's 12-page *iMac Product Environmental Report* (October 2024). Most of the useful
facts in this report are not in plain sentences: they sit in a stacked carbon chart, in tables, in icon call-outs,
and in footnotes that limit the claims they are attached to. This project makes all of those searchable and keeps
each footnote attached to the figure it qualifies.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # then set GEMINI_API_KEY
python -m scripts.ingest        # extract, vision-read the pages, chunk, embed, save to data/index
python -m scripts.ask "How much less energy does iMac use compared to the ENERGY STAR requirement?"
python -m eval.run_eval --compare
```

Gemini is used for answering, query rewriting, reranking and page vision. Embeddings run locally (BAAI bge-small),
so ingesting and searching do not use the embedding quota.

## What is extracted, and how

The report has four kinds of content, and no single extractor handles them all. The pipeline routes each kind to
the method that works for it.

| Content | Example in this report | Method | Why |
|---|---|---|---|
| Prose | "iMac uses 58 percent less energy..." | Text layer, column-aware reading order | Reads cleanly |
| Footnote markers | "...ENERGY STAR.¹⁸" | Superscript spans (PyMuPDF flag bit), kept as `[^18]` | Keeps each claim tied to its note |
| Endnotes | "18 Energy consumption and energy efficiency values..." | Parsed as a sequence 1, 2, 3, ..., 19, with pages | The numbers alone are ambiguous ("2030", "ISO 14040") |
| Tables | Power consumption (p. 12), carbon footprint (p. 10) | Vision pass on the page image, cached | See below |
| Charts | Carbon bar chart (p. 2) | Vision pass | Chart labels are scattered in the text layer |
| Call-outs | "100% recycled aluminum in the stand" | Vision pass plus the text layer | Often icon-based |

**Why not PyMuPDF's table finder?** On this report `find_tables()` split each table into header and body fragments
(pp. 10 and 12 each came back as three pieces). The text layer also returns a table's labels and values as
separate runs. So tables go through the vision model, which reads each page image into rows.

**Why the vision output is trusted, and checked:** the prompt requires numbers copied exactly and "not printed"
where a chart segment has no label. Every table number with two or more digits is checked against the page's text
layer, and none were missing. The three table pages (pp. 10 and 12) were also compared with the rendered page by eye. Vision results are cached per page in `data/cache/vision/`, so a re-run costs no API calls.

## Chunking: one chunk per element

Fixed-size windows cut tables and charts in half, and a chart's labels mean nothing without its values. So each
element is its own chunk: a prose paragraph, a table, a chart, a call-out, or an endnote.

What makes a chunk correct for a question is its footnotes. Each footnote marker in an element is resolved to its
endnote text and appended to that element's chunk:

```
Greenhouse gas emissions ... Total product footprint 346 kg CO2e ...
Footnotes:
[^10] Greenhouse gas emissions were calculated ... based on iMac (two ports) with 512GB storage configuration.
```

Endnotes are also indexed as their own chunks, so a question about a definition ("ENERGY STAR", "Apple Trade In")
can find the note directly.

Result: 111 chunks (71 text, 16 call-outs, 3 tables, 2 charts, 19 footnotes), 21 of them carrying linked footnotes.

## Retrieval

1. **Rewrite:** a small Gemini model turns the question into a search query. It only rewrites.
2. **Hybrid search:** dense cosine (bge-small) and BM25, top 25 each.
3. **Fuse:** Reciprocal Rank Fusion, k = 60.
4. **Rerank:** an LLM listwise rerank of the top 10 chunks. On by default; see the results below.
5. **Answer:** the top 5 chunks, numbered and labelled with their type and page. The prompt tells the model to keep
   footnote limits next to the figures they qualify, and to say when two sources conflict instead of picking one.

## Evaluation (`eval/`)

`sample_questions.json` has the three sample questions from the brief and six more on the same report (table cells,
a footnote-linking test, and an out-of-scope refusal). That is nine questions in total. Each one lists the pages where its
evidence sits, and a `note` with the answer and the page it comes from, checked against the PDF.

| Run | Reranker | Passed | Retrieval hit | Answer pass |
|---|---|---|---|---|
| `20261026-132230` | off | 8/9 | 100% | 89% |
| `20261026-132230` | **on** | **9/9** | 100% | 100% |

The one question that separates them is q07 ("Which configuration does the 346 kg figure refer to?"). With the
reranker on, the answer names the 256GB configuration from the table, quotes footnote 10's 512GB, and says the two
conflict. With it off, the footnote chunk falls outside the top five, and the answer gives the 256GB figure with no
caveat. That is the footnote-linking behaviour the brief asks for, so the reranker stays on.

**A note on the meeting notes.** The reference artifact reported that a reranker lowered scores on this kind of
document. Here it raised them. The meeting's warning is a reason to measure, not a reason to skip the reranker, and
that is what this project does. Nine questions is a small set, though; treat the margin as indicative.

### Checks on the answers

- q02 (the sample "total footprint and product use vs manufacturing" question) answers 346 kg, production 50%,
  product use 45%, and mentions footnote 10's different configuration.
- "Manufacturing" maps to the table's **Production** row (50%). The report's definitions (p. 9) describe Production
  as extraction, production and transport of materials, plus the manufacture and assembly of parts. The answer
  reports the 50% figure under that label, so the mapping is left to the reader.
- The power table questions (q04, q05) check cells in a 3×4 grid, which the text layer alone cannot give.

## The configuration conflict in the source

The report's table on p. 10 labels 346 kg CO2e as "iMac (two ports) 256GB", but footnote 10 says the life cycle
assessment was "based on iMac (two ports) with 512GB storage configuration". The 391 kg figure in the next table is
for "iMac (four ports) 512GB". The report does not resolve the conflict, and neither does this system. It reports
both and says they differ.

## Limitations and next steps

- **Vision depends on the model.** The table pages were compared with the rendered pages, and the numbers were
  checked against the text layer. A wrong value that also appears in the text layer would pass that check, and
  chart pages were not reviewed cell by cell. Spot-check new documents.
- **Chart segments without printed values** are stored as "not printed". The stacked bar segments on p. 2 have
  no numbers of their own, so only their totals (445 kg, −99 kg, 346 kg) are searchable values.
- **Endnote parsing depends on the sequence.** It works because the notes run 1 to 19 with no gaps. A document with
  restarting numbering would need a different rule.
- **Footnote linking covers superscripts only.** A footnote marker typed as plain text would be missed.
- **Nine questions** is a smoke test, not a benchmark.

## Layout

```
app/
  extract.py   text layer, column-aware order, [^n] footnote markers, endnote sequence parsing
  vision.py    per-page vision extraction of tables and charts, cached, number cross-check
  chunker.py   one chunk per element, footnotes appended
  embedder.py  local bge-small embeddings
  retriever.py rewrite, BM25 + dense, RRF, optional rerank
  answer.py    grounded answer with numbered, typed sources
  llm.py       Gemini generation (text, JSON, image)
scripts/       ingest.py, ask.py
server/        FastAPI backend for the viewer (server/main.py)
frontend/      React + Vite viewer, one page per pipeline stage
eval/          sample_questions.json, run_eval.py, results/
data/raw/      iMac_PER_Oct2024.pdf
data/cache/vision/  one JSON per page
```

## Step-by-step viewer

A browser view of the same pipeline: the ingest stages (pages, endnotes, vision output, chunks, index) and,
for each question, every retrieval stage (rewrite, dense, BM25, fusion, rerank, context, answer).

```bash
uvicorn server.main:app --port 8000        # backend (from this folder, with the venv active)
cd frontend && npm install && npm run dev  # frontend on http://localhost:5173
```

The viewer reads the index and vision cache that `scripts.ingest` wrote, so run ingest first. It does not
re-run ingestion. Each question makes the same Gemini calls as `scripts.ask` (rewrite, optional rerank, answer).
