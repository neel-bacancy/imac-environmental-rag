import { useState, type ReactNode } from 'react'
import { pageImageUrl, pageLabel, type ChunkItem, type Overview, type PageRow } from '../api'
import { Step } from './Step'

interface Props {
  overview: Overview | null
  chunks: ChunkItem[]
}

export function IngestView({ overview, chunks }: Props) {
  const [page, setPage] = useState(1)
  const [kind, setKind] = useState('all')
  const [search, setSearch] = useState('')

  if (!overview) return <p className="empty">Loading the ingest overview…</p>

  const selected = overview.pages.find((p) => p.page === page) ?? overview.pages[0]
  const kinds = Object.keys(overview.chunks.by_kind)
  const needle = search.trim().toLowerCase()
  const shown = chunks.filter((c) => (kind === 'all' || c.kind === kind) && (!needle || c.text.toLowerCase().includes(needle)))
  const maxKind = Math.max(1, ...Object.values(overview.chunks.by_kind))

  return (
    <div className="ingest">
      <p className="intro">
        Ingestion runs once with <code>python -m scripts.ingest</code>. This view shows what that run produced, stage by stage. It reads the
        stored index and the vision cache, so it makes no model calls.
      </p>

      <Step
        n={1}
        title="Read the report"
        what={`${overview.pdf.name} has ${overview.pdf.pages} pages. Each page is read from its text layer and rendered as an image for the vision pass. Pick a page to see both.`}
      >
        <div className="thumbs">
          {overview.pages.map((p) => (
            <button key={p.page} className={`thumb ${p.page === selected.page ? 'active' : ''}`} onClick={() => setPage(p.page)}>
              <img src={pageImageUrl(p.page)} alt="" loading="lazy" />
              <span>{pageLabel(p.page)}</span>
            </button>
          ))}
        </div>
        <PageDetail row={selected} issues={overview.vision.issues} />
      </Step>

      <Step
        n={2}
        title="Endnotes"
        what="Prose is read in reading order, and superscript markers stay attached to their words. The endnotes at the back are parsed as the sequence 1, 2, 3… so each footnote can be linked to the figure it limits."
      >
        <details className="details" open>
          <summary>{overview.endnotes.length} endnotes parsed</summary>
          <ol className="endnotes">
            {overview.endnotes.map((e) => (
              <li key={e.number}>
                <span className="rank">{e.number}</span>
                <span className="muted">{pageLabel(e.page)}</span>
                <span>{e.text}</span>
              </li>
            ))}
          </ol>
        </details>
      </Step>

      <Step
        n={3}
        title="Tables and charts from vision"
        what="Tables, charts and callouts cannot be rebuilt from the text layer, so each page image is read by the model into structured rows. Its numbers are then checked against the text layer."
      >
        <p>
          <strong>{overview.vision.elements}</strong> elements extracted from the pages (shown per page in step 1).
        </p>
        {overview.vision.issues.length === 0 ? (
          <p className="muted">Every table number was found in the text layer.</p>
        ) : (
          <div className="notice">
            <p>
              {overview.vision.issues.length} table number(s) were not found in the text layer. They may be drawn rather than printed text, so
              review them:
            </p>
            <ul>
              {overview.vision.issues.map((issue) => (
                <li key={issue}>{issue}</li>
              ))}
            </ul>
          </div>
        )}
      </Step>

      <Step
        n={4}
        title="Chunk: one chunk per element"
        what={`Each paragraph, table, chart, callout and endnote becomes one chunk. The footnotes it refers to are appended, so a figure and its limit are retrieved together. ${overview.chunks.footnote_linked} of ${overview.chunks.total} chunks carry footnotes.`}
      >
        <div className="kind-bars">
          {kinds.map((k) => (
            <div key={k} className="kind-row">
              <span className={`kind kind-${k}`}>{k}</span>
              <span className="bar-track">
                <span className="bar-fill" style={{ width: `${(overview.chunks.by_kind[k] / maxKind) * 100}%` }} />
              </span>
              <span className="score">{overview.chunks.by_kind[k]}</span>
            </div>
          ))}
        </div>

        <div className="filters">
          <select value={kind} onChange={(e) => setKind(e.target.value)}>
            <option value="all">All kinds</option>
            {kinds.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
          <input type="search" placeholder="Filter chunk text…" value={search} onChange={(e) => setSearch(e.target.value)} />
          <span className="muted">
            {shown.length} of {chunks.length}
          </span>
        </div>
        <div className="chunk-list">
          {shown.map((c) => (
            <details key={c.id} className="chunk">
              <summary>
                <code>{c.id}</code>
                <span className={`kind kind-${c.kind}`}>{c.kind}</span>
                <span className="muted">{pageLabel(c.page)}</span>
                <span className="preview">{c.text.split('\n')[0]}</span>
              </summary>
              <pre>{c.text}</pre>
            </details>
          ))}
        </div>
      </Step>

      <Step
        n={5}
        title="Embed and index"
        what="Every chunk is embedded once with a local model, and its tokens go into a BM25 index. Queries search both. The settings below control retrieval."
      >
        <div className="stats">
          <div>
            <span className="muted">Dense vectors</span>
            <strong>
              {overview.index.vectors[0]} × {overview.index.vectors[1]}
            </strong>
          </div>
          <div>
            <span className="muted">Embedding model</span>
            <strong>{overview.index.embedding_model}</strong>
          </div>
          <div>
            <span className="muted">BM25 documents</span>
            <strong>{chunks.length}</strong>
          </div>
        </div>
        <table className="grid narrow">
          <tbody>
            <Row label="Dense candidates (top k)" value={overview.index.dense_k} />
            <Row label="BM25 candidates (top k)" value={overview.index.bm25_k} />
            <Row label="RRF constant k" value={overview.index.rrf_k} />
            <Row label="Candidates to reranker" value={overview.index.candidates} />
            <Row label="Chunks to answer model" value={overview.index.final_k} />
            <Row label="Reranker" value={overview.index.use_reranker ? 'on' : 'off'} />
            <Row label="Answer model" value={overview.models.answer} />
            <Row label="Router model (rewrite, rerank)" value={overview.models.router} />
          </tbody>
        </table>
      </Step>
    </div>
  )
}

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <tr>
      <td>{label}</td>
      <td>
        <strong>{value}</strong>
      </td>
    </tr>
  )
}

function PageDetail({ row, issues }: { row: PageRow; issues: string[] }) {
  const mine = issues.filter((i) => i.startsWith(`p${row.page} `))
  return (
    <div className="page-detail">
      <img className="page-img" src={pageImageUrl(row.page)} alt={`Page ${row.page}`} />
      <div className="page-facts">
        <h4>
          {pageLabel(row.page)}
          {row.title ? ` · ${row.title}` : ''}
        </h4>
        {row.role === 'endnotes' ? (
          <p className="muted">Endnote page. Its text is parsed in step 2, not chunked as prose.</p>
        ) : (
          <div className="stats small">
            <div>
              <span className="muted">Prose blocks</span>
              <strong>{row.prose_blocks}</strong>
            </div>
            <div>
              <span className="muted">Words</span>
              <strong>{row.words}</strong>
            </div>
            <div>
              <span className="muted">Footnote markers</span>
              <strong>{row.footnote_markers.length ? row.footnote_markers.join(', ') : 'none'}</strong>
            </div>
          </div>
        )}

        <h5>Vision pass</h5>
        {row.visuals.length === 0 ? (
          <p className="muted">No tables, charts or callouts on this page.</p>
        ) : (
          row.visuals.map((v, i) => (
            <div key={i} className="visual">
              <div>
                <span className={`kind kind-${v.kind}`}>{v.kind}</span> <strong>{v.title}</strong>
              </div>
              <pre>{v.text}</pre>
            </div>
          ))
        )}
        {mine.length > 0 && (
          <div className="notice">
            {mine.map((m) => (
              <p key={m}>{m}</p>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
