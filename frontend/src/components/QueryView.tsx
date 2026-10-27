import { useState, type FormEvent, type ReactNode } from 'react'
import { askQuestion, pageLabel, type ChunkItem, type Overview, type QueryResult, type Scored } from '../api'
import { AnswerText } from './AnswerText'
import { Step } from './Step'

interface Props {
  overview: Overview | null
  chunkById: Map<string, ChunkItem>
}

export function QueryView({ overview, chunkById }: Props) {
  const [question, setQuestion] = useState('')
  const [rerankChoice, setRerankChoice] = useState<boolean | null>(null)
  const [result, setResult] = useState<QueryResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const rerank = rerankChoice ?? overview?.index.use_reranker ?? true

  const ask = async (text: string) => {
    const q = text.trim()
    if (q.length < 3 || loading) return
    setQuestion(q)
    setLoading(true)
    setError(null)
    try {
      setResult(await askQuestion(q, rerank))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    ask(question)
  }

  const cite = (n: number) => document.getElementById(`src-${n}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })

  return (
    <div className="query">
      <form className="ask" onSubmit={onSubmit}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              ask(question)
            }
          }}
          placeholder="Ask about the iMac's energy, carbon footprint, materials or packaging…"
          rows={2}
          disabled={loading}
        />
        <div className="ask-row">
          <label className="toggle">
            <input type="checkbox" checked={rerank} onChange={(e) => setRerankChoice(e.target.checked)} disabled={loading} />
            Use the reranker (one extra model call)
          </label>
          <button type="submit" disabled={loading || question.trim().length < 3}>
            {loading ? 'Running…' : 'Ask'}
          </button>
        </div>
      </form>

      <div className="samples">
        {overview?.sample_questions.map((q) => (
          <button key={q} type="button" className="chip" onClick={() => ask(q)} disabled={loading}>
            {q}
          </button>
        ))}
      </div>

      {error && <div className="error">{error}</div>}
      {loading && <div className="running">Running the pipeline: rewrite, search, fuse, rerank, then answer…</div>}
      {!result && !loading && !error && <p className="empty">Ask a question, or pick a sample, to see each step of the pipeline.</p>}

      {result && <Results r={result} overview={overview} chunkById={chunkById} onCite={cite} />}
    </div>
  )
}

function Results({
  r,
  overview,
  chunkById,
  onCite,
}: {
  r: QueryResult
  overview: Overview | null
  chunkById: Map<string, ChunkItem>
  onCite: (n: number) => void
}) {
  const t = r.timings_ms
  const finalK = overview?.index.final_k ?? r.sources.length
  const total = Object.values(t).reduce((a, b) => a + b, 0)
  const denseMax = r.dense[0]?.score ?? 1
  const bm25Max = r.bm25[0]?.score ?? 1
  const afterRerank = new Map(r.rerank_order.map((id, i) => [id, i + 1]))
  const chunkTag = (id: string) => {
    const c = chunkById.get(id)
    return c ? <span className="tag">{`${c.kind} · ${pageLabel(c.page)}`}</span> : null
  }

  return (
    <div className="results">
      <div className="summary">
        <span>
          Total <strong>{(total / 1000).toFixed(1)} s</strong>
        </span>
        <span>rewrite {t.rewrite_ms} ms</span>
        <span>embed {t.embed_ms} ms</span>
        <span>search {t.search_ms} ms</span>
        <span>rerank {t.rerank_ms} ms</span>
        <span>generation {t.generation_ms} ms</span>
      </div>

      <Step n={1} title="Your question" what="The question as it was sent to the pipeline.">
        <blockquote className="question">{r.question}</blockquote>
      </Step>

      <Step n={2} title="Rewrite into a search query" what="A small Gemini model turns the question into search terms. It only rewrites; it never answers." timing={t.rewrite_ms}>
        <div className="kv">
          <span>Search query</span>
          <code>{r.rewritten_query}</code>
        </div>
        <Details label="Rewrite prompt">
          <pre>{r.rewrite_prompt}</pre>
        </Details>
      </Step>

      <Step
        n={3}
        title="Dense search: meaning"
        what={`The query is embedded with ${overview?.index.embedding_model ?? 'the local embedding model'} and compared with every chunk vector by cosine similarity. The top ${r.dense.length} are kept. Bars are relative to the top result.`}
        timing={t.embed_ms}
      >
        <ScoreList rows={r.dense} max={denseMax} decimals={3} chunkTag={chunkTag} />
      </Step>

      <Step n={4} title="BM25 search: keywords" what="Classic keyword ranking over the same chunks, using the question and the rewritten query. Only chunks with a nonzero score appear.">
        <ScoreList rows={r.bm25} max={bm25Max} decimals={2} chunkTag={chunkTag} />
      </Step>

      <Step
        n={5}
        title="Fuse the two lists"
        what={`Reciprocal Rank Fusion: each chunk gets 1 / (60 + rank) from every list it appears in. Chunks found by both searches are highlighted. The top ${r.fused.length} go to the reranker.`}
      >
        <table className="grid">
          <thead>
            <tr>
              <th>#</th>
              <th>Chunk</th>
              <th>Dense rank</th>
              <th>BM25 rank</th>
              <th>RRF score</th>
            </tr>
          </thead>
          <tbody>
            {r.fused.map((f) => (
              <tr key={f.chunk_id} className={f.dense_rank && f.bm25_rank ? 'both' : ''}>
                <td>{f.rank}</td>
                <td>
                  <code>{f.chunk_id}</code> {chunkTag(f.chunk_id)}
                </td>
                <td>{f.dense_rank ?? '—'}</td>
                <td>{f.bm25_rank ?? '—'}</td>
                <td>{f.score.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Step>

      <Step
        n={6}
        title="Rerank"
        what={
          r.reranked
            ? `A Gemini model reads the candidates and orders them by how well they answer the question. The top ${finalK} go on to the answer model.`
            : 'The reranker is off for this question, so the fusion order is used as it is.'
        }
        timing={r.reranked ? t.rerank_ms : undefined}
      >
        {r.reranked && (
          <table className="grid">
            <thead>
              <tr>
                <th>Chunk</th>
                <th>Fusion rank</th>
                <th>After rerank</th>
                <th>Move</th>
              </tr>
            </thead>
            <tbody>
              {r.fused.map((f) => {
                const after = afterRerank.get(f.chunk_id) ?? f.rank
                const move = after < f.rank ? `↑ ${f.rank - after}` : after > f.rank ? `↓ ${after - f.rank}` : '='
                return (
                  <tr key={f.chunk_id} className={after <= finalK ? 'kept' : ''}>
                    <td>
                      <code>{f.chunk_id}</code> {chunkTag(f.chunk_id)}
                    </td>
                    <td>{f.rank}</td>
                    <td>{after}</td>
                    <td>{move}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </Step>

      <Step
        n={7}
        title="Context for the answer model"
        what={`The top ${r.sources.length} chunks, in rerank order, are numbered S1 to S${r.sources.length} and sent to ${r.model} with the question. Chunks the answer cites are marked.`}
      >
        <div className="sources">
          {r.sources.map((s) => (
            <article key={s.index} id={`src-${s.index}`} className={`source ${s.cited ? 'cited' : ''}`}>
              <header>
                <span className="tag-s">S{s.index}</span>
                <span className="tag">{`${s.kind} · ${pageLabel(s.page)}`}</span>
                {s.section && <span className="muted">{s.section}</span>}
                <span className="muted mono">RRF {s.score.toFixed(4)}</span>
                {s.cited && <span className="badge">cited</span>}
              </header>
              <pre>{s.text}</pre>
            </article>
          ))}
        </div>
        <Details label="Full prompt sent to the model">
          <div className="prompt-label">System prompt</div>
          <pre>{r.system_prompt}</pre>
          <div className="prompt-label">User message</div>
          <pre>{r.user_prompt}</pre>
        </Details>
      </Step>

      <Step n={8} title="Answer" what="The model writes the answer from the sources only. Click a citation to jump to its source." timing={t.generation_ms}>
        {!r.found && <div className="notice">The model reports that the report does not contain this information.</div>}
        <AnswerText text={r.answer} onCite={onCite} />
        <div className="stats">
          <div>
            <span className="muted">Sources cited</span>
            <strong>{r.sources.filter((s) => s.cited).length}</strong>
          </div>
          <div>
            <span className="muted">Model</span>
            <strong>{r.model}</strong>
          </div>
        </div>
      </Step>
    </div>
  )
}

function ScoreList({
  rows,
  max,
  decimals,
  chunkTag,
}: {
  rows: Scored[]
  max: number
  decimals: number
  chunkTag: (id: string) => ReactNode
}) {
  return (
    <ol className="score-list">
      {rows.map((row) => (
        <li key={row.chunk_id}>
          <span className="rank">{row.rank}</span>
          <code>{row.chunk_id}</code>
          <span>{chunkTag(row.chunk_id)}</span>
          <span className="bar-track">
            <span className="bar-fill" style={{ width: `${Math.max(0, (row.score / max) * 100)}%` }} />
          </span>
          <span className="score">{row.score.toFixed(decimals)}</span>
        </li>
      ))}
    </ol>
  )
}

function Details({ label, children }: { label: string; children: ReactNode }) {
  return (
    <details className="details">
      <summary>{label}</summary>
      {children}
    </details>
  )
}
