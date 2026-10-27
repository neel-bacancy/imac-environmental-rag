import { useEffect, useMemo, useState } from 'react'
import { getChunks, getOverview, type ChunkItem, type Overview } from './api'
import { IngestView } from './components/IngestView'
import { QueryView } from './components/QueryView'
import './App.css'

type Tab = 'ask' | 'ingest'

function App() {
  const [tab, setTab] = useState<Tab>('ask')
  const [overview, setOverview] = useState<Overview | null>(null)
  const [chunks, setChunks] = useState<ChunkItem[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([getOverview(), getChunks()])
      .then(([o, c]) => {
        setOverview(o)
        setChunks(c)
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  const chunkById = useMemo(() => new Map(chunks.map((c) => [c.id, c])), [chunks])

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>iMac Environmental Report: RAG</h1>
          <p className="sub">
            Every step from the PDF to the answer, shown as it runs. Answers come from Apple's iMac Product Environmental Report (October 2024).
          </p>
        </div>
        <div className={`status ${overview ? 'ok' : error ? 'down' : ''}`}>
          {overview ? `${overview.chunks.total} chunks · ${overview.models.answer}` : error ? 'Backend offline' : 'Connecting…'}
        </div>
      </header>

      <nav className="tabs">
        <button className={tab === 'ask' ? 'active' : ''} onClick={() => setTab('ask')}>
          Ask a question
        </button>
        <button className={tab === 'ingest' ? 'active' : ''} onClick={() => setTab('ingest')}>
          How the index was built
        </button>
      </nav>

      {error && (
        <div className="error">
          {error}. Start the backend from the project root: <code>uvicorn server.main:app --port 8000</code>
        </div>
      )}

      <div hidden={tab !== 'ask'}>
        <QueryView overview={overview} chunkById={chunkById} />
      </div>
      <div hidden={tab !== 'ingest'}>
        <IngestView overview={overview} chunks={chunks} />
      </div>
    </div>
  )
}

export default App
