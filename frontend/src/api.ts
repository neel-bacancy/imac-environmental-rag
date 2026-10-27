export interface Visual {
  kind: string
  title: string
  text: string
}

export interface PageRow {
  page: number
  role: 'body' | 'endnotes'
  title: string | null
  prose_blocks: number
  words: number
  footnote_markers: number[]
  visuals: Visual[]
}

export interface Endnote {
  number: number
  page: number
  text: string
}

export interface Overview {
  pdf: { name: string; pages: number }
  pages: PageRow[]
  endnotes: Endnote[]
  vision: { elements: number; issues: string[] }
  chunks: { total: number; by_kind: Record<string, number>; footnote_linked: number }
  index: {
    vectors: number[]
    embedding_model: string
    dense_k: number
    bm25_k: number
    rrf_k: number
    candidates: number
    final_k: number
    use_reranker: boolean
  }
  models: { answer: string; router: string }
  sample_questions: string[]
}

export interface ChunkItem {
  id: string
  kind: string
  page: number
  section: string | null
  text: string
  footnote_numbers: number[]
}

export interface Scored {
  chunk_id: string
  rank: number
  score: number
}

export interface Fused {
  chunk_id: string
  rank: number
  score: number
  dense_rank: number | null
  bm25_rank: number | null
}

export interface Source {
  index: number
  chunk_id: string
  kind: string
  page: number
  section: string | null
  text: string
  score: number
  cited: boolean
}

export interface QueryResult {
  question: string
  rewrite_prompt: string
  rewritten_query: string
  system_prompt: string
  user_prompt: string
  answer: string
  found: boolean
  model: string
  timings_ms: Record<string, number>
  reranked: boolean
  dense: Scored[]
  bm25: Scored[]
  fused: Fused[]
  rerank_order: string[]
  sources: Source[]
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path)
  if (!res.ok) throw new Error(await errorText(res))
  return res.json() as Promise<T>
}

async function errorText(res: Response): Promise<string> {
  const body = await res.json().catch(() => null)
  return typeof body?.detail === 'string' ? body.detail : `Request failed (${res.status})`
}

export const getOverview = () => get<Overview>('/api/overview')

export const getChunks = () => get<ChunkItem[]>('/api/chunks')

export const askQuestion = async (question: string, rerank: boolean): Promise<QueryResult> => {
  const res = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, rerank }),
  })
  if (!res.ok) throw new Error(await errorText(res))
  return res.json() as Promise<QueryResult>
}

export const pageImageUrl = (page: number) => `/api/pages/${page}/image`

export const pageLabel = (page: number) => `p. ${page}`
