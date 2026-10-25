// Shared pieces of the memory-egemma scale test: plan loading, store layout, embedders.
// Uses the package's compiled code (dsh-plugins/packages/memory-egemma/lib) unchanged.
import { createHash } from 'node:crypto'
import { createReadStream, existsSync, readFileSync, statSync } from 'node:fs'
import { createInterface } from 'node:readline'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../../..')
export const PKG = join(ROOT, 'dsh-plugins/packages/memory-egemma')
/** Inputs and outputs; SCALE_OUT points a dry run elsewhere (inputs linked into it). */
export const OUT = process.env.SCALE_OUT ? resolve(process.env.SCALE_OUT) : join(ROOT, 'build/scale')

export const { MemoryStore } = await import(join(PKG, 'lib/store.js'))
export const { Memory, clip, estimateTokens, formatDate, DEFAULTS } = await import(join(PKG, 'lib/memory.js'))
export const pipeline = await import(join(PKG, 'lib/pipeline.js'))
export const { HttpEmbedder, EMBEDDINGGEMMA_PREFIXES } = await import(join(PKG, 'lib/embedding-http.js'))

/** The dsh plugin's defaults (src/index.ts Config): 600 tokens per message, 2048-token recall budget, no facts. */
export const MEMORY_OPTIONS = { messageTokens: 600, budgetTokens: 2048, segmentTokens: 2500, segmentIdleMinutes: 30,
  updateLinks: true, redactSecrets: true }

export const EGEMMA_PORT = Number(process.env.SCALE_EGEMMA_PORT ?? 8296)
export const FAKE_PORT = 8396
export const MODEL_IDS = {
  real: `http:http://127.0.0.1:${EGEMMA_PORT}/v1|embeddinggemma-300m`,
  synthetic: `http:http://127.0.0.1:${FAKE_PORT}/v1|synthetic-768`,
}
export const DIMS = 768

// Timeline: the newest session starts at END; sessions are GAP minutes apart (15 per day), messages 1 min apart.
export const END = new Date(2026, 9, 20, 9, 0).getTime()
export const GAP_MS = 96 * 60_000
export const MSG_MS = 60_000

export function loadPlan() {
  return JSON.parse(readFileSync(join(OUT, 'plan.json'), 'utf8'))
}

export async function loadSessions(wanted = null) {
  const out = new Map()
  const rl = createInterface({ input: createReadStream(join(OUT, 'sessions.jsonl')), crlfDelay: Infinity })
  for await (const line of rl) {
    if (!line) continue
    const s = JSON.parse(line)
    if (!wanted || wanted.has(s.id)) out.set(s.id, s.turns)
  }
  return out
}

export function sha1(text) {
  return createHash('sha1').update(text).digest('hex')
}

/** dsh session ids are UUIDs; a deterministic one per source session. */
export function uuidFor(orig) {
  const h = createHash('sha256').update(`scale|${orig}`).digest('hex')
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20, 32)}`
}

/** Same as the dsh plugin's message id (src/index.ts shortId). */
export function shortId(session, seq) {
  return `${createHash('sha256').update(session).digest('hex').slice(0, 8)}-${seq}`
}

/** Source session ids in timeline order: S fillers, or S - |E| fillers with a question's evidence at `age` (0..1). */
export function layout(plan, size, planted = null, age = null) {
  if (!planted) return plan.fillers.slice(0, size)
  const n = planted.evidence.length
  const fill = plan.fillers.slice(0, size - n)
  const at = Math.min(fill.length, Math.max(0, Math.round(age * fill.length)))
  return [...fill.slice(0, at), ...planted.evidence, ...fill.slice(at)]
}

/** Capture-ready messages of a timeline; the newest session starts at `end`. */
export function messagesOf(order, sessions, end = END) {
  const out = []
  order.forEach((orig, k) => {
    const session = uuidFor(orig)
    const start = end - (order.length - 1 - k) * GAP_MS
    const turns = sessions.get(orig)
    if (!turns) throw new Error(`session ${orig} missing from sessions.jsonl`)
    turns.forEach((t, i) => out.push({ id: shortId(session, i + 1), session, role: t.role, text: t.content, at: start + i * MSG_MS,
      orig, index: i, hasAnswer: t.has_answer }))
  })
  return out
}

// -- embedders --------------------------------------------------------------------------------

function unit(values) {
  let n = 0
  for (const v of values) n += v * v
  n = Math.sqrt(n) || 1
  for (let k = 0; k < values.length; k++) values[k] /= n
  return values
}

/** Deterministic pseudo-random unit vectors (per text): ranking cost is the same as for real vectors. */
export function syntheticEmbedder(model = MODEL_IDS.synthetic) {
  const vector = text => {
    let h = 2166136261
    for (let k = 0; k < text.length; k++) h = Math.imul(h ^ text.charCodeAt(k), 16777619)
    let x = (h >>> 0) || 1
    const v = new Float32Array(DIMS)
    for (let k = 0; k < DIMS; k++) {
      x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0
      v[k] = x / 2147483648 - 1
    }
    return unit(v)
  }
  return { model: () => model, async embed(input) { return input.map(vector) } }
}

/** Vectors computed earlier by embed_windows.mjs (keyed by sha1 of the exact input text). */
export function cachedEmbedder(dir = join(OUT, 'vectors'), model = MODEL_IDS.real) {
  const load = name => {
    const index = JSON.parse(readFileSync(join(dir, `${name}.json`), 'utf8'))
    const buf = readFileSync(join(dir, `${name}.f32`))
    const data = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)
    return { rows: new Map(Object.entries(index.rows)), data, dims: index.dims }
  }
  const sets = { document: load('documents'), query: existsSync(join(dir, 'queries.json')) ? load('queries') : null }
  return {
    model: () => model,
    has(text, task = 'document') { return sets[task]?.rows.has(sha1(text)) ?? false },
    async embed(input, task) {
      const set = sets[task]
      return input.map(text => {
        const row = set?.rows.get(sha1(text))
        if (row === undefined) throw new Error(`no cached ${task} vector for ${JSON.stringify(text.slice(0, 80))}`)
        return set.data.slice(row * set.dims, (row + 1) * set.dims)
      })
    },
  }
}

/** The sidecar's embedder (HttpEmbedder, batch 32) against our own egemma-server. */
export function liveEmbedder(port = EGEMMA_PORT, model = 'embeddinggemma-300m') {
  return new HttpEmbedder({ baseUrl: `http://127.0.0.1:${port}/v1`, model, apiKey: '', ...EMBEDDINGGEMMA_PREFIXES,
    timeoutMs: 120000, batch: 32 })
}

/** Wraps an embedder to time each call (ms per task). */
export function timed(embedder) {
  const t = { document: 0, query: 0, calls: 0, last: 0 }
  return {
    times: t,
    model: () => embedder.model(),
    async embed(input, task, signal) {
      const s = performance.now()
      try {
        return await embedder.embed(input, task, signal)
      } finally {
        t.last = performance.now() - s
        t[task] += t.last
        t.calls++
      }
    },
  }
}

export function fileSizes(dbPath) {
  const size = p => (existsSync(p) ? statSync(p).size : 0)
  const main = size(dbPath), wal = size(`${dbPath}-wal`), shm = size(`${dbPath}-shm`)
  return { main, wal, shm, total: main + wal + shm }
}

export function percentile(values, q) {
  if (!values.length) return null
  const s = [...values].sort((a, b) => a - b)
  const pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos)
  return s[lo] + (s[hi] - s[lo]) * (pos - lo)
}

export function summary(values) {
  return { n: values.length, p50: percentile(values, 0.5), p95: percentile(values, 0.95), max: values.length ? Math.max(...values) : null,
    mean: values.length ? values.reduce((a, b) => a + b, 0) / values.length : null }
}

export function mb(bytes) {
  return Math.round(bytes / 1048576 * 10) / 10
}

export function memoryNow() {
  const m = process.memoryUsage()
  return { rss: m.rss, heapUsed: m.heapUsed, heapTotal: m.heapTotal, external: m.external, arrayBuffers: m.arrayBuffers,
    maxRss: process.resourceUsage().maxRSS * 1024 }
}
