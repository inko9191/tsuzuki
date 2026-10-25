// GPU step: embeds every collected window text (build/scale/windows.jsonl) and the latency/accuracy queries with
// EmbeddingGemma 300m on our own egemma-server (port 8296), through the package's HttpEmbedder (client-side
// EmbeddingGemma prefixes, L2-normalized), exactly as the sidecar does. Resumable.
//
// Throughput is measured as backfill runs it (batches of 32, one request at a time) on the first --sequential
// batches; the rest goes 4 requests at a time to finish sooner.
//
//   node scripts/eval/scale/embed_windows.mjs [--sequential 100] [--concurrency 4]
import { appendFileSync, existsSync, mkdirSync, readFileSync, statSync, truncateSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { DIMS, MODEL_IDS, OUT, liveEmbedder, summary } from './common.mjs'

const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }
const sequential = Number(arg('sequential', '100'))
const concurrency = Number(arg('concurrency', '4'))
const dir = join(OUT, 'vectors')
mkdirSync(dir, { recursive: true })
const log = msg => { const line = `${new Date().toISOString()} ${msg}`; console.log(line); appendFileSync(join(OUT, 'embed.log'), line + '\n') }

const tokens = JSON.parse(readFileSync(join(OUT, 'window-tokens.json'), 'utf8'))
const windows = readFileSync(join(OUT, 'windows.jsonl'), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l))
const refused = windows.filter(w => tokens.documents[w.h] > tokens.limit).map(w => w.h)
const f32 = join(dir, 'documents.f32'), orderFile = join(dir, 'documents.order')
let done = existsSync(orderFile) ? readFileSync(orderFile, 'utf8').split('\n').filter(Boolean) : []
if (existsSync(f32)) {
  const rows = statSync(f32).size / (DIMS * 4)
  if (rows < done.length) done = done.slice(0, Math.floor(rows))
  truncateSync(f32, done.length * DIMS * 4)
  writeFileSync(orderFile, done.map(h => h + '\n').join(''))
}
const have = new Set(done)
const todo = windows.filter(w => !have.has(w.h) && tokens.documents[w.h] <= tokens.limit)
log(`windows ${windows.length}, done ${done.length}, todo ${todo.length}, refused (> ${tokens.limit} tokens) ${refused.length}`)

const embedder = liveEmbedder()
const batches = []
for (let k = 0; k < todo.length; k += 32) batches.push(todo.slice(k, k + 32))
let written = 0
const stats = { sequential: { batches: 0, windows: 0, tokens: 0, ms: 0 }, parallel: { batches: 0, windows: 0, tokens: 0, ms: 0 } }
const writeBatch = (batch, vecs) => {
  const buf = Buffer.alloc(batch.length * DIMS * 4)
  vecs.forEach((v, j) => Buffer.from(v.buffer, v.byteOffset, v.byteLength).copy(buf, j * DIMS * 4))
  appendFileSync(f32, buf)
  appendFileSync(orderFile, batch.map(w => w.h + '\n').join(''))
  written += batch.length
}
const tok = batch => batch.reduce((a, w) => a + tokens.documents[w.h], 0)

let next = 0
const t0 = performance.now()
// Sequential part: as Memory.backfill (one request of 32 at a time).
for (; next < Math.min(sequential, batches.length); next++) {
  const batch = batches[next]
  const s = performance.now()
  const vecs = await embedder.embed(batch.map(w => w.text), 'document')
  stats.sequential.ms += performance.now() - s
  stats.sequential.batches++; stats.sequential.windows += batch.length; stats.sequential.tokens += tok(batch)
  writeBatch(batch, vecs)
  if (next % 25 === 24) log(`sequential ${next + 1}/${batches.length} batches, ${(stats.sequential.windows / stats.sequential.ms * 1000).toFixed(1)} windows/s`)
}
// Parallel part: results are written in completion order (each batch with its own hashes).
const p0 = performance.now()
const workers = Array.from({ length: concurrency }, async () => {
  while (next < batches.length) {
    const batch = batches[next++]
    const vecs = await embedder.embed(batch.map(w => w.text), 'document')
    writeBatch(batch, vecs)
    stats.parallel.batches++; stats.parallel.windows += batch.length; stats.parallel.tokens += tok(batch)
    if (stats.parallel.batches % 100 === 0) {
      const ms = performance.now() - p0
      log(`parallel ${stats.parallel.batches} batches, ${(stats.parallel.windows / ms * 1000).toFixed(1)} windows/s, ${written} written`)
    }
  }
})
await Promise.all(workers)
stats.parallel.ms = performance.now() - p0
log(`documents done in ${((performance.now() - t0) / 1000).toFixed(0)} s`)

// Index of all rows written so far (this run and earlier ones).
const order = readFileSync(orderFile, 'utf8').split('\n').filter(Boolean)
writeFileSync(join(dir, 'documents.json'), JSON.stringify({ dims: DIMS, model: MODEL_IDS.real, rows: Object.fromEntries(order.map((h, i) => [h, i])),
  refused }))

// Queries one at a time, as a recall embeds its query.
const queries = readFileSync(join(OUT, 'queries.jsonl'), 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l))
const qbuf = Buffer.alloc(queries.length * DIMS * 4)
const qms = []
for (const [i, q] of queries.entries()) {
  const s = performance.now()
  const [v] = await embedder.embed([q.text], 'query')
  qms.push(performance.now() - s)
  Buffer.from(v.buffer, v.byteOffset, v.byteLength).copy(qbuf, i * DIMS * 4)
}
writeFileSync(join(dir, 'queries.f32'), qbuf)
writeFileSync(join(dir, 'queries.json'), JSON.stringify({ dims: DIMS, model: MODEL_IDS.real, rows: Object.fromEntries(queries.map((q, i) => [q.h, i])) }))

const rate = s => ({ ...s, windowsPerSecond: s.windows / s.ms * 1000, tokensPerSecond: s.tokens / s.ms * 1000 })
const result = { sequential: rate(stats.sequential), parallel: rate(stats.parallel), concurrency, refused: refused.length,
  totalRows: order.length, queryEmbedMs: summary(qms) }
writeFileSync(join(OUT, 'embed-throughput.json'), JSON.stringify(result, null, 1))
log(JSON.stringify(result))
