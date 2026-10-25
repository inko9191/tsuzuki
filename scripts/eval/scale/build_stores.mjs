// Builds the perf stores: S LongMemEval-S filler sessions (nested: every store holds the first S fillers), dated
// 96 minutes apart (15 sessions a day), through the package's own code:
//   1. Memory.capture() for every message (turns, segments; timed: the storage cost of ingest at growing size),
//   2. the windows capture wrote are dropped and Memory.backfill() indexes the evaluated p2 windows (see the
//      report: capture's own windows are empty from the 4th message of a session on),
//   3. VACUUM INTO a compact copy: the size an append-only store of this content has.
//
//   node scripts/eval/scale/build_stores.mjs --vectors synthetic|real [--sizes 50,500,2000,5000]
// synthetic: deterministic random unit vectors (CPU only; ranking cost equals real vectors)
// real:      vectors from build/scale/vectors (embed_windows.mjs, EmbeddingGemma 300m on egemma-server)
import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { Memory, MemoryStore, MEMORY_OPTIONS, MODEL_IDS, OUT, cachedEmbedder, fileSizes, layout, loadPlan, loadSessions,
  messagesOf, summary, syntheticEmbedder, timed } from './common.mjs'

const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }
const vectors = arg('vectors', 'synthetic')
const sizes = arg('sizes', '50,500,2000,5000').split(',').map(Number)
const root = join(OUT, `stores-${vectors}`)
mkdirSync(root, { recursive: true })

const plan = loadPlan()
const sessions = await loadSessions(new Set(plan.fillers.slice(0, Math.max(...sizes))))
const source = vectors === 'real' ? cachedEmbedder() : syntheticEmbedder(MODEL_IDS.synthetic)
// A window egemma-server refuses (over 2,048 tokens) has no vector: it gets a placeholder here and is deleted after
// the backfill, so it stays unindexed as it would in use (capture fails for that message).
const refused = new Set()
const embedder = {
  model: () => source.model(),
  async embed(input, task) {
    const out = []
    for (const t of input) {
      try { out.push((await source.embed([t], task))[0]) } catch { refused.add(t); out.push(new Float32Array(768)) }
    }
    return out
  },
}
// Capture still embeds its own windows; they are deleted before backfill, so any vector will do.
const dummy = syntheticEmbedder(embedder.model())

for (const size of sizes) {
  const dir = join(root, `s${size}`)
  rmSync(dir, { recursive: true, force: true })
  mkdirSync(dir, { recursive: true })
  const dbPath = join(dir, 'build.db')
  const store = new MemoryStore(dbPath)
  const messages = messagesOf(layout(plan, size), sessions)
  const capture = new Memory(store, dummy, null, MEMORY_OPTIONS)
  const per = [], blocks = []
  let blockStart = performance.now(), blockN = 0
  const t0 = performance.now()
  for (const m of messages) {
    const s = performance.now()
    await capture.capture(m)
    per.push(performance.now() - s)
    if (++blockN === 2000) {
      const now = performance.now()
      blocks.push({ upTo: per.length, msPerMessage: (now - blockStart) / blockN })
      blockStart = now; blockN = 0
    }
  }
  const captureMs = performance.now() - t0
  capture.closeAll()
  capture.close()
  const shipped = store.stats().units
  store.db.exec('DELETE FROM units')
  const timedEmbedder = timed(embedder)
  const indexer = new Memory(store, timedEmbedder, null, MEMORY_OPTIONS)
  const b0 = performance.now()
  const indexed = await indexer.backfill(32)
  const backfillMs = performance.now() - b0
  indexer.close()
  let dropped = 0
  if (refused.size) {
    const del = store.db.prepare('DELETE FROM units WHERE id = ?')
    for (const r of store.db.prepare('SELECT id, text FROM units').all()) if (refused.has(r.text)) { del.run(r.id); dropped++ }
  }
  const stats = store.stats()
  store.db.exec('PRAGMA wal_checkpoint(TRUNCATE)')
  const before = fileSizes(dbPath)
  const finalPath = join(dir, 'memory.db')
  store.db.exec(`VACUUM INTO '${finalPath.replaceAll("'", "''")}'`)
  store.close()
  rmSync(dbPath, { force: true }); rmSync(`${dbPath}-wal`, { force: true }); rmSync(`${dbPath}-shm`, { force: true })
  // Reopen once in WAL mode so the copy is in the state the package leaves it in.
  new MemoryStore(finalPath).close()
  const result = {
    size, vectors, model: embedder.model(), sessions: size, messages: messages.length, stats, shippedUnits: shipped, refusedWindows: dropped,
    capture: { ms: captureMs, messagesPerSecond: messages.length / captureMs * 1000, perMessageMs: summary(per), blocks },
    backfill: { ms: backfillMs, units: indexed, unitsPerSecond: indexed / backfillMs * 1000, embedMs: timedEmbedder.times.document },
    files: { beforeVacuum: before, vacuumed: fileSizes(finalPath) },
  }
  writeFileSync(join(dir, 'build.json'), JSON.stringify(result, null, 1))
  console.log(JSON.stringify({ size, messages: messages.length, units: stats.units, captureS: (captureMs / 1000).toFixed(1),
    capturePerMsg: result.capture.perMessageMs.p50.toFixed(3), backfillS: (backfillMs / 1000).toFixed(1),
    mb: (result.files.vacuumed.total / 1048576).toFixed(1), mbBefore: (before.total / 1048576).toFixed(1) }))
}
