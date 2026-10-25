// One measurement process for one store (run by run_bench.mjs, a fresh process per store so RSS and cold start are
// clean). Follows what the dsh plugin and the sidecar do on start: open the store, backfill(), then recall.
//
//   node --expose-gc scripts/eval/scale/bench_child.mjs --store build/scale/stores-synthetic/s5000 --vectors synthetic
//        [--ingest 120] [--out file.json]
import { copyFileSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { Memory, MemoryStore, MEMORY_OPTIONS, MODEL_IDS, OUT, estimateTokens as estimate, fileSizes, liveEmbedder,
  loadPlan, loadSessions, memoryNow, messagesOf, pipeline as p, summary, syntheticEmbedder, timed } from './common.mjs'

const bootMs = performance.now()  // ms since the process started (Node startup + imports)
const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }
const storeDir = arg('store')
const vectors = arg('vectors', 'synthetic')
const ingestN = Number(arg('ingest', '120'))
const outPath = arg('out', null)
const gc = globalThis.gc ?? (() => {})

const plan = loadPlan()
const queries = plan.queries.map(q => q.text)

function makeEmbedder() {
  // real: the sidecar's own embedder against our egemma-server, for queries and for anything backfill finds missing.
  return vectors === 'synthetic' ? syntheticEmbedder(MODEL_IDS.synthetic) : liveEmbedder()
}

const result = { store: storeDir, vectors, pid: process.pid, bootMs, memory: {} }
const snap = (label, collect = false) => {
  if (collect) gc()
  result.memory[label] = memoryNow()
}
snap('boot')

const dbPath = join(storeDir, 'memory.db')
result.files = fileSizes(dbPath)
let t = performance.now()
const store = new MemoryStore(dbPath)
result.openMs = performance.now() - t
const embedder = timed(makeEmbedder())
const memory = new Memory(store, embedder, null, MEMORY_OPTIONS)
result.stats = store.stats()

// Startup: backfill (both hosts run it before the first recall can proceed).
t = performance.now()
try {
  result.startupBackfill = { indexed: await memory.backfill(), ms: 0 }
} catch (error) {
  result.startupBackfill = { error: String(error).slice(0, 200), ms: 0 }
}
result.startupBackfill.ms = performance.now() - t
snap('afterBackfill')

// First recall: reloads every unit from SQLite when backfill succeeded (backfill clears the cache).
const embedBefore = embedder.times.query
t = performance.now()
const first = await memory.retrieve(queries[0], { rerank: false, budgetTokens: MEMORY_OPTIONS.budgetTokens })
result.firstRecall = { ms: performance.now() - t, embedMs: embedder.times.query - embedBefore, ids: first.ids.length,
  sinceProcessStartMs: performance.now() }
snap('afterFirstRecall')

// Warm recalls.
const totals = [], embeds = [], locals = [], textTokens = []
for (const q of queries) {
  const e0 = embedder.times.query
  const s = performance.now()
  const r = await memory.retrieve(q, { rerank: false, budgetTokens: MEMORY_OPTIONS.budgetTokens })
  const ms = performance.now() - s
  const em = embedder.times.query - e0
  totals.push(ms); embeds.push(em); locals.push(ms - em); textTokens.push(r.ids.length)
}
result.warm = { total: summary(totals), embed: summary(embeds), local: summary(locals), idsPerRecall: summary(textTokens) }
snap('afterWarm')
snap('afterWarmGc', true)

// Where the local time goes: a step-by-step copy of Memory.retrieve (same calls, same order), checked against it.
async function profiled(query) {
  const st = {}
  const lap = (name, s) => { st[name] = performance.now() - s }
  let s = performance.now()
  const units = memory.indexUnits()
  lap('indexUnits', s)
  s = performance.now()
  const [qv] = await embedder.embed([query], 'query')
  lap('embedQuery', s)
  s = performance.now()
  const model = embedder.model()
  const kept = units.filter(u => u.model === model && u.vector.length === qv.length)
  lap('filterUnits', s)
  s = performance.now()
  const fused = p.rank(kept, kept.map(u => u.vector), qv)
  lap('rankCosineTopRrf', s)
  s = performance.now()
  const order = p.distinct(fused.map(([i]) => i), kept)
  lap('distinct', s)
  s = performance.now()
  const turns = memory.allTurns()
  const plain = turns.map(x => memory.asTurn(x))
  lap('clipAllTurns', s)
  s = performance.now()
  const raw = p.rawIds(order, kept)
  const exclude = plain.filter(() => false).map(x => x.id)
  const lookup = new Map(plain.map(x => [x.id, { ...x, session: `${x.session}|${x.date.slice(0, 10)}` }]))
  lap('lookupAllTurns', s)
  s = performance.now()
  const { order: ranked, notes } = p.versioned(raw, memory.factRows(), memory.store.links())
  lap('factRowsVersioned', s)
  s = performance.now()
  const sel = p.selectMemory(ranked, lookup, exclude, MEMORY_OPTIONS.budgetTokens, x => estimate(x), ts => p.renderDated(ts, notes))
  lap('selectRender', s)
  return { steps: st, ids: sel.ids }
}
const steps = [], mismatches = []
for (const q of queries.slice(0, 20)) {
  const a = await profiled(q)
  const b = await memory.retrieve(q, { rerank: false, budgetTokens: MEMORY_OPTIONS.budgetTokens })
  if (JSON.stringify(a.ids) !== JSON.stringify(b.ids)) mismatches.push(q)
  steps.push(a.steps)
}
result.breakdown = Object.fromEntries(Object.keys(steps[0]).map(k => [k, summary(steps.map(s => s[k]))]))
result.breakdownMismatches = mismatches.length

// Cold loads in isolation.
t = performance.now(); const unitRows = store.units(); result.unitsLoadMs = performance.now() - t
t = performance.now(); store.turns(); result.turnsLoadMs = performance.now() - t
result.unitCount = unitRows.length
result.unitTextChars = unitRows.reduce((a, u) => a + u.text.length, 0)

// The 10-minute backfill tick: scans every turn; a successful run clears the unit cache, so the next recall reloads.
t = performance.now()
try {
  await memory.backfill()
  result.tickBackfillMs = performance.now() - t
} catch (error) {
  result.tickBackfillMs = performance.now() - t
  result.tickBackfillError = String(error).slice(0, 200)
}
t = performance.now()
await memory.retrieve(queries[1], { rerank: false, budgetTokens: MEMORY_OPTIONS.budgetTokens })
result.recallAfterTickMs = performance.now() - t
snap('afterTick')
memory.close()
store.close()

// Ingest at this size: capture held-out messages into a copy of the store (the live path, one window per message).
if (ingestN > 0) {
  const tmp = join(OUT, 'tmp', `ingest-${process.pid}`)
  mkdirSync(tmp, { recursive: true })
  const copy = join(tmp, 'memory.db')
  copyFileSync(dbPath, copy)
  const istore = new MemoryStore(copy)
  const iemb = timed(vectors === 'synthetic' ? syntheticEmbedder(MODEL_IDS.synthetic) : liveEmbedder())
  const imem = new Memory(istore, iemb, null, MEMORY_OPTIONS)
  const sessions = await loadSessions(new Set(plan.holdout))
  const msgs = messagesOf(plan.holdout, sessions, Date.now()).slice(0, ingestN)
  const per = [], emb = []
  let failed = 0
  for (const m of msgs) {
    const e0 = iemb.times.document
    const s = performance.now()
    try {
      await imem.capture(m)
    } catch {
      failed++
      continue
    }
    per.push(performance.now() - s)
    emb.push(iemb.times.document - e0)
  }
  // The evaluated p2 window of each message (what a fixed capture would embed), embedded one at a time.
  const fixedEmb = []
  if (vectors === 'real') {
    const turns = istore.turns().slice(-msgs.length)
    const bySession = new Map()
    for (const x of istore.turns()) if (turns.some(y => y.session === x.session)) bySession.set(x.session, [...(bySession.get(x.session) ?? []), x])
    const windows = new Map()
    for (const list of bySession.values()) for (const u of p.windowUnitsP2(list.map(x => imem.asTurn(x)))) windows.set(u.id, u.text)
    const live = timed(liveEmbedder())
    for (const x of turns) {
      const e0 = live.times.document
      try {
        await live.embed([windows.get(`p2|${x.id}`)], 'document')
        fixedEmb.push(live.times.document - e0)
      } catch {
        failed++
      }
    }
  }
  result.ingest = { messages: msgs.length, failed, perMessageMs: summary(per), embedMs: summary(emb),
    storageMs: summary(per.map((x, k) => x - emb[k])), fixedWindowEmbedMs: summary(fixedEmb) }
  imem.close()
  istore.close()
  rmSync(tmp, { recursive: true, force: true })
}
snap('end')
result.wallMs = performance.now()
const json = JSON.stringify(result, null, 1)
if (outPath) writeFileSync(outPath, json)
else process.stdout.write(json + '\n')
