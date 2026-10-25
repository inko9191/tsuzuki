// Prototype of the smallest fix for recall cost at scale, measured outside the package (nothing in the package is
// changed). Same results as Memory.retrieve (checked id-for-id on every query), different bookkeeping:
//   - the ranked unit order and one contiguous Float32Array per unit set are built once (appended on capture),
//     instead of re-sorting every unit on each recall,
//   - the best 100 per set are kept with a bounded heap instead of sorting all scores,
//   - only the turns that can be selected (the ranked candidates) are clipped and looked up, instead of every stored
//     turn on each recall,
//   - unit texts are not kept in memory (retrieval never reads them).
// `--int8` also scores against int8 rows (one scale per row; ranking can differ, so ids are compared, not required).
//
//   node --expose-gc scripts/eval/scale/fix_prototype.mjs --store build/scale/stores-synthetic/s5000 --vectors synthetic [--int8] [--lean]
import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { Memory, MemoryStore, MEMORY_OPTIONS, MODEL_IDS, DIMS, estimateTokens, liveEmbedder, loadPlan,
  memoryNow, pipeline as p, summary, syntheticEmbedder, timed } from './common.mjs'

const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }
const storeDir = arg('store')
const vectors = arg('vectors', 'synthetic')
const useInt8 = process.argv.includes('--int8')
const outPath = arg('out', null)
const gc = globalThis.gc ?? (() => {})
const queries = loadPlan().queries.map(q => q.text)
const base = vectors === 'synthetic' ? syntheticEmbedder(MODEL_IDS.synthetic) : liveEmbedder()
const embedder = timed(base)
const result = { store: storeDir, vectors, int8: useInt8, memory: {} }
const snap = (label, collect = false) => { if (collect) gc(); result.memory[label] = memoryNow() }

const store = new MemoryStore(join(storeDir, 'memory.db'))
const memory = new Memory(store, embedder, null, MEMORY_OPTIONS)

class FastIndex {
  constructor(memory) {
    const t = performance.now()
    const model = embedder.model()
    this.units = memory.indexUnits().filter(u => u.model === model && u.vector.length === DIMS)
    this.sets = {}
    for (const set of ['p2', 'fact']) {
      const members = []
      this.units.forEach((u, i) => { if (u.set === set) members.push(i) })
      const rows = members.length
      if (useInt8) {
        const q = new Int8Array(rows * DIMS), scale = new Float32Array(rows)
        members.forEach((i, r) => {
          const v = this.units[i].vector
          let m = 0
          for (let k = 0; k < DIMS; k++) m = Math.max(m, Math.abs(v[k]))
          const s = m / 127 || 1
          scale[r] = s
          for (let k = 0; k < DIMS; k++) q[r * DIMS + k] = Math.round(v[k] / s)
        })
        this.sets[set] = { members, q, scale }
        members.forEach(i => { this.units[i].vector = null })
      } else {
        const matrix = new Float32Array(rows * DIMS)
        members.forEach((i, r) => {
          matrix.set(this.units[i].vector, r * DIMS)
          this.units[i].vector = matrix.subarray(r * DIMS, (r + 1) * DIMS)
        })
        this.sets[set] = { members, matrix }
      }
    }
    for (const u of this.units) delete u.text
    this.turns = new Map(memory.allTurns().map(x => [x.id, x]))
    this.buildMs = performance.now() - t
  }

  /** p.top(members, scores, POOL) without sorting everything: same scores (same summation order), same ties. */
  top(set, qv) {
    const s = this.sets[set]
    if (!s || !s.members.length) return null
    const n = s.members.length, keep = Math.min(p.POOL, n)
    const heapS = new Float64Array(keep), heapI = new Int32Array(keep)
    let size = 0
    // worse(a, b): a ranks after b (lower score, or equal score and later index)
    const worse = (sa, ia, sb, ib) => sa < sb || (sa === sb && ia > ib)
    const siftDown = j => {
      for (;;) {
        const l = 2 * j + 1, r = l + 1
        let w = j
        if (l < size && worse(heapS[l], heapI[l], heapS[w], heapI[w])) w = l
        if (r < size && worse(heapS[r], heapI[r], heapS[w], heapI[w])) w = r
        if (w === j) return
        ;[heapS[j], heapS[w]] = [heapS[w], heapS[j]]; [heapI[j], heapI[w]] = [heapI[w], heapI[j]]
        j = w
      }
    }
    for (let r = 0; r < n; r++) {
      let score = 0
      if (s.matrix) {
        const m = s.matrix, o = r * DIMS
        for (let k = 0; k < DIMS; k++) score += m[o + k] * qv[k]
      } else {
        const q = s.q, o = r * DIMS
        for (let k = 0; k < DIMS; k++) score += q[o + k] * qv[k]
        score *= s.scale[r]
      }
      if (size < keep) {
        heapS[size] = score; heapI[size] = r; size++
        for (let j = size - 1; j > 0;) {
          const parent = (j - 1) >> 1
          if (!worse(heapS[j], heapI[j], heapS[parent], heapI[parent])) break
          ;[heapS[j], heapS[parent]] = [heapS[parent], heapS[j]]; [heapI[j], heapI[parent]] = [heapI[parent], heapI[j]]
          j = parent
        }
      } else if (worse(heapS[0], heapI[0], score, r)) {
        heapS[0] = score; heapI[0] = r
        siftDown(0)
      }
    }
    const out = Array.from({ length: size }, (_, j) => [heapI[j], heapS[j]])
    out.sort((a, b) => (b[1] - a[1]) || (a[0] - b[0]))
    return out.map(([r, score]) => [s.members[r], score])
  }

  async retrieve(query, options = {}) {
    const [qv] = await embedder.embed([query], 'query')
    const lists = ['p2', 'fact'].map(set => this.top(set, qv)).filter(Boolean)
    const fused = p.rrf(lists.length ? lists : [[]])
    const order = p.distinct(fused.map(([i]) => i), this.units)
    const raw = p.rawIds(order, this.units)
    const { order: ranked, notes } = MEMORY_OPTIONS.updateLinks
      ? p.versioned(raw, memory.factRows(), store.links()) : { order: raw, notes: new Map() }
    const excluded = new Set(options.excludeSessions ?? [])
    const exclude = [], lookup = new Map()
    for (const id of ranked) {
      const t = this.turns.get(id)
      if (!t) continue
      const plain = memory.asTurn(t)
      if (excluded.has(t.session) || options.inContext?.(plain)) exclude.push(id)
      lookup.set(id, { ...plain, session: `${plain.session}|${plain.date.slice(0, 10)}` })
    }
    return p.selectMemory(ranked, lookup, exclude, options.budgetTokens ?? MEMORY_OPTIONS.budgetTokens, estimateTokens, ts => p.renderDated(ts, notes))
  }
}

/** --lean: units streamed from SQLite straight into one matrix (no unit text, no per-unit buffers). */
function leanLoad() {
  const n = store.unitCount()
  const big = new Float32Array(n * DIMS)
  const units = []
  let r = 0
  for (const row of store.db.prepare('SELECT id, kind, session, turns, vector, model FROM units ORDER BY rowid').iterate()) {
    const v = row.vector
    big.set(new Float32Array(v.buffer, v.byteOffset, v.byteLength / 4), r * DIMS)
    units.push({ id: row.id, set: row.kind, session: row.session, turns: JSON.parse(row.turns), model: row.model,
      vector: big.subarray(r * DIMS, (r + 1) * DIMS) })
    r++
  }
  memory.unitCache = units
}
const lean = process.argv.includes('--lean')
result.lean = lean
snap('boot')
let t = performance.now()
if (lean) leanLoad()
else memory.allUnits()
memory.allTurns()
result.loadMs = performance.now() - t
const index = new FastIndex(memory)
result.indexBuildMs = index.buildMs
snap('afterBuild', true)
t = performance.now()
await index.retrieve(queries[0])
result.firstRecallMs = performance.now() - t
const totals = [], locals = []
for (const q of queries) {
  const e0 = embedder.times.query, s = performance.now()
  await index.retrieve(q)
  const ms = performance.now() - s
  totals.push(ms); locals.push(ms - (embedder.times.query - e0))
}
result.warm = { total: summary(totals), local: summary(locals) }
snap('afterWarm')
snap('afterWarmGc', true)

// Same answers as the package? (Memory.retrieve over the same cache; int8 is compared, not required to match.)
let same = 0
for (const q of useInt8 ? [] : queries) {
  const a = await index.retrieve(q)
  const b = await memory.retrieve(q, { rerank: false, budgetTokens: MEMORY_OPTIONS.budgetTokens })
  if (JSON.stringify(a.ids) === JSON.stringify(b.ids)) same++
}
result.identicalToPackage = useInt8 ? 'n/a (int8 ranks differently)' : `${same}/${queries.length}`
const json = JSON.stringify(result, null, 1)
if (outPath) writeFileSync(outPath, json)
const m = x => (x / 1048576).toFixed(0)
console.log(JSON.stringify({ store: storeDir, int8: useInt8, lean, loadMs: result.loadMs.toFixed(0), buildMs: result.indexBuildMs.toFixed(0),
  p50: result.warm.total.p50.toFixed(2), p95: result.warm.total.p95.toFixed(2), localP50: result.warm.local.p50.toFixed(2),
  rssWarmMB: m(result.memory.afterWarm.rss), heapGcMB: m(result.memory.afterWarmGc.heapUsed), abMB: m(result.memory.afterWarmGc.arrayBuffers),
  identical: result.identicalToPackage }))
