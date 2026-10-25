// Diagnosis of the accuracy drop at scale (exploratory, no LLM, CPU only). Same stores, questions and planting as
// accuracy.mjs (mode fixed, real cached vectors), then for each (question, age):
//   - where the first answer message ranks in the full dense order (recall@K; K = messages, and units for the
//     reranker's 30 candidates): how much a perfect second stage could win back;
//   - larger budgets (4,096 / 8,192 tokens) with Memory.retrieve;
//   - an oracle date filter: only sessions within ±D days of the evidence sessions are candidates (the ceiling of a
//     date-range filter that knew the right range);
//   - lexical fusion: BM25 over the same windows fused with the dense list by RRF (as windows and facts are fused).
// Checks first that its own selection reproduces Memory.retrieve and the ids in accuracy-real.json.
//
//   node scripts/eval/scale/diagnose.mjs [--sizes 50,500,2000,5000] [--out build/scale/diagnose-real.json]
import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { Memory, MemoryStore, MEMORY_OPTIONS, OUT, END, GAP_MS, PKG, cachedEmbedder, estimateTokens, layout, loadPlan, loadSessions,
  messagesOf, pipeline as p, shortId, uuidFor } from './common.mjs'

const { windowOrder } = await import(join(PKG, 'lib/unit-index.js'))
const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }
const sizes = arg('sizes', '50,500,2000,5000').split(',').map(Number)
const outPath = arg('out', join(OUT, 'diagnose-real.json'))
const BUDGETS = [4096, 8192]
const DATE_DAYS = [1, 7, 30]
const RANK_KS = [10, 20, 50, 100, 200, 500]
const RERANK_UNITS = p.RERANK_CANDIDATES
const DAY = 86_400_000

const plan = loadPlan()
const planted = plan.planted
const ages = Object.entries(plan.ages)
const sessions = await loadSessions(new Set([...plan.fillers.slice(0, Math.max(...sizes)), ...planted.flatMap(q => q.evidence)]))
const embedder = cachedEmbedder()
const model = embedder.model()
const reference = new Map(JSON.parse(readFileSync(join(OUT, 'accuracy-real.json'), 'utf8')).rows
  .filter(r => r.mode === 'fixed').map(r => [`${r.size}|${r.qid}|${r.age}`, r.retrievedIds]))

/** Records the messages (no embedding) and stores each session's p2 windows with cached vectors (Memory.backfill's windows). */
async function addSessions(store, msgs) {
  const adder = new Memory(store, embedder, null, MEMORY_OPTIONS)
  for (const m of msgs) adder.record(m)
  let refused = 0
  for (const s of new Set(msgs.map(m => m.session))) {
    const turns = store.sessionTurns(s).map(t => adder.asTurn(t))
    for (const u of p.windowUnitsP2(turns)) {
      let vector
      try {
        [vector] = await embedder.embed([u.text], 'document')
      } catch {
        refused++  // over egemma-server's 2,048 tokens: never indexed (accuracy.mjs deletes them)
        continue
      }
      store.addUnit({ ...u, vector, model })
    }
  }
  adder.close()
  return refused
}

function evidenceMessages(q, size, age) {
  const at = Math.min(size, Math.round(age * size))
  const start = END - (size - 1 - at) * GAP_MS - GAP_MS / 2
  return q.evidence.flatMap((orig, j) => messagesOf([orig], sessions, start + j * (GAP_MS / 2) / q.evidence.length))
}

const answerIds = q => new Set(q.evidence.flatMap(orig => q.answer_turns[orig].map(i => shortId(uuidFor(orig), i + 1))))

// -- BM25 over window texts ------------------------------------------------------------------
const STOP = new Set(('a an the and or but of to in on at for with by from is are was were be been being i me my mine we our you your ' +
  'he she it its they them their this that these those do does did have has had what which who whom how when where why can could ' +
  'would should will shall may might must just so as if than then there here about into over not no yes am also any all some ' +
  'user assistant').split(' '))
const tokens = text => (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter(t => t.length > 1 && !STOP.has(t))
const K1 = 1.2, B = 0.75

function termCounts(text) {
  const tf = new Map()
  for (const t of tokens(text)) tf.set(t, (tf.get(t) ?? 0) + 1)
  return tf
}

/** Inverted index of the filler windows; evidence windows are scored on the fly against the combined statistics. */
function lexicon(units) {
  const postings = new Map(), lengths = new Float64Array(units.length)
  units.forEach((u, i) => {
    const tf = termCounts(u.text)
    for (const [t, n] of tf) {
      let list = postings.get(t)
      if (!list) postings.set(t, list = [])
      list.push(i, n)
    }
    lengths[i] = [...tf.values()].reduce((a, b) => a + b, 0)
  })
  return { ids: units.map(u => u.id), postings, lengths, total: lengths.reduce((a, b) => a + b, 0) }
}

/** BM25 scores by unit id for `query` over the filler windows plus `extra` (evidence) windows. */
function bm25(lex, extra, query) {
  const extraTf = extra.map(u => termCounts(u.text))
  const extraLen = extraTf.map(tf => [...tf.values()].reduce((a, b) => a + b, 0))
  const n = lex.ids.length + extra.length
  const avg = (lex.total + extraLen.reduce((a, b) => a + b, 0)) / n
  const scores = new Map()
  for (const t of new Set(tokens(query))) {
    const list = lex.postings.get(t) ?? []
    const df = list.length / 2 + extraTf.filter(tf => tf.has(t)).length
    if (!df) continue
    const idf = Math.log(1 + (n - df + 0.5) / (df + 0.5))
    const add = (id, tf, len) => scores.set(id, (scores.get(id) ?? 0) + idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * len / avg)))
    for (let k = 0; k < list.length; k += 2) add(lex.ids[list[k]], list[k + 1], lex.lengths[list[k]])
    extra.forEach((u, j) => { const tf = extraTf[j].get(t); if (tf) add(u.id, tf, extraLen[j]) })
  }
  return scores
}

// -- selection exactly as Memory.retrieve does it, from a ranked list of unit indices --------------
function select(memory, units, ranked, budget) {
  const order = p.distinct(ranked, units)
  const raw = p.rawIds(order, units)
  const { order: final, notes } = p.versioned(raw, memory.factRows(), memory.store.links())
  const turns = memory.turnIndex()
  const lookup = new Map()
  for (const id of final) {
    const t = turns.get(id)
    if (!t) continue
    const plain = memory.asTurn(t)
    lookup.set(id, { ...plain, session: `${plain.session}|${plain.date.slice(0, 10)}` })
  }
  return p.selectMemory(final, lookup, [], budget, estimateTokens, ts => p.renderDated(ts, notes)).ids
}

/** Top `n` unit indices of `members` by `score` (ties: index order, as pipeline.top). */
function topOf(members, score, n = p.POOL) {
  return p.top(members, members.map(score), n)
}

const results = { model, sizes, ages: plan.ages, budgets: BUDGETS, dateDays: DATE_DAYS, rankKs: RANK_KS, rows: [], builds: [], checks: { retrieve: 0, reference: 0, rows: 0 } }
for (const size of sizes) {
  const t0 = performance.now()
  const store = new MemoryStore(':memory:')
  const fillerMsgs = messagesOf(layout(plan, size), sessions)
  const refused = await addSessions(store, fillerMsgs)
  const fillerUnits = store.units().filter(u => u.set === 'p2')
  const lex = lexicon(fillerUnits)
  const fillerIds = new Set(fillerUnits.map(u => u.id))
  const startOf = new Map()
  for (const m of fillerMsgs) if (!startOf.has(m.session)) startOf.set(m.session, m.at)
  results.builds.push({ size, units: fillerUnits.length, refused, buildS: (performance.now() - t0) / 1000 })
  const t1 = performance.now()
  const memory = new Memory(store, embedder, null, MEMORY_OPTIONS)
  for (const q of planted) {
    const answers = answerIds(q)
    const [qv] = await embedder.embed([q.question], 'query')
    const hit = ids => ids.some(id => answers.has(id))
    for (const [ageName, age] of ages) {
      store.db.exec('SAVEPOINT plant')
      const ev = evidenceMessages(q, size, age)
      await addSessions(store, ev)
      const evStart = new Map()
      for (const m of ev) if (!evStart.has(m.session)) evStart.set(m.session, m.at)
      // One Memory per size, its caches dropped per row: a closed Memory that embedded a query stays reachable (the
      // AbortSignal.any of its requests), so a fresh one per row ran out of heap at 500 sessions.
      memory.unitCache = memory.turnCache = memory.turnById = memory.ranked = null
      const row = { size, age: ageName, qid: q.qid, type: q.type, evidenceSessions: q.evidence.length }

      const base = await memory.retrieve(q.question, { rerank: false, budgetTokens: MEMORY_OPTIONS.budgetTokens })
      row.base = { hit: hit(base.ids), selected: base.ids.length }
      results.checks.rows++
      if (JSON.stringify(base.ids) === JSON.stringify(reference.get(`${size}|${q.qid}|${ageName}`))) results.checks.reference++
      for (const budget of BUDGETS) {
        const r = await memory.retrieve(q.question, { rerank: false, budgetTokens: budget })
        row[`budget${budget}`] = { hit: hit(r.ids), selected: r.ids.length }
      }

      // The ranked index's unit order (windows by the capture order of the message they centre on).
      const turns = memory.turnIndex()
      const units = windowOrder(memory.allUnits().filter(u => u.set === 'p2' && u.model === model && u.vector.length === qv.length),
        id => turns.get(id)?.seq).map(([u]) => u)
      const all = units.map((_, i) => i)
      const dense = units.map(u => p.dot(u.vector, qv))
      const denseTop = topOf(all, i => dense[i])
      const mine = select(memory, units, denseTop.map(([i]) => i), MEMORY_OPTIONS.budgetTokens)
      if (JSON.stringify(mine) === JSON.stringify(base.ids)) results.checks.retrieve++

      // Full dense order: rank of the first answer message, and of the first unit showing one.
      const full = p.distinct(p.top(all, dense, all.length).map(([i]) => i), units, Number.POSITIVE_INFINITY)
      const messages = p.rawIds(full, units)
      const m = messages.findIndex(id => answers.has(id))
      const u = full.findIndex(i => units[i].turns.some(id => answers.has(id)))
      row.messageRank = m < 0 ? null : m + 1
      row.unitRank = u < 0 ? null : u + 1
      row.recallAt = Object.fromEntries(RANK_KS.map(k => [k, m >= 0 && m < k]))
      row.inRerankCandidates = u >= 0 && u < RERANK_UNITS

      // Oracle date filter: candidates only from sessions starting within ±D days of an evidence session.
      const evTimes = [...evStart.values()]
      for (const days of DATE_DAYS) {
        const near = units.flatMap((x, i) => {
          if (!fillerIds.has(x.id)) return [i]
          const s = startOf.get(x.session)
          return evTimes.some(t => Math.abs(s - t) <= days * DAY) ? [i] : []
        })
        const ids = select(memory, units, topOf(near, i => dense[i]).map(([i]) => i), MEMORY_OPTIONS.budgetTokens)
        row[`date${days}`] = { hit: hit(ids), candidates: near.length }
      }

      // BM25 fused with dense (RRF), and BM25 alone.
      const index = new Map(units.map((x, i) => [x.id, i]))
      const lexScores = bm25(lex, units.filter(x => !fillerIds.has(x.id)), q.question)
      const scored = [...lexScores.keys()].flatMap(id => index.has(id) ? [index.get(id)] : [])
      const lexTop = p.top(scored, scored.map(i => lexScores.get(units[i].id)))
      const fused = p.rrf([denseTop, lexTop])
      row.hybrid = { hit: hit(select(memory, units, fused.map(([i]) => i), MEMORY_OPTIONS.budgetTokens)) }
      row.bm25 = { hit: hit(select(memory, units, lexTop.map(([i]) => i), MEMORY_OPTIONS.budgetTokens)) }
      const fusedFull = p.distinct(fused.map(([i]) => i), units, Number.POSITIVE_INFINITY)
      const hm = p.rawIds(fusedFull, units).findIndex(id => answers.has(id))
      row.hybridMessageRank = hm < 0 ? null : hm + 1

      results.rows.push(row)
      if (process.env.DIAG_HEAP && results.rows.length % 30 === 0 && globalThis.gc) {
        globalThis.gc()
        console.error(`heap ${results.rows.length} rows: ${(process.memoryUsage().heapUsed / 1048576).toFixed(0)} MB`)
      }
      store.db.exec('ROLLBACK TO plant')
      store.db.exec('RELEASE plant')
    }
  }
  const rows = results.rows.filter(r => r.size === size)
  const mean = f => (rows.reduce((a, r) => a + Number(f(r)), 0) / rows.length).toFixed(3)
  console.log(JSON.stringify({ size, units: fillerUnits.length, evalS: ((performance.now() - t1) / 1000).toFixed(0), base: mean(r => r.base.hit),
    b4096: mean(r => r.budget4096.hit), b8192: mean(r => r.budget8192.hit), at50: mean(r => r.recallAt[50]), at200: mean(r => r.recallAt[200]),
    rerank30: mean(r => r.inRerankCandidates), date7: mean(r => r.date7.hit), hybrid: mean(r => r.hybrid.hit), checks: results.checks }))
  memory.close()
  store.close()
}
writeFileSync(outPath, JSON.stringify(results))
console.log(`wrote ${outPath}`)
