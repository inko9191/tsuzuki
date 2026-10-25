// Accuracy drift: 60 LongMemEval-S questions with known answers, each planted (its evidence sessions) at three ages
// (5 %, 50 %, 95 % of the timeline) into a store of S filler sessions, recalled with Memory.retrieve (the dsh
// plugin's recall: dense order, 2,048-token budget). No LLM: the measure is whether the recall hands back the
// messages LongMemEval marks `has_answer`.
//
// One base store per (mode, size) holds the S fillers; each (question, age) adds the evidence inside a SAVEPOINT,
// recalls, and rolls back. `--validate 50,500` also rebuilds those sizes from scratch with the evidence captured in
// timeline order and checks the recalled ids are the same.
//
//   node scripts/eval/scale/accuracy.mjs --vectors real [--sizes 50,500,2000,5000] [--modes fixed,shipped]
//        [--int8] [--validate 50,500] [--out build/scale/accuracy-real.json]
// modes: fixed   = the evaluated p2 windows (what Memory.backfill builds)
//        shipped = the windows Memory.capture builds (identical to fixed since commit 790e13f)
//        prefix  = the windows capture built before 790e13f (empty from a session's 4th message on), emulated
import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { Memory, MemoryStore, MEMORY_OPTIONS, MODEL_IDS, OUT, END, GAP_MS, MSG_MS, cachedEmbedder, layout, loadPlan, loadSessions,
  messagesOf, pipeline as p, sha1, shortId, syntheticEmbedder, uuidFor } from './common.mjs'

const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }
const vectors = arg('vectors', 'real')
const sizes = arg('sizes', '50,500,2000,5000').split(',').map(Number)
const modes = arg('modes', 'fixed,shipped').split(',')
const validate = arg('validate', '').split(',').filter(Boolean).map(Number)
const int8 = process.argv.includes('--int8')
const limitQ = Number(arg('limit', '0'))
const outPath = arg('out', join(OUT, `accuracy-${vectors}.json`))

const plan = loadPlan()
const planted = limitQ ? plan.planted.slice(0, limitQ) : plan.planted
const ages = Object.entries(plan.ages)
const maxSize = Math.max(...sizes)
const sessions = await loadSessions(new Set([...plan.fillers.slice(0, maxSize), ...planted.flatMap(q => q.evidence)]))

const base = vectors === 'real' ? cachedEmbedder() : syntheticEmbedder(MODEL_IDS.synthetic)
const model = base.model()
const dummy = syntheticEmbedder(model)
/** Cached vectors; a text egemma-server refused (over 2,048 tokens) gets a zero vector and is deleted afterwards. */
const refused = new Set()
const lenient = {
  model: () => model,
  async embed(input, task) {
    return Promise.all(input.map(async t => {
      try {
        return (await base.embed([t], task))[0]
      } catch {
        refused.add(t)
        return new Float32Array(768)
      }
    }))
  },
}
const dropRefused = store => {
  if (!refused.size) return 0
  const del = store.db.prepare('DELETE FROM units WHERE id = ?')
  let n = 0
  for (const r of store.db.prepare('SELECT id, text FROM units').all()) if (refused.has(r.text)) { del.run(r.id); n++ }
  return n
}

function quantized(v) {
  let m = 0
  for (const x of v) m = Math.max(m, Math.abs(x))
  const s = m / 127 || 1
  return Float32Array.from(v, x => Math.round(x / s) * s)
}

/** Stores the windows of `sessionIds` the way `mode` builds them. Messages are captured with Memory.capture. */
async function addSessions(store, msgs, mode) {
  if (mode === 'prefix') {
    // capture() before commit 790e13f: windowUnitsP2 over [previous, current] with their in-session `pos`, so the
    // 3rd message got a window of itself only and the 4th and later an empty window with no turns.
    const adder = new Memory(store, dummy, null, MEMORY_OPTIONS)
    for (const m of msgs) await adder.capture(m)
    adder.close()
    const ids = [...new Set(msgs.map(m => m.session))]
    const del = store.db.prepare('DELETE FROM units WHERE session = ?')
    for (const s of ids) del.run(s)
    for (const s of ids) {
      const turns = store.sessionTurns(s).map(t => adder.asTurn(t))
      const units = turns.map((t, k) => p.windowUnitsP2([...(k > 0 ? [turns[k - 1]] : []), t]).at(-1))
      const vecs = await lenient.embed(units.map(u => u.text), 'document')
      units.forEach((u, k) => store.addUnit({ ...u, vector: vecs[k], model }))
    }
  } else if (mode === 'shipped') {
    const adder = new Memory(store, lenient, null, MEMORY_OPTIONS)
    for (const m of msgs) await adder.capture(m)
    adder.close()
  } else {
    const adder = new Memory(store, dummy, null, MEMORY_OPTIONS)
    for (const m of msgs) await adder.capture(m)
    adder.close()
    const ids = [...new Set(msgs.map(m => m.session))]
    const del = store.db.prepare('DELETE FROM units WHERE session = ?')
    for (const s of ids) del.run(s)
    // Exactly Memory.backfill's windows: windowUnitsP2 over each whole session's clipped turns.
    for (const s of ids) {
      const turns = store.sessionTurns(s).map(t => adder.asTurn(t))
      const units = p.windowUnitsP2(turns)
      const vecs = await lenient.embed(units.map(u => u.text), 'document')
      units.forEach((u, k) => store.addUnit({ ...u, vector: vecs[k], model }))
    }
  }
  return dropRefused(store)
}

async function baseStore(mode, size, order) {
  const store = new MemoryStore(':memory:')
  const t0 = performance.now()
  await addSessions(store, messagesOf(order, sessions), mode)
  return { store, buildMs: performance.now() - t0 }
}

function answerIds(q) {
  return q.evidence.flatMap(orig => q.answer_turns[orig].map(i => shortId(uuidFor(orig), i + 1)))
}

/** Rank of the first answer message in the raw (dense, distinct, pool 100) order; null when outside the pool. */
function firstAnswerRank(memory, qv, answers) {
  const units = memory.indexUnits().filter(u => u.model === model && u.vector.length === qv.length)
  const order = p.distinct(p.rank(units, units.map(u => u.vector), qv).map(([i]) => i), units)
  const raw = p.rawIds(order, units)
  const k = raw.findIndex(id => answers.has(id))
  return k < 0 ? null : k + 1
}

function score(q, r, answers, rank) {
  const got = new Set(r.ids)
  const hits = [...answers].filter(id => got.has(id))
  const bySession = q.evidence.filter(orig => q.answer_turns[orig].length)
  const covered = bySession.filter(orig => q.answer_turns[orig].some(i => got.has(shortId(uuidFor(orig), i + 1))))
  const ans = q.answer.trim().toLowerCase()
  return { hit: hits.length > 0, answerTurnRecall: hits.length / answers.size, sessionsCovered: covered.length / bySession.length,
    allSessions: covered.length === bySession.length, firstRank: rank, ids: r.ids.length,
    answerInText: ans.length <= 40 ? r.text.toLowerCase().includes(ans) : null }
}

/** Evidence messages dated at `age` of a timeline of `size` fillers (just before filler round(age*size)). */
function evidenceMessages(q, size, age) {
  const at = Math.min(size, Math.round(age * size))
  const fillerStart = k => END - (size - 1 - k) * GAP_MS
  const start = fillerStart(at) - GAP_MS / 2
  const out = []
  q.evidence.forEach((orig, j) => {
    const m = messagesOf([orig], sessions, start + j * (GAP_MS / 2) / q.evidence.length)
    out.push(...m)
  })
  return out
}

const results = { vectors, model, sizes, modes, ages: plan.ages, planted: planted.length, int8, rows: [], builds: [], validation: [] }
for (const mode of modes) {
  for (const size of sizes) {
    const { store, buildMs } = await baseStore(mode, size, layout(plan, size))
    const memory = new Memory(store, base, null, MEMORY_OPTIONS)
    memory.allUnits(); memory.allTurns()
    const nUnits = memory.unitCache.length, nTurns = memory.turnCache.length
    const maxRow = store.db.prepare('SELECT COALESCE(MAX(rowid), 0) AS r FROM units').get().r
    const maxSeq = store.db.prepare('SELECT COALESCE(MAX(seq), 0) AS s FROM turns').get().s
    let memory8 = null
    if (int8) {
      memory8 = new Memory(store, base, null, MEMORY_OPTIONS)
      memory8.unitCache = memory.unitCache.map(u => ({ ...u, vector: quantized(u.vector) }))
      memory8.turnCache = memory.turnCache
    }
    results.builds.push({ mode, size, units: nUnits, turns: nTurns, buildMs })
    const t0 = performance.now()
    for (const q of planted) {
      const answers = new Set(answerIds(q))
      const [qv] = await base.embed([q.question], 'query')
      for (const [ageName, age] of ages) {
        store.db.exec('SAVEPOINT plant')
        await addSessions(store, evidenceMessages(q, size, age), mode)
        const newUnits = store.db.prepare('SELECT * FROM units WHERE rowid > ? ORDER BY rowid').all(maxRow).map(r => ({
          id: r.id, set: r.kind, session: r.session, text: r.text, turns: JSON.parse(r.turns),
          vector: new Float32Array(r.vector.slice().buffer), model: r.model }))
        const newTurns = store.db.prepare('SELECT * FROM turns WHERE seq > ? ORDER BY seq').all(maxSeq)
        memory.unitCache.push(...newUnits)
        memory.turnCache.push(...newTurns)
        const r = await memory.retrieve(q.question, { rerank: false, budgetTokens: MEMORY_OPTIONS.budgetTokens })
        const row = { mode, size, age: ageName, qid: q.qid, type: q.type, evidenceSessions: q.evidence.length,
          ...score(q, r, answers, firstAnswerRank(memory, qv, answers)), retrievedIds: r.ids }
        if (memory8) {
          memory8.unitCache.push(...newUnits.map(u => ({ ...u, vector: quantized(u.vector) })))
          const r8 = await memory8.retrieve(q.question, { rerank: false, budgetTokens: MEMORY_OPTIONS.budgetTokens })
          row.int8 = { ...score(q, r8, answers, null), sameIds: JSON.stringify(r8.ids) === JSON.stringify(r.ids) }
          memory8.unitCache.length = nUnits
        }
        results.rows.push(row)
        store.db.exec('ROLLBACK TO plant')
        store.db.exec('RELEASE plant')
        memory.unitCache.length = nUnits
        memory.turnCache.length = nTurns
      }
    }
    const rows = results.rows.filter(r => r.mode === mode && r.size === size)
    const mean = f => rows.reduce((a, r) => a + Number(f(r)), 0) / rows.length
    console.log(JSON.stringify({ mode, size, units: nUnits, buildS: (buildMs / 1000).toFixed(1), evalS: ((performance.now() - t0) / 1000).toFixed(1),
      hit: mean(r => r.hit).toFixed(3), allSessions: mean(r => r.allSessions).toFixed(3), turnRecall: mean(r => r.answerTurnRecall).toFixed(3),
      ...(int8 ? { hitInt8: mean(r => r.int8.hit).toFixed(3), sameIdsInt8: mean(r => r.int8.sameIds).toFixed(3) } : {}) }))
    memory.close()
    store.close()

    if (validate.includes(size)) {
      // Rebuild with the evidence captured in timeline order (true seq order) and compare the recalled ids.
      let same = 0, n = 0
      for (const q of planted) {
        for (const [ageName, age] of ages) {
          const fill = plan.fillers.slice(0, size)
          const at = Math.min(size, Math.round(age * size))
          const msgs = [...messagesOf(fill, sessions)]
          const ev = evidenceMessages(q, size, age)
          const cut = msgs.findIndex(m => m.session === uuidFor(fill[at] ?? '\u0000'))
          const ordered = cut < 0 ? [...msgs, ...ev] : [...msgs.slice(0, cut), ...ev, ...msgs.slice(cut)]
          const s2 = new MemoryStore(':memory:')
          await addSessions(s2, ordered, mode)
          const m2 = new Memory(s2, base, null, MEMORY_OPTIONS)
          const r = await m2.retrieve(q.question, { rerank: false, budgetTokens: MEMORY_OPTIONS.budgetTokens })
          const row = results.rows.find(x => x.mode === mode && x.size === size && x.qid === q.qid && x.age === ageName)
          n++
          if (JSON.stringify(r.ids) === JSON.stringify(row.retrievedIds)) same++
          m2.close(); s2.close()
        }
      }
      results.validation.push({ mode, size, compared: n, identical: same })
      console.log(JSON.stringify({ validate: { mode, size, compared: n, identical: same } }))
    }
  }
}
results.refusedWindows = refused.size
writeFileSync(outPath, JSON.stringify(results))
console.log(`wrote ${outPath}`)
