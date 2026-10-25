// SCR1 (exploratory screening, dev questions only): LLM-free recall changes at 5,000 sessions, set up as in S2
// (fillers re-dated per question, answer sessions at their own dates; windows only). Arms, all on the dsh plugin's
// stored windows and the same query vector unless noted:
//   A      dense, budget 2,048 (shipped)
//   A1024  dense, budget 1,024 (less text for a small reader)
//   CTX80  the window's previous message clipped to 80 tokens in the excerpt (more windows in 2,048)
//   CTR    only each window's centre message in the excerpt
//   HUB    CSLS hubness correction: 2·cos(w, q) − mean of w's 10 highest cosines to probe questions
//   CEN    cosine after subtracting the mean window vector
//   ROLE   RRF of the dense list with a dense list of assistant-centred windows ("you recommended", "remind me") or of
//          user-centred windows (all other questions)
//   QA     query embedded with EmbeddingGemma's "question answering" task prompt instead of "search result"
// Probe questions: LongMemEval-S questions used nowhere else (--probes). Writes texts for the E2B answer step.
//
//   node scripts/eval/scale/screen.mjs --questions <dev-questions.json> --probes <probes.json> --out <file> [--size 5000]
import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { Memory, MemoryStore, MEMORY_OPTIONS, END, GAP_MS, MSG_MS, PKG, cachedEmbedder, clip, estimateTokens, formatDate,
  layout, loadPlan, loadSessions, messagesOf, pipeline as p, shortId, uuidFor, HttpEmbedder, EMBEDDINGGEMMA_PREFIXES } from './common.mjs'

const { windowOrder } = await import(join(PKG, 'lib/unit-index.js'))
const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }
const questions = JSON.parse(readFileSync(arg('questions'), 'utf8'))
const probes = JSON.parse(readFileSync(arg('probes'), 'utf8'))
const size = Number(arg('size', '5000'))
const outPath = arg('out')
const port = Number(arg('embed-port', '8299'))
const HUB_K = 10, CONTEXT_TOKENS = 80

function parseDate(s) {
  const m = /^(\d{4})\/(\d{2})\/(\d{2}) \(\w{3}\) (\d{2}):(\d{2})$/.exec(s)
  return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]).getTime()
}

const plan = loadPlan()
const sessions = await loadSessions(new Set(plan.fillers.slice(0, size)))
const cache = cachedEmbedder()
const model = cache.model()
const live = new HttpEmbedder({ baseUrl: `http://127.0.0.1:${port}/v1`, model: 'embeddinggemma-300m', apiKey: '', ...EMBEDDINGGEMMA_PREFIXES, timeoutMs: 120000, batch: 32 })
const liveQA = new HttpEmbedder({ baseUrl: `http://127.0.0.1:${port}/v1`, model: 'embeddinggemma-300m', apiKey: '', ...EMBEDDINGGEMMA_PREFIXES,
  queryPrefix: 'task: question answering | query: ', timeoutMs: 120000, batch: 32 })
const embedder = {
  model: () => model,
  async embed(input, task, signal) {
    const out = []
    for (const text of input) out.push(cache.has(text, task) ? (await cache.embed([text], task))[0] : (await live.embed([text], task, signal))[0])
    return out
  },
}
const OPTIONS = { ...MEMORY_OPTIONS, queryTimeoutMs: 60_000 }
const dot = p.dot

async function addWindows(store, adder, session) {
  const turns = store.sessionTurns(session).map(t => adder.asTurn(t))
  for (const u of p.windowUnitsP2(turns)) {
    let vector
    try { [vector] = await embedder.embed([u.text], 'document') } catch { continue }
    store.addUnit({ ...u, vector, model })
  }
}

function selectStd(memory, units, ranked, budget) {
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
  return p.selectMemory(final, lookup, [], budget, estimateTokens, ts => p.renderDated(ts, notes))
}

/** Like selectStd, but a window's previous message is clipped to `context` tokens (0: left out) unless it is a centre. */
function selectCtx(memory, units, ranked, budget, context) {
  const order = p.distinct(ranked, units)
  const turns = memory.turnIndex()
  const chosen = new Map(), full = new Set()
  const render = () => p.renderDated([...chosen.values()])
  outer: for (const i of order) {
    const [centre, prev] = units[i].turns
    for (const [id, isCentre] of [[centre, true], ...(prev ? [[prev, false]] : [])]) {
      if (!isCentre && context === 0) continue
      if (full.has(id) || (!isCentre && chosen.has(id))) continue
      const t = turns.get(id)
      if (!t) continue
      const plain = memory.asTurn(t)
      const before = chosen.get(id)
      chosen.set(id, { ...plain, text: isCentre ? plain.text : clip(plain.text, context), session: `${plain.session}|${plain.date.slice(0, 10)}` })
      if (estimateTokens(render()) > budget) {
        if (before) chosen.set(id, before)
        else chosen.delete(id)
        break outer
      }
      if (isCentre) full.add(id)
    }
  }
  return { text: render(), ids: [...chosen.keys()] }
}

const topOf = (members, score) => p.top(members, members.map(score))
const YOU = /\byou (?:recommended|suggested|mentioned|told|said|gave|provided|listed|shared)\b|\bremind me\b/i

// -- stores ---------------------------------------------------------------------------------------------------
const store = new MemoryStore(':memory:')
const fillerMsgs = messagesOf(layout(plan, size), sessions)
const adder = new Memory(store, embedder, null, OPTIONS)
for (const m of fillerMsgs) adder.record(m)
for (const s of new Set(fillerMsgs.map(m => m.session))) await addWindows(store, adder, s)
const fillerTurns = store.turns().map(t => ({ id: t.id, at: t.at }))
const redate = store.db.prepare('UPDATE turns SET at = ?, date = ? WHERE id = ?')
const probeVectors = []
for (const q of probes) probeVectors.push((await live.embed([q], 'query'))[0])
const fillerUnits = store.units().filter(u => u.set === 'p2')
const mean = new Float32Array(768)
for (const u of fillerUnits) for (let k = 0; k < 768; k++) mean[k] += u.vector[k] / fillerUnits.length
const meanSq = dot(mean, mean)
function hubness(v) {
  const sims = probeVectors.map(pv => dot(v, pv)).sort((a, b) => b - a)
  return sims.slice(0, HUB_K).reduce((a, b) => a + b, 0) / HUB_K
}
const hubOf = new Map(fillerUnits.map(u => [u.id, hubness(u.vector)]))
const meanDotOf = new Map(fillerUnits.map(u => [u.id, dot(u.vector, mean)]))

const ARMS = ['A', 'A1024', 'CTX80', 'CTR', 'HUB', 'CEN', 'ROLE', 'QA']
const results = { size, arms: ARMS, probes: probes.length, rows: [] }
const memory = new Memory(store, embedder, null, OPTIONS)
for (const q of questions) {
  const qd = parseDate(q.question_date)
  store.db.exec('SAVEPOINT plant')
  const delta = qd - GAP_MS - END
  for (const t of fillerTurns) redate.run(t.at + delta, formatDate(t.at + delta), t.id)
  const answers = new Set()
  const evidence = new Set()
  const msgs = []
  for (const ev of q.evidence) {
    const session = uuidFor(ev.id), start = parseDate(ev.date)
    evidence.add(session)
    ev.turns.forEach((t, i) => {
      const id = shortId(session, i + 1)
      if (t.has_answer) answers.add(id)
      msgs.push({ id, session, role: t.role, text: t.content, at: start + i * MSG_MS })
    })
  }
  for (const m of msgs) adder.record(m)
  for (const ev of q.evidence) await addWindows(store, adder, uuidFor(ev.id))
  memory.unitCache = memory.turnCache = memory.turnById = memory.ranked = null
  const answerBySession = q.evidence.map(ev => ev.turns.flatMap((t, i) => t.has_answer ? [shortId(uuidFor(ev.id), i + 1)] : [])).filter(l => l.length)
  const turns = memory.turnIndex()
  const score = r => {
    const got = new Set(r.ids)
    const fromEvidence = r.ids.filter(id => evidence.has(turns.get(id)?.session)).length
    return { text: r.text, ids: r.ids, hit: r.ids.some(id => answers.has(id)), allSessions: answerBySession.every(l => l.some(id => got.has(id))),
      evidenceShare: r.ids.length ? fromEvidence / r.ids.length : 0, tokens: estimateTokens(r.text) }
  }
  const [qv] = await embedder.embed([q.question], 'query')
  const [qa] = await liveQA.embed([q.question], 'query')
  const units = windowOrder(memory.allUnits().filter(u => u.set === 'p2' && u.model === model && u.vector.length === 768),
    id => turns.get(id)?.seq).map(([u]) => u)
  const all = units.map((_, i) => i)
  const dense = units.map(u => dot(u.vector, qv))
  const denseTop = topOf(all, i => dense[i]).map(([i]) => i)
  const row = { qid: q.qid, type: q.type, routed: YOU.test(q.question) ? 'assistant' : 'user' }
  const shipped = await memory.retrieve(q.question, { rerank: false, budgetTokens: OPTIONS.budgetTokens })
  row.A = score(selectStd(memory, units, denseTop, OPTIONS.budgetTokens))
  row.selectionMatchesRetrieve = JSON.stringify(row.A.ids) === JSON.stringify(shipped.ids)
  row.A1024 = score(selectStd(memory, units, denseTop, 1024))
  row.CTX80 = score(selectCtx(memory, units, denseTop, OPTIONS.budgetTokens, CONTEXT_TOKENS))
  row.CTR = score(selectCtx(memory, units, denseTop, OPTIONS.budgetTokens, 0))
  const hub = units.map(u => hubOf.get(u.id) ?? hubness(u.vector))
  row.HUB = score(selectStd(memory, units, topOf(all, i => 2 * dense[i] - hub[i]).map(([i]) => i), OPTIONS.budgetTokens))
  const qMean = dot(qv, mean)
  row.CEN = score(selectStd(memory, units, topOf(all, i => {
    const vm = meanDotOf.get(units[i].id) ?? dot(units[i].vector, mean)
    return (dense[i] - vm - qMean + meanSq) / Math.sqrt(Math.max(1e-9, 1 - 2 * vm + meanSq))
  }).map(([i]) => i), OPTIONS.budgetTokens))
  const role = row.routed
  const subset = all.filter(i => turns.get(units[i].turns[0])?.role === role)
  row.ROLE = score(selectStd(memory, units, p.rrf([topOf(all, i => dense[i]), topOf(subset, i => dense[i])]).map(([i]) => i), OPTIONS.budgetTokens))
  const qaScores = units.map(u => dot(u.vector, qa))
  row.QA = score(selectStd(memory, units, topOf(all, i => qaScores[i]).map(([i]) => i), OPTIONS.budgetTokens))
  results.rows.push(row)
  store.db.exec('ROLLBACK TO plant')
  store.db.exec('RELEASE plant')
  memory.unitCache = memory.turnCache = memory.turnById = memory.ranked = null
  console.error(`${results.rows.length}/${questions.length}`)
}
adder.close()
memory.close()
store.close()
writeFileSync(outPath, JSON.stringify(results))
console.error(`wrote ${outPath}`)
