// S3 recall inputs (docs/MEMORY_SCALE_S3_PREREG.md): on a copy of a filler store built by s3_build.mjs (fillers with
// facts and update links), each question's answer sessions are added at their own dates (fillers and their segments
// re-dated as in S2), their facts extracted and update-checked by the plugin (Memory.drain with the chat model), and
// two recalls are taken:
//   A  windows only, dense, budget 2,048 (S2's A: what the plugin recalls without a chat model)
//   P  the plugin's recall with facts: windows and facts fused (RRF), update notes, budget 2,048 (setting P)
// No answers here: the texts go to longmemeval_s3.py.
//
//   node scripts/eval/scale/s3_retrieve.mjs --questions <file> --db <fillers.sqlite> --work <copy.sqlite> --out <file> --chat <url>
import { copyFileSync, existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { Memory, MemoryStore, MEMORY_OPTIONS, END, GAP_MS, MSG_MS, PKG, cachedEmbedder, estimateTokens, formatDate, pipeline as p,
  shortId, uuidFor, HttpEmbedder, EMBEDDINGGEMMA_PREFIXES } from './common.mjs'

const { windowOrder } = await import(join(PKG, 'lib/unit-index.js'))
const { OpenAIChat, LLAMA_CPP_EXTRA } = await import(join(PKG, 'lib/llm.js'))
const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }
const questions = JSON.parse(readFileSync(arg('questions'), 'utf8'))
const work = arg('work')
const port = Number(arg('embed-port', '8299'))
const log = (...a) => console.error(new Date().toISOString().slice(11, 19), ...a)

function parseDate(s) {
  const m = /^(\d{4})\/(\d{2})\/(\d{2}) \(\w{3}\) (\d{2}):(\d{2})$/.exec(s)
  if (!m) throw new Error(`bad date ${s}`)
  return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]).getTime()
}

const cache = cachedEmbedder()
const model = cache.model()
const live = new HttpEmbedder({ baseUrl: `http://127.0.0.1:${port}/v1`, model: 'embeddinggemma-300m', apiKey: '', ...EMBEDDINGGEMMA_PREFIXES, timeoutMs: 120000, batch: 32 })
const embedder = {
  model: () => model,
  async embed(input, task, signal) {
    const out = new Array(input.length), miss = []
    for (let k = 0; k < input.length; k++) {
      if (cache.has(input[k], task)) out[k] = (await cache.embed([input[k]], task))[0]
      else miss.push(k)
    }
    if (miss.length) {
      const vs = await live.embed(miss.map(k => input[k]), task, signal)
      miss.forEach((k, j) => { out[k] = vs[j] })
    }
    return out
  },
}
const chat = new OpenAIChat({ baseUrl: arg('chat'), model: '', apiKey: '', timeoutMs: 600_000, extraBody: LLAMA_CPP_EXTRA })
const OPTIONS = { ...MEMORY_OPTIONS, queryTimeoutMs: 60_000, log: (level, message) => { if (level === 'warn') log('warn', message) } }

for (const f of [work, `${work}-wal`, `${work}-shm`]) if (existsSync(f)) rmSync(f)
copyFileSync(arg('db'), work)
const store = new MemoryStore(work)
if (!store.getMeta('s3.complete')) throw new Error('filler store not complete')
// The store's finishSegment wraps its writes in BEGIN/COMMIT. Inside the per-question SAVEPOINT those become a nested
// savepoint (the same all-or-nothing write), so everything a question adds still rolls back.
const exec = store.db.exec.bind(store.db)
store.db.exec = sql => exec(sql === 'BEGIN' ? 'SAVEPOINT inner' : sql === 'COMMIT' ? 'RELEASE inner' : sql === 'ROLLBACK' ? 'ROLLBACK TO inner; RELEASE inner' : sql)
const fillerTurns = store.turns().map(t => ({ id: t.id, at: t.at }))
const fillerSegments = store.db.prepare('SELECT id, date FROM segments').all().map(s => ({ id: s.id, at: parseDate(s.date) }))
const redateTurn = store.db.prepare('UPDATE turns SET at = ?, date = ? WHERE id = ?')
const redateSegment = store.db.prepare('UPDATE segments SET date = ? WHERE id = ?')
const linkCount = () => store.db.prepare('SELECT COUNT(*) AS n FROM links').get().n
const memory = new Memory(store, embedder, chat, OPTIONS)
const reset = () => { memory.unitCache = memory.turnCache = memory.turnById = memory.ranked = null }

/** Windows only, no facts or notes: S2's A. */
function windowsOnly(qv, budget) {
  const turns = memory.turnIndex()
  const units = windowOrder(memory.allUnits().filter(u => u.set === 'p2' && u.model === model && u.vector.length === qv.length),
    id => turns.get(id)?.seq).map(([u]) => u)
  const all = units.map((_, i) => i)
  const order = p.distinct(p.top(all, units.map(u => p.dot(u.vector, qv))).map(([i]) => i), units)
  const lookup = new Map()
  for (const id of p.rawIds(order, units)) {
    const t = turns.get(id)
    if (!t) continue
    const plain = memory.asTurn(t)
    lookup.set(id, { ...plain, session: `${plain.session}|${plain.date.slice(0, 10)}` })
  }
  return p.selectMemory(p.rawIds(order, units), lookup, [], budget, estimateTokens, ts => p.renderDated(ts))
}

const rows = []
for (const q of questions) {
  const qd = parseDate(q.question_date)
  store.db.exec('SAVEPOINT plant')
  const delta = qd - GAP_MS - END
  for (const t of fillerTurns) redateTurn.run(t.at + delta, formatDate(t.at + delta), t.id)
  for (const s of fillerSegments) redateSegment.run(formatDate(s.at + delta), s.id)
  const answers = new Set()
  const evidence = [...q.evidence].sort((a, b) => parseDate(a.date) - parseDate(b.date))
  for (const ev of evidence) {
    const session = uuidFor(ev.id), start = parseDate(ev.date)
    ev.turns.forEach((t, i) => {
      const id = shortId(session, i + 1)
      if (t.has_answer) answers.add(id)
      memory.record({ id, session, role: t.role, text: t.content, at: start + i * MSG_MS })
    })
    memory.maybeCloseSegment(session, true)
  }
  reset()
  for (const ev of evidence) {
    const session = uuidFor(ev.id)
    const turns = store.sessionTurns(session).map(t => memory.asTurn(t))
    for (const u of p.windowUnitsP2(turns)) {
      let vector
      try { [vector] = await embedder.embed([u.text], 'document') } catch { continue }
      store.addUnit({ ...u, vector, model })
    }
  }
  reset()
  const links0 = linkCount()
  const t0 = performance.now()
  for (let tries = 0; store.pendingSegments(3).length; tries++) {
    if (tries > 30) throw new Error('evidence segments still pending')
    await memory.drain()
    if (store.pendingSegments(3).length) await new Promise(r => setTimeout(r, 2000))
  }
  const evidenceSessions = new Set(evidence.map(ev => uuidFor(ev.id)))
  const evidenceFacts = [...store.facts().entries()].filter(([seg]) => evidenceSessions.has(seg.split('@')[0])).reduce((n, [, f]) => n + f.length, 0)
  reset()
  const answerBySession = evidence.map(ev => ev.turns.flatMap((t, i) => t.has_answer ? [shortId(uuidFor(ev.id), i + 1)] : [])).filter(l => l.length)
  const score = r => {
    const got = new Set(r.ids)
    return { text: r.text, ids: r.ids, hit: r.ids.some(id => answers.has(id)), allSessions: answerBySession.every(l => l.some(id => got.has(id))) }
  }
  const [qv] = await embedder.embed([q.question], 'query')
  const row = { qid: q.qid, type: q.type, evidenceFacts, evidenceLinks: linkCount() - links0, extractSeconds: (performance.now() - t0) / 1000 }
  row.A = score(windowsOnly(qv, MEMORY_OPTIONS.budgetTokens))
  const pr = await memory.retrieve(q.question, { rerank: false })
  row.P = { ...score(pr), indentedLines: (pr.text.match(/^ {2}/gm) ?? []).length }
  rows.push(row)
  store.db.exec('ROLLBACK TO plant')
  store.db.exec('RELEASE plant')
  reset()
  log(`${rows.length}/${questions.length}`)
}
memory.close()
store.close()
for (const f of [work, `${work}-wal`, `${work}-shm`]) if (existsSync(f)) rmSync(f)
writeFileSync(arg('out'), JSON.stringify({ model, rows }))
log('wrote', arg('out'))
