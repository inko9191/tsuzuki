// S2 recall inputs (docs/MEMORY_SCALE_S2_PREREG.md): for each fresh LongMemEval-S question, its answer sessions at
// their own dates among S filler sessions (the scale test's fillers, re-dated per question so that the newest one
// starts 96 minutes before the question), recalled by the dsh plugin's Memory (windows only, no facts) in five ways:
//   A    dense, budget 2,048 (the shipped default)
//   B    dense, budget 8,192
//   C30  the shipped LLM rerank (top 30 windows, keep 10) with the chat model at --chat, then 2,048
//   C100 the same prompt over the whole dense pool (top 100 windows, up to 200 messages), then 2,048
//   D    only sessions inside the date range the question names (rule-based, `dateRange`), else A; then 2,048
// No answers here: the texts go to longmemeval_s2.py, which asks Gemma 4 E2B and the U2 judge.
//
//   node scripts/eval/scale/s2_retrieve.mjs --questions build/eval/memory/s2/questions.json --out <file>
//        --chat http://127.0.0.1:8080/v1 [--sizes 50,5000] [--embed-port 8299]
//   node scripts/eval/scale/s2_retrieve.mjs --parse <questions.json>     # date ranges only (no stores, no models)
import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { Memory, MemoryStore, MEMORY_OPTIONS, END, GAP_MS, MSG_MS, PKG, cachedEmbedder, liveEmbedder, estimateTokens, formatDate,
  layout, loadPlan, loadSessions, messagesOf, pipeline as p, shortId, uuidFor } from './common.mjs'

const { windowOrder } = await import(join(PKG, 'lib/unit-index.js'))
const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }
const DAY = 86_400_000
const WIDE = 100

// -- the date range a question names (rule-based; designed on the questions earlier experiments used) --------------
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
const NUMBER = { a: 1, an: 1, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11,
  twelve: 12, couple: 2, few: 3, several: 3 }
const count = w => /^\d+$/.test(w) ? Number(w) : NUMBER[w.toLowerCase()]

/** "2023/05/20 (Sat) 02:21" in local time (as Memory's formatDate writes it). */
export function parseDate(s) {
  const m = /^(\d{4})\/(\d{2})\/(\d{2}) \(\w{3}\) (\d{2}):(\d{2})$/.exec(s)
  if (!m) throw new Error(`bad date ${s}`)
  return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]).getTime()
}

/**
 * [lo, hi] in epoch ms (local) for a past period the question names relative to its date, or null. First rule wins:
 * N days/weeks/months ago (±2 days / ±7 days / ±15 days around), yesterday, last weekend, last week or last <weekday>
 * (14 days back), the last/past N days/weeks/months (back to then, plus a margin), this month, and month names after a
 * preposition (from 3 days before the most recent occurrence to the question date). Future periods
 * ("this weekend", "next ...") name no range.
 */
export function dateRange(question, questionDate) {
  const qd = parseDate(questionDate)
  const day0 = new Date(qd); day0.setHours(0, 0, 0, 0)
  const d0 = day0.getTime(), end = d0 + DAY - 1
  const back = (from, to) => [d0 - from * DAY, to === null ? end : d0 - to * DAY + DAY - 1]
  const num = '(\\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple of|few|several)'
  let m
  if ((m = new RegExp(`\\b${num} days? ago\\b`, 'i').exec(question))) {
    const n = count(m[1].split(' ')[0]); return { range: back(n + 2, Math.max(n - 2, 0)), rule: 'days ago' }
  }
  if ((m = new RegExp(`\\b${num} weeks? ago\\b`, 'i').exec(question))) {
    const n = count(m[1].split(' ')[0]); return { range: back(7 * n + 7, Math.max(7 * n - 7, 0)), rule: 'weeks ago' }
  }
  if ((m = new RegExp(`\\b${num} months? ago\\b`, 'i').exec(question))) {
    const n = count(m[1].split(' ')[0]); return { range: back(30 * n + 15, Math.max(30 * n - 15, 0)), rule: 'months ago' }
  }
  if (/\byesterday\b/i.test(question)) return { range: back(2, null), rule: 'yesterday' }
  if (/\blast weekend\b/i.test(question)) return { range: back(10, null), rule: 'last weekend' }
  if (/\blast (week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/i.test(question)) return { range: back(14, null), rule: 'last week' }
  if ((m = new RegExp(`\\b(?:last|past)\\s+(?:${num}\\s+)?(days?|weeks?|months?)\\b`, 'i').exec(question))) {
    const n = m[1] ? count(m[1].split(' ')[0]) : 1
    const unit = m[2].toLowerCase().replace(/s$/, '')
    const days = unit === 'day' ? n + 2 : unit === 'week' ? 7 * n + 3 : 31 * n + 5
    return { range: back(days, null), rule: `past ${unit}s` }
  }
  if (/\bthis month\b/i.test(question)) {
    const start = new Date(day0); start.setDate(1)
    return { range: [start.getTime() - 3 * DAY, end], rule: 'this month' }
  }
  const mentions = [...question.matchAll(new RegExp(`\\b(in|of|on|from|to|during|since|between|and|through|until|early|late|mid)\\s+(?:the\\s+month\\s+of\\s+)?(${MONTHS.join('|')})\\b`, 'g'))]
  if (mentions.length) {
    const qdate = new Date(qd)
    let lo = Infinity, hi
    for (const [, , name] of mentions) {
      const month = MONTHS.indexOf(name)
      const year = month <= qdate.getMonth() ? qdate.getFullYear() : qdate.getFullYear() - 1
      const start = new Date(year, month, 1).getTime()
      lo = Math.min(lo, start - 3 * DAY)
    }
    // People talk about a month's events during it or later (dev: sessions on the question date about "December").
    hi = end
    return { range: [lo, hi], rule: 'month names' }
  }
  return null
}

if (process.argv.includes('--parse')) {
  const qs = JSON.parse(readFileSync(arg('parse'), 'utf8'))
  for (const q of qs) {
    const r = dateRange(q.question, q.question_date)
    const inside = r && q.evidence_dates ? q.evidence_dates.filter(d => parseDate(d) >= r.range[0] && parseDate(d) <= r.range[1]).length : null
    console.log(`${r ? `${formatDate(r.range[0]).slice(0, 10)}..${formatDate(r.range[1]).slice(0, 10)} ${r.rule}` : '-'.padEnd(10)}`
      + `${inside === null ? '' : ` [answer sessions inside ${inside}/${q.evidence_dates.length}]`} | ${q.question_date.slice(0, 10)} | ${q.question}`)
  }
  process.exit(0)
}

// -- stores and recall ------------------------------------------------------------------------------------------
const questions = JSON.parse(readFileSync(arg('questions'), 'utf8'))
const outPath = arg('out')
const sizes = arg('sizes', '50,5000').split(',').map(Number)
const chatUrl = arg('chat')
const plan = loadPlan()
const sessions = await loadSessions(new Set(plan.fillers.slice(0, Math.max(...sizes))))
const cache = cachedEmbedder()
const live = liveEmbedder(Number(arg('embed-port', '8299')))
const model = cache.model()
const stats = { cachedTexts: 0, liveTexts: 0, refusedWindows: 0, refusedAnswerWindows: 0 }
/** Cached vectors where the scale test has them (fillers); the live egemma-server (same model) for the rest. */
const embedder = {
  model: () => model,
  async embed(input, task, signal) {
    const out = []
    for (const text of input) {
      if (cache.has(text, task)) { stats.cachedTexts++; out.push((await cache.embed([text], task))[0]) }
      else { stats.liveTexts++; out.push((await live.embed([text], task, signal))[0]) }
    }
    return out
  },
}

const { OpenAIChat, LLAMA_CPP_EXTRA } = await import(join(PKG, 'lib/llm.js'))
const inner = new OpenAIChat({ baseUrl: chatUrl, model: '', apiKey: '', timeoutMs: 600_000, extraBody: LLAMA_CPP_EXTRA })
const calls = []
/** The chat model, recording each rerank reply and its wall time. */
const chat = new Proxy(inner, {
  get(target, key) {
    if (key === 'complete') {
      return async (messages, options) => {
        const t0 = performance.now()
        const reply = await target.complete(messages, options)
        calls.push({ ms: performance.now() - t0, reply })
        return reply
      }
    }
    const v = Reflect.get(target, key, target)
    return typeof v === 'function' ? v.bind(target) : v
  },
})
// Generous timeouts: the experiment measures the methods, not the shipped time limits (rerank wall time is recorded).
const OPTIONS = { ...MEMORY_OPTIONS, rerankTimeoutMs: 600_000, queryTimeoutMs: 60_000 }

async function addWindows(store, adder, session, answers) {
  const turns = store.sessionTurns(session).map(t => adder.asTurn(t))
  for (const u of p.windowUnitsP2(turns)) {
    let vector
    try {
      [vector] = await embedder.embed([u.text], 'document')
    } catch {
      stats.refusedWindows++
      if (answers && u.turns.some(id => answers.has(id))) stats.refusedAnswerWindows++
      continue
    }
    store.addUnit({ ...u, vector, model })
  }
}

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
  return p.selectMemory(final, lookup, [], budget, estimateTokens, ts => p.renderDated(ts, notes))
}

/** The shipped rerank (Memory.retrieve with rerank) generalised to the first `n` pool windows. */
async function rerankWide(memory, units, order, question, n) {
  const turns = memory.turnIndex()
  const ids = new Set(order.slice(0, n).flatMap(i => units[i].turns))
  const view = [...ids].flatMap(id => turns.get(id) ?? []).sort((a, b) => a.seq - b.seq).map(t => memory.asTurn(t, OPTIONS.rerankMessageTokens))
  const index = new Map(view.map((t, k) => [t.id, k]))
  const lines = order.slice(0, n).map((i, k) => {
    const own = [...units[i].turns].sort((a, b) => index.get(a) - index.get(b))
    return `[${k + 1}] ${own.map(id => view[index.get(id)].text).join('\n')}`
  })
  const messages = [{ role: 'system', content: p.SYSTEM }, { role: 'user', content: p.PROMPTS.rerank.user(question, lines.join('\n\n')) }]
  const before = calls.length
  const reply = await chat.complete(messages, { maxTokens: p.PROMPTS.rerank.maxTokens, schema: p.PROMPTS.rerank.schema, name: 'rerank' })
  const picks = p.parseRerank(reply, Math.min(order.length, n))
  const candidates = order.slice(0, n)
  const chosen = [...new Set((picks ?? []).map(j => candidates[j]).filter(i => i !== undefined))]
  const picked = new Set(chosen)
  return { order: [...chosen, ...candidates.filter(i => !picked.has(i)), ...order.slice(n)], parsed: picks !== null,
    picks: picks?.length ?? 0, ms: calls[before].ms }
}

const results = { model, sizes, arms: ['A', 'B', 'C30', 'C100', 'D'], rows: [], builds: [], stats }
for (const size of sizes) {
  const t0 = performance.now()
  const store = new MemoryStore(':memory:')
  const fillerMsgs = messagesOf(layout(plan, size), sessions)
  const adder = new Memory(store, embedder, null, OPTIONS)
  for (const m of fillerMsgs) adder.record(m)
  for (const s of new Set(fillerMsgs.map(m => m.session))) await addWindows(store, adder, s, null)
  const fillerTurns = store.turns().map(t => ({ id: t.id, at: t.at }))
  const fillerSessions = new Set(fillerMsgs.map(m => m.session))
  const redate = store.db.prepare('UPDATE turns SET at = ?, date = ? WHERE id = ?')
  results.builds.push({ size, units: store.units().length, buildS: (performance.now() - t0) / 1000 })
  const memory = new Memory(store, embedder, chat, OPTIONS)
  for (const q of questions) {
    const qd = parseDate(q.question_date)
    store.db.exec('SAVEPOINT plant')
    // Fillers end 96 minutes before the question; the answer sessions keep their own dates.
    const delta = qd - GAP_MS - END
    for (const t of fillerTurns) redate.run(t.at + delta, formatDate(t.at + delta), t.id)
    const answers = new Set()
    const evidenceMsgs = []
    for (const ev of q.evidence) {
      const session = uuidFor(ev.id), start = parseDate(ev.date)
      ev.turns.forEach((t, i) => {
        const id = shortId(session, i + 1)
        if (t.has_answer) answers.add(id)
        evidenceMsgs.push({ id, session, role: t.role, text: t.content, at: start + i * MSG_MS })
      })
    }
    for (const m of evidenceMsgs) adder.record(m)
    for (const ev of q.evidence) await addWindows(store, adder, uuidFor(ev.id), answers)
    memory.unitCache = memory.turnCache = memory.turnById = memory.ranked = null
    const answerBySession = q.evidence.map(ev => ev.turns.flatMap((t, i) => t.has_answer ? [shortId(uuidFor(ev.id), i + 1)] : []))
      .filter(list => list.length)
    const score = ids => {
      const got = new Set(ids)
      return { hit: ids.some(id => answers.has(id)), allSessions: answerBySession.every(list => list.some(id => got.has(id))) }
    }
    const row = { size, qid: q.qid, type: q.type }

    const a = await memory.retrieve(q.question, { rerank: false, budgetTokens: OPTIONS.budgetTokens })
    row.A = { text: a.text, ids: a.ids, ...score(a.ids) }
    const b = await memory.retrieve(q.question, { rerank: false, budgetTokens: 8192 })
    row.B = { text: b.text, ids: b.ids, ...score(b.ids) }
    if (size === Math.max(...sizes)) {
      const before = calls.length
      const c = await memory.retrieve(q.question, { rerank: true, budgetTokens: OPTIONS.budgetTokens })
      const call = calls[before]
      row.C30 = { text: c.text, ids: c.ids, ...score(c.ids), reranked: c.reranked, ms: call?.ms ?? null,
        parsed: call ? p.parseRerank(call.reply, Math.min(c.candidates, p.RERANK_CANDIDATES)) !== null : false }

      const [qv] = await embedder.embed([q.question], 'query')
      const turns = memory.turnIndex()
      const units = windowOrder(memory.allUnits().filter(u => u.set === 'p2' && u.model === model && u.vector.length === qv.length),
        id => turns.get(id)?.seq).map(([u]) => u)
      const all = units.map((_, i) => i)
      const dense = units.map(u => p.dot(u.vector, qv))
      const pool = p.distinct(p.top(all, dense).map(([i]) => i), units)
      const check = select(memory, units, p.top(all, dense).map(([i]) => i), OPTIONS.budgetTokens)
      row.selectionMatchesRetrieve = JSON.stringify(check.ids) === JSON.stringify(a.ids)
      const wide = await rerankWide(memory, units, pool, q.question, WIDE)
      const c100 = select(memory, units, wide.order, OPTIONS.budgetTokens)
      row.C100 = { text: c100.text, ids: c100.ids, ...score(c100.ids), parsed: wide.parsed, picks: wide.picks, ms: wide.ms,
        candidates: Math.min(pool.length, WIDE) }

      const named = dateRange(q.question, q.question_date)
      if (named) {
        const [lo, hi] = named.range
        const inside = all.filter(i => { const t = turns.get(units[i].turns[0]); return t && t.at >= lo && t.at <= hi })
        const d = inside.length ? select(memory, units, p.top(inside, inside.map(i => dense[i])).map(([i]) => i), OPTIONS.budgetTokens) : a
        row.D = { text: d.text, ids: d.ids, ...score(d.ids), rule: named.rule, range: named.range.map(formatDate), candidates: inside.length,
          answerInRange: q.evidence.some(ev => ev.turns.some(t => t.has_answer) && parseDate(ev.date) >= lo - DAY && parseDate(ev.date) <= hi) }
      } else {
        row.D = { ...row.A, rule: null }
      }
    }
    results.rows.push(row)
    store.db.exec('ROLLBACK TO plant')
    store.db.exec('RELEASE plant')
    memory.unitCache = memory.turnCache = memory.turnById = memory.ranked = null
    console.error(`${size} ${results.rows.filter(r => r.size === size).length}/${questions.length}`)
  }
  adder.close()
  memory.close()
  store.close()
}
writeFileSync(outPath, JSON.stringify(results))
console.error(`wrote ${outPath}`)
