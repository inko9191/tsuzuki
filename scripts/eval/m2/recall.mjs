// M2 recall (docs/MEMORY_MISFIRE_M2_PREREG.md), built like M1's (scripts/eval/m1/recall.mjs): for every user, a fresh
// memory built by the product code (v0.2.0, message windows only) from the user's whole history, then for each query
// the best cosine between the message and any stored window, and the recall the proxy adds:
//   P  memory.retrieve(message, 2048 tokens, no rerank) in the product's header;
//   W  (with --tau) the same, from the windows whose cosine is at least tau only: turns found only through windows
//      below tau are left out (they rank after every window at or above tau, so the order is unchanged); none -> ''.
// dev users have their question only; test users their probes and their question. Also the cosines of the 30 best
// windows, for exploratory reading.
//
//   node scripts/eval/m2/recall.mjs --inputs build/eval/memory/m2/inputs.json --split dev --out build/eval/memory/m2/recall-dev.json --embed 8591
//   node scripts/eval/m2/recall.mjs --inputs ... --split test --tau 0.31 --out build/eval/memory/m2/recall-test.json --embed 8591
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../../..')
const PKG = join(ROOT, 'dsh-plugins/packages/memory-egemma/lib')
const { MemoryStore } = await import(join(PKG, 'store.js'))
const { Memory } = await import(join(PKG, 'memory.js'))
const { recallText } = await import(join(PKG, 'format.js'))
const { HttpEmbedder, EMBEDDINGGEMMA_PREFIXES } = await import(join(PKG, 'embedding-http.js'))

const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }

/** "2023/05/20 (Sat) 02:21" as local epoch milliseconds. */
function parseDate(text) {
  const m = /(\d{4})\/(\d{2})\/(\d{2})\D+(\d{2}):(\d{2})/.exec(text)
  if (!m) throw new Error(`unparsed date: ${text}`)
  return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]).getTime()
}

const cosine = (a, b) => {
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
  return dot / (Math.sqrt(na * nb) || 1)
}

const split = arg('split')
const tau = arg('tau') === undefined ? null : Number(arg('tau'))
if (split === 'test' && tau === null) throw new Error('--tau is needed for the test split (the W recall)')
const outPath = arg('out')
const inputs = JSON.parse(readFileSync(arg('inputs'), 'utf8'))
const embedder = new HttpEmbedder({ baseUrl: `http://127.0.0.1:${arg('embed')}/v1`, model: 'm2-embeddinggemma', apiKey: '',
  ...EMBEDDINGGEMMA_PREFIXES, timeoutMs: 120000, batch: 32 })
// Resumable: users already in the output are skipped.
const out = existsSync(outPath) ? JSON.parse(readFileSync(outPath, 'utf8')) : {}
const users = inputs.users.filter(u => u.split === split)
const started = Date.now()
for (const [n, user] of users.entries()) {
  if (`${user.user}-q` in out) continue
  const dir = mkdtempSync(join(tmpdir(), 'm2-'))
  const memory = new Memory(new MemoryStore(join(dir, 'memory.db')), embedder, null,
    { budgetTokens: 2048, redactSecrets: true, log: () => {} })
  let last = 0
  for (const s of user.sessions) {
    const at = parseDate(s.date)
    last = Math.max(last, at)
    s.turns.forEach((t, i) => memory.record({ id: `${s.id}-${i}`, session: s.id, role: t.role, text: t.text, at: at + i * 60_000 }))
  }
  await memory.backfill()
  const windows = memory.store.units().filter(u => u.set === 'p2')
  const queries = [...user.probes.map(p => ({ id: p.id, kind: 'probe', text: p.text, now: last + 86_400_000 })),
    { id: `${user.user}-q`, kind: 'question', text: user.question, now: parseDate(user.question_date) }]
  for (const q of queries) {
    const found = await memory.retrieve(q.text, { rerank: false, budgetTokens: 2048 })
    const [vector] = await embedder.embed([q.text], 'query')
    const scored = windows.map(u => [cosine(vector, u.vector), u]).sort((a, b) => b[0] - a[0])
    const row = { user: user.user, lang: user.lang, split: user.split, kind: q.kind, message: q.text, top: scored[0]?.[0] ?? -1,
      best: scored.slice(0, 30).map(([c]) => Math.round(c * 10000) / 10000),
      P: found.text ? recallText(found.text, q.now) : '', units: found.ids.length }
    if (tau !== null) {
      const allowed = new Set(scored.filter(([c]) => c >= tau).flatMap(([, u]) => u.turns))
      const kept = allowed.size
        ? await memory.retrieve(q.text, { rerank: false, budgetTokens: 2048, inContext: t => !allowed.has(t.id) })
        : { text: '', ids: [] }
      row.W = kept.text ? recallText(kept.text, q.now) : ''
      row.unitsW = kept.ids.length
    }
    out[q.id] = row
  }
  memory.close()
  memory.store.close()
  rmSync(dir, { recursive: true, force: true })
  writeFileSync(outPath, JSON.stringify(out))
  console.error(`${n + 1}/${users.length} ${user.user}: ${windows.length} windows (${Math.round((Date.now() - started) / 1000)} s)`)
}
writeFileSync(outPath, JSON.stringify(out))
