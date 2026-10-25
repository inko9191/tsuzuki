// Enumerates the exact window texts the package embeds for every session the scale test uses (the 5,000
// largest-store fillers, the planted evidence sessions, the held-out ingest sessions), so the GPU step embeds each
// unique text once. CPU only; runs the package's own code into in-memory stores with a recording embedder.
//
// Two window sets, because the live capture path and the backfill path build different windows
// (see results/eval/memory/scale-report.md):
//   shipped  Memory.capture(): p2 window of [previous, current] with the current turn's in-session `pos`
//   fixed    Memory.backfill() over the whole session: the evaluated p2 windows (previous + current message)
//
//   node scripts/eval/scale/collect_windows.mjs [--max-size 5000]
import { writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { Memory, MemoryStore, MEMORY_OPTIONS, OUT, layout, loadPlan, loadSessions, messagesOf, sha1, syntheticEmbedder } from './common.mjs'

const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }
const maxSize = Number(arg('max-size', '5000'))

const plan = loadPlan()
const sessions = await loadSessions()
const texts = new Map()      // sha1 -> text
const tags = new Map()       // sha1 -> Set of window sets
const base = syntheticEmbedder()
const recorder = tag => ({
  model: base.model,
  async embed(input, task) {
    if (task === 'document') for (const t of input) {
      const h = sha1(t)
      texts.set(h, t)
      if (!tags.has(h)) tags.set(h, new Set())
      tags.get(h).add(tag)
    }
    return base.embed(input, task)
  },
})
const groups = {
  fillers: layout(plan, maxSize),
  evidence: [...new Set(plan.planted.flatMap(p => p.evidence))],
  holdout: plan.holdout,
}
const counts = {}
for (const [name, order] of Object.entries(groups)) {
  const store = new MemoryStore(':memory:')
  const shipped = new Memory(store, recorder('shipped'), null, MEMORY_OPTIONS)
  let n = 0
  for (const m of messagesOf(order, sessions)) {
    await shipped.capture(m)
    n++
  }
  shipped.close()
  const stats = store.stats()
  let emptyShipped = 0
  for (const u of store.units()) if (!u.text) emptyShipped++
  store.db.exec('DELETE FROM units')
  const fixed = new Memory(store, recorder('fixed'), null, MEMORY_OPTIONS)
  const indexed = await fixed.backfill()
  fixed.close()
  counts[name] = { sessions: order.length, messages: n, stored: stats.turns, unitsShipped: stats.units, emptyShipped, unitsFixed: indexed }
  store.close()
}
const lines = [...texts].map(([h, text]) => JSON.stringify({ h, text, sets: [...tags.get(h)] }))
writeFileSync(join(OUT, 'windows.jsonl'), lines.join('\n') + '\n')
const queries = plan.queries.map(q => ({ h: sha1(q.text), text: q.text, qid: q.qid }))
writeFileSync(join(OUT, 'queries.jsonl'), queries.map(q => JSON.stringify(q)).join('\n') + '\n')
const chars = [...texts.values()].reduce((a, t) => a + t.length, 0)
const bySet = tag => [...tags.values()].filter(s => s.has(tag)).length
const summary = { maxSize, unique: texts.size, uniqueFixed: bySet('fixed'), uniqueShipped: bySet('shipped'), chars,
  meanChars: Math.round(chars / texts.size), counts, queries: queries.length }
writeFileSync(join(OUT, 'windows-summary.json'), JSON.stringify(summary, null, 1))
console.log(JSON.stringify(summary))
