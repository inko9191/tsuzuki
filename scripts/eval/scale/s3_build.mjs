// S3 store build (docs/MEMORY_SCALE_S3_PREREG.md): the first --size filler sessions of the scale test in an on-disk
// store, with the dsh plugin's own fact pipeline run by the chat model at --chat (Gemma 4 E2B): segments closed as on
// shutdown (closeAll), facts extracted per segment (Memory.extract: prompt, parsing, fact units, embedding), then the
// update check per segment (Memory.link) against all earlier-dated facts. Extraction runs on --workers segments at a
// time and the update checks after all extraction; each check reads only the facts of earlier-dated segments, so the
// links are those a one-at-a-time ingest in date order would make. Resumable (pending segments, links-done file).
//
//   node scripts/eval/scale/s3_build.mjs --size 5000 --db build/eval/memory/s3/fillers-5000.sqlite --chat http://127.0.0.1:8080/v1
import { appendFileSync, existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { Memory, MemoryStore, MEMORY_OPTIONS, PKG, cachedEmbedder, layout, loadPlan, loadSessions, messagesOf, pipeline as p,
  HttpEmbedder, EMBEDDINGGEMMA_PREFIXES } from './common.mjs'

const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }
const size = Number(arg('size'))
const dbPath = arg('db')
const workers = Number(arg('workers', '4'))
const port = Number(arg('embed-port', '8299'))
const linksDone = `${dbPath}.links-done`
const log = (...a) => console.error(new Date().toISOString().slice(11, 19), ...a)

const cache = cachedEmbedder()
const model = cache.model()
const live = new HttpEmbedder({ baseUrl: `http://127.0.0.1:${port}/v1`, model: 'embeddinggemma-300m', apiKey: '', ...EMBEDDINGGEMMA_PREFIXES, timeoutMs: 120000, batch: 32 })
/** Cached vectors for the scale test's windows; the live egemma-server (same model, checked equal) for the rest. */
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
const { OpenAIChat, LLAMA_CPP_EXTRA } = await import(join(PKG, 'lib/llm.js'))
const chat = new OpenAIChat({ baseUrl: arg('chat'), model: '', apiKey: '', timeoutMs: 600_000, extraBody: LLAMA_CPP_EXTRA })
const OPTIONS = { ...MEMORY_OPTIONS, queryTimeoutMs: 60_000, log: (level, message) => { if (level === 'warn') log('warn', message) } }

const fresh = !existsSync(dbPath)
const store = new MemoryStore(dbPath)
if (fresh || store.getMeta('s3.recorded') !== String(size)) {
  if (!fresh) throw new Error(`${dbPath} exists but was not fully recorded; delete it to rebuild`)
  const plan = loadPlan()
  const sessions = await loadSessions(new Set(plan.fillers.slice(0, size)))
  const msgs = messagesOf(layout(plan, size), sessions)
  const adder = new Memory(store, embedder, null, OPTIONS)
  for (const m of msgs) adder.record(m)
  for (const s of new Set(msgs.map(m => m.session))) {
    const turns = store.sessionTurns(s).map(t => adder.asTurn(t))
    for (const u of p.windowUnitsP2(turns)) {
      let vector
      try { [vector] = await embedder.embed([u.text], 'document') } catch { continue }  // over 2,048 tokens: never indexed
      store.addUnit({ ...u, vector, model })
    }
  }
  adder.closeAll()
  adder.close()
  store.setMeta('s3.recorded', String(size))
  log('recorded', msgs.length, 'messages', store.pendingSegments(3).length, 'segments')
}

// -- extraction: `workers` segments at a time, until nothing is pending (each segment gets up to maxAttempts) ------
const extractors = Array.from({ length: workers }, () => new Memory(store, embedder, chat, { ...OPTIONS, updateLinks: false }))
for (let round = 1; ; round++) {
  const pending = store.pendingSegments(3)
  if (!pending.length) break
  log(`extraction round ${round}: ${pending.length} segments`)
  let done = 0
  const t0 = Date.now()
  await Promise.all(extractors.map(async (m, w) => {
    for (let k = w; k < pending.length; k += workers) {
      for (let tries = 0; !(await m.extract(pending[k])); tries++) {  // false: chat unavailable, no attempt spent
        if (tries > 60) throw new Error('chat model unavailable')
        await new Promise(r => setTimeout(r, 2000))
      }
      if (++done % 200 === 0) log(`extracted ${done}/${pending.length} (${((Date.now() - t0) / done).toFixed(0)} ms/segment)`)
    }
  }))
}
for (const m of extractors) m.close()
const facts = store.facts()
log('facts', [...facts.values()].reduce((n, f) => n + f.length, 0), 'in', facts.size, 'segments')

// -- update checks: every segment with facts, against all earlier-dated facts --------------------------------------
const finished = new Set(existsSync(linksDone) ? readFileSync(linksDone, 'utf8').split('\n').filter(Boolean) : [])
const todo = [...facts.keys()].filter(id => facts.get(id).length && !finished.has(id))
log('update checks', todo.length, 'to do,', finished.size, 'done')
const linkers = Array.from({ length: workers }, () => new Memory(store, embedder, chat, OPTIONS))
let checked = 0
const t1 = Date.now()
await Promise.all(linkers.map(async (m, w) => {
  for (let k = w; k < todo.length; k += workers) {
    await m.link(todo[k])
    appendFileSync(linksDone, todo[k] + '\n')
    if (++checked % 200 === 0) log(`checked ${checked}/${todo.length} (${((Date.now() - t1) / checked).toFixed(0)} ms/segment)`)
  }
}))
for (const m of linkers) m.close()
const stats = store.stats()
store.setMeta('s3.complete', new Date().toISOString())
store.db.exec('PRAGMA wal_checkpoint(TRUNCATE)')
store.close()
log('complete', JSON.stringify(stats))
