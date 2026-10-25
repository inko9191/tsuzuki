// S3 amendment (docs/MEMORY_SCALE_S3_RUNLOG.md): re-runs the update checks (Memory.link) that s3_build.mjs lost to a
// dropped connection ("cannot reach ...": the server closed a kept-alive socket), with the request retried on such
// errors. Checks that failed on the model's own reply (e.g. "generation stopped: length") are not re-run: that is how
// the plugin behaves with this model. Each check reads only earlier-dated facts, so re-running one later is the same check.
//
//   node scripts/eval/scale/s3_relink.mjs --db build/eval/memory/s3/fillers-5000.sqlite --log build/eval/memory/s3/build-5000.log --chat <url>
import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { Memory, MemoryStore, MEMORY_OPTIONS, PKG, cachedEmbedder } from './common.mjs'

const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }
const workers = Number(arg('workers', '4'))
const log = (...a) => console.error(new Date().toISOString().slice(11, 19), ...a)
const lost = [...new Set(readFileSync(arg('log'), 'utf8').split('\n')
  .flatMap(l => { const m = /update check failed for (\S+): ChatUnavailableError/.exec(l); return m ? [m[1]] : [] }))]
log('update checks lost to the connection:', lost.length)

const { OpenAIChat, LLAMA_CPP_EXTRA } = await import(join(PKG, 'lib/llm.js'))
const inner = new OpenAIChat({ baseUrl: arg('chat'), model: '', apiKey: '', timeoutMs: 600_000, extraBody: LLAMA_CPP_EXTRA })
const failed = new Set()
/** Retries a request the connection dropped (up to 5 times); any other error goes to Memory.link as before. */
const chat = new Proxy(inner, {
  get(target, key) {
    if (key === 'complete') {
      return async (messages, options) => {
        for (let attempt = 1; ; attempt++) {
          try {
            return await target.complete(messages, options)
          } catch (error) {
            if (error?.name !== 'ChatUnavailableError' || attempt >= 5) throw error
            await new Promise(r => setTimeout(r, 1000 * attempt))
          }
        }
      }
    }
    const v = Reflect.get(target, key, target)
    return typeof v === 'function' ? v.bind(target) : v
  },
})
const embedder = cachedEmbedder()  // link() only reads stored fact vectors
const store = new MemoryStore(arg('db'))
const before = store.db.prepare('SELECT COUNT(*) AS n FROM links').get().n
const OPTIONS = { ...MEMORY_OPTIONS, log: (level, message) => {
  if (level !== 'warn') return
  log('warn', message)
  const m = /update check failed for (\S+): /.exec(message)
  if (m) failed.add(m[1])
} }
const linkers = Array.from({ length: workers }, () => new Memory(store, embedder, chat, OPTIONS))
let done = 0
await Promise.all(linkers.map(async (m, w) => {
  for (let k = w; k < lost.length; k += workers) {
    await m.link(lost[k])
    if (++done % 100 === 0) log(`re-checked ${done}/${lost.length}`)
  }
}))
for (const m of linkers) m.close()
const after = store.db.prepare('SELECT COUNT(*) AS n FROM links').get().n
store.setMeta('s3.relinked', JSON.stringify({ lost: lost.length, failedAgain: failed.size, linksAdded: after - before }))
store.db.exec('PRAGMA wal_checkpoint(TRUNCATE)')
store.close()
writeFileSync(`${arg('db')}.relink.json`, JSON.stringify({ lost: lost.length, failedAgain: [...failed], linksBefore: before, linksAfter: after }, null, 1))
log('done', JSON.stringify({ lost: lost.length, failedAgain: failed.size, linksAdded: after - before }))
