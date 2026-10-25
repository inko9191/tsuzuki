// Parity of the llama.cpp embedding path (EmbeddingGemma 300m Q8_0 GGUF on llama-server --embedding) with the
// evaluated Metal engine (fp32): vector cosine on scale-test windows and queries, and the scale test's recall measure
// (answer message in the 2,048-token recall; 60 questions, middle age) at --size sessions with each embedder.
//
//   node scripts/eval/scale/llamacpp_parity.mjs --url http://127.0.0.1:8311/v1 [--size 500]
import { Memory, MemoryStore, MEMORY_OPTIONS, END, GAP_MS, cachedEmbedder, layout, loadPlan, loadSessions, messagesOf, pipeline as p,
  shortId, uuidFor, HttpEmbedder, EMBEDDINGGEMMA_PREFIXES } from './common.mjs'

const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }
const size = Number(arg('size', '500'))
const plan = loadPlan()
const sessions = await loadSessions(new Set([...plan.fillers.slice(0, size), ...plan.planted.flatMap(q => q.evidence)]))
const engine = cachedEmbedder()
const lc = new HttpEmbedder({ baseUrl: arg('url'), model: 'embeddinggemma', apiKey: '', ...EMBEDDINGGEMMA_PREFIXES, timeoutMs: 120000, batch: 32 })
const model = engine.model()

async function hits(embedder) {
  let hit = 0, n = 0, refused = 0
  const store = new MemoryStore(':memory:')
  const add = async msgs => {
    const adder = new Memory(store, embedder, null, MEMORY_OPTIONS)
    for (const m of msgs) adder.record(m)
    for (const s of new Set(msgs.map(m => m.session))) {
      const units = p.windowUnitsP2(store.sessionTurns(s).map(t => adder.asTurn(t)))
      let vectors
      try { vectors = await embedder.embed(units.map(u => u.text), 'document') } catch { vectors = [] ; for (const u of units) { try { vectors.push((await embedder.embed([u.text], 'document'))[0]) } catch { vectors.push(null); refused++ } } }
      units.forEach((u, k) => { if (vectors[k]) store.addUnit({ ...u, vector: vectors[k], model }) })
    }
    adder.close()
  }
  await add(messagesOf(layout(plan, size), sessions))
  const memory = new Memory(store, { model: () => model, embed: (i, t) => embedder.embed(i, t) }, null, MEMORY_OPTIONS)
  for (const q of plan.planted) {
    store.db.exec('SAVEPOINT q')
    const at = Math.round(0.5 * size), start = END - (size - 1 - at) * GAP_MS - GAP_MS / 2
    await add(q.evidence.flatMap((orig, j) => messagesOf([orig], sessions, start + j * (GAP_MS / 2) / q.evidence.length)))
    memory.unitCache = memory.turnCache = memory.turnById = memory.ranked = null
    const answers = new Set(q.evidence.flatMap(orig => q.answer_turns[orig].map(i => shortId(uuidFor(orig), i + 1))))
    const r = await memory.retrieve(q.question, { rerank: false })
    if (r.ids.some(id => answers.has(id))) hit++
    n++
    store.db.exec('ROLLBACK TO q'); store.db.exec('RELEASE q')
    memory.unitCache = memory.turnCache = memory.turnById = memory.ranked = null
  }
  memory.close()
  return { hit: hit / n, n, refused }
}
const t0 = Date.now()
const a = await hits(engine)
const b = await hits(lc)
console.log(JSON.stringify({ size, engine: a, llamacpp: b, seconds: Math.round((Date.now() - t0) / 1000) }))
