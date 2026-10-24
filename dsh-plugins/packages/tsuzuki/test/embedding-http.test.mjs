// The sidecar's HTTP embedder against a fake OpenAI-compatible endpoint that refuses long inputs the way
// egemma-server does (HTTP 400, code context_length_exceeded). CPU only.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { HttpEmbedder, InputTooLongError } from '../lib/embedding-http.js'
import { MemoryStore } from '../lib/store.js'
import { formatDate } from '../lib/memory.js'
import { start, defaults } from '../lib/server.js'

const PREFIXES = { queryPrefix: 'task: search result | query: ', documentPrefix: 'title: none | text: ' }

function vector(text) {
  const v = new Array(64).fill(0)
  const body = text.replace(/^task: search result \| query: /, '').replace(/^title: none \| text: /, '')
  for (const w of body.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (['the', 'i', 'my', 'is', 'user', 'assistant', 'word'].includes(w)) continue
    let h = 2166136261
    for (const c of w) h = Math.imul(h ^ c.codePointAt(0), 16777619)
    v[(h >>> 0) % 64] += 1
  }
  v[63] += 0.01  // never all zero
  return v
}

/**
 * `refuse(input)` returns an error body (and status) for an input the endpoint does not take, or null.
 * Default: egemma-server's answer for an input over `maxChars` characters.
 */
async function endpoint({ maxChars = 2000, refuse = null } = {}) {
  const seen = []
  const egemma = (t, i) => t.length > maxChars ? [400, { error: { message: `Input ${i} has ${Math.ceil(t.length / 4)} tokens including `
    + 'the task prompt and BOS/EOS; the maximum is 2048. Shorten or split the input.', type: 'invalid_request_error', param: 'input',
    code: 'context_length_exceeded' } }] : null
  const server = createServer((req, res) => {
    let raw = ''
    req.on('data', c => { raw += c })
    req.on('end', () => {
      const b = JSON.parse(raw)
      seen.push(b.input)
      for (const [i, t] of b.input.entries()) {
        const refused = (refuse ?? egemma)(t, i)
        if (refused) {
          res.writeHead(refused[0], { 'content-type': 'application/json' })
          return res.end(typeof refused[1] === 'string' ? refused[1] : JSON.stringify(refused[1]))
        }
      }
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ data: b.input.map((t, index) => ({ index, embedding: vector(t) })) }))
    })
  })
  await new Promise(r => server.listen(0, '127.0.0.1', r))
  return { url: `http://127.0.0.1:${server.address().port}/v1`, seen, close: () => new Promise(r => server.close(r)) }
}

const embedderFor = (url, batch = 32) => new HttpEmbedder({ baseUrl: url, model: 'fake', apiKey: '', ...PREFIXES, timeoutMs: 5000, batch })

function normalized(values) {
  const n = Math.hypot(...values) || 1
  return Float32Array.from(values, v => v / n)
}

test('an input longer than the model accepts is halved until accepted; the rest of its batch is embedded as usual', async () => {
  const ep = await endpoint({ maxChars: 2000 })
  try {
    const long = 'kyoto '.repeat(500)  // 3000 characters with the prefix: one halving fits
    const input = ['osaka trip', long, 'guitar lesson', 'tama the cat']
    const vectors = await embedderFor(ep.url).embed(input, 'document')
    assert.equal(vectors.length, 4)
    const half = Array.from(long).slice(0, long.length / 2).join('')
    assert.deepEqual(vectors[1], normalized(vector(PREFIXES.documentPrefix + half)), 'the leading half of the long input')
    for (const k of [0, 2, 3]) assert.deepEqual(vectors[k], normalized(vector(PREFIXES.documentPrefix + input[k])))
    // The whole batch once, then one input at a time (the long one twice: refused, then its half).
    assert.deepEqual(ep.seen.map(i => i.length), [4, 1, 1, 1, 1, 1])
    assert.ok(ep.seen.every(i => i.every(t => t.startsWith(PREFIXES.documentPrefix))), 'the prefix is never cut')
  } finally {
    await ep.close()
  }
})

test('halving is bounded; other errors are not retried input by input', async () => {
  const body = t => t.slice(PREFIXES.documentPrefix.length)
  const tooLong = [400, { error: { message: 'Input 0 has 9000 tokens', code: 'context_length_exceeded' } }]
  const short = await endpoint({ refuse: t => body(t).length > 5 ? tooLong : null })
  try {
    const [v] = await embedderFor(short.url).embed(['abcdefghijklmnopqrstuvwxyz'], 'document')
    assert.deepEqual(v, normalized(vector(PREFIXES.documentPrefix + 'abc')))
    assert.deepEqual(short.seen.map(i => body(i[0])), ['abcdefghijklmnopqrstuvwxyz', 'abcdefghijklmnopqrstuvwxyz', 'abcdefghijklm', 'abcdef', 'abc'])
  } finally {
    await short.close()
  }
  const endless = await endpoint({ refuse: () => tooLong })
  try {
    await assert.rejects(embedderFor(endless.url).embed(['abcdefghijklmnopqrstuvwxyz'], 'document'), InputTooLongError)
    assert.equal(endless.seen.length, 1 + 5, 'the batch, then the input and 4 halvings')
  } finally {
    await endless.close()
  }
  const broken = await endpoint({ refuse: () => [500, { error: { message: 'engine crashed', code: 'internal' } }] })
  try {
    await assert.rejects(embedderFor(broken.url).embed(['a', 'b'], 'document'), err => !(err instanceof InputTooLongError) && /HTTP 500/.test(err.message))
    assert.equal(broken.seen.length, 1)
  } finally {
    await broken.close()
  }
})

test('too-long answers of other OpenAI-compatible servers are recognized', async () => {
  const answers = [
    [400, { error: { message: "This model's maximum context length is 8192 tokens", code: 'context_length_exceeded' } }],  // OpenAI
    [400, { error: { code: 400, message: 'the request exceeds the available context size', type: 'exceed_context_size_error' } }],  // llama.cpp
    [500, { error: { code: 500, message: 'input is too large to process. increase the physical batch size', type: 'server_error' } }],  // older llama.cpp
    [400, { error: { message: 'the input length exceeds the context length', type: 'api_error' } }],  // Ollama
  ]
  for (const answer of answers) {
    const ep = await endpoint({ refuse: t => t.length > 40 ? answer : null })
    try {
      const [v] = await embedderFor(ep.url).embed(['x'.repeat(60)], 'document')
      assert.equal(v.length, 64, JSON.stringify(answer))
    } finally {
      await ep.close()
    }
  }
})

test('sidecar: backfill indexes every window of a store that has one longer than the model accepts', async () => {
  const ep = await endpoint({ maxChars: 2000 })
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-long-'))
  // Messages stored while the embedding server was down: none has its window yet.
  const store = new MemoryStore(join(dir, 'memory.db'))
  const at = Date.UTC(2026, 9, 1, 3, 0)
  const said = ['I collect stamps.', `My notes: ${'lorem '.repeat(600)}`, 'My cat is called Tama.', ...Array.from({ length: 40 }, (_, k) => `filler message ${k}`),
    'I moved to Kyoto last week.']
  said.forEach((text, k) => store.addTurn({ id: `m${k}`, session: k < 20 ? 'one' : 'two', role: 'user', text: `user: ${text}`, at: at + k, date: formatDate(at + k) }))
  store.close()
  const config = { ...defaults(dir), port: 0, embedding: { ...defaults(dir).embedding, baseUrl: ep.url, autoPull: false } }
  const s = await start(config)
  try {
    // Recall waits for the startup backfill.
    const recall = await fetch(`http://127.0.0.1:${s.port}/v1/memory/search`, { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ query: 'Where did I move? Kyoto', rerank: false }) }).then(r => r.json())
    assert.match(recall.text, /Kyoto/)
    const health = await fetch(`http://127.0.0.1:${s.port}/health`).then(r => r.json())
    assert.equal(health.stats.units, said.length, 'every window is indexed, the long ones from their leading part')
  } finally {
    await s.close()
    await ep.close()
  }
})
