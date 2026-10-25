// Sidecar HTTP API against a fake OpenAI-compatible embedding endpoint (CPU only).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createServer, request } from 'node:http'
import { createServer as createTcpServer } from 'node:net'
import { spawn, spawnSync } from 'node:child_process'
import { chmodSync, copyFileSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { isAbsolute, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { start, defaults, loadConfig, EngineServer, defaultConfigPath, isMainScript } from '../lib/server.js'

function vector(text) {
  const v = new Array(64).fill(0)
  const body = text.replace(/^task: search result \| query: /, '').replace(/^title: none \| text: /, '')
  for (const w of body.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (['the', 'i', 'my', 'is', 'to', 'do', 'user', 'assistant', 'where', 'now'].includes(w)) continue
    let h = 2166136261
    for (const c of w) h = Math.imul(h ^ c.codePointAt(0), 16777619)
    v[(h >>> 0) % 64] += 1
  }
  return v
}

async function fakeEmbeddings() {
  const seen = []
  const server = createServer((req, res) => {
    let raw = ''
    req.on('data', c => { raw += c })
    req.on('end', () => {
      const b = JSON.parse(raw)
      seen.push(b)
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ data: b.input.map((t, index) => ({ index, embedding: vector(t) })) }))
    })
  })
  await new Promise(r => server.listen(0, '127.0.0.1', r))
  return { url: `http://127.0.0.1:${server.address().port}/v1`, seen, close: () => new Promise(r => server.close(r)) }
}

const post = (port, path, body) => fetch(`http://127.0.0.1:${port}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify(body) }).then(r => r.json())

test('config file is created with defaults and merged one level deep', () => {
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
  const path = join(dir, 'config.json')
  const first = loadConfig(path)
  assert.equal(first.recall.budgetTokens, 2048)
  assert.equal(first.embedding.acceptModelLicense, false)
  assert.equal(JSON.parse(readFileSync(path, 'utf8')).port, 8297)
})

test('without --config the config is under os.homedir(): HOME on macOS and Linux, never the working directory', () => {
  // Paths only: nothing is read or written there.
  assert.equal(defaultConfigPath(), join(homedir(), '.tsuzuki', 'config.json'))
  const home = process.env.HOME
  try {
    if (process.platform !== 'win32') {
      process.env.HOME = join(tmpdir(), 'some-home')
      assert.equal(defaultConfigPath(), join(tmpdir(), 'some-home', '.tsuzuki', 'config.json'), 'HOME decides, as before')
    }
    delete process.env.HOME  // as on Windows, which normally has no HOME
    assert.ok(isAbsolute(defaultConfigPath()), defaultConfigPath())
    assert.equal(defaultConfigPath(), join(homedir(), '.tsuzuki', 'config.json'))
  } finally {
    if (home === undefined) delete process.env.HOME
    else process.env.HOME = home
  }
})

test('messages, recall excluding the current session, search, flush, health', async () => {
  const emb = await fakeEmbeddings()
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
  const config = { ...defaults(dir), port: 0, embedding: { ...defaults(dir).embedding, baseUrl: emb.url, autoPull: false } }
  const s = await start(config)
  try {
    assert.deepEqual(await post(s.port, '/v1/memory/messages', { session: 'old', messages: [
      { role: 'user', text: 'I moved to Kyoto and my api key is sk-proj-abcdefghijklmnopqrstuvwxyz123456', at: Date.UTC(2026, 9, 1) },
      { role: 'assistant', text: 'Kyoto is lovely.', at: Date.UTC(2026, 9, 1) + 1000 }] }), { stored: 2 })
    await post(s.port, '/v1/memory/messages', { session: 'now', messages: [{ role: 'user', text: 'Where do I live now? Kyoto?' }] })
    const recall = await post(s.port, '/v1/memory/recall', { query: 'Where do I live? Kyoto', session: 'now' })
    assert.match(recall.text, /^<memory>\nCurrent date: /)
    assert.match(recall.text, /user: I moved to Kyoto/)
    assert.ok(!recall.text.includes('Where do I live now'), 'current session excluded')
    assert.ok(!recall.text.includes('sk-proj'), 'secret redacted')
    const search = await post(s.port, '/v1/memory/search', { query: 'Kyoto' })
    assert.ok(search.text.includes('Where do I live now'), 'search includes the current session')
    assert.deepEqual(await post(s.port, '/v1/memory/flush', { session: 'old' }), { ok: true })
    const health = await fetch(`http://127.0.0.1:${s.port}/health`).then(r => r.json())
    assert.equal(health.status, 'ok')
    assert.equal(health.stats.turns, 3)
    assert.equal(health.stats.segmentsPending, 1)
    assert.ok(emb.seen.every(b => b.input.every(t => t.startsWith('task: search result | query: ') || t.startsWith('title: none | text: '))))
    assert.deepEqual(await post(s.port, '/v1/memory/recall', { query: '  ' }), { text: '', ids: [], reranked: false })
  } finally {
    await s.close()
    await emb.close()
  }
})

// Raw request with full control over Origin/Host/Content-Type (fetch would set some of them itself).
function raw(port, { method = 'POST', path = '/v1/memory/search', headers = {}, body = '{"query":"Kyoto"}' } = {}) {
  return new Promise((resolve, reject) => {
    const req = request({ host: '127.0.0.1', port, method, path, headers: { host: `127.0.0.1:${port}`, ...headers } }, res => {
      let text = ''
      res.on('data', c => { text += c })
      res.on('end', () => resolve({ status: res.statusCode, text }))
    })
    req.on('error', reject)
    if (method === 'POST') req.end(body)
    else req.end()
  })
}

test('browser and rebinding requests are refused; local JSON clients are served', async () => {
  const emb = await fakeEmbeddings()
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
  const s = await start({ ...defaults(dir), port: 0, embedding: { ...defaults(dir).embedding, baseUrl: emb.url, autoPull: false } })
  try {
    const json = { 'content-type': 'application/json' }
    assert.equal((await raw(s.port, { headers: json })).status, 200, 'a local JSON client')
    assert.equal((await raw(s.port, { method: 'GET', path: '/health' })).status, 200)
    assert.equal((await raw(s.port, { headers: { ...json, origin: 'https://evil.example' } })).status, 403, 'any Origin')
    assert.equal((await raw(s.port, { method: 'GET', path: '/health', headers: { origin: 'null' } })).status, 403)
    assert.equal((await raw(s.port, { headers: { ...json, host: `evil.example:${s.port}` } })).status, 403, 'DNS rebinding')
    assert.equal((await raw(s.port, { headers: { 'content-type': 'text/plain' }, path: '/v1/memory/messages',
      body: '{"session":"x","messages":[{"role":"user","text":"planted"}]}' })).status, 415, 'no-preflight browser POST')
    const health = JSON.parse((await raw(s.port, { method: 'GET', path: '/health' })).text)
    assert.equal(health.stats.turns, 0, 'nothing was written by the refused requests')
  } finally {
    await s.close()
    await emb.close()
  }
})

test('a non-loopback host needs a token, and the token is enforced', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
  await assert.rejects(start({ ...defaults(dir), host: '0.0.0.0', port: 0 }), /without a token/)
  const emb = await fakeEmbeddings()
  const s = await start({ ...defaults(dir), host: '127.0.0.1', token: 'secret-token', port: 0,
    embedding: { ...defaults(dir).embedding, baseUrl: emb.url, autoPull: false } })
  try {
    const json = { 'content-type': 'application/json' }
    assert.equal((await raw(s.port, { headers: json })).status, 401)
    assert.equal((await raw(s.port, { headers: { ...json, authorization: 'Bearer secret-token' } })).status, 200)
  } finally {
    await s.close()
    await emb.close()
  }
})

/** Whether `text` occurs in the sidecar's memory.db or its -wal/-shm files (raw bytes). */
function onDisk(dir, text) {
  return readdirSync(dir).filter(f => f.startsWith('memory.db')).some(f => readFileSync(join(dir, f)).includes(Buffer.from(text)))
}

test('forget: one session, messages before a date, everything; the text leaves search and the file', async () => {
  const emb = await fakeEmbeddings()
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
  const s = await start({ ...defaults(dir), port: 0, embedding: { ...defaults(dir).embedding, baseUrl: emb.url, autoPull: false } })
  try {
    const old = Date.UTC(2026, 8, 1), recent = Date.UTC(2026, 9, 20)
    await post(s.port, '/v1/memory/messages', { session: 'a', messages: [{ role: 'user', text: 'My zanzibar locker code is 4417.', at: old }] })
    await post(s.port, '/v1/memory/messages', { session: 'b', messages: [
      { role: 'user', text: 'The walrus locker is in hall B.', at: old }, { role: 'user', text: 'The quokka locker is in hall C.', at: recent }] })
    await post(s.port, '/v1/memory/messages', { session: 'c', messages: [{ role: 'user', text: 'The pelican locker is new.', at: recent }] })
    const search = async q => (await post(s.port, '/v1/memory/search', { query: q, rerank: false })).text
    assert.match(await search('zanzibar locker'), /zanzibar/)
    assert.ok(onDisk(dir, 'zanzibar'), 'sanity: on disk before forgetting')

    for (const bad of [{}, { session: 'a', all: true }, { all: false }, { before: 'last week' }, { session: '' }, []]) {
      const r = await fetch(`http://127.0.0.1:${s.port}/v1/memory/forget`, { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify(bad) })
      assert.equal(r.status, 400, JSON.stringify(bad))
      assert.match((await r.json()).error, /exactly one|must be/)
    }
    assert.equal((await fetch(`http://127.0.0.1:${s.port}/health`).then(r => r.json())).stats.turns, 4, 'a bad request deletes nothing')

    assert.deepEqual(await post(s.port, '/v1/memory/forget', { session: 'a' }),
      { turns: 1, units: 1, facts: 0, segments: 0, links: 0, vacuumed: true })
    assert.doesNotMatch(await search('zanzibar locker'), /zanzibar/)
    assert.ok(!onDisk(dir, 'zanzibar'))

    const before = await post(s.port, '/v1/memory/forget', { before: '2026-10-01T00:00:00Z' })
    assert.equal(before.turns, 1)
    assert.doesNotMatch(await search('walrus locker hall'), /walrus/)
    assert.match(await search('quokka locker hall'), /quokka/, 'the newer message of the same session is kept')
    assert.ok(!onDisk(dir, 'walrus'))
    assert.ok(onDisk(dir, 'quokka'))

    assert.deepEqual(await post(s.port, '/v1/memory/forget', { all: true }), { turns: 2, units: 2, facts: 0, segments: 0, links: 0, vacuumed: true })
    const health = await fetch(`http://127.0.0.1:${s.port}/health`).then(r => r.json())
    assert.equal(health.stats.turns, 0)
    assert.equal(health.stats.units, 0)
    for (const word of ['quokka', 'pelican']) assert.ok(!onDisk(dir, word), word)
    assert.equal(await search('pelican locker'), '')
  } finally {
    await s.close()
    await emb.close()
  }
})

test('forget is guarded like every route: no browser, no rebinding, JSON only, token when set', async () => {
  const emb = await fakeEmbeddings()
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
  const s = await start({ ...defaults(dir), token: 'secret-token', port: 0,
    embedding: { ...defaults(dir).embedding, baseUrl: emb.url, autoPull: false } })
  try {
    const auth = { authorization: 'Bearer secret-token' }
    const json = { 'content-type': 'application/json', ...auth }
    assert.equal((await raw(s.port, { path: '/v1/memory/messages', headers: json,
      body: '{"session":"a","messages":[{"role":"user","text":"My zanzibar locker code is 4417."}]}' })).status, 200)
    const forget = headers => raw(s.port, { path: '/v1/memory/forget', headers, body: '{"all":true}' })
    assert.equal((await forget({ ...json, origin: 'https://evil.example' })).status, 403, 'any Origin')
    assert.equal((await forget({ ...json, host: `evil.example:${s.port}` })).status, 403, 'DNS rebinding')
    assert.equal((await forget({ 'content-type': 'text/plain', ...auth })).status, 415, 'no-preflight browser POST')
    assert.equal((await forget({ 'content-type': 'application/json' })).status, 401, 'no token')
    assert.equal((await forget({ 'content-type': 'application/json', authorization: 'Bearer wrong' })).status, 401)
    const health = JSON.parse((await raw(s.port, { method: 'GET', path: '/health', headers: auth })).text)
    assert.equal(health.stats.turns, 1, 'nothing was deleted by the refused requests')
    assert.ok(onDisk(dir, 'zanzibar'))
    const ok = await forget(json)
    assert.equal(ok.status, 200)
    assert.equal(JSON.parse(ok.text).turns, 1)
    assert.ok(!onDisk(dir, 'zanzibar'))
  } finally {
    await s.close()
    await emb.close()
  }
})

// -- release fixes (readiness test 2026-10-23: R3, R5, R6, R7, R11, R14, R21) ---------------------------------------

/** A port nothing listens on (bound once, then released). */
async function freePort() {
  const s = createTcpServer()
  await new Promise(r => s.listen(0, '127.0.0.1', r))
  const { port } = s.address()
  await new Promise(r => s.close(r))
  return port
}

/** Accepts connections and never answers (a hung model server). */
async function blackHole() {
  const sockets = new Set()
  const server = createTcpServer(socket => { sockets.add(socket); socket.on('error', () => {}) })
  await new Promise(r => server.listen(0, '127.0.0.1', r))
  return { url: `http://127.0.0.1:${server.address().port}/v1`,
    close: () => { for (const s of sockets) s.destroy(); return new Promise(r => server.close(r)) } }
}

/** Embeddings that can be switched off (`down`: 503) or made to hang (`hang`: never answer), counting requests. */
async function switchableEmbeddings() {
  const state = { down: false, hang: false, calls: 0 }
  const server = createServer((req, res) => {
    let raw = ''
    req.on('data', c => { raw += c })
    req.on('end', () => {
      state.calls++
      if (state.hang) return
      if (state.down) { res.writeHead(503, { 'content-type': 'application/json' }); return res.end('{"error":"loading"}') }
      const b = JSON.parse(raw)
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ data: b.input.map((t, index) => ({ index, embedding: vector(t) })) }))
    })
  })
  await new Promise(r => server.listen(0, '127.0.0.1', r))
  return { url: `http://127.0.0.1:${server.address().port}/v1`, state, close: () => new Promise(r => { server.closeAllConnections(); server.close(r) }) }
}

const health = port => fetch(`http://127.0.0.1:${port}/health`).then(r => r.json())
const postRaw = (port, path, body) => fetch(`http://127.0.0.1:${port}${path}`, { method: 'POST',
  headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }).then(async r => ({ status: r.status, body: await r.json() }))

/** Runs `fn` with the sidecar's log lines (stderr) collected. */
async function withLog(fn) {
  const lines = []
  const write = process.stderr.write
  process.stderr.write = (chunk, ...rest) => { lines.push(String(chunk)); return write.call(process.stderr, chunk, ...rest) }
  try {
    await fn(lines)
  } finally {
    process.stderr.write = write
  }
  return lines
}

test('malformed JSON or a body that is not an object is a 400, not a 500 (R21)', async () => {
  const emb = await fakeEmbeddings()
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
  const s = await start({ ...defaults(dir), port: 0, embedding: { ...defaults(dir).embedding, baseUrl: emb.url, autoPull: false } })
  try {
    const json = { 'content-type': 'application/json' }
    for (const body of ['{"session": "a", "messages": [', 'not json', '[1, 2]', '"text"']) {
      const r = await raw(s.port, { path: '/v1/memory/messages', headers: json, body })
      assert.equal(r.status, 400, body)
      assert.match(JSON.parse(r.text).error, /malformed JSON|must be a JSON object/)
    }
  } finally {
    await s.close()
    await emb.close()
  }
})

test('an engine that cannot start: the sidecar stays up, stores messages and says why in /health (R3, R7)', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
  const engineRoot = mkdtempSync(join(tmpdir(), 'no-engine-'))  // no .venv, no models: a fresh clone without step 2
  const s = await start({ ...defaults(dir), port: 0, embedding: { ...defaults(dir).embedding,
    engine: { engineRoot, modelDir: join(engineRoot, 'models/embeddinggemma-300m'), modelName: 'embeddinggemma-300m', port: await freePort() } } })
  try {
    assert.deepEqual((await postRaw(s.port, '/v1/memory/messages', { session: 'a', messages: [{ role: 'user', text: 'I moved to Kyoto.' }] })).body,
      { stored: 1 })
    let h
    for (let k = 0; k < 50 && (h = await health(s.port)).embedding.ok !== false; k++) await new Promise(r => setTimeout(r, 20))
    assert.equal(h.status, 'ok')
    assert.equal(h.embedding.ok, false)
    assert.match(h.embedding.lastError, /\.venv\/bin\/python does not exist/)
    assert.equal(h.stats.turns, 1)
    assert.equal(h.stats.unindexed, 1)
    const recall = await postRaw(s.port, '/v1/memory/recall', { query: 'Kyoto', session: 'b' })
    assert.equal(recall.status, 503)
    assert.equal(recall.body.cause, 'embedding-unavailable')
    assert.match(recall.body.error, /does not exist/)
  } finally {
    await s.close()
  }
})

test('EngineServer.ensure fails with the reason, never with an unhandled spawn error (R3)', async () => {
  const root = mkdtempSync(join(tmpdir(), 'engine-'))
  const modelDir = join(root, 'model')
  mkdirSync(modelDir)
  const notExecutable = join(root, 'python-not-executable')
  writeFileSync(notExecutable, '#!/bin/sh\nexit 0\n')
  const exits = join(root, 'python-exits')
  writeFileSync(exits, '#!/bin/sh\necho "no module egemma_server" >&2\nexit 3\n')
  chmodSync(exits, 0o755)
  const spec = async python => ({ engineRoot: root, modelDir, modelName: 'embeddinggemma-300m', port: await freePort(), python })
  await assert.rejects(new EngineServer(await spec(join(root, 'missing')), root).ensure(), /missing does not exist/)
  await assert.rejects(new EngineServer({ ...await spec(exits), modelDir: join(root, 'nope') }, root).ensure(), /model directory .* does not exist/)
  await assert.rejects(new EngineServer(await spec(notExecutable), root).ensure(), /cannot start egemma-server .*EACCES/)
  const started = Date.now()
  await assert.rejects(new EngineServer(await spec(exits), root).ensure(), /exited \(3\) before it was ready/)
  assert.ok(Date.now() - started < 5000, 'a server that exits is not waited for 120 s')
  assert.match(readFileSync(join(root, 'egemma-server.log'), 'utf8'), /no module egemma_server/)
})

test('a hung embedding model: messages are stored at once, recall gives up within the query limit, close() does not wait (R5, R7)', async () => {
  const emb = await switchableEmbeddings()
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
  const s = await start({ ...defaults(dir), port: 0,
    embedding: { ...defaults(dir).embedding, baseUrl: emb.url, autoPull: false, queryTimeoutMs: 500 } })
  let closed = false
  try {
    await post(s.port, '/v1/memory/messages', { session: 'old', messages: [{ role: 'user', text: 'I like green tea.' }] })
    for (let k = 0; k < 50 && (await health(s.port)).stats.unindexed; k++) await new Promise(r => setTimeout(r, 20))
    emb.state.hang = true
    let t = Date.now()
    assert.deepEqual((await postRaw(s.port, '/v1/memory/messages', { session: 'a', messages: [
      { role: 'user', text: 'My locker code is 4417.' }, { role: 'assistant', text: 'Noted.' }] })).body, { stored: 2 })
    assert.ok(Date.now() - t < 500, `messages answered after ${Date.now() - t} ms`)
    assert.equal((await postRaw(s.port, '/v1/memory/flush', { session: 'a' })).status, 200)
    t = Date.now()
    const recall = await postRaw(s.port, '/v1/memory/recall', { query: 'locker code', session: 'b' })
    const took = Date.now() - t
    assert.equal(recall.status, 503)
    assert.equal(recall.body.cause, 'embedding-timeout')
    assert.match(recall.body.error, /did not answer the query within 500 ms/)
    assert.ok(took < 2500, `recall answered after ${took} ms`)
    const h = await health(s.port)
    assert.equal(h.embedding.ok, false)
    assert.equal(h.embedding.url, emb.url)
    assert.equal(h.stats.unindexed, 2)
    t = Date.now()
    await s.close()
    closed = true
    assert.ok(Date.now() - t < 1000, `close() took ${Date.now() - t} ms with requests in flight`)
  } finally {
    if (!closed) await s.close()
    await emb.close()
  }
})

test('after an embedding outage, the first answer indexes what was stored meanwhile; the log names the URL once (R7, R11)', async () => {
  const emb = await switchableEmbeddings()
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
  let s
  const lines = await withLog(async () => {
    s = await start({ ...defaults(dir), port: 0, embedding: { ...defaults(dir).embedding, baseUrl: emb.url, autoPull: false } })
    try {
      await post(s.port, '/v1/memory/messages', { session: 'old', messages: [{ role: 'user', text: 'I like green tea.' }] })
      for (let k = 0; k < 50 && (await health(s.port)).stats.unindexed; k++) await new Promise(r => setTimeout(r, 20))
      emb.state.down = true
      await post(s.port, '/v1/memory/messages', { session: 'old', messages: [{ role: 'user', text: 'My bike is a red Bianchi.' }] })
      let h
      for (let k = 0; k < 50 && (h = await health(s.port)).embedding.ok !== false; k++) await new Promise(r => setTimeout(r, 20))
      assert.equal(h.stats.unindexed, 1)
      assert.match(h.embedding.lastError, new RegExp(`POST ${emb.url}/embeddings: HTTP 503`))
      assert.equal((await postRaw(s.port, '/v1/memory/recall', { query: 'red Bianchi bike', session: 'now' })).status, 503)
      emb.state.down = false
      await post(s.port, '/v1/memory/recall', { query: 'green tea', session: 'now' })  // the first answer after the outage
      for (let k = 0; k < 100 && (h = await health(s.port)).stats.unindexed; k++) await new Promise(r => setTimeout(r, 20))
      assert.equal(h.stats.unindexed, 0, 'indexed right after, not at the 10-minute backfill')
      assert.equal(h.embedding.ok, true)
      assert.match((await post(s.port, '/v1/memory/recall', { query: 'red Bianchi bike', session: 'now' })).text, /red Bianchi/)
    } finally {
      await s.close()
      await emb.close()
    }
  })
  const outage = lines.filter(l => /embedding model unavailable/.test(l))
  assert.equal(outage.length, 1, 'logged once per outage')
  assert.match(outage[0], new RegExp(`${emb.url}/embeddings`))
  assert.ok(lines.some(l => /answers again/.test(l)))
})

test('the Gemma license notice appears only when the endpoint is an Ollama server without the model (R7)', async () => {
  for (const [kind, tags, expected] of [['not Ollama', null, false], ['Ollama without the model', { models: [] }, true],
    ['Ollama with the model', { models: [{ name: 'embeddinggemma:latest' }] }, false]]) {
    const server = createServer((req, res) => {
      if (req.url === '/api/tags' && tags) { res.writeHead(200, { 'content-type': 'application/json' }); return res.end(JSON.stringify(tags)) }
      res.writeHead(404); res.end()
    })
    await new Promise(r => server.listen(0, '127.0.0.1', r))
    const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
    const lines = await withLog(async () => {
      const s = await start({ ...defaults(dir), port: 0, embedding: { ...defaults(dir).embedding, baseUrl: `http://127.0.0.1:${server.address().port}/v1` } })
      for (let k = 0; k < 50 && (await health(s.port)).embedding.ok === null; k++) await new Promise(r => setTimeout(r, 20))
      await s.close()
    })
    await new Promise(r => server.close(r))
    assert.equal(lines.some(l => /Gemma Terms of Use/.test(l)), expected, kind)
  }
})

test('search with a hung chat model: no answer at all skips the rerank at once; a slow rerank stops at its limit (R6)', async () => {
  const emb = await fakeEmbeddings()
  const hole = await blackHole()
  // Answers GET /models, never answers a completion (a server stuck on generation).
  const sockets = new Set()
  const stuck = createServer((req, res) => {
    sockets.add(res)
    if (req.method === 'GET') { res.writeHead(200, { 'content-type': 'application/json' }); return res.end('{"data":[{"id":"m"}]}') }
  })
  await new Promise(r => stuck.listen(0, '127.0.0.1', r))
  const run = async (chatUrl, rerankTimeoutMs) => {
    const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
    const s = await start({ ...defaults(dir), port: 0, embedding: { ...defaults(dir).embedding, baseUrl: emb.url, autoPull: false },
      chat: { ...defaults(dir).chat, baseUrl: chatUrl, model: 'm', server: 'openai', rerankTimeoutMs }, facts: { ...defaults(dir).facts, intervalSeconds: 3600 } })
    try {
      await post(s.port, '/v1/memory/messages', { session: 'a', messages: [{ role: 'user', text: 'The quokka locker is in hall C.' }] })
      const t = Date.now()
      const r = await postRaw(s.port, '/v1/memory/search', { query: 'quokka locker', rerank: true })  // off by default
      return { ...r, ms: Date.now() - t, health: await health(s.port) }
    } finally {
      await s.close()
    }
  }
  try {
    const silent = await run(hole.url, 15000)
    assert.equal(silent.status, 200)
    assert.equal(silent.body.reranked, false)
    assert.match(silent.body.text, /quokka/, 'the dense order')
    assert.ok(silent.ms < 3000, `search took ${silent.ms} ms`)
    assert.equal(silent.health.chat.ok, false)
    assert.match(silent.health.chat.lastError, /does not answer/)
    const slow = await run(`http://127.0.0.1:${stuck.address().port}/v1`, 400)
    assert.equal(slow.status, 200)
    assert.equal(slow.body.reranked, false)
    assert.match(slow.body.text, /quokka/)
    assert.ok(slow.ms < 2000, `search took ${slow.ms} ms`)
    assert.match(slow.health.chat.lastError, /no answer within 400 ms/)
  } finally {
    for (const r of sockets) r.destroy?.()
    stuck.closeAllConnections()
    await new Promise(r => stuck.close(r))
    await hole.close()
    await emb.close()
  }
})

test('the tsuzuki-memory command runs the sidecar, and says what is missing before the build (R14)', async () => {
  const bin = fileURLToPath(new URL('../bin/tsuzuki-memory.mjs', import.meta.url))
  const emb = await fakeEmbeddings()
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
  writeFileSync(join(dir, 'config.json'), JSON.stringify({ port: 0, embedding: { baseUrl: emb.url, autoPull: false } }))
  const child = spawn(process.execPath, [bin, '--config', join(dir, 'config.json')], { stdio: ['ignore', 'pipe', 'pipe'] })
  try {
    const ready = await new Promise((resolve, reject) => {
      let out = ''
      child.stdout.on('data', c => { out += c; if (out.includes('\n')) resolve(JSON.parse(out.split('\n')[0])) })
      child.once('exit', code => reject(new Error(`exited ${code}`)))
    })
    assert.equal(ready.event, 'ready')
    assert.equal((await health(ready.port)).status, 'ok')
  } finally {
    child.kill('SIGTERM')
    await new Promise(r => child.once('exit', r))
    await emb.close()
  }
  const fresh = mkdtempSync(join(tmpdir(), 'unbuilt-'))
  mkdirSync(join(fresh, 'bin'))
  copyFileSync(bin, join(fresh, 'bin', 'tsuzuki-memory.mjs'))
  const unbuilt = spawn(process.execPath, [join(fresh, 'bin', 'tsuzuki-memory.mjs')], { stdio: ['ignore', 'pipe', 'pipe'] })
  let err = ''
  unbuilt.stderr.on('data', c => { err += c })
  const code = await new Promise(r => unbuilt.once('exit', r))
  assert.equal(code, 1)
  assert.match(err, /not built yet .*run build/)
})

test('node lib/server.js runs the sidecar, as the Hermes provider starts it; on Windows too', async () => {
  // Node gives the script with the platform's separator: C:\...\lib\server.js on Windows.
  const url = 'file:///C:/src/tsuzuki/dsh-plugins/packages/tsuzuki/lib/server.js'
  assert.equal(isMainScript('C:\\src\\tsuzuki\\dsh-plugins\\packages\\tsuzuki\\lib\\server.js', url, '\\'), true)
  assert.equal(isMainScript('/home/me/tsuzuki/dsh-plugins/packages/tsuzuki/lib/server.js',
    'file:///home/me/tsuzuki/dsh-plugins/packages/tsuzuki/lib/server.js', '/'), true)
  for (const [script, separator] of [[undefined, '\\'], ['C:\\src\\tsuzuki\\dsh-plugins\\packages\\tsuzuki\\bin\\tsuzuki-memory.mjs', '\\'],
    ['/home/me/tsuzuki/dsh-plugins/packages/tsuzuki/test/server.test.mjs', '/']]) {
    assert.equal(isMainScript(script, url, separator), false, String(script))  // imported: the bin and tests call main() themselves
  }
  const emb = await fakeEmbeddings()
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
  writeFileSync(join(dir, 'config.json'), JSON.stringify({ port: 0, embedding: { baseUrl: emb.url, autoPull: false } }))
  const script = fileURLToPath(new URL('../lib/server.js', import.meta.url))
  const child = spawn(process.execPath, [script, '--config', join(dir, 'config.json')], { stdio: ['ignore', 'pipe', 'pipe'] })
  const exited = new Promise(r => child.once('exit', r))
  try {
    const ready = await new Promise((resolve, reject) => {
      let out = ''
      child.stdout.on('data', c => { out += c; if (out.includes('\n')) resolve(JSON.parse(out.split('\n')[0])) })
      exited.then(code => reject(new Error(`exited ${code} without the ready line`)))
    })
    assert.equal(ready.event, 'ready')
    assert.equal((await health(ready.port)).status, 'ok')
  } finally {
    child.kill('SIGTERM')
    await exited
    await emb.close()
  }
})

test('search does not rerank unless asked (the evaluated setting P has no rerank)', () => {
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
  assert.equal(defaults(dir).search.rerank, false)
})

test('a new config gets a random token and private files; bodies over the limit get 413 (2026-10-25 review)', async () => {
  const { statSync } = await import('node:fs')
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
  const path = join(dir, 'conf', 'config.json')
  const config = loadConfig(path)
  assert.match(config.token, /^[0-9a-f]{48}$/)
  assert.equal(statSync(path).mode & 0o777, 0o600)
  assert.equal(loadConfig(path).token, config.token, 'kept on the next start')
  const emb = await fakeEmbeddings()
  const s = await start({ ...config, dataDir: join(dir, 'data'), port: 0, embedding: { ...config.embedding, baseUrl: emb.url, autoPull: false } })
  try {
    assert.equal(statSync(join(dir, 'data')).mode & 0o777, 0o700)
    assert.equal(statSync(join(dir, 'data', 'memory.db')).mode & 0o777, 0o600)
    const ask = (headers, body) => new Promise((resolve, reject) => {
      const req = request({ host: '127.0.0.1', port: s.port, path: '/v1/memory/recall', method: 'POST',
        headers: { 'content-type': 'application/json', ...headers } }, res => { res.resume(); res.on('end', () => resolve(res.statusCode)) })
      req.on('error', reject)
      req.end(body)
    })
    assert.equal(await ask({}, '{"query":"x"}'), 401, 'no token')
    assert.equal(await ask({ authorization: 'Bearer wrong' }, '{"query":"x"}'), 401)
    assert.equal(await ask({ authorization: `Bearer ${config.token}` }, '{"query":"x"}'), 200)
    // Over the limit: 413, or the connection closed while the client is still sending (EPIPE); the sidecar stays up.
    const big = await ask({ authorization: `Bearer ${config.token}` }, '{"query":"' + 'x'.repeat(9 * 1024 * 1024) + '"}').catch(e => e.code)
    assert.ok(big === 413 || big === 'EPIPE' || big === 'ECONNRESET', String(big))
    assert.equal(await ask({ authorization: `Bearer ${config.token}` }, '{"query":"x"}'), 200)
  } finally {
    await s.close()
    await emb.close()
  }
})

test('embedding.llamacpp and chat.llamacpp: the sidecar starts llama-server on demand, once, and uses it', async () => {
  const { unlinkSync } = await import('node:fs')
  const dir = mkdtempSync(join(tmpdir(), 'sidecar-'))
  const fake = fileURLToPath(new URL('./fixtures/fake-llama-server.mjs', import.meta.url))
  const binary = join(dir, 'llama-server')
  writeFileSync(binary, `#!/bin/sh\nexec "${process.execPath}" "${fake}" "$@"\n`)
  chmodSync(binary, 0o755)
  writeFileSync(join(dir, 'model.gguf'), 'x')
  const ports = []
  for (let k = 0; k < 2; k++) {
    const s = createTcpServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); ports.push(s.address().port); await new Promise(r => s.close(r))
  }
  const log = join(dir, 'argv.jsonl')
  process.env.FAKE_LLAMA_LOG = log
  const base = defaults(join(dir, 'data'))
  const s = await start({ ...base, port: 0,
    embedding: { ...base.embedding, autoPull: false, llamacpp: { model: join(dir, 'model.gguf'), binary, port: ports[0] } },
    chat: { ...base.chat, llamacpp: { model: join(dir, 'model.gguf'), binary, port: ports[1], args: ['-c', '4096'] } },
    facts: { ...base.facts, intervalSeconds: 1 } })
  try {
    await post(s.port, '/v1/memory/messages', { session: 'a', messages: [{ role: 'user', text: 'The quokka locker is in hall C.' }] })
    for (let i = 0; i < 100 && (await health(s.port)).stats.unindexed > 0; i++) await new Promise(r => setTimeout(r, 100))
    const r = await post(s.port, '/v1/memory/recall', { query: 'quokka locker', session: 'b' })
    assert.match(r.text, /quokka locker/)
    const started = readFileSync(log, 'utf8').trim().split('\n').map(l => JSON.parse(l))
    const embed = started.find(a => a.includes('tsuzuki-embeddinggemma'))
    assert.ok(embed.includes('--embedding') && embed.includes(String(ports[0])) && embed.includes('127.0.0.1'), JSON.stringify(embed))
    await post(s.port, '/v1/memory/flush', { session: 'a' })
    for (let i = 0; i < 50 && !readFileSync(log, 'utf8').includes('tsuzuki-chat'); i++) await new Promise(r => setTimeout(r, 200))
    const chat = readFileSync(log, 'utf8').trim().split('\n').map(l => JSON.parse(l)).find(a => a.includes('tsuzuki-chat'))
    assert.ok(chat && chat.includes('4096') && chat.includes(String(ports[1])), 'the chat server was started with its own args')
    assert.equal(started.filter(a => a.includes('tsuzuki-embeddinggemma')).length, 1, 'started once')
  } finally {
    await s.close()
    delete process.env.FAKE_LLAMA_LOG
    for (const port of ports) spawnSync('sh', ['-c', `kill $(lsof -t -iTCP:${port} -sTCP:LISTEN) 2>/dev/null; true`])
    try { unlinkSync(binary) } catch {}
  }
})
