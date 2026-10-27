// MCP server (lib/mcp.js): the protocol, two clients sharing one sidecar, and the stdio command starting the sidecar.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { spawn } from 'node:child_process'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { start, defaults } from '../lib/server.js'
import { mcpHandler, httpSidecar, PROTOCOL_VERSIONS } from '../lib/mcp.js'

const LIB = fileURLToPath(new URL('../lib/', import.meta.url))

function vector(text) {
  const v = new Array(64).fill(0)
  const body = text.replace(/^task: search result \| query: /, '').replace(/^title: none \| text: /, '')
  for (const w of body.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (['the', 'i', 'my', 'is', 'to', 'do', 'user', 'assistant', 'where', 'what', 'now'].includes(w)) continue
    let h = 2166136261
    for (const c of w) h = Math.imul(h ^ c.codePointAt(0), 16777619)
    v[(h >>> 0) % 64] += 1
  }
  return v
}

async function fakeEmbeddings() {
  const server = createServer((req, res) => {
    let raw = ''
    req.on('data', c => { raw += c })
    req.on('end', () => {
      const b = JSON.parse(raw)
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ data: b.input.map((t, index) => ({ index, embedding: vector(t) })) }))
    })
  })
  await new Promise(r => server.listen(0, '127.0.0.1', r))
  return { url: `http://127.0.0.1:${server.address().port}/v1`, close: () => new Promise(r => server.close(r)) }
}

/** A config file for a sidecar with fake embeddings and a token, as setup writes it. */
function writeConfig(dir, embeddingUrl, port) {
  const config = { ...defaults(join(dir, 'memory')), port, token: 'secret-token',
    embedding: { ...defaults(dir).embedding, baseUrl: embeddingUrl, autoPull: false } }
  const path = join(dir, 'config.json')
  writeFileSync(path, JSON.stringify(config))
  return { config, path }
}

const call = (id, name, args) => ({ jsonrpc: '2.0', id, method: 'tools/call', params: { name, arguments: args } })

test('mcp: initialize, tools, ping, errors; forget only when enabled', async () => {
  const posted = []
  const sidecar = { post: async (path, body) => { posted.push({ path, body }); return path.endsWith('recall') ? { text: '' } : { stored: 1 } } }
  const h = mcpHandler(sidecar, { forget: false })
  const init = await h.handle({ jsonrpc: '2.0', id: 1, method: 'initialize',
    params: { protocolVersion: '2025-03-26', clientInfo: { name: 'Claude Desktop' }, capabilities: {} } })
  assert.equal(init.result.protocolVersion, '2025-03-26')
  assert.deepEqual(init.result.capabilities, { tools: {} })
  assert.match(init.result.instructions, /call recall/)
  assert.match(h.session, /^mcp-claude-desktop-\d{8}T\d{6}-[0-9a-f]{6}$/)
  const other = await mcpHandler(sidecar, { forget: false }).handle({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '1999-01-01' } })
  assert.equal(other.result.protocolVersion, PROTOCOL_VERSIONS[0], 'an unknown revision gets ours')
  assert.equal(await h.handle({ jsonrpc: '2.0', method: 'notifications/initialized' }), null)
  assert.deepEqual((await h.handle({ jsonrpc: '2.0', id: 2, method: 'ping' })).result, {})
  const names = (await h.handle({ jsonrpc: '2.0', id: 3, method: 'tools/list' })).result.tools.map(t => t.name)
  assert.deepEqual(names, ['recall', 'remember'])
  assert.equal((await h.handle({ jsonrpc: '2.0', id: 4, method: 'resources/list' })).error.code, -32601)
  assert.equal((await h.handle(call(5, 'forget', { all: true }))).error.code, -32602, 'no forget tool by default')
  assert.equal(posted.length, 0)

  const empty = await h.handle(call(6, 'recall', { query: 'anything' }))
  assert.equal(empty.result.content[0].text, 'No matching memories.')
  assert.deepEqual(posted[0], { path: '/v1/memory/recall', body: { query: 'anything' } }, 'one server process serves many chats: nothing is excluded')
  const bad = await h.handle(call(7, 'remember', { messages: [{ role: 'system', text: 'x' }] }))
  assert.equal(bad.result.isError, true)
  assert.equal(h.saved, false)

  const withForget = mcpHandler(sidecar, { forget: true })
  const all = (await withForget.handle({ jsonrpc: '2.0', id: 1, method: 'tools/list' })).result.tools.map(t => t.name)
  assert.deepEqual(all, ['recall', 'remember', 'forget'])
  for (const bad of [{ session: 'current' }, {}, { all: true, before: '2026-01-01' }, { all: false }]) {
    assert.equal((await withForget.handle(call(2, 'forget', bad))).result.isError, true, JSON.stringify(bad))
  }
})

test('mcp: what one client remembers, another recalls (one sidecar, its token)', async () => {
  const emb = await fakeEmbeddings()
  const dir = mkdtempSync(join(tmpdir(), 'mcp-'))
  const { config } = writeConfig(dir, emb.url, 0)
  const s = await start(config)
  const { path } = writeConfig(dir, emb.url, s.port)
  try {
    const sidecar = httpSidecar(path, join(LIB, 'server.js'))
    const desktop = mcpHandler(sidecar, { forget: true })
    await desktop.handle({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { clientInfo: { name: 'desktop' } } })
    const saved = await desktop.handle(call(2, 'remember', { messages: [
      { role: 'user', text: 'My sister Hana is a violinist in Vienna.' },
      { role: 'assistant', text: 'Noted: Hana plays violin in Vienna.' }] }))
    assert.equal(saved.result.content[0].text, 'Saved 2 message(s) to long-term memory.')
    assert.equal(desktop.saved, true)

    const cursor = mcpHandler(sidecar, { forget: false })
    await cursor.handle({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { clientInfo: { name: 'cursor' } } })
    let found
    for (let k = 0; k < 50; k++) {  // the windows are embedded just after the messages are stored
      found = await cursor.handle(call(2, 'recall', { query: 'Hana violinist Vienna', budget_tokens: 512 }))
      if (found.result.content[0].text.includes('Vienna')) break
      await new Promise(r => setTimeout(r, 50))
    }
    assert.match(found.result.content[0].text, /^<memory>\n[\s\S]*violinist in Vienna[\s\S]*<\/memory>$/)

    const forgot = await desktop.handle(call(4, 'forget', { all: true }))
    assert.match(forgot.result.content[0].text, /^Deleted: /)
    const gone = await cursor.handle(call(3, 'recall', { query: 'Hana violinist Vienna' }))
    assert.equal(gone.result.content[0].text, 'No matching memories.')

    writeFileSync(path, JSON.stringify({ ...JSON.parse(readFileSync(path, 'utf8')), token: 'wrong' }))
    const denied = await mcpHandler(httpSidecar(path, join(LIB, 'server.js')), { forget: false }).handle(call(1, 'recall', { query: 'x' }))
    assert.equal(denied.result.isError, true)
  } finally {
    await s.close()
    await emb.close()
  }
})

test('mcp: node lib/mcp.js starts the sidecar, answers over stdio, and leaves the sidecar running', async () => {
  const emb = await fakeEmbeddings()
  const dir = mkdtempSync(join(tmpdir(), 'mcp-'))
  const port = await new Promise(r => { const s = createServer().listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => r(p)) }) })
  const { path } = writeConfig(dir, emb.url, port)
  const child = spawn(process.execPath, [join(LIB, 'mcp.js'), '--config', path], { stdio: ['pipe', 'pipe', 'pipe'] })
  let out = ''
  child.stdout.on('data', d => { out += d })
  const answer = async id => {
    for (let k = 0; k < 400; k++) {
      const line = out.split('\n').find(l => l.includes(`"id":${id}`))
      if (line) return JSON.parse(line)
      await new Promise(r => setTimeout(r, 50))
    }
    throw new Error(`no answer to ${id}: ${out}`)
  }
  let pid
  try {
    child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18', clientInfo: { name: 'test' } } }) + '\n')
    assert.equal((await answer(1)).result.serverInfo.name, 'tsuzuki')
    child.stdin.write(JSON.stringify(call(2, 'remember', { messages: [{ role: 'user', text: 'I keep bees on my roof.' }] })) + '\n')
    assert.equal((await answer(2)).result.content[0].text, 'Saved 1 message(s) to long-term memory.')
    const health = await fetch(`http://127.0.0.1:${port}/health`, { headers: { authorization: 'Bearer secret-token' } }).then(r => r.json())
    pid = health.pid
    assert.ok(pid && pid !== child.pid, 'a separate sidecar process')
    assert.equal(health.stats.turns, 1)
    child.stdin.end()
    const code = await new Promise(r => child.once('exit', r))
    assert.equal(code, 0)
    assert.equal(out.split('\n').filter(Boolean).every(l => JSON.parse(l).jsonrpc === '2.0'), true, 'stdout carries only the protocol')
    const after = await fetch(`http://127.0.0.1:${port}/health`, { headers: { authorization: 'Bearer secret-token' } }).then(r => r.json())
    assert.equal(after.pid, pid, 'the sidecar outlives the client')
  } finally {
    child.kill()
    // A test that failed before reading the pid still stops the sidecar it started.
    pid ??= (await fetch(`http://127.0.0.1:${port}/health`, { headers: { authorization: 'Bearer secret-token' } })
      .then(r => r.json()).catch(() => ({}))).pid
    if (pid) try { process.kill(pid) } catch {}
    await emb.close()
  }
})

test('mcp: a program that took the port gets no token, and the client says why', async () => {
  const seen = []
  const impostor = createServer((req, res) => {
    seen.push(req.headers.authorization ?? null)
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ status: 'ok', pid: 1 }))
  })
  await new Promise(r => impostor.listen(0, '127.0.0.1', r))
  const dir = mkdtempSync(join(tmpdir(), 'mcp-'))
  const { path } = writeConfig(dir, 'http://127.0.0.1:9/v1', impostor.address().port)
  try {
    const sidecar = httpSidecar(path, join(LIB, 'server.js'))
    assert.equal(await sidecar.health(), null)
    await assert.rejects(sidecar.post('/v1/memory/recall', { query: 'x' }), /another program answers on port/)
    assert.deepEqual(seen.filter(Boolean), [], 'the token was never sent')
  } finally {
    await new Promise(r => impostor.close(r))
  }
})
