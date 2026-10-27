// The memory from a terminal (src/cli.ts) against a real sidecar with fake embeddings: import, recent, show, search,
// export, forget, status, and the Japanese text.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { start, defaults } from '../lib/server.js'
import { httpSidecar } from '../lib/mcp.js'
import { run, language, batches, clean } from '../lib/cli.js'
import { makeZip, chatgptExport, claudeExport } from './fixtures/exports.mjs'

function vector(text) {
  const v = new Array(64).fill(0)
  const body = text.replace(/^task: search result \| query: /, '').replace(/^title: none \| text: /, '')
  for (const w of body.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (['the', 'i', 'my', 'is', 'to', 'do', 'user', 'assistant', 'where', 'what', 'a', 'in', 'we', 'are'].includes(w)) continue
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

function capture(answer = 'y') {
  const out = [], err = []
  return { out, err, io: { out: l => out.push(l), err: l => err.push(l), ask: async () => answer } }
}

test('cli: import an export, then recent, show, search, export, forget and status', async () => {
  const emb = await fakeEmbeddings()
  const dir = mkdtempSync(join(tmpdir(), 'cli-'))
  const base = defaults(join(dir, 'memory'))
  const config = { ...base, port: 0, token: 'tok', embedding: { ...base.embedding, baseUrl: emb.url, autoPull: false } }
  const s = await start(config)
  const path = join(dir, 'config.json')
  writeFileSync(path, JSON.stringify({ ...config, port: s.port }))
  const sidecar = httpSidecar(path, fileURLToPath(new URL('../lib/server.js', import.meta.url)))
  const cli = (args, answer) => { const c = capture(answer); return run(['--lang', 'en', ...args], { sidecar, io: c.io }).then(code => ({ code, ...c })) }
  try {
    const zip = join(dir, 'chatgpt.zip')
    writeFileSync(zip, makeZip({ 'conversations.json': Buffer.from(JSON.stringify(chatgptExport())) }))
    const imported = await cli(['import', zip])
    assert.equal(imported.code, 0)
    assert.match(imported.out[0], /^1 conversations, 4 messages \(ChatGPT\)\.$/)
    writeFileSync(join(dir, 'claude.json'), JSON.stringify(claudeExport()))
    assert.equal((await cli(['import', join(dir, 'claude.json')])).code, 0)
    const again = await cli(['import', zip])
    assert.equal(again.code, 0)
    for (let k = 0; k < 100 && (await sidecar.health()).stats.unindexed; k++) await new Promise(r => setTimeout(r, 20))
    assert.equal((await sidecar.health()).stats.turns, 6, 'importing again stores nothing twice')

    const recent = await cli(['recent'])
    assert.match(recent.out[0], /^2026-05-01 \d\d:\d\d  claude-c-1  \(2 messages\)$/, 'newest first')
    assert.match(recent.out[1], /We are going to Hakodate in June\./)
    assert.match(recent.out[2], /chatgpt-conv-1  \(4 messages\)$/)

    const shown = await cli(['show', 'chatgpt-conv-1'])
    assert.equal(shown.out.length, 4)
    assert.match(shown.out[0], /^2023-11-1\d \d\d:\d\d  user: I adopted a cat named Mochi\.$/)
    assert.equal((await cli(['show', 'nope'])).code, 1)

    const found = await cli(['search', 'Mochi', 'tuna'])
    assert.match(found.out.join('\n'), /^Date: 2023\/11\/1\d[\s\S]*Mochi/)
    assert.doesNotMatch(found.out.join('\n'), /<memory>|Current date/)

    const backup = join(dir, 'backup.jsonl')
    assert.equal((await cli(['export', backup])).code, 0)
    const lines = readFileSync(backup, 'utf8').trim().split('\n').map(l => JSON.parse(l))
    assert.equal(lines.length, 6)
    assert.deepEqual(Object.keys(lines[0]), ['session', 'id', 'role', 'text', 'at'])

    const kept = await cli(['forget', 'claude-c-1'], 'n')
    assert.equal(kept.out[0], 'Nothing was deleted.')
    const gone = await cli(['forget', 'claude-c-1'], 'y')
    assert.match(gone.out[0], /^Deleted 2 messages/)
    assert.equal((await sidecar.health()).stats.turns, 4)
    const noTty = await cli(['forget', '--all'], null)
    assert.equal(noTty.code, 1, 'no terminal to ask and no --yes: nothing is deleted')
    assert.equal((await sidecar.health()).stats.turns, 4)

    // The backup reads back into an empty memory, with the same conversations.
    assert.equal((await cli(['forget', '--all', '--yes'])).code, 0)
    const restored = await cli(['import', backup])
    assert.match(restored.out[0], /^2 conversations, 6 messages \(backup\)\.$/)
    for (let k = 0; k < 100 && (await sidecar.health()).stats.unindexed; k++) await new Promise(r => setTimeout(r, 20))
    assert.deepEqual((await cli(['recent'])).out.filter(l => !l.startsWith(' ')).map(l => l.split('  ')[1]), ['claude-c-1', 'chatgpt-conv-1'])

    const status = await cli(['status'])
    assert.match(status.out[0], /^The memory server is running \(pid \d+\)\.$/)
    assert.match(status.out[1], /^Stored: 6 messages/)
    const ja = capture()
    await run(['--lang=ja', 'status'], { sidecar, io: ja.io })
    assert.match(ja.out[0], /^記憶のサーバーは動いています/)
    assert.equal((await cli(['nonsense'])).code, 2)
  } finally {
    await s.close()
    await emb.close()
  }
})

test('cli: language from --lang, else from the system', () => {
  assert.equal(language(['--lang', 'ja'], {}), 'ja')
  assert.equal(language(['--lang=en'], { LANG: 'ja_JP.UTF-8' }), 'en')
  assert.equal(language([], { LANG: 'ja_JP.UTF-8' }), 'ja')
  assert.equal(language([], { LC_ALL: 'en_US.UTF-8', LANG: 'ja_JP.UTF-8' }), 'en')
})

test('cli: printed text has no control characters; imports go in requests of bounded size', () => {
  assert.equal(clean('a\x1b]52;c;ZXZpbA==\x07b\x1b[2Jc\nd\te'), 'a]52;c;ZXZpbA==b[2Jc\nd\te')
  const long = 'x'.repeat(300_000)
  const parts = batches([...Array(30)].map((_, i) => ({ role: 'user', text: i === 0 ? long : 'y'.repeat(200_000) })))
  assert.equal(parts.flat().length, 30)
  assert.ok(parts.length > 1)
  for (const part of parts) assert.ok(Buffer.byteLength(JSON.stringify(part)) <= 4 * 2 ** 20 + 300_000)
  assert.ok(parts[0][0].text.endsWith(' […]') && parts[0][0].text.length < long.length, 'a message too long to keep whole is cut')
})
