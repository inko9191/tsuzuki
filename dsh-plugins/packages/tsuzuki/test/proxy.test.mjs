// OpenAI-compatible proxy (src/proxy.ts) in the sidecar, against a fake model server and fake embeddings.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createServer, request } from 'node:http'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { start, defaults } from '../lib/server.js'
import { contentKey, messageId, textOf, withRecall } from '../lib/proxy.js'

function vector(text) {
  const v = new Array(64).fill(0)
  const body = text.replace(/^task: search result \| query: /, '').replace(/^title: none \| text: /, '')
  for (const w of body.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (['the', 'i', 'my', 'is', 'to', 'do', 'user', 'assistant', 'where', 'what', 'now', 'a', 's'].includes(w)) continue
    let h = 2166136261
    for (const c of w) h = Math.imul(h ^ c.codePointAt(0), 16777619)
    v[(h >>> 0) % 64] += 1
  }
  return v
}

async function serve(handler) {
  const server = createServer(handler)
  await new Promise(r => server.listen(0, '127.0.0.1', r))
  return { url: `http://127.0.0.1:${server.address().port}`, close: () => new Promise(r => { server.closeAllConnections(); server.close(r) }) }
}

const fakeEmbeddings = () => serve((req, res) => {
  let raw = ''
  req.on('data', c => { raw += c })
  req.on('end', () => {
    const b = JSON.parse(raw)
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ data: b.input.map((t, index) => ({ index, embedding: vector(t) })) }))
  })
})

/** A model server: answers "Reply to: <last user text without the memory block>", streamed when asked. */
async function fakeModel() {
  const seen = []
  const server = await serve((req, res) => {
    let raw = ''
    req.on('data', c => { raw += c })
    req.on('end', () => {
      seen.push({ path: req.url, method: req.method, auth: req.headers.authorization, body: raw ? JSON.parse(raw) : null })
      if (req.url === '/v1/models') {
        res.writeHead(200, { 'content-type': 'application/json' })
        return res.end(JSON.stringify({ data: [{ id: 'local-model' }] }))
      }
      const b = JSON.parse(raw)
      const text = textOf(b.messages.at(-1).content).replace(/^<memory>[\s\S]*<\/memory>\s*/, '')
      const reply = `Reply to: ${text}`
      if (!b.stream) {
        res.writeHead(200, { 'content-type': 'application/json' })
        return res.end(JSON.stringify({ choices: [{ index: 0, message: { role: 'assistant', content: reply }, finish_reason: 'stop' }] }))
      }
      res.writeHead(200, { 'content-type': 'text/event-stream' })
      for (const piece of reply.match(/.{1,7}/g)) {
        res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: piece } }] })}\n\n`)
      }
      res.end('data: [DONE]\n\n')
    })
  })
  return { ...server, seen }
}

async function setup(proxy = {}) {
  const emb = await fakeEmbeddings()
  const model = await fakeModel()
  const dir = mkdtempSync(join(tmpdir(), 'proxy-'))
  const base = defaults(dir)
  const s = await start({ ...base, port: 0, token: 'tok',
    embedding: { ...base.embedding, baseUrl: `${emb.url}/v1`, autoPull: false },
    proxy: { ...base.proxy, enabled: true, port: 0, upstream: `${model.url}/v1`, ...proxy } })
  const url = `http://127.0.0.1:${s.proxyPort}/v1`
  const health = () => fetch(`http://127.0.0.1:${s.port}/health`, { headers: { authorization: 'Bearer tok' } }).then(r => r.json())
  const chat = (body, headers = {}) => fetch(`${url}/chat/completions`, { method: 'POST',
    headers: { 'content-type': 'application/json', authorization: 'Bearer tok', ...headers }, body: JSON.stringify(body) })
  const close = async () => { await s.close(); await model.close(); await emb.close() }
  return { s, url, model, chat, health, close }
}

/** Waits until the stored messages are indexed (their windows are embedded just after the reply). */
async function indexed(health) {
  for (let k = 0; k < 100; k++) {
    if ((await health()).stats.unindexed === 0) return
    await new Promise(r => setTimeout(r, 20))
  }
}

test('proxy: helpers', () => {
  assert.equal(textOf([{ type: 'text', text: 'a' }, { type: 'image_url', image_url: { url: 'x' } }, { type: 'text', text: 'b' }]), 'a\nb')
  assert.equal(textOf(null), '')
  assert.equal(contentKey('user', ' hi there '), contentKey('user', 'hi there'))
  assert.notEqual(contentKey('user', 'hi'), contentKey('assistant', 'hi'))
  const id = messageId(contentKey('user', 'hi'), 'chat-0123456789abcdef', 1)
  assert.ok(id.startsWith(`px-${contentKey('user', 'hi')}-`) && id.endsWith('-1'))
  const parts = withRecall([{ role: 'user', content: [{ type: 'text', text: 'q' }] }], '<memory>m</memory>')
  assert.deepEqual(parts[0].content, [{ type: 'text', text: '<memory>m</memory>' }, { type: 'text', text: 'q' }])
  assert.equal(withRecall([{ role: 'user', content: 'q' }], 'M')[0].content, 'M\n\nq')
})

test('proxy: a chat remembers, another chat recalls (streamed), the history is stored once', async () => {
  const t = await setup({ apiKey: 'upstream-key' })
  try {
    const first = [{ role: 'system', content: 'You are helpful.' }, { role: 'user', content: 'I adopted a cat named Mochi last week.' }]
    const r1 = await t.chat({ model: 'local-model', messages: first })
    assert.equal(r1.status, 200)
    assert.equal((await r1.json()).choices[0].message.content, 'Reply to: I adopted a cat named Mochi last week.')
    assert.doesNotMatch(JSON.stringify(t.model.seen[0].body), /<memory>/, 'nothing to recall yet')
    assert.equal(t.model.seen[0].auth, 'Bearer upstream-key', 'the upstream key, never the proxy token')
    await indexed(t.health)
    assert.equal((await t.health()).stats.turns, 2, 'the message and the reply')

    // The same chat again, with its history: only the new messages are stored.
    const again = [...first, { role: 'assistant', content: 'Reply to: I adopted a cat named Mochi last week.' },
      { role: 'user', content: 'She likes tuna.' }]
    await (await t.chat({ model: 'local-model', messages: again })).json()
    await indexed(t.health)
    assert.equal((await t.health()).stats.turns, 4)
    const sentAgain = t.model.seen.at(-1).body.messages
    assert.deepEqual(sentAgain.slice(0, 3), again.slice(0, 3), 'earlier messages are forwarded unchanged (prompt cache)')
    assert.doesNotMatch(JSON.stringify(sentAgain), /<memory>/, 'the current chat is not recalled into itself')

    // A new chat, streamed: the recall is put before its last message; the client gets the stream as sent.
    const r2 = await t.chat({ model: 'local-model', stream: true, messages: [{ role: 'user', content: "What is my cat Mochi's favorite food?" }] })
    assert.match(r2.headers.get('content-type'), /event-stream/)
    const stream = await r2.text()
    assert.match(stream, /data: \[DONE\]/)
    const sent = t.model.seen.at(-1).body.messages
    assert.equal(sent.length, 1)
    assert.match(sent[0].content, /^<memory>\n[\s\S]*Mochi[\s\S]*tuna[\s\S]*<\/memory>\n\nWhat is my cat Mochi's favorite food\?$/)
    await indexed(t.health)
    assert.equal((await t.health()).stats.turns, 6, 'the streamed reply was stored too')
  } finally {
    await t.close()
  }
})

test('proxy: pass-through, opt-out, app tasks, token and browser refusals, upstream down', async () => {
  const t = await setup()
  try {
    const models = await fetch(`${t.url}/models`, { headers: { authorization: 'Bearer tok' } })
    assert.deepEqual(await models.json(), { data: [{ id: 'local-model' }] })
    assert.equal(t.model.seen[0].auth, undefined, 'no upstream key configured: no Authorization sent')

    await (await t.chat({ messages: [{ role: 'user', content: 'Private: do not keep this.' }] }, { 'x-memory': 'off' })).json()
    await (await t.chat({ messages: [{ role: 'user', content: '### Task:\nGenerate a concise title for the chat.' }] })).json()
    assert.equal((await t.health()).stats.turns, 0, 'neither was stored')

    assert.equal((await fetch(`${t.url}/models`)).status, 401)
    const browser = await fetch(`${t.url}/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json',
      authorization: 'Bearer tok', origin: 'https://example.com' }, body: '{}' })
    assert.equal(browser.status, 403)
    assert.equal((await fetch(`http://127.0.0.1:${t.s.proxyPort}/health`, { headers: { authorization: 'Bearer tok' } })).status, 404)
    // A chat app in a Docker Desktop container calls host.docker.internal; other host names stay refused (DNS rebinding).
    const as = host => new Promise((resolve, reject) => {
      request({ port: t.s.proxyPort, host: '127.0.0.1', path: '/v1/models', headers: { host: `${host}:${t.s.proxyPort}`, authorization: 'Bearer tok' } },
        res => { res.resume(); resolve(res.statusCode) }).on('error', reject).end()
    })
    assert.equal(await as('host.docker.internal'), 200)
    assert.equal(await as('evil.example'), 403)
    assert.equal(await new Promise((resolve, reject) => {
      request({ port: t.s.port, host: '127.0.0.1', path: '/health', headers: { host: `host.docker.internal:${t.s.port}`, authorization: 'Bearer tok' } },
        res => { res.resume(); resolve(res.statusCode) }).on('error', reject).end()
    }), 403, 'only the proxy accepts it, not the memory API')

    await t.model.close()
    const down = await t.chat({ messages: [{ role: 'user', content: 'Anyone there?' }] })
    assert.equal(down.status, 502)
    assert.match((await down.json()).error.message, /did not answer/)
  } finally {
    await t.close().catch(() => {})
  }
})

test('proxy: two chats that open alike stay apart and recall each other; changing system prompts and trimmed history store nothing twice', async () => {
  const t = await setup()
  const ask = async messages => (await (await t.chat({ messages })).json()).choices[0].message.content
  const sessions = async () => (await fetch(`http://127.0.0.1:${t.s.port}/v1/memory/sessions`, { method: 'POST',
    headers: { 'content-type': 'application/json', authorization: 'Bearer tok' }, body: '{}' }).then(r => r.json())).sessions
  try {
    const a1 = await ask([{ role: 'user', content: 'Hello!' }])
    await ask([{ role: 'user', content: 'Hello!' }, { role: 'assistant', content: a1 }, { role: 'user', content: 'My bike is a red Bianchi.' }])
    const b1 = await ask([{ role: 'user', content: 'Hello!' }])
    assert.equal((await sessions()).length, 2, 'the second "Hello!" chat is a conversation of its own')
    await indexed(t.health)
    await ask([{ role: 'user', content: 'Hello!' }, { role: 'assistant', content: b1 }, { role: 'user', content: 'What bike do I ride, a Bianchi?' }])
    assert.match(textOf(t.model.seen.at(-1).body.messages.at(-1).content), /<memory>[\s\S]*red Bianchi[\s\S]*<\/memory>/,
      'chat B recalls chat A, though both opened with the same message')
    assert.equal((await sessions()).length, 2)

    // An app that puts the time into the system prompt and drops old messages.
    const system = n => ({ role: 'system', content: `You are helpful. Now: 2026-10-27 10:0${n}` })
    const c1 = await ask([system(0), { role: 'user', content: 'I am planning a trip to Hakodate.' }])
    const c2 = await ask([system(1), { role: 'user', content: 'I am planning a trip to Hakodate.' }, { role: 'assistant', content: c1 },
      { role: 'user', content: 'We go in June.' }])
    await ask([system(2), { role: 'assistant', content: c2 }, { role: 'user', content: 'Book a hotel near the station.' }])
    await indexed(t.health)
    const all = await sessions()
    assert.equal(all.length, 3, 'the changing system prompt and the trimmed history kept one conversation')
    assert.equal(all[0].messages, 6, 'and stored each message once')
  } finally {
    await t.close()
  }
})

test('proxy: an upstream on another machine is refused unless allowed', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'proxy-'))
  const base = defaults(dir)
  await assert.rejects(start({ ...base, port: 0, proxy: { ...base.proxy, enabled: true, port: 0, upstream: 'https://api.example.com/v1' } }),
    /not on this machine/)
})
