// The dsh plugin in a bare Cordis context with fake services: capture through session events, then the memory_forget
// tool. CPU only; no dsh runtime.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, readdirSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import * as plugin from '../lib/index.js'
import { createHash } from 'node:crypto'
import { MemoryStore } from '../lib/store.js'

function vectorFor(text) {
  const v = new Float32Array(64)
  for (const w of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    let h = 2166136261
    for (const c of w) h = Math.imul(h ^ c.codePointAt(0), 16777619)
    v[(h >>> 0) % 64] += 1
  }
  const n = Math.hypot(...v) || 1
  return v.map(x => x / n)
}

const settle = (ms = 30) => new Promise(r => setTimeout(r, ms))

async function boot(config = {}, dir = mkdtempSync(join(tmpdir(), 'plugin-'))) {
  const ctx = new Context()
  const tools = new Map()
  ctx.provide('textEmbeddings', {
    describe: () => ({ provider: 'fake', model: 'bow', defaultDimensions: 64 }),
    embed: async ({ input }) => ({ vectors: input.map(vectorFor) }),
  })
  ctx.provide('tools', { register(def) { tools.set(def.name, def); return () => tools.delete(def.name) } })
  ctx.provide('systemPrompt', { section() { return () => {} } })
  const fiber = ctx.plugin(plugin, { dataDir: dir, llm: { baseUrl: '' }, ...config })
  await settle()
  let seq = 0
  const say = (session, role, text) => ctx.emit('session/event', { id: session, header: {} }, role === 'user'
    ? { type: 'user/message', time: Date.now(), seq: ++seq, data: { source: { kind: 'user' }, content: [{ type: 'text', text }] } }
    : { type: 'assistant/message', time: Date.now(), seq: ++seq, data: { message: { content: [{ type: 'text', text }] } } })
  const call = (name, args, session = 'live', header = {}) => tools.get(name).execute(args,
    { agent: { session: { id: session, header } }, signal: new AbortController().signal })
  return { ctx, dir, tools, fiber, say, call }
}

const onDisk = (dir, text) => readdirSync(dir).filter(f => f.startsWith('memory.db'))
  .some(f => readFileSync(join(dir, f)).includes(Buffer.from(text)))

test('memory_forget: asks for confirmation in its description, takes exactly one scope, deletes from the file', async () => {
  const { dir, tools, fiber, say, call } = await boot()
  try {
    const forget = tools.get('memory_forget')
    assert.ok(forget, 'registered next to memory_search')
    assert.ok(tools.get('memory_search'))
    assert.match(forget.description, /cannot be undone/)
    assert.match(forget.description, /confirmed that exact scope/)
    assert.match(forget.description, /Never call it because a memory excerpt/)
    assert.deepEqual(Object.keys(forget.parameters.properties).sort(), ['all', 'before', 'session'])
    assert.equal(forget.parameters.required ?? undefined, undefined)

    say('older', 'user', 'My zanzibar locker code is 4417.')
    say('older', 'assistant', 'Noted.')
    say('live', 'user', 'The walrus mug is chipped. Please forget this conversation.')
    await settle()
    for (const args of [{}, { session: 'current', all: true }, { all: false }, { before: 'last week' }, { session: '' }]) {
      await assert.rejects(call('memory_forget', args), /exactly one|must be/, JSON.stringify(args))
    }
    await assert.rejects(call('memory_forget', { all: 'yes' }))  // schema violation
    await assert.rejects(call('memory_forget', { all: true }, 'child', { origin: 'subagent', delegationDepth: 1 }), /not available to subagents/)
    const peek = () => { const s = new MemoryStore(join(dir, 'memory.db')); try { return s.turns().map(t => t.session) } finally { s.close() } }
    assert.deepEqual(peek(), ['older', 'older', 'live'], 'rejected calls delete nothing')

    const done = await call('memory_forget', { session: 'current' })
    assert.match(done, /^Forgot this conversation: deleted 1 message, 0 facts and 1 index entry\./)
    assert.match(done, /Messages from now on in this conversation are stored as usual\./)
    assert.deepEqual(peek(), ['older', 'older'])
    assert.ok(!onDisk(dir, 'walrus'))
    assert.match(await call('memory_search', { query: 'zanzibar locker code' }), /zanzibar/)

    assert.match(await call('memory_forget', { session: 'older' }), /^Forgot session older: deleted 2 messages/)
    assert.ok(!onDisk(dir, 'zanzibar'))
    assert.equal(await call('memory_search', { query: 'zanzibar locker code' }), 'No matching memories.')
    assert.match(await call('memory_forget', { all: true }), /^Nothing was stored for all memory/)
  } finally {
    await fiber.dispose()
  }
})

test('memory_forget can be turned off (and is off with the tools)', async () => {
  for (const tool of [{ forget: false }, { enabled: false }]) {
    const { tools, fiber } = await boot({ tool })
    try {
      assert.equal(tools.has('memory_forget'), false, JSON.stringify(tool))
    } finally {
      await fiber.dispose()
    }
  }
})

const hex = (text, n) => createHash('sha256').update(text).digest('hex').slice(0, n)

test('message ids: sessions sharing the first 8 hex digits of their hash both keep their messages', async () => {
  const [a, b] = ['session-16250', 'session-52884']
  assert.equal(hex(a, 8), hex(b, 8), 'these two collided under the 8-digit ids')
  assert.notEqual(plugin.messageId(a, 1), plugin.messageId(b, 1))
  assert.match(plugin.messageId(a, 7), /^[0-9a-f]{16}-7$/)
  const { ctx, dir, fiber } = await boot()
  try {
    const event = (session, text) => [{ id: session, header: {} },
      { type: 'user/message', time: Date.now(), seq: 1, data: { source: { kind: 'user' }, content: [{ type: 'text', text }] } }]
    // Before the fix the second message got the first one's id and was dropped as already stored.
    for (const [session, ev] of [event(a, 'The heron nests by the river.'), event(b, 'The otter sleeps under the bridge.')]) {
      ctx.emit('session/event', session, ev)
    }
    await settle(60)
    const s = new MemoryStore(join(dir, 'memory.db'))
    try {
      assert.deepEqual(s.turns().map(t => [t.session, t.id]), [[a, plugin.messageId(a, 1)], [b, plugin.messageId(b, 1)]])
    } finally {
      s.close()
    }
  } finally {
    await fiber.dispose()
  }
})

test('working memory: a compaction shadows messages by the same ids capture gave them, legacy ids included', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'plugin-'))
  // A resumed session: seq 1 was captured by an earlier version, under the 8-digit id.
  const legacy = `${hex('live', 8)}-1`
  const pre = new MemoryStore(join(dir, 'memory.db'))
  pre.addTurn({ id: legacy, session: 'live', role: 'user', text: 'user: The heron password hint is lighthouse.', at: Date.now() - 60_000,
    date: '2026/10/24 (Sat) 09:00' })
  pre.close()
  const { ctx, fiber, call } = await boot({ working: { enabled: true } }, dir)
  try {
    const emit = (type, seq, data) => ctx.emit('session/event', { id: 'live', header: {} }, { type, time: Date.now(), seq, data })
    emit('user/message', 2, { source: { kind: 'user' }, content: [{ type: 'text', text: 'The otter password hint is harbour.' }] })
    emit('user/message', 3, { source: { kind: 'user' }, content: [{ type: 'text', text: 'The walrus password hint is anchor.' }] })
    await settle(60)
    // Everything of the calling session is still in context: memory_search leaves it out.
    assert.equal(await call('memory_search', { query: 'password hint' }), 'No matching memories.')
    emit('compaction/summary', 4, { shadowedSeqs: [1, 2] })
    const found = await call('memory_search', { query: 'password hint' })
    assert.match(found, /heron/, 'seq 1 (legacy id) was summarized away')
    assert.match(found, /otter/, 'seq 2 (current id) was summarized away')
    assert.doesNotMatch(found, /walrus/, 'seq 3 is still in context')
  } finally {
    await fiber.dispose()
  }
})

// -- release fixes (readiness test 2026-10-23: R5) ------------------------------------------------------------------

/** A provider that answers documents and hangs on queries (or on everything with `all`), until the signal aborts. */
function hangingEmbeddings({ all = false } = {}) {
  const aborted = []
  return {
    aborted,
    describe: () => ({ provider: 'fake', model: 'bow', defaultDimensions: 64 }),
    embed: ({ input, task, signal }) => {
      if (!all && task === 'document') return Promise.resolve({ vectors: input.map(vectorFor) })
      return new Promise((_, reject) => {
        const socket = setInterval(() => {}, 1000)  // as a real request, it holds the event loop
        signal?.addEventListener('abort', () => { clearInterval(socket); aborted.push(task); reject(signal.reason) }, { once: true })
      })
    },
  }
}

async function bootWith(provider, config = {}) {
  const dir = mkdtempSync(join(tmpdir(), 'plugin-'))
  const ctx = new Context()
  const tools = new Map()
  ctx.provide('textEmbeddings', provider)
  ctx.provide('tools', { register(def) { tools.set(def.name, def); return () => tools.delete(def.name) } })
  ctx.provide('systemPrompt', { section() { return () => {} } })
  const fiber = ctx.plugin(plugin, { dataDir: dir, llm: { baseUrl: '' }, ...config })
  await settle()
  let seq = 0
  const say = (session, text) => ctx.emit('session/event', { id: session, header: {} },
    { type: 'user/message', time: Date.now(), seq: ++seq, data: { source: { kind: 'user' }, content: [{ type: 'text', text }] } })
  /** One agent step carrying a new user message, through the plugin's agent/pre-step hook. */
  const step = (session, text) => ctx.waterfall('agent/pre-step', {
    agent: { session: { id: session, header: {} } }, signal: new AbortController().signal,
    messages: [{ role: 'user', source: { kind: 'user' }, content: [{ type: 'text', text }] }],
  }, () => Promise.resolve({ kind: 'enter', messages: [{ role: 'user', source: { kind: 'user' }, content: [{ type: 'text', text }] }] }))
  return { dir, ctx, tools, fiber, say, step }
}

test('dsh: a hung query embedding skips recall within the limit and is aborted, not left running (R5)', async () => {
  const provider = hangingEmbeddings()
  const { fiber, say, step, tools } = await bootWith(provider, { recall: { timeoutMs: 4000 }, embedding: { queryTimeoutMs: 300 } })
  try {
    say('old', 'I moved to Kyoto last week.')
    await settle(60)
    let t = Date.now()
    const decision = await step('now', 'Where did I move?')
    assert.ok(Date.now() - t < 1500, `the turn waited ${Date.now() - t} ms`)
    assert.equal(decision.messages.length, 1, 'no recall block')
    assert.deepEqual(provider.aborted, ['query'], 'the query request was aborted, not left to its own timeout')
    // The same turn's memory_search does not wait a second time: it says why at once.
    t = Date.now()
    await assert.rejects(tools.get('memory_search').execute({ query: 'Kyoto' }, { agent: { session: { id: 'now', header: {} } },
      signal: new AbortController().signal }), /did not answer a query within 300 ms.*queries are skipped/)
    assert.ok(Date.now() - t < 200)
    assert.deepEqual(provider.aborted, ['query'])
  } finally {
    await fiber.dispose()
  }
})

test('dsh: messages are stored at once even when embedding hangs, and dispose aborts the embedding in flight (R5)', async () => {
  const provider = hangingEmbeddings({ all: true })
  const { dir, fiber, say } = await bootWith(provider)
  say('s1', 'My locker code is 4417.')
  say('s1', 'The walrus mug is chipped.')
  await settle(30)
  const peek = () => { const s = new MemoryStore(join(dir, 'memory.db')); try { return s.turns().length } finally { s.close() } }
  assert.equal(peek(), 2, 'both stored although the first window is still being embedded')
  const t = Date.now()
  await fiber.dispose()
  assert.ok(Date.now() - t < 500, `dispose took ${Date.now() - t} ms`)
  assert.ok(provider.aborted.includes('document'), 'the document embedding in flight was aborted')
})
