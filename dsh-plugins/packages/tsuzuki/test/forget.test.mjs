// Forgetting: deleted messages and everything derived from them can no longer be retrieved, and their text is not in
// the SQLite file (or its WAL) afterwards. Fake embeddings and a fake chat model; CPU only.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, readdirSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, dirname, join } from 'node:path'
import { MemoryStore } from '../lib/store.js'
import { Memory, forgetScope, parseBefore } from '../lib/memory.js'

function vectorFor(text) {
  const v = new Float32Array(64)
  for (const w of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (['the', 'i', 'my', 'is', 'a', 'user', 'assistant', 'said'].includes(w)) continue
    let h = 2166136261
    for (const c of w) h = Math.imul(h ^ c.codePointAt(0), 16777619)
    v[(h >>> 0) % 64] += 1
  }
  const n = Math.hypot(...v) || 1
  return v.map(x => x / n)
}

function fakeEmbedder(gate = null) {
  return {
    model: () => 'fake-64',
    async embed(input) {
      if (gate) await gate(input)
      return input.map(vectorFor)
    },
  }
}

/** Facts restate each user message; update checks link the first earlier fact to the first new one. */
function factChat({ link = false, gate = null } = {}) {
  const seen = []
  return {
    seen,
    async complete(messages, options) {
      seen.push(options.name)
      if (gate) await gate(options.name)
      if (options.name === 'fact') {
        const said = [...messages[1].content.matchAll(/^\[(M\d+)\] user: (.+)$/gm)]
        return JSON.stringify({ facts: said.map(([, id, text]) => ({ fact: `The user said: ${text}`, ids: [id] })) })
      }
      if (options.name === 'updates') return link ? '{"updates": [{"old": "E1", "new": "N1"}]}' : '{"updates": []}'
      return '{"ranking": []}'
    },
  }
}

const DAY = 86_400_000
const day = Date.UTC(2026, 9, 1, 3, 0)

function setup({ chat = factChat(), embedder = fakeEmbedder(), path } = {}) {
  path ??= join(mkdtempSync(join(tmpdir(), 'forget-')), 'memory.db')
  const store = new MemoryStore(path)
  return { path, store, chat, memory: new Memory(store, embedder, chat, { segmentIdleMinutes: 60, updateLinks: true }) }
}

/** Whether `text` occurs anywhere in the database file or its -wal/-shm companions (raw bytes). */
function onDisk(path, text) {
  const needle = Buffer.from(text)
  return readdirSync(dirname(path)).filter(f => f.startsWith(basename(path)))
    .some(f => readFileSync(join(dirname(path), f)).includes(needle))
}

/** Every text column of the store, for "not stored anywhere" checks through SQL. */
function storedText(store) {
  const rows = [
    ...store.db.prepare('SELECT text FROM turns').all(), ...store.db.prepare('SELECT text FROM units').all(),
    ...store.db.prepare('SELECT fact AS text FROM facts').all(),
  ]
  return rows.map(r => r.text).join('\n')
}

async function say(memory, session, k, text, at) {
  await memory.capture({ id: `${session}-${k}`, session, role: 'user', text, at })
}

async function recalled(memory, query) {
  return (await memory.retrieve(query, { rerank: false, budgetTokens: 4096 })).text
}

test('forget scope: exactly one of session, before (ISO 8601) or all: true', () => {
  assert.deepEqual(forgetScope({ session: 's1' }), { session: 's1' })
  assert.deepEqual(forgetScope({ all: true }), { all: true })
  assert.equal(forgetScope({ before: '2026-10-01' }).before, new Date(2026, 9, 1).getTime(), 'a date alone is local midnight')
  assert.equal(forgetScope({ before: '2026-10-01T09:00:00+09:00' }).before, Date.UTC(2026, 9, 1, 0, 0))
  assert.equal(forgetScope({ before: '2026-10-01T09:00:00Z' }).before, Date.UTC(2026, 9, 1, 9, 0))
  for (const bad of [{}, { session: 's', all: true }, { session: '' }, { session: '  ' }, { session: 7 }, { all: false },
    { all: 'true' }, { before: 'yesterday' }, { before: '2026-02-30' }, { before: '10/01/2026' }, { before: 1759276800000 },
    { session: 's', before: '2026-10-01' }]) {
    assert.equal(typeof forgetScope(bad), 'string', JSON.stringify(bad))
  }
  assert.deepEqual(forgetScope({ session: 's', before: null }), { session: 's' }, 'null counts as absent')
  assert.equal(parseBefore('2026-13-01'), null)
})

test('forget a session: its messages, windows, facts and links are gone from retrieval and from the file', async () => {
  const { path, store, memory } = setup({ chat: factChat({ link: true }) })
  await say(memory, 'old', 0, 'I keep the spare key under the zanzibar flowerpot.', day)
  await memory.capture({ id: 'old-1', session: 'old', role: 'assistant', text: 'Noted, the zanzibar flowerpot.', at: day + 1000 })
  memory.maybeCloseSegment('old', true)
  await memory.drain()
  await say(memory, 'keep', 0, 'My quokka figurine sits on the spare key shelf.', day + DAY)
  memory.maybeCloseSegment('keep', true)
  await memory.drain()
  assert.deepEqual(store.links(), [{ old: 'old@0#0', new: 'keep@0#0' }])
  assert.match(await recalled(memory, 'zanzibar flowerpot spare key'), /zanzibar/)
  assert.ok(onDisk(path, 'zanzibar'), 'sanity: the text is in the file before forgetting')

  const result = memory.forget({ session: 'old' })
  assert.deepEqual(result, { turns: 2, units: 3, facts: 1, segments: 1, links: 1, vacuumed: true })
  const text = await recalled(memory, 'zanzibar flowerpot spare key')
  assert.doesNotMatch(text, /zanzibar/)
  assert.doesNotMatch(text, /Later update/, 'no note from a link to the deleted fact')
  assert.match(text, /quokka/, 'other sessions are kept')
  assert.doesNotMatch(storedText(store), /zanzibar/)
  assert.equal(store.links().length, 0)
  assert.ok(!onDisk(path, 'zanzibar'), 'not in the database file or its WAL')
  assert.ok(onDisk(path, 'quokka'))
  assert.deepEqual(store.stats(), { turns: 1, units: 2, facts: 1, segmentsPending: 0, segmentsFailed: 0, links: 0 })
  assert.equal(await memory.backfill(), 0, 'nothing to re-index')

  // The session can go on: positions start again and it is captured and found as usual.
  await say(memory, 'old', 2, 'The walrus statue is new.', day + 2 * DAY)
  assert.equal(store.turn('old-2').pos, 0)
  assert.match(await recalled(memory, 'walrus statue'), /walrus/)
})

test('forget before a date: older messages go, a segment across the date is narrowed and its facts re-extracted', async () => {
  const { path, store, memory, chat } = setup()
  await say(memory, 'early', 0, 'The narwhal poster hangs in the hall.', day)
  await say(memory, 'span', 0, 'I parked the car near the zanzibar cafe.', day + 1000)
  await say(memory, 'span', 1, 'The walrus mug is chipped.', day + 2000)
  await say(memory, 'span', 2, 'My quokka plant needs water.', day + 2 * DAY)
  await say(memory, 'span', 3, 'The pelican kite is in the garage.', day + 2 * DAY + 1000)
  await say(memory, 'late', 0, 'An ocelot print arrived today.', day + 3 * DAY)
  for (const s of ['early', 'span', 'late']) memory.maybeCloseSegment(s, true)
  await memory.drain()
  assert.equal(store.stats().facts, 6)
  for (const word of ['narwhal', 'zanzibar', 'walrus']) assert.ok(onDisk(path, word), `sanity: ${word} on disk`)

  const { reindex, ...result } = store.forget({ before: day + DAY })
  // Windows of early-0, span-0, span-1 and span-2 (which showed span-1); facts and fact units of both touched segments.
  assert.deepEqual(result, { turns: 3, units: 4 + 5, facts: 5, segments: 2, links: 0, vacuumed: true })
  assert.deepEqual(reindex.map(t => t.id), ['span-2'], 'the first kept message showed a deleted one in its window')
  assert.deepEqual(store.pendingSegments(3).map(s => [s.id, s.firstPos, s.lastPos]), [['span@2', 2, 3]])
  assert.ok(!store.units().some(u => u.id === 'p2|span-2'))
  for (const word of ['narwhal', 'zanzibar', 'walrus']) {
    assert.doesNotMatch(storedText(store), new RegExp(word))
    assert.ok(!onDisk(path, word), `${word} not in the file`)
  }
  for (const word of ['quokka', 'pelican', 'ocelot']) assert.ok(onDisk(path, word), `${word} kept`)

  memory.forget({ before: 0 })  // nothing older than 1970: a no-op that must not disturb anything
  assert.equal(await memory.backfill(), 1, 'the window of the first kept message is rebuilt')
  assert.deepEqual(store.units().find(u => u.id === 'p2|span-2').turns, ['span-2'])
  assert.equal(store.units().find(u => u.id === 'p2|span-2').text, 'user: My quokka plant needs water.')
  const calls = chat.seen.length
  await memory.drain()
  assert.equal(chat.seen.length, calls + 1, 'the narrowed segment is extracted again')
  assert.deepEqual([...store.facts().get('span@2')].map(f => f.turns), [['span-2'], ['span-3']])
  for (const word of ['narwhal', 'zanzibar', 'walrus']) assert.doesNotMatch(await recalled(memory, `${word} hall cafe mug`), new RegExp(word))
  assert.match(await recalled(memory, 'quokka plant'), /quokka/)
  assert.equal(await memory.backfill(), 0)
})

test('forget all: nothing is left in the store or the file, and memory keeps working', async () => {
  const { path, store, memory } = setup({ chat: factChat({ link: true }) })
  await say(memory, 'a', 0, 'The zanzibar lamp is broken.', day)
  memory.maybeCloseSegment('a', true)
  await memory.drain()
  await say(memory, 'b', 0, 'The walrus lamp replaced it.', day + DAY)
  await say(memory, 'c', 0, 'Still pending narwhal message.', day + 2 * DAY)
  memory.maybeCloseSegment('b', true)
  await memory.drain()
  assert.equal(store.links().length, 1)

  const result = memory.forget({ all: true })
  assert.deepEqual(result, { turns: 3, units: 5, facts: 2, segments: 2, links: 1, vacuumed: true })
  assert.deepEqual(store.stats(), { turns: 0, units: 0, facts: 0, segmentsPending: 0, segmentsFailed: 0, links: 0 })
  assert.equal(store.db.prepare('SELECT COUNT(*) AS n FROM segments').get().n, 0)
  assert.equal(await recalled(memory, 'zanzibar walrus narwhal lamp'), '')
  for (const word of ['zanzibar', 'walrus', 'narwhal']) assert.ok(!onDisk(path, word), `${word} not in the file`)
  await say(memory, 'd', 0, 'A fresh ocelot note.', day + 3 * DAY)
  assert.match(await recalled(memory, 'ocelot'), /ocelot/)
})

test('a link through a forgotten fact is replaced by one that skips it', async () => {
  const { store, memory } = setup()
  for (const [k, s] of ['a', 'b', 'c'].entries()) {
    await say(memory, s, 0, ['I live in Osaka now.', 'I moved from Osaka to Kyoto.', 'I moved from Kyoto to Nara.'][k], day + k * DAY)
    memory.maybeCloseSegment(s, true)
  }
  await memory.drain()
  store.db.exec('DELETE FROM links')
  store.addLinks([{ old: 'a@0#0', new: 'b@0#0' }, { old: 'b@0#0', new: 'c@0#0' }, { old: 'a@0#0', new: 'gone@0#0' }])
  assert.equal(store.links().length, 2, 'links to a fact that does not exist are not stored')
  assert.equal(memory.forget({ session: 'b' }).links, 2)
  assert.deepEqual(store.links(), [{ old: 'a@0#0', new: 'c@0#0' }])
  assert.match(await recalled(memory, 'live in Osaka'), /\[a-0\] user: I live in Osaka now\.\n {2}Later update \(.+\): The user said: I moved from Kyoto to Nara\./)
})

test('a fact extraction answered after a forget stores nothing from it', async () => {
  let release
  const answered = new Promise(r => { release = r })
  let asked
  const askedP = new Promise(r => { asked = r })
  const chat = factChat({ gate: async name => { if (name === 'fact') { asked(); await answered } } })
  const { path, store, memory } = setup({ chat })
  await say(memory, 'x', 0, 'The zanzibar code is on the fridge.', day)
  memory.maybeCloseSegment('x', true)
  const drained = memory.drain()
  await askedP
  memory.forget({ session: 'x' })
  release()
  await drained
  assert.equal(store.stats().facts, 0)
  assert.equal(store.stats().units, 0)
  assert.doesNotMatch(storedText(store), /zanzibar/)
  assert.ok(!onDisk(path, 'zanzibar'))
})

test('an extraction answered after a forget narrowed its segment (same id) is dropped and redone', async () => {
  let release
  const answered = new Promise(r => { release = r })
  let asked
  const askedP = new Promise(r => { asked = r })
  let gated = true
  const chat = factChat({ gate: async name => { if (name === 'fact' && gated) { gated = false; asked(); await answered } } })
  const { path, store, memory } = setup({ chat })
  // Explicit capture times: the middle message is the oldest, so forgetting before a date keeps both ends.
  await say(memory, 'x', 0, 'The quokka shelf is new.', day + 2 * DAY)
  await say(memory, 'x', 1, 'The zanzibar code is on the fridge.', day)
  await say(memory, 'x', 2, 'The pelican lamp is new.', day + 2 * DAY)
  memory.maybeCloseSegment('x', true)
  const drained = memory.drain()
  await askedP
  memory.forget({ before: day + DAY })
  assert.deepEqual(store.pendingSegments(3).map(s => [s.id, s.firstPos, s.lastPos]), [['x@0', 0, 2]])
  release()
  await drained
  assert.equal(store.stats().facts, 0, 'the reply drew on the deleted message')
  assert.ok(!onDisk(path, 'zanzibar'))
  await memory.drain()
  assert.deepEqual([...store.facts().get('x@0')].map(f => f.turns), [['x-0'], ['x-2']])
  assert.doesNotMatch(storedText(store), /zanzibar/)
})

test('a window being embedded when its message is forgotten is not stored', async () => {
  let release
  const embedded = new Promise(r => { release = r })
  let asked
  const askedP = new Promise(r => { asked = r })
  const embedder = fakeEmbedder(async input => { if (input.some(t => t.includes('zanzibar'))) { asked(); await embedded } })
  const { path, store, memory } = setup({ embedder })
  const capture = say(memory, 'x', 0, 'The zanzibar code is on the fridge.', day)
  await askedP
  assert.equal(store.stats().turns, 1, 'the message is stored before its window is embedded')
  memory.forget({ all: true })
  release()
  await capture
  assert.equal(store.stats().units, 0)
  assert.ok(!onDisk(path, 'zanzibar'))
})

test('another connection to the same file (a second dsh process) drops its caches and writes nothing back', async () => {
  const first = setup()
  const second = setup({ path: first.path })
  await say(first.memory, 'x', 0, 'The zanzibar code is on the fridge.', day)
  await say(first.memory, 'y', 0, 'The quokka code is in the drawer.', day)
  assert.match(await recalled(second.memory, 'zanzibar code'), /zanzibar/, 'warms the second connection\'s caches')
  const result = first.memory.forget({ session: 'x' })
  assert.equal(result.vacuumed, true, 'an idle second connection does not block VACUUM')
  assert.doesNotMatch(await recalled(second.memory, 'zanzibar code'), /zanzibar/)
  assert.equal(await second.memory.backfill(), 0)
  assert.ok(!onDisk(first.path, 'zanzibar'))
  second.store.close()
})

test('when another connection holds a read open, rows are deleted at once and the file is rewritten later', async () => {
  const { path, store, memory } = setup()
  await say(memory, 'x', 0, 'The zanzibar code is on the fridge.', day)
  const reader = new MemoryStore(path)
  reader.db.exec('BEGIN')
  reader.db.prepare('SELECT COUNT(*) AS n FROM turns').get()
  const result = memory.forget({ session: 'x' })
  assert.equal(result.turns, 1)
  assert.equal(result.vacuumed, false)
  assert.equal(await recalled(memory, 'zanzibar code'), '')
  assert.doesNotMatch(storedText(store), /zanzibar/)
  reader.db.exec('COMMIT')
  assert.equal(store.retryCompact(), true, 'rewritten once the reader is done (hosts retry on every backfill)')
  assert.equal(store.retryCompact(), false, 'nothing left to rewrite')
  assert.ok(!onDisk(path, 'zanzibar'))
  reader.close()
})
