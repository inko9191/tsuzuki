// Fixes from the 5,000-session scale test (results/eval/memory/scale-report.md). CPU only, fake embeddings.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { MemoryStore } from '../lib/store.js'
import { Memory, formatDate } from '../lib/memory.js'

const tmpDb = () => join(mkdtempSync(join(tmpdir(), 'scale-')), 'memory.db')
const day = Date.UTC(2026, 9, 1, 3, 0)

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

/** Refuses any request with an input containing `poison` (as an endpoint refuses an input it cannot take). */
function pickyEmbedder() {
  const e = {
    down: false, requests: [],
    model: () => 'fake-64',
    async embed(input) {
      e.requests.push(input.length)
      if (e.down) throw new Error('connect ECONNREFUSED')
      if (input.some(t => t.includes('poison'))) throw new Error('HTTP 400 input refused')
      return input.map(vectorFor)
    },
  }
  return e
}

test('segments are looked up by session through an index, also in a store created before the index existed', () => {
  const path = tmpDb()
  const old = new MemoryStore(path)
  old.db.exec('DROP INDEX segments_session')  // a store written by an earlier version
  old.addSegment({ id: 's1@0', session: 's1', firstPos: 0, lastPos: 3, date: '2026/10/01 (Thu) 12:00' })
  old.close()
  const store = new MemoryStore(path)
  const plan = store.db.prepare('EXPLAIN QUERY PLAN SELECT COALESCE(MAX(last_pos), -1) AS p FROM segments WHERE session = ?')
    .all('s1').map(r => r.detail).join('\n')
  assert.match(plan, /USING (COVERING )?INDEX segments_session/, plan)
  assert.doesNotMatch(plan, /SCAN segments/, plan)
  assert.equal(store.segmentedUpTo('s1'), 3)
  assert.equal(store.segmentedUpTo('s2'), -1)
  store.close()
})

test('backfill: a unit the embedder refuses is skipped and logged once; the units after it are indexed', async () => {
  const embedder = pickyEmbedder()
  const logs = []
  const store = new MemoryStore(tmpDb())
  const memory = new Memory(store, embedder, null, { log: (level, m) => logs.push(`${level} ${m}`) })
  // Stored while the embedding server was down (no windows yet); message 3 is one the model cannot take.
  for (let k = 0; k < 70; k++) {
    const text = k === 3 ? 'user: poison pill' : `user: note number ${k} about topic${k}`
    store.addTurn({ id: `m${k}`, session: k < 35 ? 'a' : 'b', role: 'user', text, at: day + k, date: formatDate(day + k) })
  }
  assert.equal(await memory.backfill(), 68, 'all but the two windows showing message 3 (its own and the next one)')
  assert.deepEqual(store.units().map(u => u.id).filter(id => ['p2|m3', 'p2|m4'].includes(id)), [])
  const refused = logs.filter(l => /is not indexed: the embedder refuses it/.test(l))
  assert.deepEqual(refused.map(l => l.split(' ')[1]), ['p2|m3', 'p2|m4'])
  assert.match((await memory.retrieve('note number 69 topic69', { rerank: false })).text, /topic69/, 'units after the bad one are found')

  embedder.requests.length = 0
  assert.equal(await memory.backfill(), 0)
  assert.equal(logs.filter(l => /refuses it/.test(l)).length, 2, 'logged once per unit, not on every backfill')
  assert.deepEqual(embedder.requests, [2, 1, 1, 1], 'the refused units alone in their batch: batch, probe, one by one')

  // An embedder that is down still fails the backfill (after one probe), instead of being taken for bad units.
  embedder.down = true
  embedder.requests.length = 0
  await assert.rejects(memory.backfill(), /ECONNREFUSED/)
  assert.deepEqual(embedder.requests, [2, 1])
  embedder.down = false
  store.close()
})

test('index: one fact the embedder refuses does not keep the segment\'s other facts out', async () => {
  const embedder = pickyEmbedder()
  const chat = {
    async complete(messages, options) {
      if (options.name !== 'fact') return '{"updates": []}'
      return JSON.stringify({ facts: [{ fact: 'The user likes tea.', ids: ['M1'] }, { fact: 'The user said poison.', ids: ['M1'] },
        { fact: 'The user lives in Kyoto.', ids: ['M2'] }] })
    },
  }
  const store = new MemoryStore(tmpDb())
  const memory = new Memory(store, embedder, chat, { segmentIdleMinutes: 60, updateLinks: false })
  await memory.capture({ id: 'a1', session: 's', role: 'user', text: 'I like tea.', at: day })
  await memory.capture({ id: 'a2', session: 's', role: 'user', text: 'I live in Kyoto.', at: day + 1 })
  memory.closeAll()
  await memory.drain()
  assert.deepEqual(store.units().filter(u => u.set === 'fact').map(u => u.id), ['fact|s@0#0', 'fact|s@0#2'])
  // A capture's own window still reports its failure to the host, as before.
  await assert.rejects(memory.capture({ id: 'a3', session: 's', role: 'user', text: 'poison', at: day + 2 }), /refused/)
  store.close()
})
