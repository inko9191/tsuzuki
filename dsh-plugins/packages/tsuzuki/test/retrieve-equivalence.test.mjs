// The faster retrieve (candidates-only lookup, ranked index and vector matrix kept between calls, heap top-k) returns
// exactly what the algorithm before 2026-10-24 returned: same ids, same text, same rerank prompt. The reference below
// is that algorithm, reading everything from the store on every call (no cache). Seeded random store with score ties,
// facts, update links, multi-day sessions and clipped messages; checked again after forgets, backfills, new captures
// and new facts, so a cache that is not dropped or updated shows up as a difference. CPU only.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { MemoryStore } from '../lib/store.js'
import { Memory, clip, estimateTokens } from '../lib/memory.js'
import * as p from '../lib/pipeline.js'

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}

const VOCAB = ['kyoto', 'osaka', 'cat', 'tama', 'guitar', 'train', 'coffee', 'garden', 'rent', 'doctor', 'sister', 'plan',
  'trip', 'budget', 'python', 'deadline', 'book', 'river', 'tea', 'piano']
const DIMS = 16

// Few dimensions and a small vocabulary: many equal scores, so tie-breaking by position is exercised.
function vectorFor(text) {
  const v = new Float32Array(DIMS)
  for (const w of text.toLowerCase().match(/[a-z]+/g) ?? []) {
    const k = VOCAB.indexOf(w)
    if (k >= 0) v[k % DIMS] += 1 + (k >> 4)
  }
  if (!v.some(Boolean)) v[0] = 1
  const n = Math.hypot(...v)
  return v.map(x => x / n)
}

const embedder = { model: () => 'fake-16', async embed(input) { return input.map(vectorFor) } }

const hash = text => { let h = 2166136261; for (const c of text) h = Math.imul(h ^ c.codePointAt(0), 16777619); return h >>> 0 }

/** Facts: every user message, cut to its first words (repeats are common); update checks link E1 -> N1. */
function chatFor(prompts) {
  return {
    async complete(messages, options) {
      if (options.name === 'fact') {
        const said = [...messages[1].content.matchAll(/^\[(M\d+)\] user: (.+)$/gm)]
        return JSON.stringify({ facts: said.map(([, id, text]) => ({ fact: `The user said ${text.split(' ').slice(0, 3).join(' ')}`, ids: [id] })) })
      }
      if (options.name === 'updates') return messages[1].content.includes('[E1]') ? '{"updates": [{"old": "E1", "new": "N1"}]}' : '{"updates": []}'
      prompts?.push(messages)
      return rerankReply(messages)
    },
  }
}

function rerankReply(messages) {
  const n = (messages[1].content.match(/^\[\d+\]/gm) ?? []).length
  const h = hash(messages[1].content)
  return JSON.stringify({ ranking: [1 + h % n, 1 + (h >>> 8) % n, n, 2].slice(0, 1 + h % 4) })
}

const OPTIONS = { messageTokens: 24, rerankMessageTokens: 12, segmentTokens: 90, segmentIdleMinutes: 60, budgetTokens: 400, updateLinks: true }

// -- the algorithm before 2026-10-24, uncached -------------------------------------------------------------------------

function asTurn(t, tokens = OPTIONS.messageTokens) {
  return { id: t.id, session: t.session, pos: t.pos, date: t.date, text: clip(t.text, tokens) }
}

function splitFact(id) {
  const body = id.slice('fact|'.length), cut = body.lastIndexOf('#')
  return [body.slice(0, cut), Number(body.slice(cut + 1))]
}

function indexUnits(store) {
  const all = store.units()
  const seq = new Map(store.turns().map(t => [t.id, t.seq]))
  const windows = all.filter(u => u.set === 'p2').sort((a, b) => seq.get(a.turns[0]) - seq.get(b.turns[0]))
  const rankOf = new Map([...store.facts().keys()].map((s, i) => [s, i]))
  const facts = all.filter(u => u.set === 'fact').sort((a, b) => {
    const [sa, ja] = splitFact(a.id), [sb, jb] = splitFact(b.id)
    return (rankOf.get(sa) - rankOf.get(sb)) || (ja - jb)
  })
  return [...windows, ...facts]
}

function factRows(store) {
  const facts = store.facts(), dates = store.segmentDates()
  const ids = [...facts.keys()].filter(id => dates.has(id))
  const order = ids.map((id, i) => [id, i])
    .sort(([a, i], [b, j]) => { const da = p.when(dates.get(a)), db = p.when(dates.get(b)); return da === db ? i - j : (da < db ? -1 : 1) })
  return p.factTable(order.map(([id]) => ({ session: id, date: dates.get(id), facts: facts.get(id) })))
}

async function reference(store, query, options) {
  const units = indexUnits(store)
  if (!units.length || !query.trim()) return { text: '', ids: [], reranked: false, candidates: 0, prompt: null }
  const [qv] = await embedder.embed([query], 'query')
  const kept = units.filter(u => u.model === embedder.model() && u.vector.length === qv.length)
  let order = p.distinct(p.rank(kept, kept.map(u => u.vector), qv).map(([i]) => i), kept)
  let reranked = false, prompt = null
  if (options.rerank && order.length) {
    prompt = p.rerankMessages(query, order, kept, store.turns().map(t => asTurn(t, OPTIONS.rerankMessageTokens)))
    order = p.distinct(p.rerankedOrder(order, p.parseRerank(rerankReply(prompt), Math.min(order.length, p.RERANK_CANDIDATES))), kept)
    reranked = true
  }
  const raw = p.rawIds(order, kept)
  const plain = store.turns().map(t => asTurn(t))
  const excluded = new Set(options.excludeSessions ?? [])
  const exclude = plain.filter(t => excluded.has(t.session) || options.inContext?.(t)).map(t => t.id)
  const lookup = new Map(plain.map(t => [t.id, { ...t, session: `${t.session}|${t.date.slice(0, 10)}` }]))
  const { order: ranked, notes } = p.versioned(raw, factRows(store), store.links())
  const { text, ids } = p.selectMemory(ranked, lookup, exclude, options.budgetTokens ?? OPTIONS.budgetTokens, estimateTokens,
    ts => p.renderDated(ts, notes))
  return { text, ids, reranked, candidates: order.length, prompt }
}

// -- the randomized store ----------------------------------------------------------------------------------------------

const DAY = 86_400_000
const START = new Date(2026, 8, 1, 9, 0).getTime()

function sentence(rand, words) {
  const out = []
  for (let k = 0; k < words; k++) out.push(rand() < 0.75 ? VOCAB[Math.floor(rand() * VOCAB.length)] : ['and', 'my', 'the', 'next'][Math.floor(rand() * 4)])
  return out.join(' ')
}

async function fill(memory, rand, sessions, from, at) {
  const said = []
  for (let s = 0; s < sessions; s++) {
    const session = `s${from + s}`
    const n = 2 + Math.floor(rand() * 9)
    for (let k = 0; k < n; k++) {
      // Some messages are repeated word for word (equal windows), some are longer than the clip.
      const text = said.length && rand() < 0.15 ? said[Math.floor(rand() * said.length)] : sentence(rand, rand() < 0.2 ? 60 : 3 + Math.floor(rand() * 8))
      said.push(text)
      at += rand() < 0.1 ? DAY : 60_000  // some sessions span days
      await memory.capture({ id: `${session}-${k}`, session, role: k % 2 ? 'assistant' : 'user', text, at })
    }
    at += Math.floor(rand() * 2 * DAY)
  }
  return at
}

function queries(rand, store, n) {
  const sessions = [...new Set(store.turns().map(t => t.session))]
  return Array.from({ length: n }, (_, k) => {
    const query = sentence(rand, 1 + Math.floor(rand() * 4))
    const options = { rerank: k % 5 === 0, budgetTokens: [40, 120, 400, 1500][k % 4] }
    if (k % 3 === 1 && sessions.length) options.excludeSessions = [sessions[Math.floor(rand() * sessions.length)]]
    if (k % 4 === 2 && sessions.length) {
      const live = sessions[Math.floor(rand() * sessions.length)]
      options.inContext = t => t.session === live && hash(t.id) % 2 === 0
    }
    return [query, options]
  })
}

async function same(label, memory, store, prompts, rand, n = 40) {
  let compared = 0, nonEmpty = 0, reranked = 0
  for (const [query, options] of queries(rand, store, n)) {
    prompts.length = 0
    const got = await memory.retrieve(query, options)
    const want = await reference(store, query, options)
    const where = `${label}: ${JSON.stringify({ query, ...options, inContext: Boolean(options.inContext) })}`
    assert.deepEqual(got.ids, want.ids, where)
    assert.equal(got.text, want.text, where)
    assert.equal(got.candidates, want.candidates, where)
    assert.equal(got.reranked, want.reranked, where)
    if (want.prompt) {
      assert.deepEqual(prompts, [want.prompt], `${where}: rerank prompt`)
      reranked++
    }
    compared++
    if (want.ids.length) nonEmpty++
  }
  assert.ok(nonEmpty > n / 2, `${label}: most queries return something (${nonEmpty}/${compared})`)
  assert.ok(reranked > 0, `${label}: the rerank path ran`)
}

test('retrieve returns exactly what the uncached algorithm returns, through forgets, captures and new facts', async () => {
  const rand = mulberry32(20261024)
  const store = new MemoryStore(join(mkdtempSync(join(tmpdir(), 'equiv-')), 'memory.db'))
  const prompts = []
  const memory = new Memory(store, embedder, chatFor(prompts), OPTIONS)
  const loads = { units: 0, turns: 0 }
  const units = store.units.bind(store), turns = store.turns.bind(store)
  // Count the store reads that only Memory's caches make (the reference reads through the originals below).
  const counting = { units: () => { loads.units++; return units() }, turns: () => { loads.turns++; return turns() } }
  store.units = counting.units
  store.turns = counting.turns
  const ref = { ...Object.fromEntries(['facts', 'segmentDates', 'links'].map(k => [k, store[k].bind(store)])), units, turns }

  let at = await fill(memory, rand, 40, 0, START)
  memory.closeAll()
  await memory.drain()
  assert.ok(store.stats().facts > 20 && store.stats().links > 0, JSON.stringify(store.stats()))
  await same('initial', memory, ref, prompts, rand)

  // Warm: nothing is read again from the store, also after a backfill that finds nothing to do.
  const warm = { ...loads }
  assert.equal(await memory.backfill(), 0)
  await same('after an empty backfill', memory, ref, prompts, rand, 10)
  assert.deepEqual(loads, warm, 'an empty backfill keeps the caches')

  // New captures append to the ranked index (no reload) and are found.
  at = await fill(memory, rand, 6, 40, at)
  await same('after new captures', memory, ref, prompts, rand)
  assert.deepEqual(loads, warm, 'captures are appended, not reloaded')

  // Forgetting drops every cache; the windows that showed a forgotten message come back with backfill.
  memory.forget({ session: 's7' })
  await same('after forgetting a session', memory, ref, prompts, rand)
  // Cut a session in two: its first message goes, the second stays (its window showed the first).
  const cut = ref.turns().find(t => t.pos === 1 && t.session > 's1' && ref.turns().some(x => x.session === t.session && x.pos === 2))
  memory.forget({ before: cut.at })
  await same('after forgetting before a date', memory, ref, prompts, rand)
  assert.ok(await memory.backfill() > 0, 'a session cut by the date had a window showing a forgotten message')
  await same('after the backfill that follows', memory, ref, prompts, rand)
  await memory.drain()  // narrowed segments are extracted again: new facts while the index is warm
  await same('after re-extraction', memory, ref, prompts, rand)

  // More captures and facts with a warm index.
  at = await fill(memory, rand, 5, 46, at)
  memory.closeAll()
  await memory.drain()
  await same('after new facts', memory, ref, prompts, rand)

  // Units embedded with another model are left out (and a fresh process ranks the same).
  const victim = store.db.prepare("SELECT id FROM units WHERE kind = 'p2' ORDER BY rowid LIMIT 3").all().map(r => r.id)
  for (const u of ref.units().filter(x => victim.includes(x.id))) store.addUnit({ ...u, vector: new Float32Array(8).fill(0.5), model: 'other' })
  const second = new Memory(store, embedder, chatFor(prompts), OPTIONS)
  await same('another model', second, ref, prompts, rand)
  store.close()
})

test('the heap top-k ranks like a full sort, ties included', async () => {
  const { UnitRows } = await import('../lib/unit-index.js')
  const rand = mulberry32(7)
  for (const rows of [0, 1, 5, 99, 100, 101, 1000]) {
    const units = Array.from({ length: rows }, (_, k) => ({ id: `u${k}`, set: 'p2', session: 's', text: '', turns: [`t${k}`], model: 'm',
      vector: vectorFor(sentence(rand, 1 + Math.floor(rand() * 3))) }))
    const vectors = units.map(u => u.vector.slice())
    const index = new UnitRows(DIMS, units)
    for (let q = 0; q < 20; q++) {
      const query = vectorFor(sentence(rand, 2))
      const want = p.top(units.map((_, k) => k), vectors.map(v => p.dot(v, query)))
      assert.deepEqual(index.top(query), want, `rows ${rows}`)
    }
  }
})
