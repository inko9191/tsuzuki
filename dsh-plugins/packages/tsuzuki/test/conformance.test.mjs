// The TS port must reproduce the evaluated Python pipeline exactly (fixtures from make_fixtures.py).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import * as p from '../lib/pipeline.js'

const fx = JSON.parse(readFileSync(new URL('./fixtures/conformance.json', import.meta.url), 'utf8'))
const words = text => text.split(/\s+/).filter(Boolean).length
const plainUnits = units => units.map(({ id, set, session, text, turns }) => ({ id, set, session, text, turns }))

test('units: p2 windows then facts, same ids, texts and returned turns', () => {
  const facts = new Map(Object.entries(fx.facts))
  const units = [...p.windowUnitsP2(fx.turns), ...p.factUnits(facts)]
  assert.deepEqual(units, plainUnits(fx.units))
})

test('fact prompt and parser', () => {
  const members = fx.turns.filter(t => t.session === 'g#2')
  const { messages, validIds } = p.factMessages(members, t => t.id.split('/')[1])
  assert.deepEqual(messages, fx.fact_messages)
  assert.deepEqual(validIds, fx.fact_valid_ids)
  for (const c of fx.fact_parses) assert.deepEqual(p.parseFacts(c.reply, validIds), c.value, c.reply.slice(0, 40))
})

test('clean follows Python split/join and code-point slicing', () => {
  for (const c of fx.clean) assert.equal(p.clean(c.input), c.value, JSON.stringify(c.input))
})

test('top keeps corpus order on ties', () => {
  const { indices, scores, value } = fx.top_ties
  assert.deepEqual(p.top(indices, scores), value)
})

test('rank, RRF, distinct, rerank prompt, rerank parse/merge and budgeted selection', () => {
  const units = plainUnits(fx.units)
  const lookup = new Map(fx.turns.map(t => [t.id, t]))
  assert.ok(fx.cases.length >= 10)
  for (const c of fx.cases) {
    const vectors = c.vectors.map(v => Float32Array.from(v))
    const fused = p.rank(units, vectors, Float32Array.from(c.query))
    assert.deepEqual(fused.map(([i]) => i), c.fused.map(([i]) => i), `seed ${c.seed} fused order`)
    fused.forEach(([, s], k) => assert.ok(Math.abs(s - c.fused[k][1]) < 1e-6))
    const order = p.distinct(fused.map(([i]) => i), units)
    assert.deepEqual(order, c.order)
    assert.deepEqual(p.rerankMessages('Where do I live now?', order, units, fx.turns), c.rerank_messages)
    for (const r of c.reranks) {
      const picks = p.parseRerank(r.reply, Math.min(order.length, p.RERANK_CANDIDATES))
      assert.deepEqual(picks, r.picks, r.reply)
      const final = p.distinct(p.rerankedOrder(order, picks), units)
      assert.deepEqual(final, r.final)
      const raw = p.rawIds(final, units)
      assert.deepEqual(raw, r.raw)
      assert.deepEqual(p.selectMemory(raw, lookup, ['g/D1:3'], 25, words, p.renderFrozen), r.frozen)
      assert.deepEqual(p.selectMemory(raw, lookup, ['g/D1:3'], 25, words, ts => p.renderDated(ts)), r.dated)
    }
  }
})

test('date order and both renderings', () => {
  const t = fx.turns
  assert.deepEqual(p.sessionOrder(t), fx.session_order)
  assert.equal(p.renderFrozen([t[3], t[0], t[4], t[1]]), fx.render_frozen_mixed)
  assert.equal(p.renderDated([t[3], t[0], t[4], t[1]]), fx.render_dated_mixed)
})

test('write-time update links: table, jobs, prompt, parser, chains, versioned order and notes', () => {
  const u = fx.updates
  const facts = new Map(Object.entries(fx.facts))
  const dates = new Map(fx.turns.map(t => [t.session, t.date]))
  const table = p.factTable(u.session_order.map(s => ({ session: s, date: dates.get(s), facts: facts.get(s) ?? [] })))
  assert.deepEqual(table, u.table)
  const vectors = new Map(Object.entries(u.vectors).map(([k, v]) => [k, Float32Array.from(v)]))
  assert.deepEqual(p.updateJobs(table, vectors), u.jobs)
  const lookup = new Map(table.map(f => [f.key, f]))
  assert.deepEqual(u.jobs.map(j => p.updateMessages(j, lookup)), u.messages)
  for (const c of u.parses) assert.deepEqual(p.parseUpdates(c.reply, u.parse_job), c.value, c.reply)
  const links = [{ old: 'g#2#0', new: 'g#1#0' }, { old: 'g#2#1', new: 'g#1#1' }]
  assert.deepEqual(Object.fromEntries(p.replacements(links, lookup)), u.replacements)
  assert.deepEqual(Object.fromEntries(p.replacements([{ old: 'g#2#0', new: 'g#3#0' }, { old: 'g#3#0', new: 'g#1#0' }], lookup)), u.chain)
  const v = p.versioned(['g/D2:1', 'g/D3:1', 'g/D1:1'], table, links)
  assert.deepEqual({ order: v.order, notes: Object.fromEntries(v.notes) }, u.versioned)
})
