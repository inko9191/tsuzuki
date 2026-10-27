// Reading conversations exported from ChatGPT and Claude (src/importers.ts), from the zip or from conversations.json.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { parseBackup, parseConversations, readExport, zipEntries } from '../lib/importers.js'
import { makeZip, chatgptExport, claudeExport } from './fixtures/exports.mjs'

test('ChatGPT: the branch on screen, without hidden messages, tool calls or tool output', () => {
  const [c, ...rest] = parseConversations(chatgptExport())
  assert.equal(rest.length, 0, 'a conversation without messages is left out')
  assert.equal(c.id, 'conv-1')
  assert.equal(c.source, 'chatgpt')
  assert.deepEqual(c.messages.map(m => [m.id, m.role, m.text]), [
    ['chatgpt:u1', 'user', 'I adopted a cat named Mochi.'],
    ['chatgpt:a1', 'assistant', 'Mochi is a lovely name!'],
    ['chatgpt:u2', 'user', 'What does she eat? Plot it.'],
    ['chatgpt:a2', 'assistant', 'Cats like Mochi eat tuna.'],
  ])
  assert.equal(c.messages[0].at, 1700000000000)
})

test('Claude: human and assistant messages, text from content parts when the text is empty', () => {
  const [c] = parseConversations(claudeExport())
  assert.equal(c.source, 'claude')
  assert.deepEqual(c.messages.map(m => [m.id, m.role, m.text, m.at]), [
    ['claude:m1', 'user', 'We are going to Hakodate in June.', Date.parse('2026-05-01T09:00:00Z')],
    ['claude:m2', 'assistant', 'Hakodate in June is nice.', Date.parse('2026-05-01T09:00:05Z')],
  ])
})

test('the zip as downloaded (deflated or stored) or the json file; other zips are refused clearly', () => {
  const dir = mkdtempSync(join(tmpdir(), 'import-'))
  const json = Buffer.from(JSON.stringify(chatgptExport()))
  for (const stored of [false, true]) {
    const path = join(dir, `export-${stored}.zip`)
    writeFileSync(path, makeZip({ 'chat.html': Buffer.from('<html>'), 'conversations.json': json }, stored))
    assert.equal(readExport(path)[0].messages.length, 4)
  }
  writeFileSync(join(dir, 'conversations.json'), JSON.stringify(claudeExport()))
  assert.equal(readExport(join(dir, 'conversations.json'))[0].source, 'claude')
  writeFileSync(join(dir, 'photos.zip'), makeZip({ 'a.jpg': Buffer.from('x') }))
  assert.throws(() => readExport(join(dir, 'photos.zip')), /no conversations\.json/)
  const big = makeZip({ 'conversations.json': json })
  big.writeUInt16LE(0xffff, big.length - 22 + 10)
  assert.throws(() => zipEntries(big, () => true), /zip64/)
  assert.throws(() => parseConversations({ not: 'a list' }), /list of conversations/)
})

test('zip: sizes are bounded and entries sharing their data refused (a crafted zip cannot inflate gigabytes)', () => {
  const big = Buffer.alloc(3 * 2 ** 20, 0x20)
  const zip = makeZip({ 'conversations.json': big })
  assert.throws(() => zipEntries(zip, () => true, { entry: 2 ** 20, total: 2 ** 30 }), /too large to import/)
  // The size field lies (says 10 bytes): inflating stops at the limit.
  const lying = Buffer.from(zip)
  const cd = lying.length - 22 - (46 + 'conversations.json'.length)
  lying.writeUInt32LE(10, cd + 24)
  assert.throws(() => zipEntries(lying, () => true, { entry: 2 ** 20, total: 2 ** 30 }), /within the size limit/)
  // Two directory entries pointing at the same local data.
  const one = makeZip({ 'conversations-1.json': Buffer.from('[]') })
  const entry = one.subarray(one.length - 22 - (46 + 'conversations-1.json'.length), one.length - 22)
  const twice = Buffer.concat([one.subarray(0, one.length - 22), entry, one.subarray(one.length - 22)])
  twice.writeUInt16LE(2, twice.length - 22 + 8); twice.writeUInt16LE(2, twice.length - 22 + 10)
  twice.writeUInt32LE(entry.length * 2, twice.length - 22 + 12)
  assert.throws(() => zipEntries(twice, () => true), /share their data/)
})

test('backup: the JSON Lines export reads back as conversations with their ids', () => {
  const lines = [{ session: 'chat-1', id: 'a', role: 'user', text: 'hi', at: 1 }, { session: 'chat-1', id: 'b', role: 'assistant', text: 'yo', at: 2 },
    { session: 'chatgpt-x', id: 'c', role: 'user', text: 'other', at: 3 }, { session: 'chat-1', id: 'd', role: 'tool', text: 'skip' }]
  const back = parseBackup(lines.map(l => JSON.stringify(l)).join('\n'))
  assert.deepEqual(back.map(c => [c.id, c.source, c.messages.map(m => m.id)]), [['chat-1', 'backup', ['a', 'b']], ['chatgpt-x', 'backup', ['c']]])
})
