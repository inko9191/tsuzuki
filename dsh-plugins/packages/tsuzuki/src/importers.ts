/**
 * Conversations exported from ChatGPT (Settings > Data controls > Export data) or Claude (Settings > Privacy > Export
 * data), read from the zip the service sends or from its conversations.json, as plain conversations to store.
 *
 * ChatGPT keeps a conversation as a tree (each edit or regenerated reply starts a branch); the branch that was on
 * screen last (`current_node` back to the root) is taken. Hidden messages, tool calls and tool output, and reasoning
 * summaries are left out: only what the user and the assistant said to each other.
 */
import { createHash } from 'node:crypto'
import { closeSync, fstatSync, openSync, readSync } from 'node:fs'
import { inflateRawSync } from 'node:zlib'

export interface ImportedMessage {
  /** The service's own message id (stable across exports, so importing again stores nothing twice). */
  id: string
  role: 'user' | 'assistant'
  text: string
  /** Epoch milliseconds when the message was sent, when the export has it. */
  at?: number
}

export interface ImportedConversation {
  id: string
  title: string
  /** `backup`: the JSON Lines `memory.mjs export` writes; its conversation ids are kept as they were. */
  source: 'chatgpt' | 'claude' | 'backup'
  messages: ImportedMessage[]
}

/** Bytes that can be read at a position: a buffer, or a file read piece by piece (exports can be larger than 2 GB). */
export interface ByteSource {
  size: number
  read(position: number, length: number): Buffer
}

export function bufferSource(buffer: Buffer): ByteSource {
  return { size: buffer.length, read: (position, length) => buffer.subarray(position, position + length) }
}

/** The largest file inside a zip that is read (a JavaScript string cannot be much longer), and the most in all. */
export const ZIP_LIMITS = { entry: 512 * 2 ** 20, total: 2 ** 30 }

/**
 * The files of a zip archive whose names match `wanted` (stored or deflated entries; no zip64, no encryption). Sizes
 * are bounded (`ZIP_LIMITS`), and two entries pointing at the same data are refused, so a crafted zip cannot make it
 * inflate gigabytes.
 */
export function zipEntries(zip: Buffer | ByteSource, wanted: (name: string) => boolean, limits = ZIP_LIMITS): Map<string, Buffer> {
  const src = Buffer.isBuffer(zip) ? bufferSource(zip) : zip
  const tailStart = Math.max(0, src.size - 22 - 0xffff)
  const tail = src.read(tailStart, src.size - tailStart)
  let end = -1
  for (let i = tail.length - 22; i >= 0; i--) {
    if (tail.readUInt32LE(i) === 0x06054b50) { end = i; break }
  }
  if (end < 0) throw new Error('not a zip file (or a damaged one)')
  const count = tail.readUInt16LE(end + 10)
  const cdSize = tail.readUInt32LE(end + 12)
  const cdAt = tail.readUInt32LE(end + 16)
  if (count === 0xffff || cdAt === 0xffffffff || cdSize === 0xffffffff) {
    throw new Error('this zip is in the zip64 format; unzip it and give conversations.json instead')
  }
  if (cdAt + cdSize > src.size) throw new Error('damaged zip (central directory)')
  const cd = src.read(cdAt, cdSize)
  const out = new Map<string, Buffer>()
  const offsets = new Set<number>()
  let total = 0, at = 0
  for (let k = 0; k < count; k++) {
    if (at + 46 > cd.length || cd.readUInt32LE(at) !== 0x02014b50) throw new Error('damaged zip (central directory)')
    const flags = cd.readUInt16LE(at + 8)
    const method = cd.readUInt16LE(at + 10)
    const size = cd.readUInt32LE(at + 20)
    const unpacked = cd.readUInt32LE(at + 24)
    const nameLength = cd.readUInt16LE(at + 28)
    const next = at + 46 + nameLength + cd.readUInt16LE(at + 30) + cd.readUInt16LE(at + 32)
    const name = cd.toString('utf8', at + 46, at + 46 + nameLength)
    const local = cd.readUInt32LE(at + 42)
    if (wanted(name)) {
      if (flags & 1) throw new Error(`${name} is encrypted`)
      if (size === 0xffffffff || unpacked === 0xffffffff || local === 0xffffffff) {
        throw new Error('this zip is in the zip64 format; unzip it and give conversations.json instead')
      }
      if (offsets.has(local)) throw new Error('damaged zip (two entries share their data)')
      offsets.add(local)
      if (unpacked > limits.entry || total + unpacked > limits.total) throw new Error(`${name} is too large to import (${unpacked} bytes)`)
      const header = src.read(local, 30)
      if (header.length < 30 || header.readUInt32LE(0) !== 0x04034b50) throw new Error('damaged zip (local header)')
      const start = local + 30 + header.readUInt16LE(26) + header.readUInt16LE(28)
      if (start + size > src.size) throw new Error('damaged zip (entry data)')
      const data = src.read(start, size)
      let file: Buffer
      if (method === 0) file = Buffer.from(data)
      else if (method === 8) {
        try {
          file = inflateRawSync(data, { maxOutputLength: Math.min(limits.entry, limits.total - total) })
        } catch (error) {
          throw new Error(`${name} could not be unpacked within the size limit (${(error as Error).message})`)
        }
      } else throw new Error(`${name} uses a compression this reader does not know (method ${method}); unzip it and give the .json file`)
      total += file.length
      out.set(name, file)
    }
    at = next
  }
  return out
}

type Json = Record<string, unknown>
const isObject = (v: unknown): v is Json => !!v && typeof v === 'object' && !Array.isArray(v)

function chatgpt(c: Json): ImportedConversation {
  const mapping = isObject(c.mapping) ? c.mapping as Record<string, Json> : {}
  const created = typeof c.create_time === 'number' ? c.create_time * 1000 : undefined
  // The branch on screen: from the current node up to the root.
  const branch: Json[] = []
  const seen = new Set<string>()
  for (let node = c.current_node as string | undefined; node && mapping[node] && !seen.has(node); node = mapping[node]!.parent as string) {
    seen.add(node)
    branch.push(mapping[node]!)
  }
  if (!branch.length) {
    // No current node: every message, in time order.
    branch.push(...Object.values(mapping).sort((a, b) => Number((a.message as Json)?.create_time ?? 0) - Number((b.message as Json)?.create_time ?? 0)).reverse())
  }
  const messages: ImportedMessage[] = []
  for (const node of branch.reverse()) {
    const m = node.message
    if (!isObject(m)) continue
    const role = (m.author as Json | undefined)?.role
    if (role !== 'user' && role !== 'assistant') continue
    if ((m.metadata as Json | undefined)?.is_visually_hidden_from_conversation) continue
    if (role === 'assistant' && m.recipient !== undefined && m.recipient !== 'all') continue  // a tool call
    const content = m.content as Json | undefined
    if (content?.content_type !== 'text' && content?.content_type !== 'multimodal_text') continue
    const text = (Array.isArray(content.parts) ? content.parts : []).filter((p): p is string => typeof p === 'string').join('\n').trim()
    if (!text) continue
    const at = typeof m.create_time === 'number' ? Math.round(m.create_time * 1000) : created
    messages.push({ id: `chatgpt:${String(m.id ?? node.id)}`, role, text, ...(at !== undefined ? { at } : {}) })
  }
  const id = String(c.conversation_id ?? c.id ?? createHash('sha256').update(JSON.stringify([c.title, c.create_time])).digest('hex').slice(0, 16))
  return { id, title: String(c.title ?? ''), source: 'chatgpt', messages }
}

function claude(c: Json): ImportedConversation {
  const messages: ImportedMessage[] = []
  const list = Array.isArray(c.chat_messages) ? c.chat_messages as Json[] : []
  list.forEach((m, i) => {
    const role = m.sender === 'human' ? 'user' : m.sender === 'assistant' ? 'assistant' : null
    if (!role) return
    let text = typeof m.text === 'string' ? m.text : ''
    if (!text.trim() && Array.isArray(m.content)) {
      text = (m.content as Json[]).filter(p => p?.type === 'text' && typeof p.text === 'string').map(p => p.text as string).join('\n')
    }
    text = text.trim()
    if (!text) return
    const at = Date.parse(String(m.created_at ?? ''))
    messages.push({ id: `claude:${String(m.uuid ?? `${String(c.uuid)}-${i}`)}`, role, text, ...(Number.isFinite(at) ? { at } : {}) })
  })
  return { id: String(c.uuid ?? ''), title: String(c.name ?? ''), source: 'claude', messages }
}

/** The conversations in an export's parsed conversations.json (ChatGPT's or Claude's); empty ones are left out. */
export function parseConversations(data: unknown): ImportedConversation[] {
  if (!Array.isArray(data)) throw new Error('conversations.json should be a list of conversations')
  const out: ImportedConversation[] = []
  for (const c of data) {
    if (!isObject(c)) continue
    const conversation = 'mapping' in c ? chatgpt(c) : 'chat_messages' in c ? claude(c) : null
    if (conversation?.id && conversation.messages.length) out.push(conversation)
  }
  return out
}

/** Conversations from the JSON Lines that `memory.mjs export` writes (one message per line, conversation by conversation). */
export function parseBackup(text: string): ImportedConversation[] {
  const byId = new Map<string, ImportedConversation>()
  for (const line of text.split('\n')) {
    if (!line.trim()) continue
    const m = JSON.parse(line) as Json
    if (typeof m.session !== 'string' || (m.role !== 'user' && m.role !== 'assistant') || typeof m.text !== 'string' || !m.text.trim()) continue
    const c: ImportedConversation = byId.get(m.session) ?? { id: m.session, title: '', source: 'backup', messages: [] }
    c.messages.push({ id: String(m.id ?? `${m.session}-${c.messages.length}`), role: m.role, text: m.text, ...(typeof m.at === 'number' ? { at: m.at } : {}) })
    byId.set(m.session, c)
  }
  return [...byId.values()]
}

/**
 * The conversations of an export file: the zip ChatGPT or Claude sends, the conversations.json in it, or a backup made
 * with `memory.mjs export`.
 */
export function readExport(path: string): ImportedConversation[] {
  const fd = openSync(path, 'r')
  try {
    const size = fstatSync(fd).size
    const src: ByteSource = { size, read(position, length) {
      const buffer = Buffer.alloc(Math.max(0, Math.min(length, size - position)))
      let got = 0
      while (got < buffer.length) {
        const n = readSync(fd, buffer, got, buffer.length - got, position + got)
        if (!n) break
        got += n
      }
      return buffer.subarray(0, got)
    } }
    if (size >= 4 && src.read(0, 4).readUInt32LE(0) === 0x04034b50) {
      const entries = zipEntries(src, name => /(^|\/)conversations(-\d+)?\.json$/.test(name))
      if (!entries.size) throw new Error('there is no conversations.json in this zip: is it an export from ChatGPT or Claude?')
      return [...entries.keys()].sort().flatMap(name => parseConversations(JSON.parse(entries.get(name)!.toString('utf8'))))
    }
    if (size > ZIP_LIMITS.entry) throw new Error(`${path} is too large to read at once (${size} bytes)`)
    const text = src.read(0, size).toString('utf8')
    return text.trimStart().startsWith('{') ? parseBackup(text) : parseConversations(JSON.parse(text))
  } finally {
    closeSync(fd)
  }
}
