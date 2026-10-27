/**
 * OpenAI-compatible proxy in front of a local model server (llama.cpp, Ollama, LM Studio, vLLM...), for chat apps that
 * cannot load a plugin: point the app at the proxy instead of the server and every chat gets memory, with no tool calls.
 * For each chat completion it stores the conversation's messages, recalls from earlier conversations with the last user
 * message, adds the recall before that message, forwards the request, and stores the reply (streamed or not). Other
 * requests (/v1/models, completions, embeddings) are passed through unchanged.
 *
 * Chat APIs carry no conversation id, and apps resend, trim and reorder history and put the date into system prompts.
 * So a message is known by its role and text (its content key), and a request belongs to the stored conversation that
 * holds the most of its earlier messages; a request with no earlier message starts a new conversation (two chats that
 * both open with "hello" stay apart). The recall leaves out exactly the messages the app sent, wherever they are
 * stored, and goes into the last user message, not the system prompt: the earlier messages stay byte-identical, so the
 * server's prompt cache still covers them.
 */
import { createHash, randomBytes } from 'node:crypto'
import { request as httpRequest, type IncomingMessage, type ServerResponse } from 'node:http'
import { request as httpsRequest } from 'node:https'
import { recallText } from './format.ts'
import type { Memory } from './memory.ts'
import type { MemoryStore, StoredTurn } from './store.ts'

export interface ProxyOptions {
  memory: Memory
  store: MemoryStore
  /** OpenAI-compatible base URL of the model server, e.g. http://127.0.0.1:8080/v1. */
  upstream: string
  /** Sent upstream as a bearer token when set (the client's own Authorization is the proxy's token, never forwarded). */
  apiKey: string
  budgetTokens: number
  /** Runs `work` after the captures already queued, so messages are stored in arrival order. */
  queue<T>(work: () => T): Promise<T>
  /** Embeds the windows of newly stored messages in the background. */
  index(turns: readonly StoredTurn[]): void
  /** Waits (briefly) for indexing in progress, so a message stored just before can be recalled. */
  settled(): Promise<void>
  log(level: string, message: string): void
}

/** Largest request accepted (images sent to a vision model are base64 inside the JSON). */
export const PROXY_MAX_BODY_BYTES = 64 * 1024 * 1024
/** Most reply text kept for storing (the client still gets all of it). */
const MAX_REPLY_CHARS = 1_000_000

/** The text of a chat message's content: a string, or the text parts of an array (images and audio are left out). */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content.map(p => p && typeof p === 'object' && (p as { type?: unknown }).type === 'text'
    ? String((p as { text?: unknown }).text ?? '') : '').filter(Boolean).join('\n')
}

type ChatMessage = Record<string, unknown> & { role?: unknown, content?: unknown }

/** A message's content key: its role and text, not its position. */
export function contentKey(role: string, text: string): string {
  return createHash('sha256').update(JSON.stringify([role, text.trim()])).digest('hex').slice(0, 20)
}

/** The stored id of the `occurrence`-th message with `key` in `session`; its prefix `px-<key>-` finds it in any session. */
export function messageId(key: string, session: string, occurrence: number): string {
  return `px-${key}-${session.slice(-12)}-${occurrence}`
}

/** The messages with `block` put before the text of the last user message. */
export function withRecall(messages: readonly ChatMessage[], block: string): ChatMessage[] {
  const out = messages.slice()
  const last = out.length - 1
  const content = out[last]!.content
  const prefix = `${block}\n\n`
  out[last] = { ...out[last], content: Array.isArray(content)
    ? [{ type: 'text', text: prefix.trimEnd() }, ...content]
    : prefix + (typeof content === 'string' ? content : '') }
  return out
}

/** Open WebUI and similar apps send their own background tasks (titles, tags, follow-ups) as chats: not conversations. */
function isAppTask(text: string): boolean {
  return /^\s*### Task:/.test(text)
}

async function rawBody(req: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = []
  let size = 0
  for await (const chunk of req) {
    size += (chunk as Buffer).length
    if (size > PROXY_MAX_BODY_BYTES) throw Object.assign(new Error(`the body is larger than ${PROXY_MAX_BODY_BYTES} bytes`), { status: 413 })
    chunks.push(chunk as Buffer)
  }
  return Buffer.concat(chunks)
}

function fail(res: ServerResponse, status: number, message: string): void {
  if (res.headersSent) { res.destroy(); return }
  res.writeHead(status, { 'content-type': 'application/json' })
  res.end(JSON.stringify({ error: { message, type: 'proxy_error' } }))
}

/** Collects the reply text from a streamed (server-sent events) or plain JSON chat completion, up to a bound. */
class ReplyText {
  private buffer = ''
  private decoder = new TextDecoder()
  private full = false
  text = ''
  constructor(private readonly streamed: boolean) {}
  push(chunk: Uint8Array): void {
    if (this.full) return
    this.buffer += this.decoder.decode(chunk, { stream: true })
    if (this.streamed) {
      const lines = this.buffer.split('\n')
      this.buffer = lines.pop() ?? ''
      for (const line of lines) this.line(line)
    }
    // A reply (or one line of it) this long is not a chat message to keep.
    if (this.buffer.length > MAX_REPLY_CHARS * 2 || this.text.length > MAX_REPLY_CHARS) { this.full = true; this.text = ''; this.buffer = '' }
  }
  end(): string {
    if (this.full) return ''
    this.buffer += this.decoder.decode()
    if (this.streamed) {
      this.line(this.buffer)
    } else {
      try {
        const value = JSON.parse(this.buffer) as { choices?: Array<{ message?: { content?: unknown } }> }
        this.text = textOf(value.choices?.[0]?.message?.content)
      } catch {}
    }
    return this.text.length > MAX_REPLY_CHARS ? '' : this.text
  }
  private line(line: string): void {
    const data = line.startsWith('data:') ? line.slice(5).trim() : ''
    if (!data || data === '[DONE]') return
    try {
      const value = JSON.parse(data) as { choices?: Array<{ delta?: { content?: unknown } }> }
      const piece = value.choices?.[0]?.delta?.content
      if (typeof piece === 'string') this.text += piece
    } catch {}
  }
}

export function proxyHandler(options: ProxyOptions): (req: IncomingMessage, res: ServerResponse) => Promise<void> {
  const upstream = options.upstream.replace(/\/+$/, '')
  let recallWarned = false

  /**
   * Sends the request upstream and the answer back as it arrives; resolves with the reply text (chat completions).
   * Plain node:http: redirects are passed to the client, never followed (they could lead off this machine), and a
   * long generation on a slow machine is not cut by a client-side timeout.
   */
  function forward(req: IncomingMessage, res: ServerResponse, path: string, body: Buffer | null): Promise<string> {
    return new Promise(resolve => {
      const target = new URL(upstream + path)
      const headers: Record<string, string | number> = {}
      for (const name of ['content-type', 'accept']) {
        const value = req.headers[name]
        if (typeof value === 'string') headers[name] = value
      }
      if (body) headers['content-length'] = body.length
      if (options.apiKey) headers.authorization = `Bearer ${options.apiKey}`
      const send = target.protocol === 'https:' ? httpsRequest : httpRequest
      const out = send(target, { method: req.method, headers }, answer => {
        const type = String(answer.headers['content-type'] ?? 'application/json')
        const streamed = /event-stream/.test(type)
        res.writeHead(answer.statusCode ?? 502, { 'content-type': type, ...(streamed ? { 'cache-control': 'no-cache' } : {}) })
        const reply = new ReplyText(streamed)
        answer.on('data', (chunk: Buffer) => {
          reply.push(chunk)
          if (!res.write(chunk)) { answer.pause(); res.once('drain', () => answer.resume()) }
        })
        answer.on('end', () => {
          res.end()
          const ok = (answer.statusCode ?? 500) >= 200 && (answer.statusCode ?? 500) < 300
          resolve(ok ? reply.end() : '')
        })
        answer.on('error', error => {
          options.log('warn', `proxy: the answer from ${upstream} broke off: ${error.message}`)
          res.destroy()
          resolve('')
        })
      })
      out.on('error', error => {
        if (!res.destroyed) fail(res, 502, `the model server at ${upstream} did not answer: ${error.message}`)
        resolve('')
      })
      res.once('close', () => { if (!res.writableFinished) out.destroy() })
      out.end(body && body.length ? body : undefined)
    })
  }

  /** The stored conversation that holds the most of these earlier messages (ties: the latest), or a new one. */
  function conversation(earlier: ReadonlyArray<{ key: string }>): string {
    const score = new Map<string, { matches: number, last: number }>()
    for (const { key } of new Map(earlier.map(m => [m.key, m])).values()) {
      const sessions = new Map<string, number>()
      for (const t of options.store.turnsWithIdPrefix(`px-${key}-`)) sessions.set(t.session, Math.max(sessions.get(t.session) ?? 0, t.at))
      for (const [session, at] of sessions) {
        const s = score.get(session) ?? { matches: 0, last: 0 }
        score.set(session, { matches: s.matches + 1, last: Math.max(s.last, at) })
      }
    }
    let best: [string, { matches: number, last: number }] | undefined
    for (const entry of score) if (!best || entry[1].matches > best[1].matches || (entry[1].matches === best[1].matches && entry[1].last > best[1].last)) best = entry
    return best?.[0] ?? `chat-${randomBytes(8).toString('hex')}`
  }

  async function chat(req: IncomingMessage, res: ServerResponse, path: string, raw: Buffer): Promise<void> {
    let b: Record<string, unknown>
    try {
      b = JSON.parse(raw.toString('utf8')) as Record<string, unknown>
    } catch {
      return void await forward(req, res, path, raw)  // the server says what is wrong with it
    }
    const messages = Array.isArray(b.messages) ? b.messages as ChatMessage[] : []
    const last = messages.at(-1)
    const query = last?.role === 'user' ? textOf(last.content) : ''
    const off = String(req.headers['x-memory'] ?? '').toLowerCase() === 'off'
    if (!query.trim() || off || isAppTask(query)) return void await forward(req, res, path, raw)

    const said = messages.map(m => ({ role: String(m.role), text: textOf(m.content) }))
      .filter(m => (m.role === 'user' || m.role === 'assistant') && m.text.trim())
      .map(m => ({ ...m, key: contentKey(m.role, m.text) }))
    const seen = new Map<string, number>()
    const ids = new Set<string>()
    const stored = await options.queue(() => {
      const session = conversation(said.slice(0, -1))
      const turns = said.map(m => {
        const occurrence = seen.get(m.key) ?? 0
        seen.set(m.key, occurrence + 1)
        const id = messageId(m.key, session, occurrence)
        ids.add(id)
        return options.memory.record({ id, session, role: m.role, text: m.text })
      })
      return { session, turns }
    })
    options.index(stored.turns.filter((t): t is StoredTurn => t !== null))

    let sent = messages
    try {
      await options.settled()
      // Leave out what the app sent (the model reads it anyway), wherever it is stored.
      const found = await options.memory.retrieve(query, { rerank: false, budgetTokens: options.budgetTokens,
        inContext: t => ids.has(t.id) })
      if (found.text) sent = withRecall(messages, recallText(found.text))
      recallWarned = false
    } catch (error) {
      // The chat goes on without memory (the embedding model may be down); said once until a recall works again.
      if (!recallWarned) options.log('warn', `proxy: recall skipped: ${(error as Error).message}`)
      recallWarned = true
    }
    const reply = await forward(req, res, path, Buffer.from(JSON.stringify({ ...b, messages: sent })))
    if (!reply.trim()) return
    const key = contentKey('assistant', reply)
    const turn = await options.queue(() => options.memory.record({
      id: messageId(key, stored.session, seen.get(key) ?? 0), session: stored.session, role: 'assistant', text: reply }))
    if (turn) options.index([turn])
  }

  return async (req, res) => {
    const path = new URL(req.url ?? '/', 'http://localhost').pathname
    if (!path.startsWith('/v1/')) return fail(res, 404, 'not found: the proxy serves /v1/...')
    const rest = path.slice(3) + (req.url?.includes('?') ? req.url.slice(req.url.indexOf('?')) : '')
    try {
      const raw = req.method === 'GET' || req.method === 'HEAD' ? null : await rawBody(req)
      if (req.method === 'POST' && path === '/v1/chat/completions' && raw) return await chat(req, res, rest, raw)
      await forward(req, res, rest, raw)
    } catch (error) {
      const status = (error as { status?: number }).status ?? 500
      if (status === 500) options.log('warn', `proxy: ${(error as Error).message}`)
      fail(res, status, (error as Error).message)
    }
  }
}
