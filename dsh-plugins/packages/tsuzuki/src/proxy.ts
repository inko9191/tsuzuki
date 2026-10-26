/**
 * OpenAI-compatible proxy in front of a local model server (llama.cpp, Ollama, LM Studio, vLLM...), for chat apps that
 * cannot load a plugin: point the app at the proxy instead of the server and every chat gets memory, with no tool calls.
 * For each chat completion it stores the conversation's messages, recalls from earlier conversations with the last user
 * message, adds the recall before that message, forwards the request, and stores the reply (streamed or not). Other
 * requests (/v1/models, completions, embeddings) are passed through unchanged.
 *
 * A conversation is recognized by its system prompt and first user message (chat APIs carry no conversation id), and
 * each message by its position and text, so the history a client sends again is stored once. The recall goes into the
 * last user message, not the system prompt: the earlier messages stay byte-identical, so the server's prompt cache
 * still covers them.
 */
import { createHash } from 'node:crypto'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { recallText } from './format.ts'
import type { Memory } from './memory.ts'
import type { StoredTurn } from './store.ts'

export interface ProxyOptions {
  memory: Memory
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

/** The text of a chat message's content: a string, or the text parts of an array (images and audio are left out). */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content.map(p => p && typeof p === 'object' && (p as { type?: unknown }).type === 'text'
    ? String((p as { text?: unknown }).text ?? '') : '').filter(Boolean).join('\n')
}

type ChatMessage = Record<string, unknown> & { role?: unknown, content?: unknown }

/** The conversation's id: its system prompt and first user message (a new chat in the app gets a new id). */
export function conversationId(messages: readonly ChatMessage[]): string {
  const system = messages.filter(m => m.role === 'system' || m.role === 'developer').map(m => textOf(m.content)).join('\n')
  const first = textOf(messages.find(m => m.role === 'user')?.content)
  return `chat-${createHash('sha256').update(JSON.stringify([system, first])).digest('hex').slice(0, 16)}`
}

/** A message's id: the conversation, its position, role and text (a regenerated reply at the same position is another). */
export function messageId(session: string, position: number, role: string, text: string): string {
  return createHash('sha256').update(JSON.stringify([session, position, role, text.trim()])).digest('hex').slice(0, 16)
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

/** Collects the reply text from a streamed (server-sent events) or plain JSON chat completion. */
class ReplyText {
  private buffer = ''
  private decoder = new TextDecoder()
  text = ''
  constructor(private readonly streamed: boolean) {}
  push(chunk: Uint8Array): void {
    this.buffer += this.decoder.decode(chunk, { stream: true })
    if (!this.streamed) return
    const lines = this.buffer.split('\n')
    this.buffer = lines.pop() ?? ''
    for (const line of lines) this.line(line)
  }
  end(): string {
    this.buffer += this.decoder.decode()
    if (this.streamed) {
      this.line(this.buffer)
    } else {
      try {
        const value = JSON.parse(this.buffer) as { choices?: Array<{ message?: { content?: unknown } }> }
        this.text = textOf(value.choices?.[0]?.message?.content)
      } catch {}
    }
    return this.text
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

  /** Sends the request upstream and the answer back as it arrives; resolves with the reply text (chat completions). */
  async function forward(req: IncomingMessage, res: ServerResponse, path: string, body: Buffer | null): Promise<string> {
    const headers: Record<string, string> = {}
    for (const name of ['content-type', 'accept']) {
      const value = req.headers[name]
      if (typeof value === 'string') headers[name] = value
    }
    if (options.apiKey) headers.authorization = `Bearer ${options.apiKey}`
    const gone = new AbortController()
    res.once('close', () => { if (!res.writableFinished) gone.abort(new Error('the client closed the request')) })
    let answer: Response
    try {
      answer = await fetch(upstream + path, { method: req.method, headers, body: body && body.length ? new Uint8Array(body) : undefined,
        signal: gone.signal })
    } catch (error) {
      fail(res, 502, `the model server at ${upstream} did not answer: ${(error as Error).message}`)
      return ''
    }
    const type = answer.headers.get('content-type') ?? 'application/json'
    res.writeHead(answer.status, { 'content-type': type, ...(/event-stream/.test(type) ? { 'cache-control': 'no-cache' } : {}) })
    const reply = new ReplyText(/event-stream/.test(type))
    try {
      if (answer.body) {
        for await (const chunk of answer.body as unknown as AsyncIterable<Uint8Array>) {
          res.write(chunk)
          reply.push(chunk)
        }
      }
    } catch (error) {
      if (!gone.signal.aborted) options.log('warn', `proxy: the answer from ${upstream} broke off: ${(error as Error).message}`)
      res.destroy()
      return ''
    }
    res.end()
    return answer.ok ? reply.end() : ''
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

    const session = conversationId(messages)
    const said = messages.map((m, position) => ({ position, role: String(m.role), text: textOf(m.content) }))
      .filter(m => (m.role === 'user' || m.role === 'assistant') && m.text.trim())
    const stored = await options.queue(() => said.map(m => options.memory.record({
      id: messageId(session, m.position, m.role, m.text), session, role: m.role, text: m.text })))
    options.index(stored.filter((t): t is StoredTurn => t !== null))

    let sent = messages
    try {
      await options.settled()
      const found = await options.memory.retrieve(query, { rerank: false, excludeSessions: [session], budgetTokens: options.budgetTokens })
      if (found.text) sent = withRecall(messages, recallText(found.text))
      recallWarned = false
    } catch (error) {
      // The chat goes on without memory (the embedding model may be down); said once until a recall works again.
      if (!recallWarned) options.log('warn', `proxy: recall skipped: ${(error as Error).message}`)
      recallWarned = true
    }
    const reply = await forward(req, res, path, Buffer.from(JSON.stringify({ ...b, messages: sent })))
    if (!reply.trim()) return
    const position = messages.length
    const turn = await options.queue(() => options.memory.record({
      id: messageId(session, position, 'assistant', reply), session, role: 'assistant', text: reply }))
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
