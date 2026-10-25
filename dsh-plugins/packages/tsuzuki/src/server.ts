#!/usr/bin/env node
/**
 * Memory sidecar: the same memory as the dsh plugin, served over loopback HTTP for hosts that are not dsh
 * (Hermes Agent's memory provider, MCP bridges, scripts).
 *
 *   node lib/server.js --config ~/.hermes/tsuzuki/config.json
 *
 * Prints one JSON line `{"event":"ready","port":N}` when listening. See README "Sidecar API".
 */
import { chmodSync, closeSync, mkdirSync, openSync, readFileSync, writeFileSync, existsSync, realpathSync } from 'node:fs'
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http'
import { createHash, randomBytes, timingSafeEqual } from 'node:crypto'
import { spawn } from 'node:child_process'
import { homedir } from 'node:os'
import { dirname, join, sep } from 'node:path'
import { HttpEmbedder, EMBEDDINGGEMMA_PREFIXES, InputTooLongError, ensureOllamaModel, MODEL_NOTICES, ollamaModels } from './embedding-http.ts'
import { recallText } from './format.ts'
import { ChatUnavailableError, OpenAIChat, LLAMA_CPP_EXTRA } from './llm.ts'
import { EmbeddingTimeoutError, EmbeddingUnavailableError, Memory, describe, forgetScope, within, type ChatModel } from './memory.ts'
import { MemoryStore, type StoredTurn } from './store.ts'

export const VERSION = '0.1.0'

export interface SidecarConfig {
  host: string
  port: number
  /** Bearer token required on every request; mandatory when `host` is not a loopback address. */
  token: string
  dataDir: string
  embedding: { baseUrl: string, model: string, apiKey: string, queryPrefix: string, documentPrefix: string,
    autoPull: boolean, acceptModelLicense: boolean,
    /** Limit of a query embedding (recall, search); storing waits up to 60 s. */
    queryTimeoutMs: number,
    /** Start this repository's egemma-server (Metal engine) on demand instead of using `baseUrl`. */
    engine?: { engineRoot: string, modelDir: string, modelName: string, port: number, python?: string },
    /** Start llama.cpp's llama-server with the EmbeddingGemma GGUF on demand instead of using `baseUrl`. */
    llamacpp?: LlamaSpec }
  chat: { baseUrl: string, model: string, apiKey: string, server: 'llama.cpp' | 'openai', timeoutMs: number, rerankTimeoutMs: number,
    /** Start llama.cpp's llama-server with this chat model (GGUF) on demand instead of using `baseUrl`. */
    llamacpp?: LlamaSpec }
  recall: { budgetTokens: number }
  search: { budgetTokens: number, rerank: boolean }
  facts: { segmentTokens: number, segmentIdleMinutes: number, intervalSeconds: number, updateLinks: boolean }
  capture: { redactSecrets: boolean }
}

/** A llama-server the sidecar starts when it needs it and leaves running (other hosts may share it). */
export interface LlamaSpec {
  /** The model file (GGUF). */
  model: string
  /** The llama-server executable: a path, or a name found on PATH (default `llama-server`). */
  binary?: string
  /** Loopback port (default 8291 for embeddings, 8292 for chat). */
  port?: number
  /** Further arguments; the sidecar adds `-m`, `--host 127.0.0.1`, `--port` and `--alias` itself. */
  args?: string[]
  /** Log file of the server it starts (default: `llama-embedding.log` / `llama-chat.log` in the data folder). */
  log?: string
}

/** EmbeddingGemma: 4 slots of 2,048 tokens (its input limit); one input may fill a slot. */
export const LLAMA_EMBEDDING_ARGS = ['--embedding', '-c', '8192', '-np', '4', '-b', '2048', '-ub', '2048']
/** Fact extraction and update checks: one slot of 8,192 tokens (a segment is ~2,500 tokens plus the reply). */
export const LLAMA_CHAT_ARGS = ['-c', '8192', '-np', '1']

export function defaults(dataDir: string): SidecarConfig {
  return {
    host: '127.0.0.1', port: 8297, token: '', dataDir,
    embedding: { baseUrl: 'http://127.0.0.1:11434/v1', model: 'embeddinggemma', apiKey: '', ...EMBEDDINGGEMMA_PREFIXES,
      autoPull: true, acceptModelLicense: false, queryTimeoutMs: 3000 },
    chat: { baseUrl: '', model: '', apiKey: '', server: 'llama.cpp', timeoutMs: 180000, rerankTimeoutMs: 15000 },
    recall: { budgetTokens: 2048 },
    search: { budgetTokens: 2048, rerank: false },
    facts: { segmentTokens: 2500, segmentIdleMinutes: 30, intervalSeconds: 60, updateLinks: true },
    capture: { redactSecrets: true },
  }
}

/** Defaults overlaid with the file (one level of nesting), so a partial config is valid. */
export function loadConfig(path: string): SidecarConfig {
  const base = defaults(join(dirname(path)))
  if (!existsSync(path)) {
    // A new config gets a random token: only programs that can read this file (the user's own) can use the sidecar.
    const created = { ...base, token: randomBytes(24).toString('hex') }
    mkdirSync(dirname(path), { recursive: true, mode: 0o700 })
    writeFileSync(path, JSON.stringify(created, null, 2) + '\n', { mode: 0o600 })
    return created
  }
  const file = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>
  const out = { ...base } as Record<string, unknown>
  for (const [k, v] of Object.entries(file)) {
    out[k] = v && typeof v === 'object' && !Array.isArray(v) ? { ...(base as unknown as Record<string, object>)[k], ...v } : v
  }
  return out as unknown as SidecarConfig
}

function log(level: string, message: string): void {
  process.stderr.write(`${new Date().toISOString()} ${level} ${message}\n`)
}

/** A request the client got wrong (answered 400, not 500). */
class BadRequest extends Error {}

/** Largest request body accepted (a batch of long messages fits easily); larger ones are answered 413. */
export const MAX_BODY_BYTES = 8 * 1024 * 1024

class TooLarge extends Error {}

async function body(req: IncomingMessage): Promise<Record<string, unknown>> {
  let raw = '', size = 0
  for await (const chunk of req) {
    size += (chunk as Buffer).length
    if (size > MAX_BODY_BYTES) throw new TooLarge(`the body is larger than ${MAX_BODY_BYTES} bytes`)
    raw += chunk
  }
  if (!raw) return {}
  let value: unknown
  try {
    value = JSON.parse(raw)
  } catch (error) {
    throw new BadRequest(`malformed JSON: ${(error as Error).message}`)
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new BadRequest('the body must be a JSON object')
  return value as Record<string, unknown>
}

/** `promise`, or the signal's reason once it aborts (the promise keeps running). */
function abortable<T>(promise: Promise<T>, signal: AbortSignal | undefined): Promise<T> {
  if (!signal) return promise
  if (signal.aborted) return Promise.reject(signal.reason)
  return new Promise((resolve, reject) => {
    const stop = () => reject(signal.reason)
    signal.addEventListener('abort', stop, { once: true })
    promise.then(v => { signal.removeEventListener('abort', stop); resolve(v) }, e => { signal.removeEventListener('abort', stop); reject(e) })
  })
}

const sleep = (ms: number, signal?: AbortSignal) => abortable(new Promise<void>(r => setTimeout(r, ms)), signal)

function send(res: ServerResponse, status: number, value: unknown): void {
  res.writeHead(status, { 'content-type': 'application/json' })
  res.end(JSON.stringify(value))
}

/**
 * Keeps an egemma-server answering on its port: reuses one that serves the same model (e.g. the dsh provider's),
 * otherwise starts it detached so other hosts can share it. A start that cannot work (no Python, no model directory,
 * a spawn error, a server that exits while loading) fails `ensure()` with the reason instead of taking the sidecar down.
 */
export class EngineServer {
  private starting: Promise<void> | null = null
  private checked = 0

  constructor(private readonly spec: NonNullable<SidecarConfig['embedding']['engine']>, private readonly logDir: string) {}

  get baseUrl(): string {
    return `http://127.0.0.1:${this.spec.port}/v1`
  }

  get model(): string {
    return this.spec.modelName
  }

  private async health(): Promise<{ status?: string, model?: string } | null> {
    try {
      const r = await fetch(`http://127.0.0.1:${this.spec.port}/health`, { signal: AbortSignal.timeout(2000) })
      return await r.json() as { status?: string, model?: string }
    } catch {
      return null
    }
  }

  /**
   * Resolves once the server reports `ready` with this model; starts it when nothing answers. `signal` only stops the
   * caller's wait: a start in progress goes on, so the next call finds the server ready.
   */
  ensure(signal?: AbortSignal): Promise<void> {
    if (Date.now() - this.checked < 5000) return Promise.resolve()
    if (!this.starting) {
      const run = this.start()
      this.starting = run.finally(() => { this.starting = null })
      // A caller that gave up does not wait for the start; its failure must not go unhandled (that ends Node).
      this.starting.catch(() => {})
    }
    return abortable(this.starting, signal)
  }

  private async start(): Promise<void> {
    let h = await this.health()
    if (h && h.model && h.model !== this.spec.modelName) {
      throw new Error(`port ${this.spec.port} serves ${h.model}, not ${this.spec.modelName}`)
    }
    let failure: Error | null = null
    const logPath = join(this.logDir, 'egemma-server.log')
    if (!h) {
      const python = this.spec.python || join(this.spec.engineRoot, '.venv/bin/python')
      if (!existsSync(python)) {
        throw new Error(`cannot start egemma-server: ${python} does not exist (build the engine as the README says, set `
          + '"embedding.engine.python", or remove "embedding.engine" from the config to use Ollama)')
      }
      if (!existsSync(this.spec.modelDir)) {
        throw new Error(`cannot start egemma-server: model directory ${this.spec.modelDir} does not exist (download `
          + `${this.spec.modelName} as the README says)`)
      }
      const out = openSync(logPath, 'a')
      try {
        const child = spawn(python, ['-m', 'egemma_server', '--model-dir', this.spec.modelDir, '--model-name', this.spec.modelName,
          '--port', String(this.spec.port)], { cwd: this.spec.engineRoot, detached: true, stdio: ['ignore', out, out],
          env: { ...process.env, PYTHONPATH: join(this.spec.engineRoot, 'server') } })
        // Without a listener, a spawn error (EACCES, ENOENT) is an uncaught 'error' event and ends the sidecar.
        child.once('error', error => { failure ??= new Error(`cannot start egemma-server (${python}): ${error.message}`) })
        child.once('exit', (code, sig) => {
          failure ??= new Error(`egemma-server exited (${code ?? sig}) before it was ready; see ${logPath}`)
        })
        child.unref()
        if (child.pid) log('info', `started egemma-server (pid ${child.pid}) for ${this.spec.modelName} on port ${this.spec.port}`)
      } finally {
        closeSync(out)  // the child has its own copy
      }
    }
    for (let i = 0; i < 240 && h?.status !== 'ready'; i++) {
      if (h?.status === 'failed') throw new Error(`egemma-server failed to load (see ${logPath})`)
      if (failure && !h) throw failure  // nothing else (another host's server) took the port meanwhile
      await new Promise(r => setTimeout(r, 500))
      h = await this.health()
    }
    if (h?.status !== 'ready') throw new Error(`egemma-server not ready after 120 s (see ${logPath})`)
    this.checked = Date.now()
  }
}

const LOOPBACK = new Set(['127.0.0.1', 'localhost', '::1'])

/**
 * Why a request must be refused, or null. The sidecar is for local programs only: a web page must not be able to
 * write or read memory through the user's browser. Browsers send `Origin` on cross-site requests, a DNS-rebinding
 * page arrives with a foreign `Host`, and only JSON POSTs are accepted (a browser can send `text/plain` without a
 * CORS preflight, never `application/json`).
 */
/**
 * llama.cpp's llama-server, started on demand like egemma-server and left running when the sidecar stops (another
 * sidecar may use it; the next start reuses it). `--alias` names it, so a different server on the port is not used.
 */
export class LlamaServer {
  private starting: Promise<void> | null = null
  private checked = 0
  private readonly port: number

  constructor(private readonly spec: LlamaSpec, defaultPort: number, private readonly defaultArgs: readonly string[],
    readonly model: string, private readonly logPath: string) {
    this.port = spec.port ?? defaultPort
  }

  get baseUrl(): string {
    return `http://127.0.0.1:${this.port}/v1`
  }

  /** 'ready' (our model answers), 'loading', 'other' (another server holds the port), or null (nothing answers). */
  private async probe(): Promise<'ready' | 'loading' | 'other' | null> {
    try {
      const r = await fetch(`http://127.0.0.1:${this.port}/health`, { signal: AbortSignal.timeout(2000) })
      await r.body?.cancel()
      if (r.status === 503) return 'loading'
      if (!r.ok) return 'other'
      const m = await (await fetch(`${this.baseUrl}/models`, { signal: AbortSignal.timeout(2000) })).json() as { data?: Array<{ id?: string }> }
      return m.data?.some(d => d.id === this.model) ? 'ready' : 'other'
    } catch {
      return null
    }
  }

  ensure(signal?: AbortSignal): Promise<void> {
    if (Date.now() - this.checked < 5000) return Promise.resolve()
    if (!this.starting) {
      const run = this.start()
      this.starting = run.finally(() => { this.starting = null })
      this.starting.catch(() => {})  // a caller that gave up does not wait; an unhandled failure would end Node
    }
    return abortable(this.starting, signal)
  }

  private async start(): Promise<void> {
    let state = await this.probe()
    if (state === 'other') throw new Error(`port ${this.port} is used by another server, not llama-server "${this.model}"; set another "port"`)
    let failure: Error | null = null
    if (state === null) {
      if (!existsSync(this.spec.model)) throw new Error(`model file ${this.spec.model} does not exist (run node scripts/setup.mjs)`)
      const binary = this.spec.binary || 'llama-server'
      const out = openSync(this.logPath, 'a', 0o600)
      try {
        const child = spawn(binary, ['-m', this.spec.model, ...(this.spec.args ?? this.defaultArgs), '--host', '127.0.0.1',
          '--port', String(this.port), '--alias', this.model], { detached: true, stdio: ['ignore', out, out], windowsHide: true })
        child.once('error', error => { failure ??= new Error(`cannot start llama-server (${binary}): ${error.message}`) })
        child.once('exit', (code, sig) => { failure ??= new Error(`llama-server exited (${code ?? sig}) before it was ready; see ${this.logPath}`) })
        child.unref()
        if (child.pid) log('info', `started llama-server (pid ${child.pid}) for ${this.model} on port ${this.port}`)
      } finally {
        closeSync(out)
      }
    }
    // A 3 GB model from a slow disk can take minutes to load.
    for (let i = 0; i < 600 && state !== 'ready'; i++) {
      if (state === 'other') throw new Error(`port ${this.port} is used by another server, not llama-server "${this.model}"`)
      if (failure && state === null) throw failure
      await new Promise(r => setTimeout(r, 500))
      state = await this.probe()
    }
    if (state !== 'ready') throw new Error(`llama-server "${this.model}" not ready after 300 s (see ${this.logPath})`)
    this.checked = Date.now()
  }
}

/** The chat model behind a started-on-demand llama-server; a server that cannot start reads as unreachable (facts wait). */
function managedChat(inner: OpenAIChat, server: LlamaServer): ChatModel & { busy(): Promise<boolean> } {
  return {
    baseUrl: inner.baseUrl,
    async complete(messages, options) {
      try {
        await server.ensure(options.signal)
      } catch (error) {
        if (options.signal?.aborted) throw error
        throw new ChatUnavailableError(`cannot start the chat model: ${describe(error)}`)
      }
      return inner.complete(messages, options)
    },
    reachable: (signal?: AbortSignal) => inner.reachable(signal),
    busy: () => inner.busy(),
  }
}

/** Constant-time comparison (of digests, so the lengths are equal). */
function sameSecret(given: string, expected: string): boolean {
  const digest = (v: string) => createHash('sha256').update(v).digest()
  return timingSafeEqual(digest(given), digest(expected))
}

export function refusal(req: IncomingMessage, config: Pick<SidecarConfig, 'host' | 'token'>, port: number): [number, string] | null {
  if (req.headers.origin !== undefined) return [403, 'browser requests are not accepted']
  if (LOOPBACK.has(config.host)) {
    const allowed = new Set([`127.0.0.1:${port}`, `localhost:${port}`, `[::1]:${port}`])
    if (!allowed.has(String(req.headers.host ?? '').toLowerCase())) return [403, 'unexpected Host header']
  }
  if (config.token && !sameSecret(String(req.headers.authorization ?? ''), `Bearer ${config.token}`)) return [401, 'missing or wrong token']
  if (req.method === 'POST' && !/^application\/json\b/i.test(String(req.headers['content-type'] ?? ''))) {
    return [415, 'Content-Type must be application/json']
  }
  return null
}

export async function start(config: SidecarConfig): Promise<{ close(): Promise<void>, port: number }> {
  if (!LOOPBACK.has(config.host) && !config.token) {
    throw new Error(`refusing to listen on ${config.host} without a token (set "token" in the config)`)
  }
  mkdirSync(config.dataDir, { recursive: true, mode: 0o700 })
  const dataDir = realpathSync(config.dataDir)
  const store = new MemoryStore(join(config.dataDir, 'memory.db'))
  const engine = config.embedding.engine ? new EngineServer(config.embedding.engine, config.dataDir)
    : config.embedding.llamacpp ? new LlamaServer(config.embedding.llamacpp, 8291, LLAMA_EMBEDDING_ARGS, 'tsuzuki-embeddinggemma',
      config.embedding.llamacpp.log || join(config.dataDir, 'llama-embedding.log'))
      : null
  const embeddingUrl = engine ? engine.baseUrl : config.embedding.baseUrl
  const http = new HttpEmbedder({ ...config.embedding, ...(engine ? { baseUrl: engine.baseUrl, model: engine.model } : {}),
    timeoutMs: 60000, batch: 32 })
  const embedder = engine ? {
    model: () => http.model(),
    async embed(input: readonly string[], task: 'query' | 'document', signal?: AbortSignal) {
      await engine.ensure(signal)
      try {
        return await http.embed(input, task, signal)
      } catch (error) {
        // The input, not the server (a restart cannot help); or the caller stopped waiting.
        if (error instanceof InputTooLongError || signal?.aborted) throw error
        await sleep(1000, signal)
        await engine.ensure(signal)  // the server may have been stopped by its other owner; start it again once
        return await http.embed(input, task, signal)
      }
    },
  } : http
  const chatServer = config.chat.llamacpp ? new LlamaServer(config.chat.llamacpp, 8292, LLAMA_CHAT_ARGS, 'tsuzuki-chat',
    config.chat.llamacpp.log || join(config.dataDir, 'llama-chat.log')) : null
  const chatUrl = chatServer ? chatServer.baseUrl : config.chat.baseUrl
  const chatClient = chatUrl ? new OpenAIChat({ baseUrl: chatUrl, model: chatServer ? chatServer.model : config.chat.model,
    apiKey: config.chat.apiKey, timeoutMs: config.chat.timeoutMs,
    extraBody: chatServer || config.chat.server === 'llama.cpp' ? LLAMA_CPP_EXTRA : {} }) : null
  const chat = chatClient && chatServer ? managedChat(chatClient, chatServer) : chatClient
  let ready: Promise<void> = Promise.resolve()
  const memory = new Memory(store, embedder, chat, {
    budgetTokens: config.recall.budgetTokens, segmentTokens: config.facts.segmentTokens,
    segmentIdleMinutes: config.facts.segmentIdleMinutes, updateLinks: config.facts.updateLinks,
    redactSecrets: config.capture.redactSecrets, log: (level, m) => log(level, m),
    queryTimeoutMs: config.embedding.queryTimeoutMs, rerankTimeoutMs: config.chat.rerankTimeoutMs,
    onEmbeddingRecovered: () => { ready = backfill() },
  })

  // Ollama only: pull the model when it is missing (EmbeddingGemma 300m only after the user accepted its terms). The
  // license notice is shown only when the endpoint is an Ollama server that lacks the model.
  let embeddingReady: Promise<void> = Promise.resolve()
  if (config.embedding.autoPull && !engine) {
    const base = config.embedding.model.split(':')[0]!
    const notice = MODEL_NOTICES[base]
    const gated = base === 'embeddinggemma'
    embeddingReady = (async () => {
      try {
        const models = await ollamaModels(config.embedding.baseUrl)
        if (!models || models.some(name => name.split(':')[0] === base)) return  // not Ollama (or not up), or present
        if (gated && !config.embedding.acceptModelLicense) {
          log('warn', `${notice} Set "embedding.acceptModelLicense": true in the config to let this server pull it, `
            + `or run "ollama pull ${config.embedding.model}" yourself.`)
          return
        }
        if (notice) {
          writeFileSync(join(config.dataDir, 'MODEL-NOTICE.txt'), notice + '\n')
          log('info', notice)
        }
        const state = await ensureOllamaModel(config.embedding.baseUrl, config.embedding.model, m => log('info', m))
        if (state === 'pulled') log('info', `${config.embedding.model} is ready`)
      } catch (error) {
        log('warn', `embedding model check skipped (${embeddingUrl}): ${describe(error)}`)
      }
    })()
  }
  let stopping = false
  const quiet = (e: unknown) => stopping || e instanceof EmbeddingUnavailableError || e instanceof EmbeddingTimeoutError
  // An outage is logged once by Memory (with the URL); backfill and index failures caused by it are not repeated.
  const backfill = () => memory.backfill().then(n => { if (n) log('info', `indexed ${n} pending units`) },
    e => { if (!quiet(e)) log('warn', `backfill: ${describe(e)}`) })
  // One embedding at start, so /health (and the log) tell at once whether the embedding model answers.
  ready = embeddingReady.then(async () => { if (await memory.probeEmbedding()) await backfill() })
  const timers = [
    setInterval(() => { ready = backfill() }, 600_000),
    // While the embedding model is down, ask it every 30 s; the first answer starts the backfill (onEmbeddingRecovered).
    setInterval(() => { if (memory.embeddingState().ok === false) void memory.probeEmbedding() }, 30_000),
  ]
  let tick = () => {}
  if (chat) {
    tick = () => { void ready.then(() => memory.drain(async () => (chatServer !== null || config.chat.server === 'llama.cpp') && await chat.busy()))
      .catch(e => log('warn', `facts: ${String(e)}`)) }
    tick()
    timers.push(setInterval(tick, config.facts.intervalSeconds * 1000))
  }
  for (const t of timers) t.unref?.()

  let chain: Promise<void> = Promise.resolve()
  /** Windows of stored messages, embedded after the reply; skipped while the model is down (backfill does them then). */
  let indexing: Promise<void> = Promise.resolve()
  const indexTurns = async (turns: readonly StoredTurn[]) => {
    for (const turn of turns) {
      if (memory.embeddingState().ok === false) return
      try {
        await memory.indexTurn(turn)
      } catch (error) {
        if (!quiet(error)) log('warn', `index: ${describe(error)}`)
      }
    }
  }
  /** Waited for before a recall at most this long: a slow backfill must not hold the user's turn. */
  const READY_WAIT_MS = 500
  let boundPort = config.port
  const server = createServer(async (req, res) => {
    try {
      const refused = refusal(req, config, boundPort)
      if (refused) {
        req.resume()
        return send(res, refused[0], { error: refused[1] })
      }
      const url = new URL(req.url ?? '/', 'http://localhost')
      if (req.method === 'GET' && url.pathname === '/health') {
        // `status` is about this process; whether memory works is in `embedding` / `chat` (ok, lastError) and `unindexed`.
        const state = memory.status()
        return send(res, 200, { status: 'ok', version: VERSION, dataDir, stats: { ...memory.stats(), unindexed: state.unindexed },
          embedding: { model: embedder.model(), url: embeddingUrl, ...state.embedding },
          chat: chat ? { url: chatUrl, ...state.chat } : null })
      }
      if (req.method !== 'POST') return send(res, 404, { error: 'not found' })
      const b = await body(req)
      if (url.pathname === '/v1/memory/messages') {
        const session = String(b.session ?? '')
        const messages = Array.isArray(b.messages) ? b.messages as Array<Record<string, unknown>> : []
        if (!session) return send(res, 400, { error: 'session is required' })
        const base = Date.now()
        const items = messages.filter(m => (m.role === 'user' || m.role === 'assistant') && typeof m.text === 'string' && m.text.trim())
        // Answered once the messages are in the store; their windows are embedded afterwards, so a slow or hung
        // embedding model never holds the host (Hermes waits for this at the end of a session).
        const work = chain.then(() => items.map((m, k) => {
          const at = typeof m.at === 'number' ? m.at : base + k
          const id = typeof m.id === 'string' && m.id ? m.id
            : createHash('sha256').update(`${session}|${at}|${k}|${String(m.text)}`).digest('hex').slice(0, 16)
          return memory.record({ id, session, role: String(m.role), text: String(m.text), at })
        }))
        chain = work.then(() => {}, e => log('warn', `capture: ${describe(e)}`))
        const turns = (await work).filter(t => t !== null)
        indexing = indexing.then(() => indexTurns(turns))
        return send(res, 200, { stored: items.length })
      }
      if (url.pathname === '/v1/memory/recall' || url.pathname === '/v1/memory/search') {
        const query = String(b.query ?? '')
        if (!query.trim()) return send(res, 200, { text: '', ids: [], reranked: false })
        // A client that gives up (Hermes' own timeout) stops the model requests made for it.
        const gone = new AbortController()
        res.once('close', () => { if (!res.writableFinished) gone.abort(new Error('the client closed the request')) })
        await within(Promise.all([ready, indexing]), READY_WAIT_MS)
        const search = url.pathname.endsWith('search')
        try {
          const found = await memory.retrieve(query, {
            rerank: search ? (typeof b.rerank === 'boolean' ? b.rerank : config.search.rerank) : false,
            excludeSessions: !search && b.session ? [String(b.session)] : [],
            budgetTokens: typeof b.budget_tokens === 'number' ? b.budget_tokens : (search ? config.search : config.recall).budgetTokens,
            signal: gone.signal,
          })
          return send(res, 200, { text: found.text ? recallText(found.text) : '', ids: found.ids, reranked: found.reranked })
        } catch (error) {
          // The query could not be embedded: say so (503 and the cause), so a client can tell this from "no memories".
          if (error instanceof EmbeddingTimeoutError) return send(res, 503, { error: error.message, cause: 'embedding-timeout' })
          if (error instanceof EmbeddingUnavailableError) return send(res, 503, { error: error.message, cause: 'embedding-unavailable' })
          throw error
        }
      }
      if (url.pathname === '/v1/memory/flush') {
        if (b.session) memory.maybeCloseSegment(String(b.session), true)
        else memory.closeAll()
        tick()  // extract the closed segment's facts now, not at the next interval
        return send(res, 200, { ok: true })
      }
      if (url.pathname === '/v1/memory/forget') {
        const scope = forgetScope(b && typeof b === 'object' && !Array.isArray(b) ? b : {})
        if (typeof scope === 'string') return send(res, 400, { error: scope })
        // After the captures already queued, so a message sent just before is forgotten too.
        const work = chain.then(() => memory.forget(scope))
        chain = work.then(() => {}, e => log('warn', `forget: ${String(e)}`))
        const result = await work
        ready = backfill()  // windows of kept messages that showed a forgotten one
        return send(res, 200, result)
      }
      return send(res, 404, { error: 'not found' })
    } catch (error) {
      if (error instanceof BadRequest) return send(res, 400, { error: error.message })
      if (error instanceof TooLarge) { res.setHeader('connection', 'close'); return send(res, 413, { error: error.message }) }
      log('warn', `request failed: ${describe(error)}`)
      return send(res, 500, { error: String(error) })
    }
  })
  await new Promise<void>((resolve, reject) => { server.once('error', reject); server.listen(config.port, config.host, resolve) })
  const port = (server.address() as { port: number }).port
  boundPort = port
  return {
    port,
    async close() {
      stopping = true
      for (const t of timers) clearInterval(t)
      const closed = new Promise<void>(resolve => server.close(() => resolve()))
      memory.close()  // aborts model requests in flight, so open requests and queued indexing end now
      await closed
      await chain
      await indexing
      memory.closeAll()
      store.close()
    },
  }
}

/**
 * The config used without `--config`: `.tsuzuki/config.json` in the home directory. `os.homedir()` is HOME on macOS and
 * Linux and the user profile on Windows, which normally has no HOME (`HOME ?? '.'` put the config, and the memory, in
 * whatever directory the sidecar was started from).
 */
export function defaultConfigPath(): string {
  return join(homedir(), '.tsuzuki', 'config.json')
}

/** Command-line entry (also used by the `tsuzuki-memory` bin). */
export async function main(): Promise<void> {
  const at = process.argv.indexOf('--config')
  const path = at >= 0 ? process.argv[at + 1]! : defaultConfigPath()
  const config = loadConfig(path)
  const running = await start(config)
  process.stdout.write(JSON.stringify({ event: 'ready', port: running.port, version: VERSION }) + '\n')
  const stop = () => { void running.close().finally(() => process.exit(0)) }
  process.on('SIGTERM', stop)
  process.on('SIGINT', stop)
}

/**
 * Whether node runs this file as its script (`node lib/server.js`, as the Hermes provider starts it) rather than
 * importing it (the bin, tests). Node gives the script as an absolute path with the platform's separator, so on
 * Windows it ends in `\server.js`.
 */
export function isMainScript(script: string | undefined, moduleUrl: string, separator: string = sep): boolean {
  return moduleUrl === `file://${script}` || !!script?.endsWith(`${separator}server.js`)
}

if (isMainScript(process.argv[1], import.meta.url)) {
  main().catch(error => { log('error', String(error?.stack ?? error)); process.exit(1) })
}
