/**
 * Chat model for fact extraction and reranking over an OpenAI-compatible endpoint (the local llama.cpp
 * Qwen server in the evaluated setup): temperature 0, seed 0, JSON-schema constrained, thinking off.
 */
import type { ChatMessage } from './pipeline.ts'
import type { ChatModel } from './memory.ts'

export interface OpenAIChatConfig {
  /** e.g. `http://127.0.0.1:8080/v1`. */
  baseUrl: string
  /** Model id; empty: the first id from `GET {baseUrl}/models`. */
  model: string
  apiKey: string
  timeoutMs: number
  /** Extra request fields; the llama.cpp defaults keep the evaluated settings. */
  extraBody: Record<string, unknown>
}

export const LLAMA_CPP_EXTRA = { cache_prompt: true, chat_template_kwargs: { enable_thinking: false } }

/** The chat server is not reachable or not serving yet; background work waits instead of spending an attempt. */
export class ChatUnavailableError extends Error {
  override readonly name = 'ChatUnavailableError'
}

/** A connection reset or closed by the other side (undici: `ECONNRESET`, `EPIPE`, `UND_ERR_SOCKET`). */
export function socketReset(error: unknown): boolean {
  const cause = (error as { cause?: { code?: unknown, message?: unknown } } | null)?.cause
  const code = typeof cause?.code === 'string' ? cause.code : ''
  return code === 'ECONNRESET' || code === 'EPIPE' || code === 'UND_ERR_SOCKET' || /other side closed/.test(String(cause?.message ?? ''))
}

/**
 * `fetch` that reports an unreachable server (or one still loading) as {@link ChatUnavailableError}. A request that hit a
 * connection the server had closed is sent once more first (chat requests have no side effects).
 */
async function call(url: string, init: RequestInit, caller: AbortSignal | undefined, timeoutMs: number): Promise<Response> {
  let response: Response
  for (let attempt = 1; ; attempt++) {
    try {
      response = await fetch(url, init)
      break
    } catch (error) {
      if (caller?.aborted) throw error
      if ((error as Error).name === 'TimeoutError') {
        throw Object.assign(new Error(`${url}: no answer within ${timeoutMs} ms`, { cause: error }), { name: 'TimeoutError' })
      }
      // A kept-alive connection the server had already closed: the request never reached it, so send it once more.
      if (attempt === 1 && socketReset(error)) continue
      throw new ChatUnavailableError(`cannot reach ${url}: ${String((error as Error).cause ?? error)}`, { cause: error })
    }
  }
  if (response.status === 502 || response.status === 503) {
    throw new ChatUnavailableError(`${url}: HTTP ${response.status} ${(await response.text()).slice(0, 200)}`)
  }
  return response
}

export class OpenAIChat implements ChatModel {
  private resolved: string | null = null

  constructor(private readonly config: OpenAIChatConfig) {}

  private headers(): Record<string, string> {
    return { 'content-type': 'application/json', ...(this.config.apiKey ? { authorization: `Bearer ${this.config.apiKey}` } : {}) }
  }

  private url(path: string): string {
    return this.config.baseUrl.replace(/\/+$/, '') + path
  }

  async modelId(signal?: AbortSignal): Promise<string> {
    if (this.config.model) return this.config.model
    if (this.resolved) return this.resolved
    const response = await call(this.url('/models'), { headers: this.headers(), signal: timeout(signal, this.config.timeoutMs) }, signal,
      this.config.timeoutMs)
    if (!response.ok) throw new Error(`GET ${this.url('/models')}: HTTP ${response.status}`)
    const body = await response.json() as { data?: Array<{ id?: string }> }
    const id = body.data?.[0]?.id
    if (!id) throw new Error(`GET ${this.url('/models')} returned no model id`)
    return this.resolved = id
  }

  async complete(messages: readonly ChatMessage[], options: { maxTokens: number, schema: object, name: string, signal?: AbortSignal }): Promise<string> {
    const body = {
      model: await this.modelId(options.signal), messages, temperature: 0, seed: 0, stream: false,
      max_tokens: options.maxTokens,
      response_format: { type: 'json_schema', json_schema: { name: options.name, schema: options.schema } },
      ...this.config.extraBody,
    }
    const response = await call(this.url('/chat/completions'), {
      method: 'POST', headers: this.headers(), body: JSON.stringify(body), signal: timeout(options.signal, this.config.timeoutMs),
    }, options.signal, this.config.timeoutMs)
    if (!response.ok) {
      throw new Error(`POST ${this.url('/chat/completions')}: HTTP ${response.status} ${(await response.text()).slice(0, 200)}`)
    }
    const value = await response.json() as { choices?: Array<{ message?: { content?: string | null }, finish_reason?: string }> }
    const choice = value.choices?.[0]
    if (!choice) throw new Error('chat completion without choices')
    if (choice.finish_reason && choice.finish_reason !== 'stop') throw new Error(`generation stopped: ${choice.finish_reason}`)
    return choice.message?.content ?? ''
  }

  /**
   * Whether the server answers at all within `ms` (`GET /models`; any HTTP status counts). A server that accepts the
   * connection and never answers would otherwise hold a rerank for its whole timeout; llama.cpp and Ollama answer this
   * at once even while they generate.
   */
  async reachable(signal?: AbortSignal, ms = 1500): Promise<boolean> {
    try {
      const response = await fetch(this.url('/models'), { headers: this.headers(), signal: timeout(signal, ms) })
      await response.body?.cancel()
      return true
    } catch (error) {
      if (signal?.aborted) throw error
      return false
    }
  }

  /** The endpoint, for status reports and logs. */
  get baseUrl(): string {
    return this.config.baseUrl
  }

  /**
   * Whether a llama.cpp server is generating right now (`GET /slots`); background work waits then so the
   * agent's own request is not queued behind it. Unknown servers report not busy.
   */
  async busy(): Promise<boolean> {
    try {
      const root = this.config.baseUrl.replace(/\/+$/, '').replace(/\/v1$/, '')
      const response = await fetch(root + '/slots', { headers: this.headers(), signal: AbortSignal.timeout(1000) })
      if (!response.ok) return false
      const slots = await response.json() as Array<{ is_processing?: boolean }>
      return Array.isArray(slots) && slots.some(s => s.is_processing === true)
    } catch {
      return false
    }
  }
}

function timeout(signal: AbortSignal | undefined, ms: number): AbortSignal {
  const limit = AbortSignal.timeout(ms)
  return signal ? AbortSignal.any([signal, limit]) : limit
}
