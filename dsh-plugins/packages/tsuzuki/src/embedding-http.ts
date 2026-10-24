/**
 * Embeddings over any OpenAI-compatible `/embeddings` endpoint (Ollama, llama.cpp, egemma-server) for hosts
 * without `ctx.textEmbeddings`. EmbeddingGemma's task prompts are prepended client-side, as the evaluation did.
 */
import type { Embedder } from './memory.ts'

export interface HttpEmbedderConfig {
  baseUrl: string
  model: string
  apiKey: string
  queryPrefix: string
  documentPrefix: string
  timeoutMs: number
  batch: number
}

export const EMBEDDINGGEMMA_PREFIXES = { queryPrefix: 'task: search result | query: ', documentPrefix: 'title: none | text: ' }

/**
 * The endpoint refused an input as longer than its model accepts: HTTP 400 with `context_length_exceeded`
 * (egemma-server, OpenAI) or a message saying so (llama.cpp, Ollama).
 */
export class InputTooLongError extends Error {
  override name = 'InputTooLongError'
}

const TOO_LONG_CODES = new Set(['context_length_exceeded', 'exceed_context_size_error'])
const TOO_LONG_TEXT = /context.length|too (long|large)|exceeds? (the )?(maximum|context)|has \d+ tokens/i

function tooLong(status: number, body: string): boolean {
  if (![400, 413, 422, 500].includes(status)) return false
  try {
    const error = (JSON.parse(body) as { error?: { code?: unknown, type?: unknown, message?: unknown } | string }).error
    if (typeof error === 'object' && error !== null && (TOO_LONG_CODES.has(String(error.code)) || TOO_LONG_CODES.has(String(error.type)))) return true
  } catch {
    // not JSON: the text decides
  }
  return TOO_LONG_TEXT.test(body)
}

/** Halvings of one input before it is given up (as the dsh plugin's ServiceEmbedder: 1/16 of it at the end). */
const MAX_HALVINGS = 4

export class HttpEmbedder implements Embedder {
  constructor(private readonly config: HttpEmbedderConfig) {}

  model(): string {
    return `http:${this.config.baseUrl.replace(/\/+$/, '')}|${this.config.model}`
  }

  /**
   * One vector per input, `batch` inputs per request. A request refused because one input is too long is redone one
   * input at a time, halving an input the endpoint still refuses (bounded) so it gets the vector of its leading part,
   * as the dsh plugin's embedder does; before, the whole batch failed and backfill stopped at it.
   */
  async embed(input: readonly string[], task: 'query' | 'document', signal?: AbortSignal): Promise<Float32Array[]> {
    const prefix = task === 'query' ? this.config.queryPrefix : this.config.documentPrefix
    const out: Float32Array[] = []
    for (let k = 0; k < input.length; k += this.config.batch) {
      const part = input.slice(k, k + this.config.batch)
      try {
        out.push(...await this.post(part.map(t => prefix + t), signal))
      } catch (error) {
        if (!(error instanceof InputTooLongError)) throw error
        for (const text of part) out.push(await this.one(prefix, text, signal))
      }
    }
    return out
  }

  private async one(prefix: string, text: string, signal?: AbortSignal): Promise<Float32Array> {
    for (let attempt = 0; ; attempt++) {
      try {
        return (await this.post([prefix + text], signal))[0]!
      } catch (error) {
        if (!(error instanceof InputTooLongError) || attempt >= MAX_HALVINGS) throw error
        const chars = Array.from(text)
        text = chars.slice(0, Math.max(1, Math.floor(chars.length / 2))).join('')
      }
    }
  }

  /** Errors name the URL: `fetch failed` alone does not say which server is missing. */
  private async post(part: readonly string[], signal?: AbortSignal): Promise<Float32Array[]> {
    const url = this.config.baseUrl.replace(/\/+$/, '') + '/embeddings'
    const limit = AbortSignal.timeout(this.config.timeoutMs)
    let response: Response
    try {
      response = await fetch(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json', ...(this.config.apiKey ? { authorization: `Bearer ${this.config.apiKey}` } : {}) },
        body: JSON.stringify({ model: this.config.model, input: part }),
        signal: signal ? AbortSignal.any([signal, limit]) : limit,
      })
    } catch (error) {
      if (signal?.aborted) throw error  // the caller's limit or shutdown: the caller says what that means
      if (limit.aborted) throw new Error(`POST ${url}: no answer within ${this.config.timeoutMs} ms`, { cause: error })
      const cause = (error as Error).cause as { code?: string, message?: string } | undefined
      throw new Error(`POST ${url}: ${(error as Error).message}${cause ? ` (${cause.code ?? cause.message})` : ''}`, { cause: error })
    }
    if (!response.ok) {
      const text = await response.text()
      const message = `POST ${url}: HTTP ${response.status} ${text.slice(0, 200)}`
      throw tooLong(response.status, text) ? new InputTooLongError(message) : new Error(message)
    }
    const body = await response.json() as { data?: Array<{ index: number, embedding: number[] }> }
    const data = [...(body.data ?? [])].sort((a, b) => a.index - b.index)
    if (data.length !== part.length) throw new Error(`POST ${url}: expected ${part.length} vectors, got ${data.length}`)
    return data.map(d => normalize(d.embedding))
  }
}

function normalize(values: readonly number[]): Float32Array {
  let n = 0
  for (const v of values) n += v * v
  n = Math.sqrt(n) || 1
  return Float32Array.from(values, v => v / n)
}

/**
 * Ollama's model list (`GET /api/tags`), or null when the endpoint is not an Ollama server that answers (another
 * OpenAI-compatible server, or nothing listening).
 */
export async function ollamaModels(baseUrl: string): Promise<string[] | null> {
  const root = baseUrl.replace(/\/+$/, '').replace(/\/v1$/, '')
  try {
    const response = await fetch(root + '/api/tags', { signal: AbortSignal.timeout(5000) })
    if (!response.ok) return null
    const body = await response.json() as { models?: Array<{ name?: unknown }> }
    return Array.isArray(body.models) ? body.models.map(m => String(m.name ?? '')) : null
  } catch {
    return null
  }
}

/** Ollama only: pull `model` when it is missing. Returns 'present' | 'pulled'; throws on failure. */
export async function ensureOllamaModel(baseUrl: string, model: string, log: (message: string) => void): Promise<'present' | 'pulled'> {
  const root = baseUrl.replace(/\/+$/, '').replace(/\/v1$/, '')
  const tags = await (await fetch(root + '/api/tags', { signal: AbortSignal.timeout(5000) })).json() as { models?: Array<{ name: string }> }
  const want = model.includes(':') ? model : `${model}:latest`
  if ((tags.models ?? []).some(m => m.name === want || m.name === model)) return 'present'
  log(`pulling ${model} with Ollama (one-time download)`)
  const response = await fetch(root + '/api/pull', { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, stream: false }), signal: AbortSignal.timeout(60 * 60 * 1000) })
  const body = await response.json() as { status?: string, error?: string }
  if (!response.ok || body.status !== 'success') throw new Error(`ollama pull ${model}: ${body.error ?? response.status}`)
  return 'pulled'
}

/** License notices shown before a model is pulled for the user. */
export const MODEL_NOTICES: Record<string, string> = {
  embeddinggemma: 'EmbeddingGemma (300m) is provided under and subject to the Gemma Terms of Use found at '
    + 'ai.google.dev/gemma/terms and the Gemma Prohibited Use Policy. By downloading and using it you agree to those terms.',
  'embeddinggemma-2': 'EmbeddingGemma 2 is licensed under the Apache License 2.0.',
}
