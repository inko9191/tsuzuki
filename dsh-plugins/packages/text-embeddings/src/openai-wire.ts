/**
 * OpenAI-compatible `POST {baseUrl}/embeddings` client shared by HTTP providers.
 * Maps transport and HTTP failures onto {@link TextEmbeddingError} kinds.
 */
import { TextEmbeddingError } from './index.ts'

/** One wire call; every value is explicit (providers resolve their own defaults). */
export interface OpenAIEmbeddingsCall {
  /** Base URL including the version path, e.g. `http://127.0.0.1:8299/v1`. */
  baseUrl: string
  /** Bearer token; empty or undefined sends no Authorization header. */
  apiKey?: string
  model: string
  input: readonly string[]
  /** Sent as the standard `dimensions` field when defined. */
  dimensions?: number
  /** Additional JSON body fields (e.g. egemma-server's `task`). */
  extraBody?: Readonly<Record<string, unknown>>
  timeoutMs: number
  signal?: AbortSignal
}

/** Parsed response. */
export interface OpenAIEmbeddingsResponse {
  vectors: Float32Array[]
  model: string
  promptTokens: number
}

interface WireError { message?: string, code?: string | null, type?: string }

function classify(status: number, error: WireError): TextEmbeddingError {
  const code = error.code ?? undefined
  const message = error.message ?? `HTTP ${status}`
  if (code === 'context_length_exceeded') return new TextEmbeddingError('input-too-long', message, { status, code })
  if (status === 400 || status === 404 || status === 422) return new TextEmbeddingError('invalid-request', message, { status, ...code ? { code } : {} })
  if (status === 503 || status === 502) return new TextEmbeddingError('unavailable', message, { status, ...code ? { code } : {} })
  if (status === 504 || status === 408) return new TextEmbeddingError('timeout', message, { status, ...code ? { code } : {} })
  return new TextEmbeddingError('provider', message, { status, ...code ? { code } : {} })
}

/**
 * Call `POST {baseUrl}/embeddings` with float encoding.
 * @param call - endpoint, model, inputs, and limits.
 * @returns vectors in input order.
 * @throws {TextEmbeddingError} for every failure, including a vector count or length mismatch (`provider`).
 */
export async function postOpenAIEmbeddings(call: OpenAIEmbeddingsCall): Promise<OpenAIEmbeddingsResponse> {
  const timeout = AbortSignal.timeout(call.timeoutMs)
  const signal = call.signal ? AbortSignal.any([call.signal, timeout]) : timeout
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (call.apiKey) headers.Authorization = `Bearer ${call.apiKey}`
  const body = { ...call.extraBody, model: call.model, input: call.input, encoding_format: 'float',
    ...call.dimensions !== undefined ? { dimensions: call.dimensions } : {} }
  let response: Response
  try {
    response = await fetch(`${call.baseUrl.replace(/\/+$/, '')}/embeddings`, {
      method: 'POST', headers, body: JSON.stringify(body), signal,
    })
  } catch (error) {
    if (call.signal?.aborted) throw new TextEmbeddingError('aborted', 'embedding call aborted by the caller', { cause: error })
    if (timeout.aborted) throw new TextEmbeddingError('timeout', `no response within ${call.timeoutMs} ms`, { cause: error })
    throw new TextEmbeddingError('unavailable', `cannot reach ${call.baseUrl}: ${(error as Error).message}`, { cause: error })
  }
  let parsed: { data?: { index: number, embedding: number[] }[], model?: string, usage?: { prompt_tokens?: number }, error?: WireError }
  try {
    parsed = await response.json() as typeof parsed
  } catch (error) {
    if (call.signal?.aborted) throw new TextEmbeddingError('aborted', 'embedding call aborted by the caller', { cause: error })
    if (timeout.aborted) throw new TextEmbeddingError('timeout', `no response within ${call.timeoutMs} ms`, { cause: error })
    throw new TextEmbeddingError('provider', `HTTP ${response.status}: response is not JSON`, { status: response.status, cause: error })
  }
  if (!response.ok) throw classify(response.status, parsed.error ?? {})
  const data = parsed.data
  if (!Array.isArray(data) || data.length !== call.input.length) {
    throw new TextEmbeddingError('provider', `expected ${call.input.length} embeddings, got ${Array.isArray(data) ? data.length : 'none'}`)
  }
  const vectors = new Array<Float32Array>(data.length)
  for (const item of data) {
    if (!Number.isInteger(item.index) || item.index < 0 || item.index >= data.length || !Array.isArray(item.embedding)) {
      throw new TextEmbeddingError('provider', 'malformed embedding item in response')
    }
    vectors[item.index] = Float32Array.from(item.embedding)
  }
  const width = vectors[0]?.length ?? 0
  if (vectors.some(v => v === undefined || v.length !== width) || (call.dimensions !== undefined && width !== call.dimensions)) {
    throw new TextEmbeddingError('provider', `embedding width mismatch (expected ${call.dimensions ?? width})`)
  }
  return { vectors, model: parsed.model ?? call.model, promptTokens: parsed.usage?.prompt_tokens ?? 0 }
}
