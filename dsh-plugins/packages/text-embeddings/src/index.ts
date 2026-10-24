/**
 * Service Definition for text embeddings in DeepSeek Harness: the abstract
 * `ctx.textEmbeddings` service, its request/spec/result vocabulary, typed
 * failures, the `text-embeddings/settled` observation event, and the
 * OpenAI-compatible `/embeddings` wire helper shared by HTTP providers.
 *
 * Providers subclass {@link TextEmbeddings}; consumers declare
 * `inject: ['textEmbeddings']` and call {@link TextEmbeddings.embed}.
 */
import { Context, Service } from '@deepseek-ai/cordis'

export { postOpenAIEmbeddings } from './openai-wire.ts'
export type { OpenAIEmbeddingsCall, OpenAIEmbeddingsResponse } from './openai-wire.ts'

declare module '@deepseek-ai/cordis' {
  interface Context {
    textEmbeddings: TextEmbeddings
  }
  interface Events {
    /**
     * One embedding call settled (success or failure). Observation only.
     * @mode emit
     * @param event - provider, sizes, duration, and the failure kind when it failed.
     */
    'text-embeddings/settled'(event: TextEmbeddingSettled): void
  }
}

/** Caller request; omitted fields are filled by the provider's {@link TextEmbeddings.resolve}. */
export interface TextEmbeddingRequest {
  /** Texts to embed, one vector per entry, in order. Must be non-empty. */
  input: readonly string[]
  /**
   * Provider task name selecting the model's instruction prompt (for EmbeddingGemma:
   * `query`, `document`, `STS`, ...); `'none'` requests no prompt. Omitted: provider default.
   */
  task?: string
  /** Output dimensions; must be one of {@link TextEmbeddingModelInfo.dimensions}. Omitted: provider default. */
  dimensions?: number
  /** Cancels the call; the promise then rejects with kind `aborted`. */
  signal?: AbortSignal
}

/** Fully resolved request: every default applied, nothing implicit left for {@link TextEmbeddings.execute}. */
export interface TextEmbeddingSpec {
  input: readonly string[]
  /** Resolved task name, or `null` when the provider does not distinguish tasks. */
  task: string | null
  dimensions: number
  signal?: AbortSignal
}

/** Vectors for one call. */
export interface TextEmbeddingResult {
  /** One L2-normalized vector per input, in input order, each of length {@link dimensions}. */
  vectors: Float32Array[]
  /** Provider-reported model identifier. */
  model: string
  dimensions: number
  task: string | null
  /** Tokens consumed as reported by the provider (0 when unknown). */
  promptTokens: number
}

/** Lifecycle state of the provider behind the service. */
export type TextEmbeddingProviderStatus = 'starting' | 'ready' | 'restarting' | 'failed'

/** Static facts and current state of the active provider. */
export interface TextEmbeddingModelInfo {
  provider: string
  model: string
  /** Allowed output dimensions; the first entry is the native width. */
  dimensions: readonly number[]
  defaultDimensions: number
  /** Accepted task names; empty when the provider does not distinguish tasks. */
  tasks: readonly string[]
  defaultTask: string | null
  /** Maximum tokens per input including any prompt, when the provider enforces one. */
  maxInputTokens?: number
  status: TextEmbeddingProviderStatus
  /** Last failure description while `status` is `restarting` or `failed`. */
  lastError?: string
}

/** Observation payload of `text-embeddings/settled`. */
export interface TextEmbeddingSettled {
  provider: string
  inputs: number
  dimensions: number
  task: string | null
  durationMs: number
  promptTokens: number
  /** Present when the call failed. */
  error?: TextEmbeddingErrorKind
}

/**
 * Failure classes:
 * - `invalid-request`: the request cannot succeed as written (unknown task, bad dimensions, empty input).
 * - `input-too-long`: an input exceeds the provider's token limit; shorten or split it.
 * - `unavailable`: the provider is starting, restarting, failed, or unreachable.
 * - `timeout`: no answer within the provider's request timeout.
 * - `aborted`: the caller's signal fired.
 * - `provider`: the provider answered with an error not covered above (authentication, server error).
 */
export type TextEmbeddingErrorKind = 'invalid-request' | 'input-too-long' | 'unavailable' | 'timeout' | 'aborted' | 'provider'

const ERROR_BRAND = Symbol.for('dsh-text-embeddings.error')

/** Typed embedding failure; check with {@link TextEmbeddingError.is} across package copies. */
export class TextEmbeddingError extends Error {
  readonly [ERROR_BRAND] = true
  override readonly name = 'TextEmbeddingError'

  /**
   * @param kind - failure class callers switch on.
   * @param message - actionable description.
   * @param details - HTTP status, provider error code, and the underlying cause when known.
   */
  constructor(readonly kind: TextEmbeddingErrorKind, message: string,
    readonly details: { status?: number, code?: string, cause?: unknown } = {}) {
    super(message, details.cause === undefined ? undefined : { cause: details.cause })
  }

  /**
   * Brand check that also recognizes errors created by another copy of this package.
   * @param error - any thrown value.
   * @returns whether `error` is a {@link TextEmbeddingError}.
   */
  static is(error: unknown): error is TextEmbeddingError {
    return typeof error === 'object' && error !== null && (error as Record<symbol, unknown>)[ERROR_BRAND] === true
  }
}

/**
 * Abstract text-embedding service registered as `ctx.textEmbeddings`.
 * Exactly one provider should be loaded per composition.
 */
export abstract class TextEmbeddings extends Service {
  constructor(ctx: Context) {
    super(ctx, 'textEmbeddings')
  }

  /** @returns the active provider's facts and current status. */
  abstract describe(): TextEmbeddingModelInfo

  /**
   * Apply defaults and validate a request.
   * @param request - caller request.
   * @returns the fully resolved spec.
   * @throws {TextEmbeddingError} kind `invalid-request` for an empty input, unknown task, or unsupported dimensions.
   */
  abstract resolve(request: TextEmbeddingRequest): TextEmbeddingSpec

  /**
   * Embed a resolved spec.
   * @param spec - output of {@link resolve}.
   * @returns vectors in input order.
   * @throws {TextEmbeddingError} for every failure.
   */
  abstract execute(spec: TextEmbeddingSpec): Promise<TextEmbeddingResult>

  /**
   * Resolve, execute, and emit `text-embeddings/settled`.
   * @param request - caller request.
   * @returns vectors in input order.
   * @throws {TextEmbeddingError} for every failure.
   */
  async embed(request: TextEmbeddingRequest): Promise<TextEmbeddingResult> {
    const started = performance.now()
    const provider = this.describe().provider
    let spec: TextEmbeddingSpec | undefined
    try {
      spec = this.resolve(request)
      const result = await this.execute(spec)
      this.ctx.emit('text-embeddings/settled', {
        provider, inputs: spec.input.length, dimensions: result.dimensions, task: result.task,
        durationMs: performance.now() - started, promptTokens: result.promptTokens,
      })
      return result
    } catch (error) {
      const kind: TextEmbeddingErrorKind = TextEmbeddingError.is(error) ? error.kind : 'provider'
      this.ctx.emit('text-embeddings/settled', {
        provider, inputs: request.input.length, dimensions: spec?.dimensions ?? request.dimensions ?? 0,
        task: spec?.task ?? request.task ?? null, durationMs: performance.now() - started, promptTokens: 0, error: kind,
      })
      throw TextEmbeddingError.is(error) ? error : new TextEmbeddingError('provider', String(error), { cause: error })
    }
  }
}

/**
 * Shared validation for providers' {@link TextEmbeddings.resolve}.
 * @param request - caller request.
 * @param info - the provider's facts.
 * @returns the resolved spec.
 * @throws {TextEmbeddingError} kind `invalid-request`.
 */
export function resolveAgainst(request: TextEmbeddingRequest, info: TextEmbeddingModelInfo): TextEmbeddingSpec {
  if (!Array.isArray(request.input) || request.input.length === 0 || request.input.some(x => typeof x !== 'string')) {
    throw new TextEmbeddingError('invalid-request', 'input must be a non-empty array of strings')
  }
  const dimensions = request.dimensions ?? info.defaultDimensions
  if (!info.dimensions.includes(dimensions)) {
    throw new TextEmbeddingError('invalid-request', `dimensions must be one of ${info.dimensions.join(', ')} (got ${dimensions})`)
  }
  let task: string | null = request.task ?? info.defaultTask
  if (info.tasks.length === 0) {
    if (request.task !== undefined && request.task !== 'none') {
      throw new TextEmbeddingError('invalid-request', `provider ${info.provider} does not support tasks (got '${request.task}')`)
    }
    task = null
  } else if (task !== null && task !== 'none' && !info.tasks.some(t => t.toLowerCase() === task!.toLowerCase())) {
    throw new TextEmbeddingError('invalid-request', `unknown task '${task}'; valid: none, ${info.tasks.join(', ')}`)
  }
  return { input: request.input, task, dimensions, ...request.signal ? { signal: request.signal } : {} }
}

export default TextEmbeddings
