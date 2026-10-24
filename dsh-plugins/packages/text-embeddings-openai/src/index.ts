/**
 * Generic `ctx.textEmbeddings` provider for any OpenAI-compatible
 * `POST {baseUrl}/embeddings` service (OpenAI, SiliconFlow, Jina, a
 * self-hosted server, or an egemma-server started outside dsh).
 *
 * Tasks are expressed as text prefixes (`taskPrefixes`), because the OpenAI
 * wire format has no task field. Smaller output widths are requested with
 * the standard `dimensions` field when `sendDimensions` is true; otherwise the
 * returned vector prefix is renormalized locally, which is only meaningful
 * for Matryoshka-trained models — list those widths in `allowedDimensions`.
 */
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import {
  postOpenAIEmbeddings, resolveAgainst, TextEmbeddingError, TextEmbeddings,
  type TextEmbeddingModelInfo, type TextEmbeddingRequest, type TextEmbeddingResult, type TextEmbeddingSpec,
} from 'dsh-text-embeddings'

export interface Config {
  baseUrl: string
  apiKey: string
  model: string
  dimensions: number
  allowedDimensions: number[]
  sendDimensions: boolean
  taskPrefixes: Record<string, string>
  defaultTask: string
  timeoutMs: number
  maxInputsPerCall: number
}

/**
 * Renormalize the first `width` components of each vector (Matryoshka truncation).
 * @param vectors - L2-normalized vectors.
 * @param width - prefix length.
 * @returns new vectors of length `width` with unit norm.
 */
export function truncateNormalize(vectors: Float32Array[], width: number): Float32Array[] {
  return vectors.map((v) => {
    const out = v.slice(0, width)
    let sum = 0
    for (const x of out) sum += x * x
    const scale = 1 / Math.max(Math.sqrt(sum), 1e-12)
    for (let i = 0; i < out.length; i++) out[i] = out[i]! * scale
    return out
  })
}

export class OpenAITextEmbeddings extends TextEmbeddings {
  static Config: z<Config> = z.object({
    baseUrl: z.string().required().description('Base URL including the version path, e.g. https://api.openai.com/v1'),
    apiKey: z.string().default('').role('secret').description('Bearer token; empty sends no Authorization header'),
    model: z.string().required().description('Model name sent in every request'),
    dimensions: z.natural().required().description('Native output width of the model'),
    allowedDimensions: z.array(z.natural()).default([]).description('Additional Matryoshka widths callers may request'),
    sendDimensions: z.boolean().default(false).description('Send the dimensions field instead of truncating locally'),
    taskPrefixes: z.dict(z.string()).default({}).description('Task name -> text prefix prepended to every input'),
    defaultTask: z.string().default('none').description("Task used when a request names none ('none' = no prefix)"),
    timeoutMs: z.natural().default(60_000),
    maxInputsPerCall: z.natural().default(256).description('Inputs per HTTP request; larger requests are split'),
  })

  private readonly info: TextEmbeddingModelInfo

  /**
   * @param ctx - plugin context.
   * @param config - validated configuration.
   * @throws Error when the configuration names an invalid URL, width, or default task.
   */
  constructor(ctx: Context, readonly config: Config) {
    super(ctx)
    let url: URL
    try {
      url = new URL(config.baseUrl)
    } catch {
      throw new Error(`text-embeddings-openai: baseUrl is not a URL: ${config.baseUrl}`)
    }
    if (url.protocol !== 'http:' && url.protocol !== 'https:') throw new Error('text-embeddings-openai: baseUrl must be http(s)')
    if (config.dimensions < 1 || config.maxInputsPerCall < 1 || config.timeoutMs < 1) {
      throw new Error('text-embeddings-openai: dimensions, maxInputsPerCall and timeoutMs must be positive')
    }
    const extra = config.allowedDimensions.filter(d => d !== config.dimensions)
    if (extra.some(d => d < 1 || d > config.dimensions)) {
      throw new Error(`text-embeddings-openai: allowedDimensions must be within 1..${config.dimensions}`)
    }
    const tasks = Object.keys(config.taskPrefixes)
    if (config.defaultTask !== 'none' && !tasks.includes(config.defaultTask)) {
      throw new Error(`text-embeddings-openai: defaultTask '${config.defaultTask}' is not in taskPrefixes`)
    }
    this.info = {
      provider: 'openai-compatible', model: config.model, dimensions: [config.dimensions, ...extra],
      defaultDimensions: config.dimensions, tasks, defaultTask: tasks.length ? config.defaultTask : null, status: 'ready',
    }
  }

  describe(): TextEmbeddingModelInfo {
    return this.info
  }

  resolve(request: TextEmbeddingRequest): TextEmbeddingSpec {
    return resolveAgainst(request, this.info)
  }

  async execute(spec: TextEmbeddingSpec): Promise<TextEmbeddingResult> {
    const key = spec.task === null || spec.task === 'none'
      ? undefined
      : Object.keys(this.config.taskPrefixes).find(t => t.toLowerCase() === spec.task!.toLowerCase())
    const prefix = key === undefined ? '' : this.config.taskPrefixes[key]!
    const native = spec.dimensions === this.config.dimensions
    const vectors: Float32Array[] = []
    let promptTokens = 0
    let model = this.config.model
    for (let start = 0; start < spec.input.length; start += this.config.maxInputsPerCall) {
      const chunk = spec.input.slice(start, start + this.config.maxInputsPerCall).map(t => prefix + t)
      const response = await postOpenAIEmbeddings({
        baseUrl: this.config.baseUrl, apiKey: this.config.apiKey, model: this.config.model, input: chunk,
        timeoutMs: this.config.timeoutMs,
        ...!native && this.config.sendDimensions ? { dimensions: spec.dimensions } : native ? { dimensions: this.config.dimensions } : {},
        ...spec.signal ? { signal: spec.signal } : {},
      })
      if (!native && !this.config.sendDimensions && response.vectors[0]!.length !== this.config.dimensions) {
        throw new TextEmbeddingError('provider', `expected ${this.config.dimensions}-dimensional vectors before truncation`)
      }
      vectors.push(...(native || this.config.sendDimensions ? response.vectors : truncateNormalize(response.vectors, spec.dimensions)))
      promptTokens += response.promptTokens
      model = response.model
    }
    return { vectors, model, dimensions: spec.dimensions, task: spec.task, promptTokens }
  }
}

export default OpenAITextEmbeddings
