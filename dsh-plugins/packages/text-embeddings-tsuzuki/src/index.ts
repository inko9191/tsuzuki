/**
 * EmbeddingGemma 2 provider for `ctx.textEmbeddings`, backed by the local
 * egemma C++/Metal engine running as the `egemma-server` daemon
 * (OpenAI-compatible `POST /v1/embeddings` on loopback).
 *
 * With `autoStart` (default) the plugin owns the daemon: it spawns it through
 * `ctx.subprocess`, waits for readiness before the fiber becomes active, watches
 * `/health`, restarts it with exponential backoff, and terminates it on unload.
 * With `autoStart: false` it attaches to a daemon started elsewhere and never
 * manages its process. Other tools can share the same daemon over HTTP.
 */
import { Context, Service } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import {
  postOpenAIEmbeddings, resolveAgainst, TextEmbeddingError, TextEmbeddings,
  type TextEmbeddingModelInfo, type TextEmbeddingRequest, type TextEmbeddingResult, type TextEmbeddingSpec,
} from 'dsh-text-embeddings'
import { readFileSync } from 'node:fs'
import { isAbsolute, join } from 'node:path'
import { DaemonSupervisor, type SupervisorSpec } from './supervisor.ts'

export { DaemonSupervisor } from './supervisor.ts'
export type { SupervisorLogger, SupervisorSpec } from './supervisor.ts'

/** Matryoshka widths of EmbeddingGemma 2 and 300m; the first is the native width. */
export const ENGINE_DIMENSIONS = [768, 512, 256, 128] as const

export interface Config {
  engineRoot: string
  modelDir: string
  modelName: string
  python: string
  buildDir: string
  dtype: 'auto' | 'fp16' | 'bf16' | 'fp32'
  matmul: 'auto' | 'native' | 'mpp'
  dimensions: 768 | 512 | 256 | 128
  defaultTask: string
  host: string
  port: number
  autoStart: boolean
  apiKey: string
  maxTokens: number
  extraArgs: string[]
  startupTimeoutMs: number
  requestTimeoutMs: number
  readyWaitMs: number
  healthIntervalMs: number
  healthFailures: number
  maxRestarts: number
  restartWindowMs: number
  backoffMs: number
  maxBackoffMs: number
  graceMs: number
}

/** Daemon launch facts derived from the configuration. */
export interface ResolvedLaunch {
  origin: string
  modelDir: string
  supervisor: SupervisorSpec
}

/**
 * Derive paths, argv, and environment for egemma-server.
 * @param config - validated configuration.
 * @returns origin, model directory, and the supervisor spec.
 */
export function resolveLaunch(config: Config): ResolvedLaunch {
  const root = config.engineRoot
  const under = (value: string, fallback: string) => value ? (isAbsolute(value) ? value : join(root, value)) : join(root, fallback)
  const modelDir = under(config.modelDir, 'models/embeddinggemma-2')
  const host = config.host.includes(':') ? `[${config.host}]` : config.host
  const origin = `http://${host}:${config.port}`
  const argv = [
    under(config.python, '.venv/bin/python'), '-m', 'egemma_server',
    '--model-dir', modelDir, '--build-dir', under(config.buildDir, 'build'),
    '--host', config.host, '--port', String(config.port), '--dtype', config.dtype, '--matmul', config.matmul,
    '--model-name', config.modelName,
    '--default-dimensions', String(config.dimensions), '--max-tokens', String(config.maxTokens),
    ...config.defaultTask === 'none' ? [] : ['--default-task', config.defaultTask],
    ...config.extraArgs,
  ]
  const env: Record<string, string> = { PYTHONPATH: join(root, 'server'), PYTHONUNBUFFERED: '1' }
  if (config.apiKey) env.EGEMMA_API_KEY = config.apiKey
  return {
    origin, modelDir,
    supervisor: {
      argv, cwd: root, env, origin,
      startupTimeoutMs: config.startupTimeoutMs, healthIntervalMs: config.healthIntervalMs,
      healthFailures: config.healthFailures, maxRestarts: config.maxRestarts, restartWindowMs: config.restartWindowMs,
      backoffMs: config.backoffMs, maxBackoffMs: config.maxBackoffMs, graceMs: config.graceMs,
    },
  }
}

/**
 * Task names offered by the checkpoint (`config_sentence_transformers.json` prompts).
 * @param modelDir - EmbeddingGemma 2 checkpoint directory.
 * @returns prompt names in file order.
 * @throws Error when the file is missing or has no prompts.
 */
export function readTasks(modelDir: string): string[] {
  const path = join(modelDir, 'config_sentence_transformers.json')
  let parsed: { prompts?: Record<string, string> }
  try {
    parsed = JSON.parse(readFileSync(path, 'utf8')) as typeof parsed
  } catch (error) {
    throw new Error(`text-embeddings-tsuzuki: cannot read ${path}: ${(error as Error).message}`)
  }
  const tasks = Object.keys(parsed.prompts ?? {})
  if (!tasks.length) throw new Error(`text-embeddings-tsuzuki: ${path} defines no prompts`)
  return tasks
}

export class TsuzukiTextEmbeddings extends TextEmbeddings {
  static inject = ['subprocess']

  static Config: z<Config> = z.object({
    engineRoot: z.string().required().description('Absolute path of the tsuzuki checkout (server/, build/, .venv/)'),
    modelDir: z.string().default('').description('Checkpoint directory; default <engineRoot>/models/embeddinggemma-2'),
    modelName: z.string().default('embeddinggemma-2')
      .description("Model name the daemon serves, e.g. 'embeddinggemma-300m' with modelDir models/embeddinggemma-300m and dtype fp32"),
    python: z.string().default('').description('Python with numpy and tokenizers; default <engineRoot>/.venv/bin/python'),
    buildDir: z.string().default('').description('Directory with the _egemma_native extension; default <engineRoot>/build'),
    dtype: z.union(['auto', 'fp16', 'bf16', 'fp32']).default('auto')
      .description('auto: fp16 for EmbeddingGemma 2, fp32 for EmbeddingGemma 300m; bf16 is the faster, less exact mode for 300m'),
    matmul: z.union(['auto', 'native', 'mpp']).default('auto').description('auto: MPP on M5-class GPUs, own kernels elsewhere'),
    dimensions: z.union([768, 512, 256, 128]).default(768).description('Default output width'),
    defaultTask: z.string().default('none').description("Prompt name used when a call names none; 'none' follows the model card"),
    host: z.string().default('127.0.0.1'),
    port: z.natural().default(8299),
    autoStart: z.boolean().default(true).description('Spawn and supervise the daemon; false attaches to a running one'),
    apiKey: z.string().default('').role('secret').description('Bearer token required by the daemon (needed for non-loopback hosts)'),
    maxTokens: z.natural().default(2048).description('Per-input token limit including prompt and BOS/EOS'),
    extraArgs: z.array(z.string()).default([]).description('Additional egemma-server arguments'),
    startupTimeoutMs: z.natural().default(180_000),
    requestTimeoutMs: z.natural().default(120_000),
    readyWaitMs: z.natural().default(60_000).description('How long a call waits while the daemon starts or restarts'),
    healthIntervalMs: z.natural().default(5_000),
    healthFailures: z.natural().default(3).description('Consecutive failed health checks before a restart'),
    maxRestarts: z.natural().default(5),
    restartWindowMs: z.natural().default(600_000),
    backoffMs: z.natural().default(1_000),
    maxBackoffMs: z.natural().default(30_000),
    graceMs: z.natural().default(5_000),
  })

  private readonly launch: ResolvedLaunch
  private readonly info: Omit<TextEmbeddingModelInfo, 'status' | 'lastError'>
  private supervisor: DaemonSupervisor | undefined
  private owning: Promise<void> | undefined

  /**
   * @param ctx - plugin context with the subprocess service.
   * @param config - validated configuration.
   * @throws Error for a relative engineRoot, an unknown defaultTask, or a missing checkpoint.
   */
  constructor(ctx: Context, readonly config: Config) {
    super(ctx)
    if (!isAbsolute(config.engineRoot)) throw new Error(`text-embeddings-tsuzuki: engineRoot must be absolute: ${config.engineRoot}`)
    if (config.port < 1 || config.port > 65535) throw new Error(`text-embeddings-tsuzuki: invalid port ${config.port}`)
    this.launch = resolveLaunch(config)
    const tasks = readTasks(this.launch.modelDir)
    if (config.defaultTask !== 'none' && !tasks.some(t => t.toLowerCase() === config.defaultTask.toLowerCase())) {
      throw new Error(`text-embeddings-tsuzuki: unknown defaultTask '${config.defaultTask}'; valid: none, ${tasks.join(', ')}`)
    }
    this.info = {
      provider: 'egemma', model: config.modelName,
      dimensions: [config.dimensions, ...ENGINE_DIMENSIONS.filter(d => d !== config.dimensions)],
      defaultDimensions: config.dimensions, tasks, defaultTask: config.defaultTask, maxInputTokens: config.maxTokens,
    }
  }

  /**
   * Start (or attach to) the daemon; the yielded disposer stops an owned daemon on unload.
   * With `autoStart`, an egemma-server already serving this model on the port (e.g. one started by
   * another tool) is shared instead of duplicated; the plugin takes the port over if that daemon goes away.
   */
  async* [Service.init](): AsyncGenerator<() => Promise<void>, void, unknown> {
    const existing = await this.probe()
    if (!this.config.autoStart) {
      if (existing !== 'egemma') this.ctx.logger.warn(`text-embeddings-tsuzuki: no egemma-server answers at ${this.launch.origin}; calls fail until one does`)
      return
    }
    if (existing === 'other') {
      throw new Error(`text-embeddings-tsuzuki: ${this.launch.origin} is already in use by something other than an egemma-server ` +
        `serving ${this.config.modelName}; choose another port`)
    }
    yield () => this.supervisor?.stop() ?? Promise.resolve()
    if (existing === 'egemma') {
      this.ctx.logger.info(`text-embeddings-tsuzuki: sharing the egemma-server already serving ${this.config.modelName} at ${this.launch.origin}`)
      return
    }
    await this.own()
  }

  private own(): Promise<void> {
    this.owning ??= (async () => {
      const supervisor = new DaemonSupervisor(this.ctx.subprocess, this.launch.supervisor, {
        info: m => this.ctx.logger.info(m), warn: m => this.ctx.logger.warn(m), error: m => this.ctx.logger.error(m),
      })
      this.supervisor = supervisor
      await supervisor.start()
    })()
    return this.owning
  }

  /** A shared daemon stopped answering: start our own on the freed port (once), or report why not. */
  private async takeOver(error: unknown): Promise<boolean> {
    if (!this.config.autoStart || this.supervisor) return false
    const now = await this.probe()
    if (now === 'egemma') return true // it came back (or another tool restarted it)
    if (now === 'other') return false
    this.ctx.logger.warn(`text-embeddings-tsuzuki: the shared egemma-server at ${this.launch.origin} went away (${(error as Error).message}); starting our own`)
    try {
      await this.own()
    } catch (startError) {
      this.owning = undefined
      this.supervisor = undefined
      throw new TextEmbeddingError('unavailable', `egemma-server could not be started: ${(startError as Error).message}`, { cause: startError })
    }
    return true
  }

  private async probe(): Promise<'free' | 'egemma' | 'other'> {
    try {
      const response = await fetch(`${this.launch.origin}/health`, { signal: AbortSignal.timeout(2_000) })
      const body = await response.json().catch(() => ({})) as { model?: string }
      return body.model === this.config.modelName ? 'egemma' : 'other'
    } catch {
      return 'free'
    }
  }

  describe(): TextEmbeddingModelInfo {
    const supervisor = this.supervisor
    if (!supervisor) return { ...this.info, status: 'ready' }
    return { ...this.info, status: supervisor.status, ...supervisor.lastError ? { lastError: supervisor.lastError } : {} }
  }

  resolve(request: TextEmbeddingRequest): TextEmbeddingSpec {
    return resolveAgainst(request, this.describe())
  }

  async execute(spec: TextEmbeddingSpec): Promise<TextEmbeddingResult> {
    if (this.supervisor) {
      try {
        await this.supervisor.waitReady(this.config.readyWaitMs, spec.signal)
      } catch (error) {
        if (spec.signal?.aborted) throw new TextEmbeddingError('aborted', 'embedding call aborted by the caller', { cause: error })
        throw new TextEmbeddingError('unavailable', `egemma-server is ${this.supervisor.status}: ${(error as Error).message}`, { cause: error })
      }
    }
    const task = spec.task ?? 'none'
    const post = () => postOpenAIEmbeddings({
      baseUrl: `${this.launch.origin}/v1`, apiKey: this.config.apiKey, model: this.config.modelName, input: spec.input,
      dimensions: spec.dimensions, extraBody: { task }, timeoutMs: this.config.requestTimeoutMs,
      ...spec.signal ? { signal: spec.signal } : {},
    })
    let response: Awaited<ReturnType<typeof post>>
    try {
      response = await post()
    } catch (error) {
      if (spec.signal?.aborted || !(TextEmbeddingError.is(error) && error.kind === 'unavailable') || !await this.takeOver(error)) throw error
      response = await post()
    }
    return { vectors: response.vectors, model: response.model, dimensions: spec.dimensions, task, promptTokens: response.promptTokens }
  }
}

export default TsuzukiTextEmbeddings
