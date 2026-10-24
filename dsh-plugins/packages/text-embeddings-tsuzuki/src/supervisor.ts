/**
 * Lifecycle of one egemma-server child process: spawn through `ctx.subprocess`,
 * wait for its `{"event":"ready"}` stdout line, watch `/health`, and restart
 * with exponential backoff after a crash or a hung health check. After
 * `maxRestarts` restarts inside `restartWindowMs` the supervisor stays
 * `failed`; requests then fail fast with kind `unavailable` and the last
 * exit status plus stderr tail.
 */
import type { SubprocessHandle, SubprocessRuntime } from '@deepseek-ai/dsh-subprocess'
import type { TextEmbeddingProviderStatus } from 'dsh-text-embeddings'
import { createInterface } from 'node:readline'

/** Fully resolved launch and restart settings. */
export interface SupervisorSpec {
  argv: readonly string[]
  cwd: string
  env: Record<string, string>
  /** `http://host:port`, used for health checks. */
  origin: string
  startupTimeoutMs: number
  healthIntervalMs: number
  /** Consecutive failed health checks of a running process before it is restarted. */
  healthFailures: number
  maxRestarts: number
  restartWindowMs: number
  backoffMs: number
  maxBackoffMs: number
  graceMs: number
}

/** Minimal logger surface (the Cordis logger satisfies it). */
export interface SupervisorLogger {
  info(message: string): void
  warn(message: string): void
  error(message: string): void
}

const STDERR_TAIL_BYTES = 16_384

export class DaemonSupervisor {
  status: TextEmbeddingProviderStatus = 'starting'
  lastError: string | undefined
  private handle: SubprocessHandle | undefined
  private stopping = false
  private restarts: number[] = []
  private ready!: Promise<void>
  private settleReady!: (error?: Error) => void
  private healthTimer: NodeJS.Timeout | undefined
  private restartTimer: NodeJS.Timeout | undefined
  private healthMisses = 0
  private generation = 0

  constructor(private readonly runtime: SubprocessRuntime, private readonly spec: SupervisorSpec,
    private readonly logger: SupervisorLogger) {
    this.resetReady()
  }

  private resetReady(): void {
    this.ready = new Promise<void>((resolve, reject) => {
      this.settleReady = error => error ? reject(error) : resolve()
    })
    // A failed start with no waiting caller must not become an unhandled rejection.
    this.ready.catch(() => undefined)
  }

  /**
   * Start the first process and wait until it is ready.
   * @throws Error with the daemon's failure report when the first start fails.
   */
  async start(): Promise<void> {
    this.launch()
    await this.ready
  }

  /**
   * Wait until the daemon is ready.
   * @param timeoutMs - upper bound for the wait.
   * @param signal - caller cancellation.
   * @returns once ready.
   * @throws Error when failed, timed out, or aborted.
   */
  async waitReady(timeoutMs: number, signal?: AbortSignal): Promise<void> {
    if (this.status === 'ready') return
    if (this.status === 'failed') throw new Error(this.lastError ?? 'egemma-server failed')
    const timeout = AbortSignal.timeout(timeoutMs)
    const abort = signal ? AbortSignal.any([signal, timeout]) : timeout
    await Promise.race([this.ready, new Promise<never>((_, reject) => {
      abort.addEventListener('abort', () => reject(new Error(signal?.aborted ? 'aborted' : `not ready within ${timeoutMs} ms`)), { once: true })
    })])
  }

  /** Stop health checks and terminate the process; resolves when it has exited. */
  async stop(): Promise<void> {
    this.stopping = true
    clearInterval(this.healthTimer)
    clearTimeout(this.restartTimer)
    const handle = this.handle
    if (handle) {
      handle.terminate()
      await handle.waitForExit(AbortSignal.timeout(this.spec.graceMs * 4)).catch(() => false)
    }
    this.settleReady(new Error('egemma-server stopped'))
  }

  private launch(): void {
    const generation = ++this.generation
    this.healthMisses = 0
    let handle: SubprocessHandle
    try {
      handle = this.runtime.spawn({
        argv: this.spec.argv, cwd: this.spec.cwd, env: this.spec.env, graceMs: this.spec.graceMs,
        stdio: { stdin: 'ignore', stdout: 'pipe', stderr: { maxBytes: STDERR_TAIL_BYTES } },
      })
    } catch (error) {
      this.onExit(generation, `spawn failed: ${(error as Error).message}`)
      return
    }
    this.handle = handle
    let announced = false
    const startup = setTimeout(() => {
      if (!announced) {
        this.lastError = `egemma-server did not report ready within ${this.spec.startupTimeoutMs} ms`
        handle.terminate()
      }
    }, this.spec.startupTimeoutMs)
    const lines = createInterface({ input: handle.stdout! })
    lines.on('line', (line) => {
      let event: { event?: string, error?: string, port?: number, cold_start?: Record<string, number> }
      try {
        event = JSON.parse(line) as typeof event
      } catch {
        return // only JSON lines are protocol; anything else is diagnostics
      }
      if (event.event === 'ready' && generation === this.generation) {
        announced = true
        clearTimeout(startup)
        this.status = 'ready'
        this.lastError = undefined
        this.logger.info(`egemma-server ready on ${this.spec.origin} (cold start ${event.cold_start?.ready_s ?? '?'} s)`)
        this.settleReady()
        this.watchHealth(generation)
      } else if (event.event === 'failed') {
        this.lastError = `egemma-server ${event.error ?? 'failed'}`
      }
    })
    handle.done.then(
      outcome => this.onExit(generation, `exited (code ${outcome.exitCode ?? 'none'}, signal ${outcome.signal ?? 'none'})`),
      error => this.onExit(generation, `subprocess error: ${(error as Error).message}`),
    ).finally(() => clearTimeout(startup))
  }

  private stderrTail(): string {
    const text = this.handle?.collected.stderr?.readFrom(0).text.trim() ?? ''
    return text ? `\n--- egemma-server stderr (tail) ---\n${text.split('\n').slice(-20).join('\n')}` : ''
  }

  private onExit(generation: number, how: string): void {
    if (generation !== this.generation || this.stopping) return
    clearInterval(this.healthTimer)
    const wasReady = this.status === 'ready'
    this.lastError = `${this.lastError ? `${this.lastError}; ` : ''}${how}${this.stderrTail()}`
    const now = Date.now()
    this.restarts = this.restarts.filter(t => now - t < this.spec.restartWindowMs)
    if (!wasReady && this.restarts.length === 0 && this.status === 'starting') {
      // The very first start failed: report it to start() instead of looping on a misconfiguration.
      this.status = 'failed'
      this.logger.error(`egemma-server failed to start: ${this.lastError}`)
      this.settleReady(new Error(this.lastError))
      return
    }
    if (this.restarts.length >= this.spec.maxRestarts) {
      this.status = 'failed'
      this.logger.error(`egemma-server gave up after ${this.restarts.length} restarts in ${this.spec.restartWindowMs} ms: ${this.lastError}`)
      this.settleReady(new Error(this.lastError))
      return
    }
    const delay = Math.min(this.spec.backoffMs * 2 ** this.restarts.length, this.spec.maxBackoffMs)
    this.restarts.push(now)
    if (this.status === 'ready') this.resetReady()
    this.status = 'restarting'
    this.logger.warn(`egemma-server ${how}; restarting in ${delay} ms (${this.restarts.length}/${this.spec.maxRestarts})`)
    this.restartTimer = setTimeout(() => this.launch(), delay)
  }

  private watchHealth(generation: number): void {
    clearInterval(this.healthTimer)
    this.healthTimer = setInterval(() => {
      void (async () => {
        try {
          const response = await fetch(`${this.spec.origin}/health`, { signal: AbortSignal.timeout(this.spec.healthIntervalMs) })
          if (!response.ok) throw new Error(`health ${response.status}`)
          this.healthMisses = 0
        } catch (error) {
          if (generation !== this.generation || this.stopping) return
          if (++this.healthMisses >= this.spec.healthFailures) {
            this.lastError = `health check failed ${this.healthMisses} times: ${(error as Error).message}`
            this.logger.warn(`egemma-server unresponsive (${this.lastError}); terminating for restart`)
            this.handle?.terminate()
          }
        }
      })()
    }, this.spec.healthIntervalMs)
    this.healthTimer.unref?.()
  }
}
