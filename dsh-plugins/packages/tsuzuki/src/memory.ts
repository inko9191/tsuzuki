/**
 * Memory orchestration, independent of dsh: capture messages, cut sessions into segments, extract facts
 * in the background, embed p2 windows and facts, and retrieve dated excerpts for a query.
 *
 * Retrieval is the evaluated C-final pipeline (p2 + fact units, dense ranking fused with RRF, distinct,
 * optional LLM rerank of the top 30, budgeted selection of raw messages), rendered in date order with a
 * Date line per session block (U1 arms S/T).
 *
 * Forgetting (`forget()`) deletes in the store; this class then drops its caches, and every write that was waiting on a
 * model (embedding, fact extraction) checks that what it is about to store still exists, so nothing deleted comes back.
 */
import * as p from './pipeline.ts'
import { redact } from './redact.ts'
import type { ForgetResult, ForgetScope, MemoryStore, Segment, StoredTurn, StoredUnit } from './store.ts'
import { RankIndex, factOrder, windowOrder } from './unit-index.ts'
import { linkedSignal } from './signals.ts'

export type { ForgetResult, ForgetScope }

export interface Embedder {
  /** Unit-normalized vectors, one per input; `task` selects the model's query/document prompt. */
  embed(input: readonly string[], task: 'query' | 'document', signal?: AbortSignal): Promise<Float32Array[]>
  /** Identifier recorded with each vector; a change means stored vectors are stale. */
  model(): string
}

export interface ChatModel {
  /** Content of one JSON-constrained completion (temperature 0). */
  complete(messages: readonly p.ChatMessage[], options: { maxTokens: number, schema: object, name: string, signal?: AbortSignal }): Promise<string>
  /** Whether the server answers at all, quickly (checked before a rerank, so a hung server costs seconds, not the timeout). */
  reachable?(signal?: AbortSignal): Promise<boolean>
  /** The endpoint, named in status reports. */
  readonly baseUrl?: string
}

/** What the last call to a model did: `ok` is null before the first call. Errors name the URL that failed. */
export interface ModelStatus {
  ok: boolean | null
  lastError: string | null
  lastErrorAt: string | null
}

/** The query embedding did not answer within `queryTimeoutMs`. */
export class EmbeddingTimeoutError extends Error {
  override readonly name = 'EmbeddingTimeoutError'
}

/** The embedding model failed (down, or answered with an error); the message names the URL when the embedder gives it. */
export class EmbeddingUnavailableError extends Error {
  override readonly name = 'EmbeddingUnavailableError'
}

/** Resolves with `promise`, or with undefined after `ms` (the promise keeps running). */
export function within<T>(promise: Promise<T>, ms: number): Promise<T | undefined> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => resolve(undefined), ms)
    promise.then(v => { clearTimeout(timer); resolve(v) }, e => { clearTimeout(timer); reject(e) })
  })
}

/** An embedder's refusal of one input (too long), as opposed to a model that is down. */
function refusedInput(error: unknown): boolean {
  const e = error as { name?: unknown, kind?: unknown } | null
  return e?.name === 'InputTooLongError' || e?.kind === 'input-too-long'
}

export interface MemoryOptions {
  /** Close a segment once its messages reach about this many tokens (benchmarks' median session: ~2.4k). */
  segmentTokens: number
  /** Close a segment after this many minutes without a new message in its session. */
  segmentIdleMinutes: number
  /** Tokens of one message kept in windows, prompts and excerpts (two fit one 2048-token embedding). */
  messageTokens: number
  /** Tokens of one message shown to the reranker (30 candidates must fit its window). */
  rerankMessageTokens: number
  /** Token budget of retrieved excerpts (evaluated: 512). */
  budgetTokens: number
  maxAttempts: number
  /** After each extraction, link facts made outdated by the new segment's facts (U1 arm V). */
  updateLinks: boolean
  /**
   * Limit of one query embedding (recall, search). Storing has the embedder's own, longer limit: a query is waited for
   * by the user, a stored message is not.
   */
  queryTimeoutMs: number
  /**
   * After a query embedding timed out, further queries fail at once for this long (until any embedding answers), so a
   * hung model costs one wait, not one per recall and search of the same turn.
   */
  queryBackoffMs: number
  /** Limit of one rerank request; on timeout the dense order is kept. */
  rerankTimeoutMs: number
  /**
   * Called when the embedding model answers again after a failure (hosts start `backfill()` so messages stored while
   * it was down become searchable at once). Without it, `backfill()` is started here.
   */
  onEmbeddingRecovered?: () => void
  /** Replace API keys, passwords, tokens and card numbers with `[REDACTED:KIND]` before storing. */
  redactSecrets: boolean
  log?: (level: 'info' | 'warn', message: string) => void
}

export const DEFAULTS: MemoryOptions = {
  segmentTokens: 2500, segmentIdleMinutes: 30, messageTokens: 900, rerankMessageTokens: 250, budgetTokens: 2048, maxAttempts: 3, updateLinks: true, redactSecrets: true,
  queryTimeoutMs: 3000, queryBackoffMs: 30000, rerankTimeoutMs: 15000,
}

/** Benchmark date format, local time: `2023/05/20 (Sat) 10:00`. */
export function formatDate(ms: number): string {
  const d = new Date(ms)
  const day = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][d.getDay()]
  const two = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}/${two(d.getMonth() + 1)}/${two(d.getDate())} (${day}) ${two(d.getHours())}:${two(d.getMinutes())}`
}

/** Rough token count that does not undercount Japanese: ASCII ~4 chars per token, other characters ~1. */
export function estimateTokens(text: string): number {
  let ascii = 0, other = 0
  for (const ch of text) (ch.codePointAt(0)! < 128 ? ascii++ : other++)
  return Math.ceil(ascii / 4 + other)
}

/** Leading part of `text` within about `tokens` estimated tokens. */
export function clip(text: string, tokens: number): string {
  let used = 0, end = 0
  for (const ch of text) {
    used += ch.codePointAt(0)! < 128 ? 0.25 : 1
    if (used > tokens) return text.slice(0, end) + ' …'
    end += ch.length
  }
  return text
}

const ISO = /^(\d{4})-(\d{2})-(\d{2})(?:T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)?$/

/**
 * Epoch ms of an ISO 8601 date or date-time, or null. A date alone means local midnight (the dates memory shows are
 * local time); a date-time without an offset is local time too.
 */
export function parseBefore(value: string): number | null {
  const m = ISO.exec(value.trim())
  if (!m) return null
  if (value.trim().length === 10) {
    const d = new Date(+m[1]!, +m[2]! - 1, +m[3]!)
    return d.getFullYear() === +m[1]! && d.getMonth() === +m[2]! - 1 && d.getDate() === +m[3]! ? d.getTime() : null
  }
  const ms = Date.parse(value.trim())
  return Number.isNaN(ms) ? null : ms
}

/**
 * The scope of a forget request (tool arguments or an HTTP body): exactly one of `session` (non-empty string),
 * `before` (ISO 8601 date or date-time) or `all: true`. Returns an error message for anything else, so an ambiguous
 * request never deletes.
 */
export function forgetScope(args: { session?: unknown, before?: unknown, all?: unknown }): ForgetScope | string {
  const given = (['session', 'before', 'all'] as const).filter(k => args[k] !== undefined && args[k] !== null)
  if (given.length !== 1) return 'give exactly one of session, before or all'
  if (given[0] === 'session') {
    return typeof args.session === 'string' && args.session.trim() ? { session: args.session } : 'session must be a non-empty string'
  }
  if (given[0] === 'before') {
    const before = typeof args.before === 'string' ? parseBefore(args.before) : null
    return before === null ? 'before must be an ISO 8601 date or date-time, e.g. 2026-10-01 or 2026-10-01T09:00:00+09:00' : { before }
  }
  return args.all === true ? { all: true } : 'all must be true'
}

export interface Retrieval {
  /** Rendered excerpts (empty when nothing fits or nothing is stored). */
  text: string
  ids: string[]
  reranked: boolean
  candidates: number
}

/** Tries of one update check when the chat server cannot be reached, and the pause before the n-th retry (n × this). */
const LINK_ATTEMPTS = 4
const LINK_RETRY_MS = 1000

/** Embedded when a batch fails, to tell an embedder that is down from a unit it refuses. */
const PROBE = 'memory index probe'

export class Memory {
  private readonly options: MemoryOptions
  private readonly idleTimers = new Map<string, ReturnType<typeof setTimeout>>()
  /** Every stored unit, in insertion order. */
  private unitCache: StoredUnit[] | null = null
  /** Every stored message, in capture order, and by id. */
  private turnCache: StoredTurn[] | null = null
  private turnById: Map<string, StoredTurn> | null = null
  /**
   * The ranked windows and facts with their vectors, kept between recalls (see `RankIndex`). Exists only while the
   * three caches above do; a capture appends to it, other writes drop it.
   */
  private ranked: RankIndex | null = null
  /** Store generation the caches were read at (a forget in any connection bumps it). */
  private generation = -1
  private draining: Promise<void> | null = null
  private backfilling: Promise<number> | null = null
  private backfillAgain = false
  /** Ids of units the embedder refused on their own (logged once). */
  private readonly refused = new Set<string>()
  private closed = false
  /** Aborted by `close()`: every model request in flight stops, so a host can exit without waiting for its timeout. */
  private readonly closing = new AbortController()
  private chatDown = false
  private embedding: ModelStatus = { ok: null, lastError: null, lastErrorAt: null }
  /** Until when queries are not sent (a query timed out; see `queryBackoffMs`). */
  private queryBackoffUntil = 0
  private chatState: ModelStatus = { ok: null, lastError: null, lastErrorAt: null }

  constructor(readonly store: MemoryStore, private readonly embedder: Embedder, private readonly chat: ChatModel | null,
    options: Partial<MemoryOptions> = {}) {
    this.options = { ...DEFAULTS, ...options }
  }

  private log(level: 'info' | 'warn', message: string): void {
    this.options.log?.(level, message)
  }

  private asTurn(t: StoredTurn, tokens = this.options.messageTokens): p.Turn {
    return { id: t.id, session: t.session, pos: t.pos, date: t.date, text: clip(t.text, tokens) }
  }

  /** Drops the caches when a forget ran since they were read (here or in another process sharing the file). */
  private refresh(): void {
    const g = this.store.generation()
    if (g === this.generation) return
    this.generation = g
    this.unitCache = null
    this.turnCache = null
    this.turnById = null
    this.ranked = null
  }

  private allTurns(): StoredTurn[] {
    this.refresh()
    return this.turnCache ??= this.store.turns()
  }

  private allUnits(): StoredUnit[] {
    this.refresh()
    return this.unitCache ??= this.store.units()
  }

  private turnIndex(): Map<string, StoredTurn> {
    this.refresh()
    return this.turnById ??= new Map(this.allTurns().map(t => [t.id, t] as const))
  }

  /**
   * Whether a unit built before an await may still be stored: every message it shows still exists and, for a fact,
   * the fact is unchanged. A forget during the await makes this false, so deleted text is never written back.
   */
  private storable(u: p.Unit): boolean {
    if (!u.turns.every(id => this.store.hasTurn(id))) return false
    return u.set !== 'fact' || this.store.factText(...splitFact(u.id)) === u.text
  }

  /** The p2 window of a stored message: the stored message before it in its session, then itself. */
  private windowOf(turn: StoredTurn): p.Unit {
    const prev = this.store.previousTurn(turn.session, turn.pos)
    const pair = [...(prev ? [{ ...this.asTurn(prev), pos: 0 }] : []), { ...this.asTurn(turn), pos: prev ? 1 : 0 }]
    return p.windowUnitsP2(pair).at(-1)!
  }

  // -- capture --------------------------------------------------------------------------

  /** Stores one message and embeds its p2 window. Returns once the window is indexed (throws when embedding fails). */
  async capture(message: { id: string, session: string, role: string, text: string, at?: number }): Promise<void> {
    const turn = this.record(message)
    if (turn) await this.indexTurn(turn)
  }

  /**
   * Stores one message (redacted), synchronously, without embedding it; `indexTurn()` embeds its window. Hosts answer
   * once this returns, so a slow or hung embedding model never holds the conversation (a window not indexed is
   * embedded by `backfill()`). Returns null after `close()` or for an empty message.
   */
  record(message: { id: string, session: string, role: string, text: string, at?: number }): StoredTurn | null {
    if (this.closed || !message.text.trim()) return null
    const at = message.at ?? Date.now()
    let text = message.text.trim()
    if (this.options.redactSecrets) {
      const r = redact(text)
      if (Object.keys(r.found).length) this.log('info', `redacted ${Object.entries(r.found).map(([k, n]) => `${n} ${k}`).join(', ')} before storing`)
      text = r.text
    }
    const known = this.store.turn(message.id)
    const turn = known ?? this.store.addTurn({ id: message.id, session: message.session, role: message.role,
      text: `${message.role}: ${text}`, at, date: formatDate(at) })
    this.refresh()
    if (!known) {
      this.turnCache?.push(turn)
      this.turnById?.set(turn.id, turn)
    }
    this.maybeCloseSegment(turn.session, false)
    this.armIdle(turn.session)
    return turn
  }

  /** Embeds the p2 window of a recorded message (nothing when it was forgotten meanwhile or memory is closed). */
  async indexTurn(turn: StoredTurn): Promise<void> {
    if (this.closed || !this.store.hasTurn(turn.id)) return
    await this.index([this.windowOf(turn)])
  }

  /** Embeds and stores units not stored yet. A single unit's embedding error is thrown (capture reports it). */
  private async index(units: readonly p.Unit[]): Promise<void> {
    const fresh = units.filter(u => !this.store.hasUnit(u.id))
    if (!fresh.length) return
    const vectors = fresh.length === 1 ? await this.embed([fresh[0]!.text], 'document') : await this.embedDocuments(fresh)
    const model = this.embedder.model()
    this.refresh()
    fresh.forEach((u, k) => {
      const vector = vectors[k]
      if (!vector || !this.storable(u)) return
      const stored = { ...u, vector, model }
      this.store.addUnit(stored)
      this.unitCache?.push(stored)
      // A capture's window goes at the end of the ranked windows; anything else rebuilds what it changes.
      if (this.ranked && !this.ranked.add(stored, this.seqOf(stored.turns[0]))) this.ranked = null
    })
  }

  /** Capture order of a stored message. */
  private seqOf(id: string | undefined): number | undefined {
    if (id === undefined) return undefined
    return (this.turnById?.get(id) ?? this.store.turn(id))?.seq
  }

  /**
   * Document vectors for `units`, null for a unit the embedder refuses on its own (e.g. longer than its model accepts
   * after the embedder's own shortening). When the batch fails, a short probe tells a down embedder (the batch's error
   * is thrown, as before) from a bad unit; the units are then embedded one at a time, so one unit never keeps the
   * others out of the index. A refused unit is logged once per process and tried again by the next backfill.
   */
  private async embedDocuments(units: readonly p.Unit[]): Promise<Array<Float32Array | null>> {
    try {
      // Not counted as an outage yet: the probe below tells a down model from a unit it refuses.
      return await this.embed(units.map(u => u.text), 'document', undefined, false)
    } catch (error) {
      if (this.closed) throw error
      try {
        await this.embed([PROBE], 'document')
      } catch (down) {
        throw down  // the model is down (an EmbeddingUnavailableError naming the URL), not a unit
      }
      if (units.length === 1) {
        this.refuse(units[0]!, error)
        return [null]
      }
      const out: Array<Float32Array | null> = []
      for (const u of units) {
        try {
          out.push((await this.embed([u.text], 'document', undefined, false))[0]!)
        } catch (e) {
          out.push(null)
          this.refuse(u, e)
        }
      }
      return out
    }
  }

  /**
   * Every embedding request goes through here: `close()` aborts it; a query gets `queryTimeoutMs`; the outcome is kept
   * for `status()`, an outage is logged once, and the first answer after one starts the backfill (R11).
   */
  private async embed(input: readonly string[], task: 'query' | 'document', signal?: AbortSignal, countFailure = true): Promise<Float32Array[]> {
    if (task === 'query' && Date.now() < this.queryBackoffUntil) {
      throw new EmbeddingTimeoutError(`the embedding model did not answer a query within ${this.options.queryTimeoutMs} ms `
        + `at ${this.embedding.lastErrorAt}; queries are skipped until ${new Date(this.queryBackoffUntil).toISOString()} `
        + 'or until it answers again')
    }
    const limit = task === 'query' ? AbortSignal.timeout(this.options.queryTimeoutMs) : null
    const linked = linkedSignal([this.closing.signal, signal, limit])
    let vectors: Float32Array[]
    try {
      vectors = await this.embedder.embed(input, task, linked.signal)
    } catch (error) {
      if (this.closed || signal?.aborted) throw error  // shutdown or the caller gave up: says nothing about the model
      if (refusedInput(error) || !countFailure) throw error  // the input, not the model
      const failure = limit?.aborted
        ? new EmbeddingTimeoutError(`the embedding model did not answer the query within ${this.options.queryTimeoutMs} ms `
          + `(${this.embedder.model()})`, { cause: error })
        : new EmbeddingUnavailableError(describe(error), { cause: error })
      if (limit?.aborted) this.queryBackoffUntil = Date.now() + this.options.queryBackoffMs
      this.embeddingFailed(failure)
      throw failure
    } finally {
      linked.release()
    }
    this.embeddingAnswered()
    return vectors
  }

  private embeddingFailed(error: Error): void {
    const first = this.embedding.ok !== false
    this.embedding = { ok: false, lastError: error.message.slice(0, 500), lastErrorAt: new Date().toISOString() }
    if (first) {
      this.log('warn', `embedding model unavailable: ${error.message.slice(0, 500)}. Messages are still stored; they become `
        + 'searchable when it answers again (logged once per outage)')
    }
  }

  private embeddingAnswered(): void {
    this.queryBackoffUntil = 0
    const recovered = this.embedding.ok === false
    this.embedding = { ...this.embedding, ok: true }
    if (!recovered || this.closed) return
    this.log('info', 'embedding model answers again; indexing the messages stored meanwhile')
    if (this.options.onEmbeddingRecovered) this.options.onEmbeddingRecovered()
    else void this.backfill().catch(() => {})
  }

  /**
   * One short document embedding, for hosts that want the model's state before (or without) any traffic. Never throws;
   * true when it answered. A success after an outage starts the backfill.
   */
  async probeEmbedding(): Promise<boolean> {
    try {
      await this.embed([PROBE], 'document')
      return true
    } catch {
      return false
    }
  }

  /** Chat requests (facts, update links, rerank) go through here, for `status()`; `close()` aborts them. */
  private async complete(messages: readonly p.ChatMessage[], options: { maxTokens: number, schema: object, name: string, signal?: AbortSignal }): Promise<string> {
    const linked = linkedSignal([this.closing.signal, options.signal])
    try {
      const reply = await this.chat!.complete(messages, { ...options, signal: linked.signal })
      this.chatState = { ...this.chatState, ok: true }
      return reply
    } catch (error) {
      if (!linked.signal.aborted) this.chatFailed(error)
      throw error
    } finally {
      linked.release()
    }
  }

  private chatFailed(error: unknown): void {
    this.chatState = { ok: false, lastError: describe(error).slice(0, 500), lastErrorAt: new Date().toISOString() }
  }

  /** The embedding model's state (cheap; `status()` also counts unindexed units). */
  embeddingState(): ModelStatus {
    return { ...this.embedding }
  }

  /**
   * Embedding and chat model state, and how many messages and facts have no vector of the current model yet
   * (stored while the embedding model was down, or refused by it).
   */
  status(): { embedding: ModelStatus, chat: ModelStatus | null, unindexed: number } {
    return { embedding: { ...this.embedding }, chat: this.chat ? { ...this.chatState } : null,
      unindexed: this.closed ? 0 : this.store.unindexed(this.embedder.model()) }
  }

  private refuse(u: p.Unit, error: unknown): void {
    if (this.refused.has(u.id)) return
    this.refused.add(u.id)
    this.log('warn', `${u.id} is not indexed: the embedder refuses it (${String(error).slice(0, 300)}); `
      + 'the other units are indexed, and backfill tries it again (logged once)')
  }

  /**
   * Embeds windows and facts that are stored but not indexed (an embedding failure, or a model change), or indexed
   * with other text (windows of the third and later messages of a session, which capture built from the wrong
   * messages before 2026-10-23). A unit the embedder refuses is skipped (see `embedDocuments`), so the units after it
   * are still indexed; an embedder that is down still fails the call.
   */
  backfill(batch = 32): Promise<number> {
    if (this.backfilling) {
      // A request during a run (a forget, a recovered model) may concern units that run did not list: one more pass.
      this.backfillAgain = true
      return this.backfilling
    }
    const run = (async () => {
      let stored = 0
      do {
        this.backfillAgain = false
        stored += await this.backfillOnce(batch)
      } while (this.backfillAgain && !this.closed)
      return stored
    })()
    // Cleared after the assignment (see `drain`).
    this.backfilling = run.finally(() => { this.backfilling = null })
    return this.backfilling
  }

  private async backfillOnce(batch: number): Promise<number> {
    if (this.closed) return 0
    const model = this.embedder.model()
    const indexed = new Map(this.allUnits().filter(u => u.model === model).map(u => [u.id, u.text] as const))
    const turns = this.allTurns()
    const bySession = new Map<string, StoredTurn[]>()
    for (const t of turns) {
      const list = bySession.get(t.session)
      if (list) list.push(t)
      else bySession.set(t.session, [t])
    }
    const missing: p.Unit[] = []
    for (const list of bySession.values()) {
      // Numbered by stored order: after a forget, a session's positions can have gaps, and a window pairs a message
      // with the stored message before it (as capture does).
      for (const u of p.windowUnitsP2(list.map((t, k) => ({ ...this.asTurn(t), pos: k })))) {
        if (indexed.get(u.id) !== u.text) missing.push(u)
      }
    }
    for (const u of p.factUnits(this.store.facts())) if (indexed.get(u.id) !== u.text) missing.push(u)
    // Units the embedder refused before go last, in batches of their own, so they do not fail the others' batches.
    const order = [...missing.filter(u => !this.refused.has(u.id)), ...missing.filter(u => this.refused.has(u.id))]
    let stored = 0
    try {
      for (let k = 0; k < order.length; k += batch) {
        const part = order.slice(k, k + batch)
        const vectors = await this.embedDocuments(part)
        part.forEach((u, j) => {
          const vector = vectors[j]
          if (!vector || !this.storable(u)) return  // refused, or forgotten while it was being embedded
          this.store.addUnit({ ...u, vector, model })
          stored++
        })
      }
    } finally {
      // Read again only after a write (new or replaced vectors, also when a later batch failed): the backfill every
      // 10 minutes, which usually finds nothing, no longer makes the next recall reload every unit.
      if (stored) {
        this.unitCache = null
        this.ranked = null
      }
    }
    if (this.store.retryCompact()) this.log('info', 'store file rewritten after an earlier forget')
    return stored
  }

  // -- segments and fact extraction -------------------------------------------------------

  private armIdle(session: string): void {
    clearTimeout(this.idleTimers.get(session))
    const timer = setTimeout(() => { this.idleTimers.delete(session); this.maybeCloseSegment(session, true) },
      this.options.segmentIdleMinutes * 60_000)
    timer.unref?.()
    this.idleTimers.set(session, timer)
  }

  /** Closes the open part of a session as a segment when it is long enough, or always when `force`. */
  maybeCloseSegment(session: string, force: boolean): Segment | null {
    const from = this.store.segmentedUpTo(session) + 1
    const open = this.store.sessionTurns(session, from)
    if (!open.length) return null
    const tokens = open.reduce((n, t) => n + estimateTokens(clip(t.text, this.options.messageTokens)), 0)
    if (!force && tokens < this.options.segmentTokens) return null
    const segment = { id: `${session}@${from}`, session, firstPos: from, lastPos: open.at(-1)!.pos, date: open[0]!.date }
    this.store.addSegment(segment)
    return { ...segment, status: 'pending' }
  }

  /** Closes every open segment (shutdown, or an explicit flush). */
  closeAll(): void {
    const sessions = new Set(this.store.turns().map(t => t.session))
    for (const s of sessions) this.maybeCloseSegment(s, true)
  }

  /**
   * Extracts facts for pending segments one at a time. `shouldYield` is checked before each request so a
   * busy chat model (one slot shared with the agent) is not kept waiting.
   */
  drain(shouldYield: () => boolean | Promise<boolean> = () => false, signal?: AbortSignal): Promise<void> {
    if (!this.chat) return Promise.resolve()
    if (!this.draining) {
      // The reset runs after the assignment: with nothing pending the body finishes without awaiting, and a
      // `finally` inside it would clear `draining` before it was set, leaving a settled promise forever.
      const run = (async () => {
        const generation = this.store.generation()
        for (const segment of this.store.pendingSegments(this.options.maxAttempts)) {
          if (this.closed || signal?.aborted || this.store.generation() !== generation || await shouldYield()) break
          if (!await this.extract(segment, signal)) break
        }
      })()
      this.draining = run.finally(() => { this.draining = null })
    }
    return this.draining
  }

  /**
   * @returns false when the chat server is unavailable, or a forget ran while it answered (no attempt spent, the
   *   reply is dropped; the drain stops until the next tick).
   */
  private async extract(segment: Segment, signal?: AbortSignal): Promise<boolean> {
    const generation = this.store.generation()
    const members = this.store.segmentTurns(segment).map(t => this.asTurn(t))
    if (!members.length) {
      this.store.finishSegment(segment.id, [], undefined, this.options.maxAttempts)
      return true
    }
    const local = (t: p.Turn) => `M${t.pos - segment.firstPos + 1}`
    const { messages, validIds } = p.factMessages(members.map(t => ({ ...t, date: segment.date })), local)
    let reply: string
    try {
      reply = await this.complete(messages, { maxTokens: p.PROMPTS.fact.maxTokens, schema: p.PROMPTS.fact.schema, name: 'fact', signal })
    } catch (error) {
      if (this.closed || signal?.aborted) return false  // shutdown: no attempt spent
      if ((error as Error).name === 'ChatUnavailableError') {
        if (!this.chatDown) this.log('warn', `fact extraction waits for the chat model: ${String(error)}`)
        this.chatDown = true
        return false
      }
      this.store.finishSegment(segment.id, null, String(error), this.options.maxAttempts)
      this.log('warn', `fact extraction failed for ${segment.id}: ${String(error)}`)
      return true
    }
    if (this.chatDown) this.log('info', 'chat model reachable again; fact extraction resumed')
    this.chatDown = false
    // The reply may draw on messages deleted while the model answered.
    if (this.closed || this.store.generation() !== generation) return false
    const parsed = p.parseFacts(reply, validIds)
    if (parsed === null) {
      this.store.finishSegment(segment.id, null, 'unparseable reply', this.options.maxAttempts)
      return true
    }
    const byLocal = new Map(members.map(t => [local(t), t.id] as const))
    const facts = parsed.map(f => ({ fact: f.fact, turns: f.turns.map(id => byLocal.get(id)!) }))
    if (!this.store.finishSegment(segment.id, facts, undefined, this.options.maxAttempts)) return true
    await this.index(p.factUnits(new Map([[segment.id, facts]])))
    this.log('info', `extracted ${facts.length} facts from ${segment.id}`)
    if (this.options.updateLinks && facts.length) await this.link(segment.id, signal)
    return true
  }

  /** Facts of every extracted segment, ranked by segment date (creation order breaks ties). */
  private factRows(): p.FactRow[] {
    const facts = this.store.facts(), dates = this.store.segmentDates()
    const ids = [...facts.keys()].filter(id => dates.has(id))
    const order = ids.map((id, i) => [id, i] as const)
      .sort(([a, i], [b, j]) => { const da = p.when(dates.get(a)!), db = p.when(dates.get(b)!); return da === db ? i - j : (da < db ? -1 : 1) })
    return p.factTable(order.map(([id]) => ({ session: id, date: dates.get(id)!, facts: facts.get(id)! })))
  }

  /** One update check: this segment's facts against the most similar facts of earlier segments. */
  private async link(segmentId: string, signal?: AbortSignal): Promise<void> {
    const rows = this.factRows()
    const fresh = rows.filter(r => r.session === segmentId)
    if (!fresh.length) return
    const earlier = rows.filter(r => r.rank < fresh[0]!.rank)
    const vectors = new Map(this.allUnits().filter(u => u.set === 'fact').map(u => [u.id.slice('fact|'.length), u.vector] as const))
    if ([...fresh, ...earlier].some(r => !vectors.has(r.key))) return this.log('warn', `update check skipped for ${segmentId}: facts not indexed`)
    const job = p.updateJob(fresh, earlier, vectors)
    if (!job) return
    const messages = p.updateMessages(job, new Map(rows.map(r => [r.key, r] as const)))
    for (let attempt = 1; ; attempt++) {
      try {
        const reply = await this.complete(messages, { maxTokens: p.UPDATE.maxTokens, schema: p.UPDATE.schema, name: 'updates', signal })
        if (this.closed) return
        const links = p.parseUpdates(reply, job) ?? []
        this.store.addLinks(links)
        if (links.length) this.log('info', `${links.length} facts outdated by ${segmentId}`)
        return
      } catch (error) {
        if (this.closed || signal?.aborted) return
        // The check runs once, right after the extraction: a chat server that is briefly unreachable must not cost it
        // (before 2026-10-25 a dropped connection lost the check; 11% of them in a 5,000-session build).
        if ((error as Error).name === 'ChatUnavailableError' && attempt < LINK_ATTEMPTS) {
          await new Promise(resolve => setTimeout(resolve, LINK_RETRY_MS * attempt))
          continue
        }
        this.log('warn', `update check failed for ${segmentId}: ${String(error)}`)
        return
      }
    }
  }

  // -- retrieval ---------------------------------------------------------------------------

  /**
   * The ranked index for vectors of `model` with `dims` dimensions: p2 windows in capture order, then facts in segment
   * order (the benchmarks' build order). Built from the caches when missing, and kept until a write changes it.
   */
  private rankIndex(model: string, dims: number): RankIndex {
    const all = this.allUnits()
    let index = this.ranked
    if (!index || index.model !== model || index.dims !== dims) {
      const turns = this.turnIndex()
      const windows = all.filter(u => u.set === 'p2')
      const usable = windows.filter(u => u.model === model && u.vector.length === dims)
      index = this.ranked = new RankIndex(model, dims, windowOrder(usable, id => turns.get(id)?.seq), windows.length - usable.length)
    }
    if (!index.hasFacts()) {
      const facts = all.filter(u => u.set === 'fact')
      const usable = facts.filter(u => index!.usable(u))
      index.setFacts(factOrder(usable, [...this.store.facts().keys()]), facts.length - usable.length)
    }
    return index
  }

  /** The reranker's view of the top candidates' messages, in capture order (all `rerankMessages` reads of them). */
  private rerankTurns(order: readonly number[], units: readonly StoredUnit[]): p.Turn[] {
    const turns = this.turnIndex()
    const ids = new Set(order.slice(0, p.RERANK_CANDIDATES).flatMap(i => units[i]!.turns))
    return [...ids].flatMap(id => turns.get(id) ?? []).sort((a, b) => a.seq - b.seq)
      .map(t => this.asTurn(t, this.options.rerankMessageTokens))
  }

  /**
   * Excerpts for `query`. Messages of `excludeSessions`, and messages `inContext` reports as still in the model's
   * context (working memory: the unsummarized part of the current session), are not returned.
   * With `rerank`, the chat model reorders the top 30 candidates (evaluated C-final); without it, dense only.
   *
   * Only the ranked candidates' messages are clipped and looked up, and the ranked index is kept between calls
   * (before 2026-10-24 every stored message was clipped and every unit sorted on each call: 209 ms at 5,000 sessions).
   *
   * The query embedding is limited to `queryTimeoutMs` (throws {@link EmbeddingTimeoutError}); a rerank to
   * `rerankTimeoutMs`, and it is skipped at once when the chat server does not answer at all. Either way the dense order
   * is returned. `signal` (the caller giving up) aborts everything and throws.
   */
  async retrieve(query: string, options: { rerank: boolean, excludeSessions?: readonly string[], budgetTokens?: number,
    signal?: AbortSignal, inContext?: (turn: { id: string, session: string }) => boolean } = { rerank: false }): Promise<Retrieval> {
    const indexed = this.allUnits().some(u => u.set === 'p2' || u.set === 'fact')
    // Nothing indexed: nothing to find, unless the embedding model is down while messages wait to be indexed; then the
    // query is embedded anyway, so the caller learns why (and an answer starts the backfill).
    if (!query.trim() || (!indexed && (this.embedding.ok !== false || !this.allTurns().length))) {
      return { text: '', ids: [], reranked: false, candidates: 0 }
    }
    // The query may go to a remote embedding endpoint or the reranker: the same redaction as stored text.
    if (this.options.redactSecrets) query = redact(query).text
    const [qv] = await this.embed([query], 'query', options.signal)
    if (!indexed) return { text: '', ids: [], reranked: false, candidates: 0 }
    // After the await: a forget or a capture may have run while the query was embedded.
    const index = this.rankIndex(this.embedder.model(), qv!.length)
    if (index.skipped()) this.log('warn', `${index.skipped()} units embedded with another model are skipped`)
    const kept = index.units()
    let order = p.distinct(index.rank(qv!).map(([i]) => i), kept)
    let reranked = false
    if (options.rerank && this.chat && order.length) {
      const limit = AbortSignal.timeout(this.options.rerankTimeoutMs)
      let silent = false
      try {
        if (this.chat.reachable && !await this.chat.reachable(options.signal)) {
          silent = true
          throw new Error('the chat server does not answer (GET /models)')
        }
        const reply = await this.complete(p.rerankMessages(query, order, kept, this.rerankTurns(order, kept)),
          { maxTokens: p.PROMPTS.rerank.maxTokens, schema: p.PROMPTS.rerank.schema, name: 'rerank',
            signal: AbortSignal.any([limit, ...options.signal ? [options.signal] : []]) })
        order = p.distinct(p.rerankedOrder(order, p.parseRerank(reply, Math.min(order.length, p.RERANK_CANDIDATES))), kept)
        reranked = true
      } catch (error) {
        if (options.signal?.aborted || this.closed) throw error
        const why = limit.aborted ? `no answer within ${this.options.rerankTimeoutMs} ms` : describe(error)
        const at = this.chat.baseUrl ? ` (${this.chat.baseUrl})` : ''
        // Other failures were recorded by `complete`; these two never reached it as failures.
        if (limit.aborted || silent) this.chatFailed(new Error(`rerank: ${why}${at}`))
        this.log('warn', `rerank failed, dense order kept: ${why}${at}`)
      }
    }
    const raw = p.rawIds(order, kept)
    const { order: ranked, notes } = this.options.updateLinks
      ? p.versioned(raw, this.factRows(), this.store.links())
      : { order: raw, notes: new Map<string, string[]>() }
    // Read after the last await: a message forgotten meanwhile has no text to return.
    const turns = this.turnIndex()
    const excluded = new Set(options.excludeSessions ?? [])
    const exclude: string[] = [], lookup = new Map<string, p.Turn>()
    for (const id of ranked) {
      const t = turns.get(id)
      if (!t) continue
      const plain = this.asTurn(t)
      if (excluded.has(t.session) || options.inContext?.(plain)) exclude.push(id)
      // A dsh session can span days: excerpts get a Date line per session and day.
      lookup.set(id, { ...plain, session: `${plain.session}|${plain.date.slice(0, 10)}` })
    }
    const { text, ids } = p.selectMemory(ranked, lookup, exclude, options.budgetTokens ?? this.options.budgetTokens,
      estimateTokens, ts => p.renderDated(ts, notes))
    return { text, ids, reranked, candidates: order.length }
  }

  stats(): Record<string, number> {
    return this.store.stats()
  }

  // -- forgetting ----------------------------------------------------------------------------

  /**
   * Deletes the messages in `scope` and everything derived from them (see `MemoryStore.forget`), and the file is
   * rewritten without them. Kept messages whose window showed a deleted one lose that window until the next
   * `backfill()`, which the hosts start right after a forget (it also retries a rewrite another process blocked).
   * Hosts serialize this with their pending captures, so a message captured just before the request is covered too.
   */
  forget(scope: ForgetScope): ForgetResult {
    const { reindex, ...result } = this.store.forget(scope)
    this.refresh()
    if ('all' in scope) {
      for (const timer of this.idleTimers.values()) clearTimeout(timer)
      this.idleTimers.clear()
    } else if ('session' in scope) {
      clearTimeout(this.idleTimers.get(scope.session))
      this.idleTimers.delete(scope.session)
    }
    const what = 'all' in scope ? 'all memory' : 'session' in scope ? 'one session' : `messages before ${formatDate(scope.before)}`
    this.log('info', `forgot ${what}: ${result.turns} messages, ${result.units} index entries, ${result.facts} facts`
      + `${reindex.length ? `; ${reindex.length} windows to re-index` : ''}${result.vacuumed ? '' : '; file not yet compacted (in use)'}`)
    return result
  }

  /** Stops: nothing is stored after this, and every model request in flight is aborted. */
  close(): void {
    this.closed = true
    this.closing.abort(new Error('memory closed'))
    for (const timer of this.idleTimers.values()) clearTimeout(timer)
    this.idleTimers.clear()
  }
}

/** An error and its cause in one line (`fetch failed` alone does not say what failed). */
export function describe(error: unknown): string {
  const e = error as { message?: unknown, cause?: { message?: unknown, code?: unknown } } | null
  const message = typeof e?.message === 'string' ? e.message : String(error)
  const cause = e?.cause?.message ?? e?.cause?.code
  return cause && !message.includes(String(cause)) ? `${message} (${String(cause)})` : message
}

function splitFact(id: string): [string, number] {
  const body = id.slice('fact|'.length)
  const cut = body.lastIndexOf('#')
  return [body.slice(0, cut), Number(body.slice(cut + 1))]
}
