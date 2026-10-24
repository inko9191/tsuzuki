/**
 * The evaluated memory pipeline, ported from the Python research code with identical rules:
 * `scripts/eval/locomo_memory_methods.py` (units, fact/rerank prompts and parsers, ranking, RRF,
 * distinct, rerank merge), `locomo_answer_production.py` (rerank passages), `locomo_answer_methods.py`
 * (frozen excerpt rendering and budgeted selection) and `longmemeval_u1_methods.py` (date-ordered
 * rendering, update links). `test/conformance.test.mjs` replays fixtures produced by the Python code.
 *
 * Pure functions only: no I/O, no model calls.
 */

/** One stored message. `id` is unique; `session` groups messages written together; `pos` is the index in it. */
export interface Turn {
  id: string
  session: string
  pos: number
  /** Session date, `YYYY/MM/DD (Ddd) HH:MM` like the benchmarks. */
  date: string
  /** `role: content`, e.g. `user: I moved to Kyoto.` */
  text: string
}

/** What gets embedded and ranked; it returns raw turns only. */
export interface Unit {
  id: string
  set: 'p2' | 'fact'
  session: string
  text: string
  turns: string[]
}

export interface Fact {
  fact: string
  turns: string[]
}

export const POOL = 100
export const RRF_K = 60
export const RERANK_CANDIDATES = 30
export const RERANK_KEEP = 10
export const MAX_TEXT = 600
export const SYSTEM = 'You prepare text for a search index over personal chat logs. Answer with JSON only.'

export const PROMPTS = {
  fact: {
    maxTokens: 1500,
    schema: {
      type: 'object', properties: {
        facts: {
          type: 'array', items: {
            type: 'object', properties: { fact: { type: 'string' }, ids: { type: 'array', items: { type: 'string' } } },
            required: ['fact', 'ids'],
          },
        },
      }, required: ['facts'],
    },
    user: (date: string, messages: string) => `Chat session date: ${date}\n\nMessages (id, speaker: text):\n${messages}\n\n`
      + 'List the facts in this session that someone might later ask about: events, plans, activities, '
      + 'possessions, preferences, relationships, feelings and when things happened. Write each fact as one '
      + 'self-contained sentence that names the person, with dates resolved from the session date where '
      + 'possible. For each fact give the ids of the messages that state it. '
      + 'Return JSON: {"facts": [{"fact": "...", "ids": ["..."]}]}',
  },
  rerank: {
    maxTokens: 120,
    schema: {
      type: 'object', properties: { ranking: { type: 'array', items: { type: 'integer' }, maxItems: RERANK_KEEP } },
      required: ['ranking'],
    },
    user: (question: string, passages: string) => `Question about a long chat between two friends:\n${question}\n\n`
      + `Retrieved passages:\n${passages}\n\n`
      + `Choose up to ${RERANK_KEEP} passages that are most useful for answering the question, most useful `
      + 'first. Passages that together answer it are all useful. Return JSON: {"ranking": [numbers]}',
  },
} as const

export interface ChatMessage {
  role: 'system' | 'user'
  content: string
}

// -- text helpers (Python semantics) -------------------------------------------------

/** Python `str.split()` whitespace (includes \x1c-\x1f and unicode spaces). */
const PY_SPACE = /[\s\x1c-\x1f\x85]+/u

/** `' '.join(value.split())[:600]`, slicing by code points like Python. */
export function clean(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const joined = value.split(PY_SPACE).filter(Boolean).join(' ')
  if (!joined) return null
  return Array.from(joined).slice(0, MAX_TEXT).join('')
}

/** The JSON object in a reply, tolerating a thinking block or a code fence around it. */
export function jsonObject(text: string | null | undefined): Record<string, unknown> | null {
  const stripped = (text ?? '').replace(/<think>[\s\S]*?<\/think>/g, '').trim()
  const start = stripped.indexOf('{'), end = stripped.lastIndexOf('}')
  if (start < 0 || end < start) return null
  try {
    const value: unknown = JSON.parse(stripped.slice(start, end + 1))
    return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : null
  } catch {
    return null
  }
}

function unique<T>(items: Iterable<T>): T[] {
  return [...new Set(items)]
}

// -- units ---------------------------------------------------------------------------

/** Turns grouped by session, preserving first-appearance order (Python `defaultdict(list)`). */
export function sessionsOf(turns: readonly Turn[]): Map<string, Turn[]> {
  const out = new Map<string, Turn[]>()
  for (const t of turns) {
    const list = out.get(t.session)
    if (list) list.push(t)
    else out.set(t.session, [t])
  }
  return out
}

/** `p2`: the previous turn plus the centre, never across a session; centre returned first. */
export function windowUnitsP2(turns: readonly Turn[]): Unit[] {
  const sessions = sessionsOf(turns)
  return turns.map(turn => {
    const session = sessions.get(turn.session)!
    const i = turn.pos
    const span = session.slice(Math.max(0, i - 1), i + 1)
    const order = [...span].sort((a, b) => (Math.abs(a.pos - i) - Math.abs(b.pos - i)) || (a.pos - b.pos))
    return { id: `p2|${turn.id}`, set: 'p2' as const, session: turn.session,
      text: span.map(t => t.text).join('\n'), turns: order.map(t => t.id) }
  })
}

/** One unit per extracted fact; it returns exactly the turns it cites. Keys follow `facts` insertion order. */
export function factUnits(facts: ReadonlyMap<string, readonly Fact[]>): Unit[] {
  const units: Unit[] = []
  for (const [session, items] of facts) {
    items.forEach((fact, j) => units.push({ id: `fact|${session}#${j}`, set: 'fact', session, text: fact.fact, turns: [...fact.turns] }))
  }
  return units
}

// -- fact extraction ----------------------------------------------------------------

/**
 * Prompt for one session (or one session segment). `localId` maps a turn to the short id shown to the
 * model; the benchmarks show `D{session}:{turn}`.
 */
export function factMessages(members: readonly Turn[], localId: (t: Turn) => string): { messages: ChatMessage[], validIds: string[] } {
  const ids = members.map(localId)
  const lines = members.map((t, k) => `[${ids[k]}] ${t.text}`).join('\n')
  return {
    messages: [{ role: 'system', content: SYSTEM }, { role: 'user', content: PROMPTS.fact.user(members[0]?.date || 'unknown', lines) }],
    validIds: ids,
  }
}

/** Validated facts citing only shown ids, or null when the reply is not the expected JSON. */
export function parseFacts(reply: string, validIds: readonly string[]): Fact[] | null {
  const value = jsonObject(reply)
  if (value === null || !Array.isArray(value.facts)) return null
  const allowed = new Set(validIds)
  const facts: Fact[] = []
  for (const item of value.facts.slice(0, 60)) {
    if (typeof item !== 'object' || item === null || Array.isArray(item)) continue
    const record = item as Record<string, unknown>
    const text = clean(record.fact)
    if (!text || !Array.isArray(record.ids)) continue
    const ids = unique(record.ids.filter((i): i is string => typeof i === 'string' && allowed.has(i.trim())).map(i => i.trim()))
    if (ids.length) facts.push({ fact: text, turns: ids })
  }
  return facts
}

// -- ranking ---------------------------------------------------------------------------

export type Ranked = Array<[number, number]>

/** Best n by score; ties keep corpus order (numpy stable argsort of -score). */
export function top(indices: readonly number[], scores: ArrayLike<number>, n = POOL): Ranked {
  const order = indices.map((_, j) => j).sort((a, b) => (scores[b]! - scores[a]!) || (a - b))
  return order.slice(0, n).map(j => [indices[j]!, scores[j]!])
}

export function rrf(lists: readonly Ranked[]): Ranked {
  if (lists.length === 1) return lists[0]!
  const score = new Map<number, number>()
  for (const items of lists) {
    items.forEach(([item], k) => score.set(item, (score.get(item) ?? 0) + 1 / (RRF_K + k + 1)))
  }
  return [...score.entries()].sort((a, b) => (b[1] - a[1]) || (a[0] - b[0]))
}

/** Dense cosine ranking per unit set, fused with RRF (C-final: sets p2 and fact, no post-processing). */
export function rank(units: readonly Unit[], vectors: readonly Float32Array[], query: Float32Array,
  sets: readonly Unit['set'][] = ['p2', 'fact']): Ranked {
  const lists: Ranked[] = []
  for (const set of sets) {
    const members = units.flatMap((u, i) => u.set === set ? [i] : [])
    if (!members.length) continue
    lists.push(top(members, members.map(i => dot(vectors[i]!, query))))
  }
  return rrf(lists.length ? lists : [[]])
}

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0
  for (let k = 0; k < a.length; k++) s += a[k]! * b[k]!
  return s
}

/** Units in rank order, skipping any unit that returns no turn not already returned. */
export function distinct(order: readonly number[], units: readonly Unit[], limit = POOL): number[] {
  const seen = new Set<string>(), out: number[] = []
  for (const i of order) {
    const fresh = units[i]!.turns.filter(t => !seen.has(t))
    if (fresh.length) {
      out.push(i)
      fresh.forEach(t => seen.add(t))
      if (out.length === limit) break
    }
  }
  return out
}

/** Passages for the reranker: each candidate's turns in conversation order, numbered from 1. */
export function rerankMessages(question: string, order: readonly number[], units: readonly Unit[], turns: readonly Turn[]): ChatMessage[] {
  const index = new Map(turns.map((t, n) => [t.id, n] as const))
  const lines = order.slice(0, RERANK_CANDIDATES).map((i, k) => {
    const ids = [...units[i]!.turns].sort((a, b) => index.get(a)! - index.get(b)!)
    return `[${k + 1}] ${ids.map(id => turns[index.get(id)!]!.text).join('\n')}`
  })
  return [{ role: 'system', content: SYSTEM }, { role: 'user', content: PROMPTS.rerank.user(question, lines.join('\n\n')) }]
}

/** Zero-based picks, or null when the reply is not the expected JSON. Python treats booleans as 0/1. */
export function parseRerank(reply: string, nPassages: number): number[] | null {
  const value = jsonObject(reply)
  if (value === null || !Array.isArray(value.ranking)) return null
  const picks: number[] = []
  for (const raw of value.ranking) {
    const v = typeof raw === 'boolean' ? Number(raw) : raw
    if (typeof v === 'number' && Number.isInteger(v) && v >= 1 && v <= nPassages) picks.push(v - 1)
  }
  return unique(picks).slice(0, RERANK_KEEP)
}

/** Reranker's picks first (valid, unique), then the remaining candidates, then the rest of the order. */
export function rerankedOrder(order: readonly number[], picks: readonly number[] | null): number[] {
  const candidates = order.slice(0, RERANK_CANDIDATES)
  const chosen = unique((picks ?? []).map(j => candidates[j]).filter((i): i is number => i !== undefined))
  const picked = new Set(chosen)
  return [...chosen, ...candidates.filter(i => !picked.has(i)), ...order.slice(RERANK_CANDIDATES)]
}

/** Raw turns in final order, first occurrence wins. */
export function rawIds(order: readonly number[], units: readonly Unit[]): string[] {
  return unique(order.flatMap(i => units[i]!.turns))
}

// -- excerpt rendering and budgeted selection -----------------------------------------

/** Frozen C rendering: a Date line at each session's first appearance, in the given order. */
export function renderFrozen(turns: readonly Turn[]): string {
  const paid = new Set<string>(), lines: string[] = []
  for (const t of turns) {
    if (!paid.has(t.session)) {
      lines.push(`Date: ${t.date || 'unknown'}`)
      paid.add(t.session)
    }
    lines.push(`[${t.id}] ${t.text}`)
  }
  return lines.join('\n')
}

const DATE = /^(\d{4})\/(\d{2})\/(\d{2}) \(\w{3}\) (\d{2}):(\d{2})$/

/** Sort key of a benchmark-style date; unparseable dates sort last in their given order. */
export function when(date: string): number {
  const m = DATE.exec(date)
  return m ? Date.UTC(+m[1]!, +m[2]! - 1, +m[3]!, +m[4]!, +m[5]!) : Number.POSITIVE_INFINITY
}

/** Sessions by date; equal dates keep first-appearance order. */
export function sessionOrder(turns: readonly Turn[]): string[] {
  const first = new Map<string, [number, number]>()
  turns.forEach((t, n) => { if (!first.has(t.session)) first.set(t.session, [when(t.date), n]) })
  return [...first.keys()].sort((a, b) => {
    const [da, na] = first.get(a)!, [db, nb] = first.get(b)!
    return da === db ? na - nb : (da < db ? -1 : 1)
  })
}

/** Date-ordered rendering (U1 S/T/V): a Date line opens every session block; notes follow their message. */
export function renderDated(turns: readonly Turn[], notes?: ReadonlyMap<string, readonly string[]>): string {
  const position = new Map(sessionOrder(turns).map((s, i) => [s, i] as const))
  const sorted = [...turns].sort((a, b) => (position.get(a.session)! - position.get(b.session)!) || (a.pos - b.pos))
  const lines: string[] = []
  let current: string | null = null
  for (const t of sorted) {
    if (t.session !== current) {
      lines.push(`Date: ${t.date || 'unknown'}`)
      current = t.session
    }
    lines.push(`[${t.id}] ${t.text}`)
    for (const n of notes?.get(t.id) ?? []) lines.push(`  ${n}`)
  }
  return lines.join('\n')
}

/**
 * Frozen budgeted selection: walk the rank order, skip excluded or repeated turns, stop at the first
 * turn whose addition would exceed `budget` tokens (no skipping ahead).
 */
export function selectMemory(order: readonly string[], lookup: ReadonlyMap<string, Turn>, exclude: readonly string[],
  budget: number, count: (text: string) => number, render: (turns: readonly Turn[]) => string): { text: string, ids: string[] } {
  const seen = new Set(exclude), chosen: Turn[] = []
  for (const id of order) {
    if (seen.has(id)) continue
    const turn = lookup.get(id)
    if (!turn) continue
    const trial = [...chosen, turn]
    if (count(render(trial)) > budget) break
    chosen.push(turn)
    seen.add(id)
  }
  return { text: render(chosen), ids: chosen.map(t => t.id) }
}

// -- write-time update links (U1 arm V) -------------------------------------------------

/** A fact with the date order of its session (segment); `key` is `session#j` like the fact units. */
export interface FactRow {
  key: string
  session: string
  rank: number
  date: string
  text: string
  turns: string[]
}

export interface UpdateJob {
  session: string
  date: string
  earlier: string[]
  new: string[]
}

export const TOP_EARLIER = 2
export const MAX_EARLIER = 20
export const MAX_NOTES = 2

export const UPDATE = {
  maxTokens: 300,
  schema: {
    type: 'object', properties: {
      updates: {
        type: 'array', maxItems: MAX_EARLIER, items: {
          type: 'object', properties: { old: { type: 'string' }, new: { type: 'string' } },
          required: ['old', 'new'], additionalProperties: false,
        },
      },
    }, required: ['updates'], additionalProperties: false,
  },
  user: (earlier: string, date: string, newer: string) => `Earlier facts from a personal chat history:\n${earlier}\n\n`
    + `New facts from a later chat session dated ${date}:\n${newer}\n\n`
    + 'An earlier fact is outdated when a new fact changes any detail in it: where someone lives or works, '
    + 'a number, amount or count, a status, possession, preference, habit, schedule, plan or date. The detail '
    + 'can be about the user or anyone else, and it can be mentioned in passing. It is not outdated when the new '
    + 'fact only adds information, repeats it, or is about something else. List every outdated earlier fact with '
    + 'the new fact that changes it. '
    + 'Return JSON: {"updates": [{"old": "E1", "new": "N1"}]} with an empty list if none.',
} as const

/** Facts of `sessions` (already in date order) keyed `session#j`, each with its session's date rank. */
export function factTable(sessions: ReadonlyArray<{ session: string, date: string, facts: readonly Fact[] }>): FactRow[] {
  return sessions.flatMap(({ session, date, facts }, rank) =>
    facts.map((f, j) => ({ key: `${session}#${j}`, session, rank, date, text: f.fact, turns: [...f.turns] })))
}

/** The check for one later session: its facts against the most similar facts of earlier sessions. */
export function updateJob(fresh: readonly FactRow[], earlier: readonly FactRow[],
  vectors: ReadonlyMap<string, Float32Array>): UpdateJob | null {
  if (!fresh.length || !earlier.length) return null
  const best = new Map<number, number>()
  for (const f of fresh) {
    const v = vectors.get(f.key)!
    const sims = earlier.map(e => dot64(vectors.get(e.key)!, v))
    const order = sims.map((_, i) => i).sort((a, b) => (sims[b]! - sims[a]!) || (a - b)).slice(0, TOP_EARLIER)
    for (const i of order) best.set(i, Math.max(best.get(i) ?? -2, sims[i]!))
  }
  const picked = [...best.keys()].sort((a, b) => (best.get(b)! - best.get(a)!) || (a - b)).slice(0, MAX_EARLIER)
    .sort((a, b) => (earlier[a]!.rank - earlier[b]!.rank) || (a - b))
  return { session: fresh[0]!.session, date: fresh[0]!.date, earlier: picked.map(i => earlier[i]!.key), new: fresh.map(f => f.key) }
}

/** All checks over a table (one per later session), as the U1 runner builds them. */
export function updateJobs(table: readonly FactRow[], vectors: ReadonlyMap<string, Float32Array>): UpdateJob[] {
  const ranks = [...new Set(table.map(f => f.rank))].sort((a, b) => a - b)
  return ranks.flatMap(r => {
    const job = updateJob(table.filter(f => f.rank === r), table.filter(f => f.rank < r), vectors)
    return job ? [job] : []
  })
}

function dot64(a: Float32Array, b: Float32Array): number {
  let s = 0
  for (let k = 0; k < a.length; k++) s += a[k]! * b[k]!
  return s
}

export function updateMessages(job: UpdateJob, facts: ReadonlyMap<string, FactRow>): ChatMessage[] {
  const earlier = job.earlier.map((k, i) => `[E${i + 1}] (${facts.get(k)!.date}) ${facts.get(k)!.text}`).join('\n')
  const newer = job.new.map((k, i) => `[N${i + 1}] ${facts.get(k)!.text}`).join('\n')
  return [{ role: 'system', content: SYSTEM }, { role: 'user', content: UPDATE.user(earlier, job.date, newer) }]
}

/** Valid label pairs only; null when the reply is not the expected JSON. */
export function parseUpdates(text: string, job: UpdateJob): Array<{ old: string, new: string }> | null {
  const value = jsonObject(text)
  if (value === null || !Array.isArray(value.updates)) return null
  const links: Array<{ old: string, new: string }> = [], seen = new Set<string>()
  for (const item of value.updates) {
    if (typeof item !== 'object' || item === null || Array.isArray(item)) continue
    const record = item as Record<string, unknown>
    const old = pyStr(record.old).trim(), neu = pyStr(record.new).trim()
    if (!(old[0] === 'E' && neu[0] === 'N' && /^\d+$/.test(old.slice(1)) && /^\d+$/.test(neu.slice(1)))) continue
    const i = Number(old.slice(1)), j = Number(neu.slice(1))
    if (!(i >= 1 && i <= job.earlier.length && j >= 1 && j <= job.new.length)) continue
    const pair = [job.earlier[i - 1]!, job.new[j - 1]!] as const
    const key = JSON.stringify(pair)
    if (!seen.has(key)) {
      seen.add(key)
      links.push({ old: pair[0], new: pair[1] })
    }
  }
  return links
}

/** Python `str(value)` for the JSON values a model returns (`None` for null, `True`/`False`). */
function pyStr(value: unknown): string {
  if (value === undefined) return ''
  if (value === null) return 'None'
  if (value === true) return 'True'
  if (value === false) return 'False'
  return typeof value === 'string' ? value : JSON.stringify(value)
}

/** Each outdated fact -> its newest replacements, following chains to facts that are not outdated. */
export function replacements(links: ReadonlyArray<{ old: string, new: string }>, facts: ReadonlyMap<string, FactRow>): Map<string, string[]> {
  const next = new Map<string, string[]>()
  for (const l of links) next.set(l.old, [...(next.get(l.old) ?? []), l.new])
  const result = new Map<string, string[]>()
  for (const old of next.keys()) {
    const frontier = [...next.get(old)!], seen = new Set([old]), final: string[] = []
    while (frontier.length) {
      const k = frontier.shift()!
      if (seen.has(k)) continue
      seen.add(k)
      if (next.has(k)) frontier.push(...next.get(k)!)
      else final.push(k)
    }
    const newest = unique(final).sort((a, b) => (facts.get(b)!.rank - facts.get(a)!.rank) || (a < b ? -1 : a > b ? 1 : 0)).slice(0, MAX_NOTES)
    if (newest.length) result.set(old, newest)
  }
  return result
}

/** Rank order with each outdated message followed by the messages of its replacements, plus notes. */
export function versioned(rawIds: readonly string[], table: readonly FactRow[],
  links: ReadonlyArray<{ old: string, new: string }>): { order: string[], notes: Map<string, string[]> } {
  const facts = new Map(table.map(f => [f.key, f] as const))
  const newer = replacements(links, facts)
  const byTurn = new Map<string, string[]>()
  for (const f of table) for (const t of f.turns) byTurn.set(t, [...(byTurn.get(t) ?? []), f.key])
  const notes = new Map<string, string[]>(), order: string[] = [], emitted = new Set<string>()
  for (const id of rawIds) {
    if (!emitted.has(id)) { order.push(id); emitted.add(id) }
    const repl = unique((byTurn.get(id) ?? []).flatMap(k => newer.get(k) ?? []))
    if (!repl.length) continue
    notes.set(id, repl.map(r => `Later update (${facts.get(r)!.date}): ${facts.get(r)!.text}`))
    for (const r of repl) for (const t of facts.get(r)!.turns) if (!emitted.has(t)) { order.push(t); emitted.add(t) }
  }
  return { order, notes }
}
