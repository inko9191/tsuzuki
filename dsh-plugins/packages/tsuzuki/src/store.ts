/**
 * Durable memory store on the built-in `node:sqlite` (no native addon to build).
 *
 * - `turns`: every captured message, in capture order (`seq`), grouped into sessions.
 * - `units`: embedded p2 windows and extracted facts with their vectors; a unit returns raw turns only.
 * - `segments`: slices of a session handed to fact extraction (the benchmarks' "session").
 * - `links`: write-time update links between facts (old fact outdated by new fact).
 *
 * `forget()` deletes messages and everything derived from them, then rewrites the file (VACUUM) so the text is gone
 * from disk, not only from queries. Every forget bumps `generation()`, which other connections (another dsh process
 * sharing the file) use to drop their caches and to discard work started before it.
 */
import { DatabaseSync } from 'node:sqlite'
import type { Fact, Turn, Unit } from './pipeline.ts'

export interface StoredTurn extends Turn {
  seq: number
  role: string
  /** Epoch milliseconds when captured. */
  at: number
}

export interface Segment {
  id: string
  session: string
  firstPos: number
  lastPos: number
  date: string
  status: 'pending' | 'done' | 'failed'
}

export interface StoredUnit extends Unit {
  vector: Float32Array
  model: string
}

/** What to forget: one session, every message captured before `before` (epoch ms), or everything. */
export type ForgetScope = { session: string } | { before: number } | { all: true }

/** Rows a forget deleted. */
export interface ForgetResult {
  /** Messages. */
  turns: number
  /** Embedded windows and facts. */
  units: number
  facts: number
  /** Segments deleted, or reset to pending because only some of their messages were deleted. */
  segments: number
  /** Update links deleted (a link through a deleted fact is replaced by one that skips it). */
  links: number
  /**
   * The file was rewritten (VACUUM) and its WAL emptied, so the deleted text is no longer on disk. False when another
   * connection was reading and kept SQLite from doing that: the rows are deleted anyway (their cells zeroed,
   * `secure_delete`), and `retryCompact()` rewrites the file later.
   */
  vacuumed: boolean
}

const SCHEMA = `
PRAGMA journal_mode = WAL;
PRAGMA secure_delete = ON;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS turns (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, session TEXT NOT NULL, pos INTEGER NOT NULL,
  date TEXT NOT NULL, at INTEGER NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session, pos);
CREATE TABLE IF NOT EXISTS units (
  rowid INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, kind TEXT NOT NULL, session TEXT NOT NULL,
  text TEXT NOT NULL, turns TEXT NOT NULL, vector BLOB NOT NULL, model TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS segments (
  id TEXT PRIMARY KEY, session TEXT NOT NULL, first_pos INTEGER NOT NULL, last_pos INTEGER NOT NULL,
  date TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, error TEXT);
-- Every capture asks for its session's last segmented position: without this, a scan of every segment.
CREATE INDEX IF NOT EXISTS segments_session ON segments(session, last_pos);
CREATE TABLE IF NOT EXISTS facts (
  segment TEXT NOT NULL, j INTEGER NOT NULL, fact TEXT NOT NULL, turns TEXT NOT NULL, PRIMARY KEY (segment, j));
CREATE TABLE IF NOT EXISTS links (old TEXT NOT NULL, new TEXT NOT NULL, PRIMARY KEY (old, new));
`

function blob(vector: Float32Array): Uint8Array {
  return new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength)
}

function floats(value: Uint8Array): Float32Array {
  // Copy: the returned buffer may be reused by the driver.
  return new Float32Array(value.slice().buffer)
}

/** `[segment, j]` of a fact key `segment#j` (a session id may itself contain `#`). */
function splitKey(key: string): [string, number] {
  const cut = key.lastIndexOf('#')
  return [key.slice(0, cut), Number(key.slice(cut + 1))]
}

export class MemoryStore {
  readonly db: DatabaseSync
  /** A forget deleted rows but could not rewrite the file yet. */
  private uncompacted = false

  constructor(readonly path: string) {
    this.db = new DatabaseSync(path)
    this.db.exec(SCHEMA)
  }

  close(): void {
    this.db.close()
  }

  getMeta(key: string): string | undefined {
    const row = this.db.prepare('SELECT value FROM meta WHERE key = ?').get(key) as { value: string } | undefined
    return row?.value
  }

  setMeta(key: string, value: string): void {
    this.db.prepare('INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value').run(key, value)
  }

  /** Bumped by every forget, in any connection to this file. */
  generation(): number {
    return Number(this.getMeta('generation') ?? 0)
  }

  // -- turns ----------------------------------------------------------------------------

  /** Appends a message to its session; returns the stored turn (idempotent on `id`). */
  addTurn(turn: Omit<StoredTurn, 'seq' | 'pos'>): StoredTurn {
    const existing = this.turn(turn.id)
    if (existing) return existing
    const row = this.db.prepare('SELECT COALESCE(MAX(pos) + 1, 0) AS pos FROM turns WHERE session = ?').get(turn.session) as { pos: number }
    this.db.prepare('INSERT INTO turns(id, session, pos, date, at, role, text) VALUES (?, ?, ?, ?, ?, ?, ?)')
      .run(turn.id, turn.session, row.pos, turn.date, turn.at, turn.role, turn.text)
    return this.turn(turn.id)!
  }

  turn(id: string): StoredTurn | undefined {
    return this.db.prepare('SELECT * FROM turns WHERE id = ?').get(id) as StoredTurn | undefined
  }

  hasTurn(id: string): boolean {
    return this.db.prepare('SELECT 1 FROM turns WHERE id = ?').get(id) !== undefined
  }

  /** The stored message just before position `pos` of `session` (its p2 window partner), if any. */
  previousTurn(session: string, pos: number): StoredTurn | undefined {
    return this.db.prepare('SELECT * FROM turns WHERE session = ? AND pos < ? ORDER BY pos DESC LIMIT 1').get(session, pos) as StoredTurn | undefined
  }

  /** All turns in capture order (sessions appear in the order they started). */
  turns(): StoredTurn[] {
    return this.db.prepare('SELECT * FROM turns ORDER BY seq').all() as unknown as StoredTurn[]
  }

  sessionTurns(session: string, fromPos = 0): StoredTurn[] {
    return this.db.prepare('SELECT * FROM turns WHERE session = ? AND pos >= ? ORDER BY pos').all(session, fromPos) as unknown as StoredTurn[]
  }

  // -- units ----------------------------------------------------------------------------

  addUnit(unit: StoredUnit): void {
    this.db.prepare('INSERT OR REPLACE INTO units(id, kind, session, text, turns, vector, model) VALUES (?, ?, ?, ?, ?, ?, ?)')
      .run(unit.id, unit.set, unit.session, unit.text, JSON.stringify(unit.turns), blob(unit.vector), unit.model)
  }

  hasUnit(id: string): boolean {
    return this.db.prepare('SELECT 1 FROM units WHERE id = ?').get(id) !== undefined
  }

  /** Units in insertion order: the RRF tie-break uses this index, as the benchmarks used build order. */
  units(): StoredUnit[] {
    const rows = this.db.prepare('SELECT * FROM units ORDER BY rowid').all() as Array<Record<string, unknown>>
    return rows.map(r => ({ id: r.id as string, set: r.kind as Unit['set'], session: r.session as string,
      text: r.text as string, turns: JSON.parse(r.turns as string) as string[], vector: floats(r.vector as Uint8Array),
      model: r.model as string }))
  }

  unitCount(): number {
    return (this.db.prepare('SELECT COUNT(*) AS n FROM units').get() as { n: number }).n
  }

  // -- segments and facts -----------------------------------------------------------------

  /** Highest position already assigned to a segment in this session, or -1. */
  segmentedUpTo(session: string): number {
    const row = this.db.prepare('SELECT COALESCE(MAX(last_pos), -1) AS p FROM segments WHERE session = ?').get(session) as { p: number }
    return row.p
  }

  addSegment(segment: Omit<Segment, 'status'>): void {
    this.db.prepare("INSERT OR IGNORE INTO segments(id, session, first_pos, last_pos, date, status) VALUES (?, ?, ?, ?, ?, 'pending')")
      .run(segment.id, segment.session, segment.firstPos, segment.lastPos, segment.date)
  }

  pendingSegments(maxAttempts: number): Segment[] {
    const rows = this.db.prepare("SELECT * FROM segments WHERE status = 'pending' AND attempts < ? ORDER BY rowid").all(maxAttempts) as Array<Record<string, unknown>>
    return rows.map(r => ({ id: r.id as string, session: r.session as string, firstPos: r.first_pos as number,
      lastPos: r.last_pos as number, date: r.date as string, status: r.status as Segment['status'] }))
  }

  segmentTurns(segment: Segment): StoredTurn[] {
    return this.db.prepare('SELECT * FROM turns WHERE session = ? AND pos BETWEEN ? AND ? ORDER BY pos')
      .all(segment.session, segment.firstPos, segment.lastPos) as unknown as StoredTurn[]
  }

  /**
   * Records a finished extraction: facts (possibly none) and done, or a failed attempt.
   * @returns false when the segment no longer exists (a forget removed it meanwhile); nothing is written then.
   */
  finishSegment(id: string, facts: readonly Fact[] | null, error?: string, maxAttempts = 3): boolean {
    this.db.exec('BEGIN')
    try {
      let found: boolean
      if (facts) {
        found = this.db.prepare("UPDATE segments SET status = 'done', attempts = attempts + 1, error = NULL WHERE id = ?").run(id).changes > 0
        if (found) {
          facts.forEach((f, j) => this.db.prepare('INSERT OR REPLACE INTO facts(segment, j, fact, turns) VALUES (?, ?, ?, ?)')
            .run(id, j, f.fact, JSON.stringify(f.turns)))
        }
      } else {
        found = this.db.prepare("UPDATE segments SET attempts = attempts + 1, error = ?, status = CASE WHEN attempts + 1 >= ? THEN 'failed' ELSE 'pending' END WHERE id = ?")
          .run(error ?? 'unknown', maxAttempts, id).changes > 0
      }
      this.db.exec('COMMIT')
      return found
    } catch (e) {
      this.db.exec('ROLLBACK')
      throw e
    }
  }

  /** Facts keyed by segment, in segment creation order (the order fact units are built). */
  facts(): Map<string, Fact[]> {
    const rows = this.db.prepare('SELECT f.segment, f.fact, f.turns FROM facts f JOIN segments s ON s.id = f.segment ORDER BY s.rowid, f.j').all() as Array<Record<string, string>>
    const out = new Map<string, Fact[]>()
    for (const r of rows) {
      const list = out.get(r.segment!) ?? []
      list.push({ fact: r.fact!, turns: JSON.parse(r.turns!) as string[] })
      out.set(r.segment!, list)
    }
    return out
  }

  /** Text of fact `j` of `segment`, if it is stored. */
  factText(segment: string, j: number): string | undefined {
    return (this.db.prepare('SELECT fact FROM facts WHERE segment = ? AND j = ?').get(segment, j) as { fact: string } | undefined)?.fact
  }

  /** Date of every segment with extracted facts, in creation order. */
  segmentDates(): Map<string, string> {
    const rows = this.db.prepare("SELECT id, date FROM segments WHERE status = 'done' ORDER BY rowid").all() as Array<{ id: string, date: string }>
    return new Map(rows.map(r => [r.id, r.date] as const))
  }

  /** Stores links between facts that exist (a link computed before a forget may name a deleted fact). */
  addLinks(links: ReadonlyArray<{ old: string, new: string }>): void {
    const stmt = this.db.prepare('INSERT OR IGNORE INTO links(old, new) VALUES (?, ?)')
    const stored = (key: string) => this.factText(...splitKey(key)) !== undefined
    for (const l of links) if (stored(l.old) && stored(l.new)) stmt.run(l.old, l.new)
  }

  links(): Array<{ old: string, new: string }> {
    return (this.db.prepare('SELECT old, new FROM links ORDER BY rowid').all() as Array<{ old: string, new: string }>)
      .map(r => ({ old: r.old, new: r.new }))
  }

  /** Messages without a window vector, and facts without a vector, of embedding model `model`. */
  unindexed(model: string): number {
    const n = (sql: string) => (this.db.prepare(sql).get(model) as { n: number }).n
    return n("SELECT COUNT(*) AS n FROM turns t WHERE NOT EXISTS (SELECT 1 FROM units u WHERE u.id = 'p2|' || t.id AND u.model = ?)")
      + n("SELECT COUNT(*) AS n FROM facts f WHERE NOT EXISTS (SELECT 1 FROM units u WHERE u.id = 'fact|' || f.segment || '#' || f.j AND u.model = ?)")
  }

  stats(): Record<string, number> {
    const n = (sql: string) => (this.db.prepare(sql).get() as { n: number }).n
    return {
      turns: n('SELECT COUNT(*) AS n FROM turns'), units: n('SELECT COUNT(*) AS n FROM units'),
      facts: n('SELECT COUNT(*) AS n FROM facts'), segmentsPending: n("SELECT COUNT(*) AS n FROM segments WHERE status = 'pending'"),
      segmentsFailed: n("SELECT COUNT(*) AS n FROM segments WHERE status = 'failed'"), links: n('SELECT COUNT(*) AS n FROM links'),
    }
  }

  // -- forgetting ------------------------------------------------------------------------------

  /**
   * Deletes the messages in `scope` and everything derived from them, in one transaction:
   * - every window or fact unit that cites a deleted message;
   * - the facts (and fact units) of every segment that held a deleted message. A segment whose messages are all gone
   *   is deleted; one that keeps some is narrowed to them and set pending, so facts are extracted again from what is
   *   left (a fact may draw on any message of its segment, not only the ones it cites);
   * - update links touching a deleted fact; a chain old -> deleted -> new keeps a direct link old -> new.
   * Then VACUUM and a WAL checkpoint rewrite the file without the deleted text.
   *
   * @returns counts, and `reindex`: kept messages whose p2 window showed a deleted one. Their windows are dropped;
   *   `Memory.backfill()` embeds them again without the deleted message.
   */
  forget(scope: ForgetScope): ForgetResult & { reindex: StoredTurn[] } {
    const db = this.db
    const n = (sql: string) => (db.prepare(sql).get() as { n: number }).n
    const count = () => ({ turns: n('SELECT COUNT(*) AS n FROM turns'), units: n('SELECT COUNT(*) AS n FROM units'),
      facts: n('SELECT COUNT(*) AS n FROM facts') })
    let before: ReturnType<typeof count>, after: ReturnType<typeof count>
    let segments = 0, links = 0, reindex: string[] = []
    db.exec('BEGIN IMMEDIATE')
    try {
      before = count()
      if ('all' in scope) {
        segments = n('SELECT COUNT(*) AS n FROM segments')
        links = n('SELECT COUNT(*) AS n FROM links')
        db.exec('DELETE FROM turns; DELETE FROM units; DELETE FROM segments; DELETE FROM facts; DELETE FROM links;')
      } else {
        db.exec('CREATE TEMP TABLE IF NOT EXISTS forget_turns (id TEXT PRIMARY KEY); DELETE FROM temp.forget_turns;'
          + 'CREATE TEMP TABLE IF NOT EXISTS forget_segments (id TEXT PRIMARY KEY, session TEXT NOT NULL); DELETE FROM temp.forget_segments;')
        if ('session' in scope) db.prepare('INSERT INTO temp.forget_turns SELECT id FROM turns WHERE session = ?').run(scope.session)
        else db.prepare('INSERT INTO temp.forget_turns SELECT id FROM turns WHERE at < ?').run(scope.before)
        db.exec(`INSERT OR IGNORE INTO temp.forget_segments
          SELECT s.id, s.session FROM segments s JOIN turns t ON t.session = s.session AND t.pos BETWEEN s.first_pos AND s.last_pos
          JOIN temp.forget_turns d ON d.id = t.id`)
        const cites = 'EXISTS (SELECT 1 FROM json_each(units.turns) j JOIN temp.forget_turns d ON d.id = j.value)'
        reindex = (db.prepare(`SELECT json_extract(turns, '$[0]') AS centre FROM units WHERE kind = 'p2' AND ${cites}
          AND json_extract(turns, '$[0]') NOT IN (SELECT id FROM temp.forget_turns)`).all() as Array<{ centre: string }>).map(r => r.centre)
        db.exec(`DELETE FROM units WHERE ${cites}`)
        db.exec("DELETE FROM units WHERE kind = 'fact' AND session IN (SELECT id FROM temp.forget_segments)")
        db.exec('DELETE FROM facts WHERE segment IN (SELECT id FROM temp.forget_segments)')
        db.exec('DELETE FROM turns WHERE id IN (SELECT id FROM temp.forget_turns)')
        const touched = db.prepare('SELECT s.id, s.session, s.first_pos, s.last_pos FROM segments s JOIN temp.forget_segments f ON f.id = s.id')
          .all() as Array<{ id: string, session: string, first_pos: number, last_pos: number }>
        for (const s of touched) {
          const kept = db.prepare('SELECT MIN(pos) AS a, MAX(pos) AS b FROM turns WHERE session = ? AND pos BETWEEN ? AND ?')
            .get(s.session, s.first_pos, s.last_pos) as { a: number | null, b: number | null }
          if (kept.a === null) {
            db.prepare('DELETE FROM segments WHERE id = ?').run(s.id)
          } else {
            const date = (db.prepare('SELECT date FROM turns WHERE session = ? AND pos = ?').get(s.session, kept.a) as { date: string }).date
            db.prepare("UPDATE segments SET id = ?, first_pos = ?, last_pos = ?, date = ?, status = 'pending', attempts = 0, error = NULL WHERE id = ?")
              .run(`${s.session}@${kept.a}`, kept.a, kept.b, date, s.id)
          }
        }
        segments = touched.length
        links = this.relink()
        db.exec('DROP TABLE temp.forget_turns; DROP TABLE temp.forget_segments;')
      }
      db.prepare("INSERT INTO meta(key, value) VALUES ('generation', '1') ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1").run()
      after = count()
      db.exec('COMMIT')
    } catch (e) {
      db.exec('ROLLBACK')
      throw e
    }
    const vacuumed = this.compact()
    this.uncompacted = !vacuumed
    return {
      turns: before.turns - after.turns, units: before.units - after.units, facts: before.facts - after.facts, segments, links,
      vacuumed, reindex: reindex.map(id => this.turn(id)).filter((t): t is StoredTurn => t !== undefined),
    }
  }

  /** Retries the rewrite an earlier forget could not do. True when it ran now. */
  retryCompact(): boolean {
    if (!this.uncompacted) return false
    this.uncompacted = !this.compact()
    return !this.uncompacted
  }

  /** Deletes links whose fact is gone, bridging old -> (gone ...) -> new as old -> new. Returns the links deleted. */
  private relink(): number {
    const exists = new Set((this.db.prepare('SELECT segment, j FROM facts').all() as Array<{ segment: string, j: number }>)
      .map(r => `${r.segment}#${r.j}`))
    const all = this.links()
    const dangling = all.filter(l => !exists.has(l.old) || !exists.has(l.new))
    if (!dangling.length) return 0
    const next = new Map<string, string[]>()
    for (const l of all) next.set(l.old, [...(next.get(l.old) ?? []), l.new])
    const bridges: Array<{ old: string, new: string }> = []
    for (const l of dangling) {
      if (!exists.has(l.old) || exists.has(l.new)) continue
      const frontier = [l.new], seen = new Set(frontier)
      while (frontier.length) {
        const k = frontier.shift()!
        if (exists.has(k)) { if (k !== l.old) bridges.push({ old: l.old, new: k }); continue }
        for (const m of next.get(k) ?? []) if (!seen.has(m)) { seen.add(m); frontier.push(m) }
      }
    }
    const remove = this.db.prepare('DELETE FROM links WHERE old = ? AND new = ?')
    for (const l of dangling) remove.run(l.old, l.new)
    this.addLinks(bridges)
    return dangling.length
  }

  /** VACUUM, then checkpoint and empty the WAL. False when another connection prevented either. */
  private compact(): boolean {
    try {
      this.db.exec('VACUUM')
      const r = this.db.prepare('PRAGMA wal_checkpoint(TRUNCATE)').get() as { busy: number }
      return r.busy === 0
    } catch {
      return false
    }
  }
}
