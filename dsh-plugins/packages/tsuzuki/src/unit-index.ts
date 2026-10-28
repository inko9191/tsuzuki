/**
 * The ranked unit index, kept between recalls instead of being rebuilt on every one.
 *
 * Units are in the evaluated build order, which breaks score ties: p2 windows in capture order, then facts in segment
 * order. Each set's vectors live in one contiguous matrix that the units' `vector` fields view. `rank()` returns
 * exactly what `pipeline.rank` returns over the same units: every score is the same products summed in the same order
 * as `pipeline.dot`, and the best `POOL` of each set are kept with a bounded heap (ties by position, as the stable
 * sort in `pipeline.top`) instead of sorting every score.
 */
import * as p from './pipeline.ts'
import type { StoredUnit } from './store.ts'

/** Rows allocated for `rows` units: a little room to append captures without copying the matrix each time. */
function capacity(rows: number): number {
  return rows + Math.max(64, rows >> 4)
}

/** One unit set in rank order, with its vectors in one matrix. */
export class UnitRows {
  readonly units: StoredUnit[] = []
  private data: Float32Array

  constructor(readonly dims: number, units: readonly StoredUnit[] = []) {
    this.data = new Float32Array(capacity(units.length) * dims)
    for (const u of units) this.push(u)
  }

  /** Appends `u` as the last row; its `vector` then views the matrix (its own buffer can be freed). */
  push(u: StoredUnit): void {
    const n = this.units.length, d = this.dims
    if ((n + 1) * d > this.data.length) {
      const grown = new Float32Array(capacity(n + 1) * d)
      grown.set(this.data.subarray(0, n * d))
      this.data = grown
      this.units.forEach((x, r) => { x.vector = grown.subarray(r * d, (r + 1) * d) })
    }
    this.data.set(u.vector, n * d)
    u.vector = this.data.subarray(n * d, (n + 1) * d)
    this.units.push(u)
  }

  /** `pipeline.top` over every row: the best `n` as [row, score], highest score first, ties by row. */
  top(query: Float32Array, n = p.POOL): p.Ranked {
    const rows = this.units.length, d = this.dims, m = this.data
    const keep = Math.min(n, rows)
    if (!keep) return []
    // Min-heap of the kept rows; the root is the one that ranks last (lowest score, then highest row).
    const hs = new Float64Array(keep), hr = new Int32Array(keep)
    let size = 0
    const below = (i: number, j: number) => hs[i]! < hs[j]! || (hs[i] === hs[j] && hr[i]! > hr[j]!)
    const swap = (i: number, j: number) => {
      const s = hs[i]!, r = hr[i]!
      hs[i] = hs[j]!; hr[i] = hr[j]!; hs[j] = s; hr[j] = r
    }
    for (let r = 0; r < rows; r++) {
      let score = 0
      for (let k = 0, o = r * d; k < d; k++) score += m[o + k]! * query[k]!
      if (size < keep) {
        hs[size] = score; hr[size] = r
        for (let j = size++; j > 0;) {
          const parent = (j - 1) >> 1
          if (!below(j, parent)) break
          swap(j, parent)
          j = parent
        }
      } else if (score > hs[0]!) {  // equal scores keep the earlier row, which is already in
        hs[0] = score; hr[0] = r
        for (let j = 0; ;) {
          const l = 2 * j + 1, rt = l + 1
          let w = j
          if (l < size && below(l, w)) w = l
          if (rt < size && below(rt, w)) w = rt
          if (w === j) break
          swap(j, w)
          j = w
        }
      }
    }
    const out: p.Ranked = []
    for (let j = 0; j < size; j++) out.push([hr[j]!, hs[j]!])
    return out.sort((a, b) => (b[1] - a[1]) || (a[0] - b[0]))
  }
}

/** Windows ordered by the capture order (`seq`) of the message they centre on; unknown ones last. */
export function windowOrder(windows: readonly StoredUnit[], seqOf: (id: string) => number | undefined): Array<[StoredUnit, number]> {
  return windows.map(u => [u, seqOf(u.turns[0] ?? '') ?? Number.POSITIVE_INFINITY] as [StoredUnit, number])
    .sort((a, b) => a[1] - b[1])
}

/** Facts ordered by their segment's creation order, then by position in the segment (`fact|segment#j`). */
export function factOrder(facts: readonly StoredUnit[], segments: readonly string[]): StoredUnit[] {
  const rankOf = new Map(segments.map((s, i) => [s, i] as const))
  const key = (u: StoredUnit): [number, number] => {
    const body = u.id.slice('fact|'.length), cut = body.lastIndexOf('#')
    return [rankOf.get(body.slice(0, cut)) ?? Number.POSITIVE_INFINITY, Number(body.slice(cut + 1))]
  }
  const keyed = facts.map(u => [u, ...key(u)] as const)
  return keyed.sort((a, b) => (a[1] - b[1]) || (a[2] - b[2])).map(([u]) => u)
}

/** Windows and facts of one embedding model and dimension, ranked as `pipeline.rank` ranks them. */
export class RankIndex {
  readonly windows: UnitRows
  private facts: UnitRows | null = null
  private skippedWindows: number
  private skippedFacts = 0
  private lastSeq: number
  private list: StoredUnit[] | null = null

  /** `windows` in rank order with their centre's `seq`; `skipped`: windows of another model or dimension. */
  constructor(readonly model: string, readonly dims: number, windows: ReadonlyArray<[StoredUnit, number]>, skipped: number) {
    this.windows = new UnitRows(dims, windows.map(([u]) => u))
    this.lastSeq = windows.length ? windows.at(-1)![1] : Number.NEGATIVE_INFINITY
    this.skippedWindows = skipped
  }

  usable(u: StoredUnit): boolean {
    return u.model === this.model && u.vector.length === this.dims
  }

  /** False until `setFacts` (after a fact was stored, facts are ordered again from the store). */
  hasFacts(): boolean {
    return this.facts !== null
  }

  setFacts(facts: readonly StoredUnit[], skipped: number): void {
    this.facts = new UnitRows(this.dims, facts)
    this.skippedFacts = skipped
    this.list = null
  }

  /** Units left out because another model (or dimension) embedded them. */
  skipped(): number {
    return this.skippedWindows + this.skippedFacts
  }

  /** Units by the indices `rank()` returns: the windows, then the facts. */
  units(): StoredUnit[] {
    return this.list ??= [...this.windows.units, ...this.facts!.units]
  }

  /**
   * `order`: `pipeline.rank(units(), vectors, query)`. `best`: the highest score, that is the cosine with the query
   * (vectors are unit length), of a ranked unit that `counts`; -Infinity when there is none.
   */
  rank(query: Float32Array, counts: (u: StoredUnit) => boolean = () => true): { order: p.Ranked, best: number } {
    const lists: p.Ranked[] = []
    if (this.windows.units.length) lists.push(this.windows.top(query))
    const offset = this.windows.units.length
    if (this.facts!.units.length) lists.push(this.facts!.top(query).map(([r, s]) => [r + offset, s] as [number, number]))
    const units = this.units()
    // Each list is sorted best first.
    const best = Math.max(Number.NEGATIVE_INFINITY, ...lists.map(l => l.find(([i]) => counts(units[i]!))?.[1] ?? Number.NEGATIVE_INFINITY))
    return { order: p.rrf(lists.length ? lists : [[]]), best }
  }

  /**
   * Takes a unit just stored (`seq`: its centre message's, for a window). A window of the newest message is appended;
   * a fact makes the facts be ordered again. False when the unit cannot be placed (the caller drops this index).
   */
  add(u: StoredUnit, seq: number | undefined): boolean {
    if (u.set === 'fact') {
      this.facts = null
      this.list = null
      return true
    }
    if (u.set !== 'p2') return true
    if (!this.usable(u)) {
      this.skippedWindows++
      return true
    }
    if (seq === undefined || !(seq > this.lastSeq)) return false
    this.windows.push(u)
    this.lastSeq = seq
    this.list = null
    return true
  }
}
