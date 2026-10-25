// Runs bench_child.mjs in a fresh process per store, sequentially, and writes one JSON per store plus a summary.
//
//   node scripts/eval/scale/run_bench.mjs --vectors synthetic|real [--sizes 50,500,2000,5000] [--ingest 120] [--tag name]
import { spawnSync } from 'node:child_process'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { loadavg } from 'node:os'
import { join } from 'node:path'
import { OUT } from './common.mjs'

const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }
const vectors = arg('vectors', 'synthetic')
const sizes = arg('sizes', '50,500,2000,5000').split(',').map(Number)
const ingest = arg('ingest', '120')
const tag = arg('tag', vectors)
const dir = join(OUT, `bench-${tag}`)
mkdirSync(dir, { recursive: true })

const rows = []
for (const size of sizes) {
  const out = join(dir, `s${size}.json`)
  const load = loadavg()
  const started = Date.now()
  const r = spawnSync(process.execPath, ['--no-warnings', '--expose-gc', join(import.meta.dirname, 'bench_child.mjs'),
    '--store', join(OUT, `stores-${vectors}`, `s${size}`), '--vectors', vectors, '--ingest', ingest, '--out', out],
  { stdio: ['ignore', 'inherit', 'inherit'], timeout: 3600_000 })
  if (r.status !== 0) throw new Error(`bench for ${size} exited ${r.status}`)
  const res = JSON.parse(readFileSync(out, 'utf8'))
  res.loadavgBefore = load
  res.loadavgAfter = loadavg()
  res.spawnToExitMs = Date.now() - started
  writeFileSync(out, JSON.stringify(res, null, 1))
  const m = x => (x / 1048576).toFixed(0)
  const row = {
    size, units: res.unitCount, dbMB: m(res.files.total),
    warmP50: res.warm.total.p50.toFixed(1), warmP95: res.warm.total.p95.toFixed(1), localP50: res.warm.local.p50.toFixed(1),
    embedP50: res.warm.embed.p50.toFixed(1), firstRecallMs: res.firstRecall.ms.toFixed(0),
    coldStartMs: res.firstRecall.sinceProcessStartMs.toFixed(0), backfillMs: res.startupBackfill.ms.toFixed(0),
    tickMs: res.tickBackfillMs.toFixed(0), afterTickMs: res.recallAfterTickMs.toFixed(0),
    rssWarmMB: m(res.memory.afterWarm.rss), maxRssMB: m(res.memory.end.maxRss), heapGcMB: m(res.memory.afterWarmGc.heapUsed),
    ingestMs: res.ingest ? res.ingest.perMessageMs.p50.toFixed(2) : null, load: load[0].toFixed(1),
  }
  rows.push(row)
  console.log(JSON.stringify(row))
}
writeFileSync(join(dir, 'summary.json'), JSON.stringify(rows, null, 1))
