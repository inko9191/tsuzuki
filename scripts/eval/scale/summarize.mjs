// Markdown tables of everything the scale test produced (missing parts are skipped). Writes build/scale/summary.md.
//
//   node scripts/eval/scale/summarize.mjs
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { OUT, percentile } from './common.mjs'

const read = p => (existsSync(p) ? JSON.parse(readFileSync(p, 'utf8')) : null)
const mb = b => (b / 1048576).toFixed(0)
const f = (x, d = 1) => (x === null || x === undefined || Number.isNaN(x) ? '—' : Number(x).toFixed(d))
const out = []
const table = (head, rows) => {
  out.push(`|${head.join('|')}|`, `|${head.map((_, i) => (i ? '---:' : '---')).join('|')}|`, ...rows.map(r => `|${r.join('|')}|`), '')
}
const sizesIn = dir => (existsSync(dir) ? readdirSync(dir).filter(n => /^s\d+(\.json)?$/.test(n)).map(n => Number(n.replace(/\D/g, ''))).sort((a, b) => a - b) : [])

for (const v of ['real', 'synthetic']) {
  const dir = join(OUT, `bench-${v}`)
  const sizes = sizesIn(dir)
  if (!sizes.length) continue
  out.push(`## 性能（${v === 'real' ? '実ベクトル・クエリは egemma-server で埋め込み' : '合成ベクトル・クエリ埋め込みなし'}）`, '')
  const rows = sizes.map(s => {
    const r = read(join(dir, `s${s}.json`)), b = read(join(OUT, `stores-${v}`, `s${s}`, 'build.json'))
    return [s, r.unitCount, mb(r.files.total), f(r.warm.total.p50), f(r.warm.total.p95), f(r.warm.embed.p50), f(r.warm.local.p50),
      f(r.firstRecall.sinceProcessStartMs, 0), f(r.startupBackfill.ms, 0), f(r.firstRecall.ms, 0), f(r.tickBackfillMs, 0), f(r.recallAfterTickMs, 0),
      mb(r.memory.afterWarm.rss), mb(r.memory.afterTick.maxRss), mb(r.memory.afterWarmGc.heapUsed), mb(r.memory.afterWarmGc.arrayBuffers),
      b ? f(b.capture.perMessageMs.p50, 2) : '—', r.ingest ? f(r.ingest.perMessageMs.p50, 2) : '—']
  })
  table(['セッション', '窓', 'DB MB', '想起 p50 ms', '想起 p95 ms', 'うちクエリ埋め込み p50', 'うち計算 p50', '起動→初回想起 ms',
    '起動 backfill ms', '初回想起 ms', '定期 backfill ms', 'その次の想起 ms', 'RSS（100回後）MB', '最大 RSS（取り込み前）MB', 'GC後ヒープ MB',
    'GC後ベクトル MB', '取り込み（構築時）ms/件', '取り込み（その規模で）ms/件'], rows)
  const steps = Object.keys(read(join(dir, `s${sizes[0]}.json`)).breakdown)
  out.push('想起1回の内訳（p50 ms、20問）', '')
  table(['セッション', ...steps], sizes.map(s => { const r = read(join(dir, `s${s}.json`)); return [s, ...steps.map(k => f(r.breakdown[k].p50, 1))] }))
  const ing = sizes.map(s => read(join(dir, `s${s}.json`))).filter(r => r.ingest?.fixedWindowEmbedMs?.n)
  if (ing.length) {
    out.push('取り込み1件（実サーバー、1件ずつ）', '')
    table(['セッション', 'capture 計 p50 ms', 'うち埋め込み p50', 'うち保存 p50', '正しい p2 窓の埋め込み p50 ms', '同 p95 ms', '失敗'],
      ing.map(r => [r.unitCount, f(r.ingest.perMessageMs.p50), f(r.ingest.embedMs.p50), f(r.ingest.storageMs.p50, 2),
        f(r.ingest.fixedWindowEmbedMs.p50), f(r.ingest.fixedWindowEmbedMs.p95), r.ingest.failed ?? 0]))
  }
}

const emb = read(join(OUT, 'embed-throughput.json'))
if (emb) {
  out.push('## 一括埋め込み（backfill と同じ32件ずつ）', '')
  table(['方式', 'バッチ', '窓', 'トークン', '窓/秒', 'トークン/秒'], [
    ['1要求ずつ（backfill と同じ）', emb.sequential.batches, emb.sequential.windows, emb.sequential.tokens, f(emb.sequential.windowsPerSecond), f(emb.sequential.tokensPerSecond, 0)],
    [`${emb.concurrency}要求並行`, emb.parallel.batches, emb.parallel.windows, emb.parallel.tokens, f(emb.parallel.windowsPerSecond), f(emb.parallel.tokensPerSecond, 0)]])
  out.push(`クエリ埋め込み（1件ずつ、${emb.queryEmbedMs.n}問）: p50 ${f(emb.queryEmbedMs.p50)} ms, p95 ${f(emb.queryEmbedMs.p95)} ms。2,048トークン超で拒否: ${emb.refused} 窓。`, '')
}

for (const v of ['real', 'synthetic']) {
  const dir = join(OUT, `fix-${v}`)
  if (!existsSync(dir)) continue
  const files = readdirSync(dir).filter(n => n.endsWith('.json')).sort((a, b) => Number(a.replace(/\D/g, '')) - Number(b.replace(/\D/g, '')))
  out.push(`## 修正案の試作（${v}）`, '')
  table(['ストア', 'int8', 'lean', '読込 ms', '索引構築 ms', '想起 p50 ms', 'p95 ms', 'うち計算 p50', 'RSS（100回後）MB', 'GC後ヒープ MB', 'ベクトル MB', 'パッケージと同じ結果'],
    files.map(n => { const r = read(join(dir, n)); return [n.replace('.json', ''), r.int8 ? 'yes' : 'no', r.lean ? 'yes' : 'no', f(r.loadMs, 0), f(r.indexBuildMs, 0),
      f(r.warm.total.p50), f(r.warm.total.p95), f(r.warm.local.p50), mb(r.memory.afterWarm.rss), mb(r.memory.afterWarmGc.heapUsed),
      mb(r.memory.afterWarmGc.arrayBuffers), r.identicalToPackage] }))
}

const acc = read(join(OUT, 'accuracy-real.json'))
const accPrefix = read(join(OUT, 'accuracy-real-prefix.json'))
if (acc && accPrefix) acc.rows.push(...accPrefix.rows)
if (acc) {
  out.push('## 精度の変化（60問 × 3つの古さ、取り出し段階）', '')
  const groups = new Map()
  for (const r of acc.rows) {
    const k = `${r.mode}|${r.size}`
    if (!groups.has(k)) groups.set(k, [])
    groups.get(k).push(r)
  }
  const mean = (rows, fn) => rows.reduce((a, r) => a + Number(fn(r)), 0) / rows.length
  const rank = rows => { const v = rows.map(r => r.firstRank ?? 1e9); return percentile(v, 0.5) }
  table(['窓', 'セッション', '件数', '答えの発言を1つ以上', '全根拠セッション', '答えの発言の再現率', '最初の答えの順位（中央値）', '答え文字列が想起文に含まれる', '空の想起', 'int8: 1つ以上', 'int8: 同じ結果'],
    [...groups].map(([k, rows]) => {
      const [mode, size] = k.split('|')
      const textRows = rows.filter(r => r.answerInText !== null)
      const r50 = rank(rows)
      return [mode, size, rows.length, f(mean(rows, r => r.hit), 3), f(mean(rows, r => r.allSessions), 3), f(mean(rows, r => r.answerTurnRecall), 3),
        r50 >= 1e9 ? '圏外' : f(r50, 0), textRows.length ? `${f(mean(textRows, r => r.answerInText), 3)}（${textRows.length}）` : '—',
        f(mean(rows, r => r.ids === 0), 3),
        rows[0].int8 ? f(mean(rows, r => r.int8.hit), 3) : '—', rows[0].int8 ? f(mean(rows, r => r.int8.sameIds), 3) : '—']
    }))
  out.push('古さ別（答えの発言を1つ以上）', '')
  const ages = Object.keys(acc.ages)
  table(['窓', 'セッション', ...ages], [...groups].map(([k, rows]) => {
    const [mode, size] = k.split('|')
    return [mode, size, ...ages.map(a => { const x = rows.filter(r => r.age === a); return f(mean(x, r => r.hit), 3) })]
  }))
  out.push('問題の種類別（答えの発言を1つ以上、fixed）', '')
  const types = [...new Set(acc.rows.map(r => r.type))].sort()
  table(['セッション', ...types], [...groups].filter(([k]) => k.startsWith('fixed|')).map(([k, rows]) =>
    [k.split('|')[1], ...types.map(t => { const x = rows.filter(r => r.type === t); return f(mean(x, r => r.hit), 2) })]))
  if (acc.validation?.length) out.push(`検証（作り直したストアとの一致）: ${acc.validation.map(v => `${v.mode} ${v.size}: ${v.identical}/${v.compared}`).join(', ')}`, '')
}

writeFileSync(join(OUT, 'summary.md'), out.join('\n'))
console.log(out.join('\n'))
