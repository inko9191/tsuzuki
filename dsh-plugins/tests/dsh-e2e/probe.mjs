// Test-only consumer: proves a real dsh composition exposes ctx.textEmbeddings,
// writes the observed results to config.out, then asks the launcher to exit.
import { writeFileSync } from 'node:fs'

export const name = 'text-embeddings-probe'
export const inject = ['textEmbeddings']

export function apply(ctx, config) {
  const settled = []
  ctx.on('text-embeddings/settled', e => settled.push(e))
  ctx.effect(() => {
    const timer = setTimeout(async () => {
      const report = { ok: false }
      try {
        const service = ctx.textEmbeddings
        report.info = service.describe()
        const q = await service.embed({ input: ['Which planet is known as the Red Planet?', '赤い惑星はどれ？'], task: 'query', dimensions: 256 })
        const d = await service.embed({ input: ['Mars is often called the Red Planet.'], task: 'document' })
        report.query = { model: q.model, n: q.vectors.length, width: q.vectors[0].length, promptTokens: q.promptTokens }
        report.document = { model: d.model, width: d.vectors[0].length }
        try {
          await service.embed({ input: ['x '.repeat(5000)] })
        } catch (e) {
          report.tooLong = { kind: e.kind, code: e.details?.code }
        }
        report.settled = settled.map(e => e.error ?? 'ok')
        report.ok = true
      } catch (e) {
        report.error = String(e?.stack ?? e)
      }
      writeFileSync(config.out, JSON.stringify(report, null, 2))
      ctx.get('appExit')?.(report.ok ? 0 : 1)
    }, 10)
    return () => clearTimeout(timer)
  })
}
