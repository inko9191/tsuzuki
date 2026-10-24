// Keyless consumer of the real prime-memory tools; no LLM calls or benchmarks.
import assert from 'node:assert/strict'
import { writeFileSync } from 'node:fs'
import { setTimeout as delay } from 'node:timers/promises'

export const name = 'prime-memory-probe'
export const inject = ['tools', 'textEmbeddings']

export function apply(ctx, config) {
  ctx.effect(() => {
    const controller = new AbortController()
    const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(60_000)])
    const report = { ok: false, model: 'embeddinggemma-2', dimensions: 768, strategy: 'hybrid' }
    const stats = async (stage) => {
      const response = await fetch(`${config.origin}/stats`, { signal })
      assert.ok(response.ok, `/stats HTTP ${response.status}`)
      const snapshot = await response.json()
      writeFileSync(`${config.out}/stats-${stage}.json`, JSON.stringify(snapshot, null, 2) + '\n')
      return snapshot
    }
    const execute = (name, args) => ctx.tools.execute({
      callId: `prime-swap-${name}`, name, arguments: args, signal,
    })
    // Wait for prime-memory's async initialization and tool registration.
    void (async () => {
      try {
        while (!ctx.tools.get('memory_add') || !ctx.tools.get('memory_search')) {
          await delay(100, undefined, { signal })
        }
        report.provider = ctx.textEmbeddings.describe()
        const before = await stats('before')
        const content = 'Stage1 swap check: Mars is the Red Planet. 火星は赤い惑星として知られる。'
        report.add = await execute('memory_add', { content, type: 'work_fact', scene: 'stage1-swap' })
        assert.equal(report.add.isError, false, 'memory_add failed')
        assert.ok(report.add.value?.id, 'memory_add did not write a memory (check live.memoryMutate)')
        const afterAdd = await stats('after-add')
        assert.ok(afterAdd.requests > before.requests, 'memory_add did not call remote embeddings')
        report.search = await execute('memory_search', { query: 'Mars Red Planet 火星', limit: 5 })
        assert.equal(report.search.isError, false, 'memory_search failed')
        assert.ok(report.search.value?.items?.some(item => item.id === report.add.value.id && item.content === content),
          'memory_search did not return the added memory')
        const afterSearch = await stats('after-search')
        assert.ok(afterSearch.requests > afterAdd.requests, 'memory_search did not call remote embeddings')
        assert.equal(afterSearch.errors, before.errors, 'daemon recorded an embedding error')
        report.requests = { before: before.requests, afterAdd: afterAdd.requests, afterSearch: afterSearch.requests }
        report.ok = true
      } catch (error) {
        report.error = String(error?.stack ?? error)
      }
      writeFileSync(`${config.out}/probe.json`, JSON.stringify(report, null, 2) + '\n')
      ctx.get('appExit')?.(report.ok ? 0 : 1)
    })()
    return () => controller.abort()
  })
}
