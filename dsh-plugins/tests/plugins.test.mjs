// Unit and Cordis-composition tests for the three packages (built lib/ output).
import { Context } from '@deepseek-ai/cordis'
import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, it } from 'node:test'
import { postOpenAIEmbeddings, resolveAgainst, TextEmbeddingError } from 'dsh-text-embeddings'
import OpenAIProvider, { truncateNormalize } from 'dsh-text-embeddings-openai'
import TsuzukiProvider, { DaemonSupervisor, resolveLaunch } from 'dsh-text-embeddings-tsuzuki'
import { TestRuntime } from './fixtures/runtime.mjs'

const FAKE = fileURLToPath(new URL('./fixtures/fake-daemon.mjs', import.meta.url))
const MODEL_DIR = fileURLToPath(new URL('../../models/embeddinggemma-2', import.meta.url))
// The provider reads the task names from the checkpoint; tests that load it are skipped without the downloaded
// EmbeddingGemma 2 (scripts/fetch_model.sh). EmbeddingGemma 300m is optional (scripts/eval/fetch_v1_model.py) too.
const HAS_V2 = existsSync(`${MODEL_DIR}/config_sentence_transformers.json`)
// The real-engine tests also need the built engine and its Python environment (README, Embeddings, Mac); they use the GPU.
const HAS_ENGINE = HAS_V2 && existsSync(fileURLToPath(new URL('../../.venv/bin/python', import.meta.url)))
const V1_DIR = fileURLToPath(new URL('../../models/embeddinggemma-300m', import.meta.url))
const settle = (ms = 30) => new Promise(r => setTimeout(r, ms))

/** Start an HTTP server whose handler is chosen per test; returns base URL and closer. */
async function stub(handler) {
  const server = createServer(handler)
  await new Promise(r => server.listen(0, '127.0.0.1', r))
  return { baseUrl: `http://127.0.0.1:${server.address().port}/v1`, close: () => new Promise(r => server.close(r)) }
}
const json = (res, status, body) => { res.writeHead(status, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(body)) }
const echo = dims => (req, res) => {
  let b = ''; req.on('data', c => { b += c }); req.on('end', () => {
    const q = JSON.parse(b)
    json(res, 200, { model: q.model, usage: { prompt_tokens: 2 }, data: q.input.map((t, index) => ({ index, embedding: Array.from({ length: q.dimensions ?? dims }, (_, i) => (i + 1) * (t.length + 1)) })) })
  })
}

describe('openai wire', () => {
  it('returns vectors in index order', async () => {
    const s = await stub((req, res) => json(res, 200, { model: 'm', usage: { prompt_tokens: 5 }, data: [{ index: 1, embedding: [0, 1] }, { index: 0, embedding: [1, 0] }] }))
    const r = await postOpenAIEmbeddings({ baseUrl: s.baseUrl, model: 'm', input: ['a', 'b'], timeoutMs: 1000 })
    assert.deepEqual(r.vectors.map(v => Array.from(v)), [[1, 0], [0, 1]])
    assert.equal(r.promptTokens, 5)
    await s.close()
  })
  for (const [status, code, kind] of [[400, 'context_length_exceeded', 'input-too-long'], [400, 'unknown_task', 'invalid-request'],
    [404, 'model_not_found', 'invalid-request'], [401, 'invalid_api_key', 'provider'], [503, 'not_ready', 'unavailable'],
    [500, 'internal_error', 'provider'], [504, 'timeout', 'timeout']]) {
    it(`maps HTTP ${status} ${code} to ${kind}`, async () => {
      const s = await stub((req, res) => json(res, status, { error: { message: 'm', code } }))
      await assert.rejects(postOpenAIEmbeddings({ baseUrl: s.baseUrl, model: 'm', input: ['a'], timeoutMs: 1000 }),
        e => TextEmbeddingError.is(e) && e.kind === kind && e.details.status === status)
      await s.close()
    })
  }
  it('maps refused connections, timeouts, aborts and count mismatches', async () => {
    await assert.rejects(postOpenAIEmbeddings({ baseUrl: 'http://127.0.0.1:1/v1', model: 'm', input: ['a'], timeoutMs: 1000 }), e => e.kind === 'unavailable')
    const slow = await stub(() => {})
    await assert.rejects(postOpenAIEmbeddings({ baseUrl: slow.baseUrl, model: 'm', input: ['a'], timeoutMs: 100 }), e => e.kind === 'timeout')
    const controller = new AbortController(); setTimeout(() => controller.abort(), 50)
    await assert.rejects(postOpenAIEmbeddings({ baseUrl: slow.baseUrl, model: 'm', input: ['a'], timeoutMs: 5000, signal: controller.signal }), e => e.kind === 'aborted')
    await slow.close()
    const short = await stub((req, res) => json(res, 200, { data: [{ index: 0, embedding: [1] }] }))
    await assert.rejects(postOpenAIEmbeddings({ baseUrl: short.baseUrl, model: 'm', input: ['a', 'b'], timeoutMs: 1000 }), e => e.kind === 'provider')
    await short.close()
  })
})

describe('resolveAgainst', () => {
  const info = { provider: 'p', model: 'm', dimensions: [768, 256], defaultDimensions: 768, tasks: ['query', 'document'], defaultTask: 'none', status: 'ready' }
  it('applies defaults and validates', () => {
    assert.deepEqual(resolveAgainst({ input: ['x'] }, info), { input: ['x'], task: 'none', dimensions: 768 })
    assert.equal(resolveAgainst({ input: ['x'], task: 'QUERY', dimensions: 256 }, info).dimensions, 256)
    for (const bad of [{ input: [] }, { input: ['x'], dimensions: 100 }, { input: ['x'], task: 'nope' }]) {
      assert.throws(() => resolveAgainst(bad, info), e => e.kind === 'invalid-request')
    }
    assert.throws(() => resolveAgainst({ input: ['x'], task: 'query' }, { ...info, tasks: [] }), e => e.kind === 'invalid-request')
  })
  it('renormalizes Matryoshka prefixes', () => {
    const [v] = truncateNormalize([Float32Array.from([3, 4, 12])], 2)
    assert.deepEqual(Array.from(v), [0.6000000238418579, 0.800000011920929])
  })
})

describe('openai provider in a Cordis context', () => {
  it('registers ctx.textEmbeddings, prefixes tasks, truncates, emits settled, and unregisters on dispose', async () => {
    const s = await stub(echo(4))
    const ctx = new Context()
    const settled = []
    ctx.on('text-embeddings/settled', e => settled.push(e))
    const fiber = ctx.plugin(OpenAIProvider, { baseUrl: s.baseUrl, model: 'm', dimensions: 4, allowedDimensions: [2], taskPrefixes: { query: 'q: ' } })
    await settle()
    const service = ctx.get('textEmbeddings')
    const plain = await service.embed({ input: ['ab'] })
    const query = await service.embed({ input: ['ab'], task: 'query', dimensions: 2 })
    assert.equal(plain.vectors[0].length, 4)
    assert.equal(query.vectors[0].length, 2)
    // 'q: ab' is 5 chars: echo gives (i+1)*6, truncated to [6,12] and renormalized
    assert.ok(Math.abs(query.vectors[0][0] - 6 / Math.hypot(6, 12)) < 1e-6)
    await assert.rejects(service.embed({ input: ['x'], task: 'document' }), e => e.kind === 'invalid-request')
    assert.deepEqual(settled.map(e => e.error ?? 'ok'), ['ok', 'ok', 'invalid-request'])
    await fiber.dispose()
    assert.equal(ctx.get('textEmbeddings'), undefined)
    await s.close()
  })
  it('fails loud on invalid configuration', async () => {
    const ctx = new Context()
    const fiber = ctx.plugin(OpenAIProvider, { baseUrl: 'not a url', model: 'm', dimensions: 4 })
    await settle()
    assert.equal(ctx.get('textEmbeddings'), undefined)
    await fiber.dispose()
  })
})

const spec = (port, extra = {}) => ({
  argv: [process.execPath, FAKE, '--port', String(port)], cwd: process.cwd(), env: {}, origin: `http://127.0.0.1:${port}`,
  startupTimeoutMs: 5000, healthIntervalMs: 50, healthFailures: 2, maxRestarts: 2, restartWindowMs: 60_000,
  backoffMs: 20, maxBackoffMs: 100, graceMs: 500, ...extra,
})
const quiet = { info() {}, warn() {}, error() {} }
let nextPort = 18400 + Math.floor(Math.random() * 500)

describe('daemon supervisor', () => {
  it('starts, and stop() terminates the process', async () => {
    const runtime = new TestRuntime()
    const sup = new DaemonSupervisor(runtime, spec(nextPort++), quiet)
    await sup.start()
    assert.equal(sup.status, 'ready')
    await sup.stop()
    assert.notEqual(runtime.spawned[0].exitCode ?? runtime.spawned[0].signalCode, null)
  })
  it('reports a failed first start with the daemon error and stderr tail', async () => {
    const runtime = new TestRuntime()
    const sup = new DaemonSupervisor(runtime, { ...spec(nextPort++), env: { FAKE_MODE: 'fail' } }, quiet)
    await assert.rejects(sup.start(), e => /FileNotFoundError/.test(e.message) && /traceback/.test(e.message))
    assert.equal(sup.status, 'failed')
  })
  it('restarts after crashes with backoff, then gives up', async () => {
    const runtime = new TestRuntime()
    const sup = new DaemonSupervisor(runtime, { ...spec(nextPort++), env: { FAKE_MODE: 'crash', FAKE_CRASH_MS: '60' } }, quiet)
    await sup.start()
    for (let i = 0; i < 100 && sup.status !== 'failed'; i++) await settle(20)
    assert.equal(sup.status, 'failed')
    assert.equal(runtime.spawned.length, 3) // first start + maxRestarts
    assert.match(sup.lastError, /code 3/)
    await assert.rejects(sup.waitReady(100), /code 3/)
    await sup.stop()
  })
  it('restarts a process whose health checks stop answering', async () => {
    const runtime = new TestRuntime()
    const sup = new DaemonSupervisor(runtime, { ...spec(nextPort++), env: { FAKE_MODE: 'hang' }, maxRestarts: 1 }, quiet)
    await sup.start()
    for (let i = 0; i < 100 && runtime.spawned.length < 2; i++) await settle(20)
    assert.ok(runtime.spawned.length >= 2)
    await sup.stop()
  })
})

describe('tsuzuki engine provider in a Cordis context', () => {
  const engineRoot = fileURLToPath(new URL('../..', import.meta.url)).replace(/\/$/, '')
  it('derives the daemon command line from engineRoot', () => {
    const launch = resolveLaunch({ engineRoot, modelDir: '', modelName: 'embeddinggemma-2', python: '', buildDir: '', dtype: 'fp16', matmul: 'auto', dimensions: 256,
      defaultTask: 'query', host: '127.0.0.1', port: 9, autoStart: true, apiKey: 'k', maxTokens: 512, extraArgs: ['--batch-wait-ms', '1'],
      startupTimeoutMs: 1, requestTimeoutMs: 1, readyWaitMs: 1, healthIntervalMs: 1, healthFailures: 1, maxRestarts: 1,
      restartWindowMs: 1, backoffMs: 1, maxBackoffMs: 1, graceMs: 1 })
    assert.equal(launch.modelDir, MODEL_DIR.replace(/\/$/, ''))
    assert.deepEqual(launch.supervisor.argv.slice(1, 3), ['-m', 'egemma_server'])
    assert.ok(launch.supervisor.argv.includes('--default-task') && launch.supervisor.argv.at(-1) === '1')
    assert.equal(launch.supervisor.env.EGEMMA_API_KEY, 'k')
    const name = launch.supervisor.argv.indexOf('--model-name')
    assert.equal(launch.supervisor.argv[name + 1], 'embeddinggemma-2')
  })
  it('owns the real egemma-server: starts it, embeds with task and dimensions, stops it on dispose', { timeout: 300_000, skip: !HAS_ENGINE && 'needs the built engine' }, async () => {
    const port = nextPort++
    const ctx = new Context()
    ctx.provide('subprocess', new TestRuntime())
    const fiber = ctx.plugin(TsuzukiProvider, { engineRoot, port, healthIntervalMs: 1000 })
    let service
    for (let i = 0; i < 600 && !(service = ctx.get('textEmbeddings')); i++) await settle(100)
    const info = service.describe()
    assert.equal(info.status, 'ready')
    assert.ok(info.tasks.includes('query') && info.tasks.includes('document'))
    const r = await service.embed({ input: ['Which planet is red?', '火星は赤い'], task: 'query', dimensions: 256 })
    assert.equal(r.vectors.length, 2)
    assert.equal(r.vectors[0].length, 256)
    assert.equal(r.model, 'embeddinggemma-2:query:256')
    assert.ok(Math.abs(Math.hypot(...r.vectors[0]) - 1) < 1e-5)
    await assert.rejects(service.embed({ input: ['x '.repeat(3000)] }), e => e.kind === 'input-too-long')
    await fiber.dispose()
    await assert.rejects(fetch(`http://127.0.0.1:${port}/health`))
  })
  it('serves EmbeddingGemma 300m when modelDir/modelName select it (dtype auto = fp32)', { timeout: 300_000, skip: !(HAS_ENGINE && existsSync(`${V1_DIR}/model.safetensors`)) }, async () => {
    const port = nextPort++
    const ctx = new Context()
    ctx.provide('subprocess', new TestRuntime())
    const fiber = ctx.plugin(TsuzukiProvider, { engineRoot, port, healthIntervalMs: 1000, modelDir: V1_DIR,
      modelName: 'embeddinggemma-300m' })
    let service
    for (let i = 0; i < 600 && !(service = ctx.get('textEmbeddings')); i++) await settle(100)
    const info = service.describe()
    assert.equal(info.status, 'ready')
    assert.equal(info.model, 'embeddinggemma-300m')
    const r = await service.embed({ input: ['Which planet is red?'], task: 'query', dimensions: 128 })
    assert.equal(r.model, 'embeddinggemma-300m:query:128')
    const stats = await (await fetch(`http://127.0.0.1:${port}/stats`)).json()
    assert.equal(stats.engine.dtype, 'fp32')
    assert.equal(r.vectors[0].length, 128)
    assert.ok(Math.abs(Math.hypot(...r.vectors[0]) - 1) < 1e-5)
    await fiber.dispose()
    await assert.rejects(fetch(`http://127.0.0.1:${port}/health`))
  })
  it('shares an egemma-server already serving the model, and takes the port over when it goes away', { skip: !HAS_V2 && 'needs models/embeddinggemma-2' }, async () => {
    let shared = 0
    const s = await stub((req, res) => {
      if (req.url === '/health') return json(res, 200, { status: 'ready', model: 'embeddinggemma-2' })
      shared++
      echo(768)(req, res)
    })
    const port = Number(new URL(s.baseUrl).port)
    const dir = mkdtempSync(join(tmpdir(), 'tsuzuki-fake-python-'))
    const python = join(dir, 'python') // stands in for `python -m egemma_server ...`
    writeFileSync(python, `#!/bin/sh\nexec "${process.execPath}" "${FAKE}" "$@"\n`, { mode: 0o755 })
    const ctx = new Context()
    ctx.provide('subprocess', new TestRuntime())
    const fiber = ctx.plugin(TsuzukiProvider, { engineRoot, port, python, healthIntervalMs: 1000 })
    let service
    for (let i = 0; i < 100 && !(service = ctx.get('textEmbeddings')); i++) await settle(20)
    assert.equal(service.describe().status, 'ready')
    await service.embed({ input: ['a'], task: 'query' })
    assert.equal(shared, 1, 'the running daemon is used, not duplicated')
    await s.close()
    const r = await service.embed({ input: ['b'], task: 'query', dimensions: 256 })
    assert.equal(r.model, 'embeddinggemma-2:query:256', 'answered by the daemon the plugin started on the freed port')
    assert.equal(shared, 1)
    const owned = await (await fetch(`http://127.0.0.1:${port}/health`)).json()
    await fiber.dispose()
    await assert.rejects(fetch(`http://127.0.0.1:${port}/health`), 'the taken-over daemon is owned and stopped on unload')
    assert.ok(owned.pid)
    rmSync(dir, { recursive: true, force: true })
  })
  it('refuses to start over a port held by something else', { skip: !HAS_V2 && 'needs models/embeddinggemma-2' }, async () => {
    const s = await stub((req, res) => json(res, 200, { status: 'ready', model: 'another-model' }))
    const port = Number(new URL(s.baseUrl).port)
    const ctx = new Context()
    ctx.provide('subprocess', new TestRuntime())
    const fiber = ctx.plugin(TsuzukiProvider, { engineRoot, port })
    await settle(200)
    assert.equal(ctx.get('textEmbeddings'), undefined)
    await fiber.dispose()
    // attach mode uses the running daemon without owning it
    const attach = ctx.plugin(TsuzukiProvider, { engineRoot, port, autoStart: false })
    await settle(100)
    assert.equal(ctx.get('textEmbeddings').describe().provider, 'egemma')
    await attach.dispose()
    await s.close()
  })
})
