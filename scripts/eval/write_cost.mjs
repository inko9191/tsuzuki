// Write-time cost of facts in normal use: one conversation at a time, as the sidecar does it after each conversation.
// Uses the package's compiled code unchanged (Memory, MemoryStore, OpenAIChat, HttpEmbedder) with the sidecar's
// defaults, and starts two llama-servers with the sidecar's own arguments (LLAMA_EMBEDDING_ARGS, LLAMA_CHAT_ARGS, and
// `--reasoning off` as scripts/setup.mjs adds it). Per conversation: capture every message (store + embed its
// window), close the conversation's segments (as the 30-minute idle close does), then drain: fact extraction, fact
// embedding and the update check, one segment at a time. Each chat request is timed.
//
// Conversations: the first --sessions filler sessions of the scale test (LongMemEval-S; build/scale/plan.json and
// sessions.jsonl, made by scripts/eval/scale/plan.py), in plan order, 96 minutes apart.
//
//   node scripts/eval/write_cost.mjs --models ~/.tsuzuki/models --sessions 60 --out results/eval/memory/write-cost.jsonl
import { appendFileSync, createReadStream, mkdtempSync, openSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:net'
import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import { dirname, join } from 'node:path'
import { cpus, tmpdir, totalmem } from 'node:os'
import { fileURLToPath } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..')
const PKG = join(ROOT, 'dsh-plugins/packages/tsuzuki/lib')
const { MemoryStore } = await import(join(PKG, 'store.js'))
const { Memory, estimateTokens } = await import(join(PKG, 'memory.js'))
const { OpenAIChat, LLAMA_CPP_EXTRA } = await import(join(PKG, 'llm.js'))
const { HttpEmbedder, EMBEDDINGGEMMA_PREFIXES } = await import(join(PKG, 'embedding-http.js'))
const { LLAMA_EMBEDDING_ARGS, LLAMA_CHAT_ARGS } = await import(join(PKG, 'server.js'))

const arg = (name, fallback) => { const i = process.argv.indexOf(`--${name}`); return i >= 0 ? process.argv[i + 1] : fallback }
const models = arg('models')
const binary = arg('llama-server', 'llama-server')
const count = Number(arg('sessions', '60'))
const out = arg('out')
if (!models || !out) throw new Error('usage: --models DIR --out FILE [--sessions 60] [--llama-server BIN] [--scale build/scale]')
const scale = arg('scale', join(ROOT, 'build/scale'))

const plan = JSON.parse(readFileSync(join(scale, 'plan.json'), 'utf8'))
const wanted = plan.fillers.slice(0, count)
const sessions = new Map()
for await (const line of createInterface({ input: createReadStream(join(scale, 'sessions.jsonl')), crlfDelay: Infinity })) {
  if (!line) continue
  const s = JSON.parse(line)
  if (wanted.includes(s.id)) sessions.set(s.id, s.turns)
}

const freePort = () => new Promise((resolve, reject) => {
  const s = createServer().once('error', reject).listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => resolve(port)) })
})
const work = mkdtempSync(join(tmpdir(), 'write-cost-'))
const children = []
async function llamaServer(model, args, alias) {
  const port = await freePort()
  const log = openSync(join(work, `${alias}.log`), 'a')
  const child = spawn(binary, ['-m', join(models, model), ...args, '--host', '127.0.0.1', '--port', String(port), '--alias', alias],
    { stdio: ['ignore', log, log] })
  children.push(child)
  for (let k = 0; ; k++) {
    if (child.exitCode !== null) throw new Error(`llama-server exited; see ${join(work, `${alias}.log`)}`)
    try { if ((await fetch(`http://127.0.0.1:${port}/health`)).ok) return port } catch {}
    if (k > 600) throw new Error(`llama-server ${alias} not ready after 5 minutes`)
    await new Promise(r => setTimeout(r, 500))
  }
}
const stop = () => { for (const c of children) c.kill() }
process.on('exit', stop)

const version = await new Promise(resolve => {
  const p = spawn(binary, ['--version'], { stdio: ['ignore', 'pipe', 'pipe'] })
  let text = ''
  p.stdout.on('data', d => { text += d }); p.stderr.on('data', d => { text += d })
  p.on('close', () => resolve(text.trim().split('\n').filter(l => l.startsWith('version')).join(' ')))
})
const embedPort = await llamaServer('embeddinggemma-300M-Q8_0.gguf', LLAMA_EMBEDDING_ARGS, 'write-cost-embeddinggemma')
const chatPort = await llamaServer('gemma-4-E2B_q4_0-it.gguf', [...LLAMA_CHAT_ARGS, '--reasoning', 'off'], 'write-cost-chat')

const embedder = new HttpEmbedder({ baseUrl: `http://127.0.0.1:${embedPort}/v1`, model: 'write-cost-embeddinggemma', apiKey: '',
  ...EMBEDDINGGEMMA_PREFIXES, timeoutMs: 60000, batch: 32 })
const client = new OpenAIChat({ baseUrl: `http://127.0.0.1:${chatPort}/v1`, model: 'write-cost-chat', apiKey: '',
  timeoutMs: 180000, extraBody: LLAMA_CPP_EXTRA })
let calls = []
/** The sidecar's chat client, each request timed (name: 'fact' = extraction, 'updates' = update check). */
const chat = {
  baseUrl: client.baseUrl,
  reachable: signal => client.reachable(signal),
  async complete(messages, options) {
    const t0 = performance.now()
    try {
      const reply = await client.complete(messages, options)
      calls.push({ name: options.name, ms: performance.now() - t0, ok: true })
      return reply
    } catch (error) {
      calls.push({ name: options.name, ms: performance.now() - t0, ok: false, error: String(error).slice(0, 200) })
      throw error
    }
  },
}

const warnings = []
const memory = new Memory(new MemoryStore(join(work, 'memory.db')), embedder, chat, {
  budgetTokens: 2048, segmentTokens: 2500, segmentIdleMinutes: 30, updateLinks: true, redactSecrets: true,
  queryTimeoutMs: 3000, rerankTimeoutMs: 15000, log: (level, m) => { if (level === 'warn') warnings.push(m) },
})

writeFileSync(out, JSON.stringify({ header: true, machine: `${cpus()[0]?.model}, ${Math.round(totalmem() / 2 ** 30)} GB`,
  platform: process.platform, node: process.version, llamaServer: version,
  models: ['embeddinggemma-300M-Q8_0.gguf', 'gemma-4-E2B_q4_0-it.gguf'], sessions: count,
  embeddingArgs: LLAMA_EMBEDDING_ARGS, chatArgs: [...LLAMA_CHAT_ARGS, '--reasoning', 'off'] }) + '\n')
const start = new Date(2026, 0, 1, 9, 0).getTime()
for (const [k, id] of wanted.entries()) {
  const turns = sessions.get(id)
  if (!turns) throw new Error(`session ${id} missing from sessions.jsonl`)
  const session = `conversation-${String(k).padStart(3, '0')}`
  calls = []
  warnings.length = 0
  const t0 = performance.now()
  for (const [i, t] of turns.entries()) {
    await memory.capture({ id: `${session}-${i}`, session, role: t.role, text: t.content, at: start + k * 96 * 60_000 + i * 60_000 })
  }
  const t1 = performance.now()
  memory.closeAll()
  await memory.drain()
  const t2 = performance.now()
  const tokens = turns.reduce((n, t) => n + estimateTokens(t.content), 0)
  appendFileSync(out, JSON.stringify({ k, id, messages: turns.length, tokens, captureMs: t1 - t0, factsMs: t2 - t1, calls,
    warnings: [...warnings] }) + '\n')
  console.log(`${k}: ${tokens} tokens, capture ${((t1 - t0) / 1000).toFixed(1)} s, facts ${((t2 - t1) / 1000).toFixed(1)} s`
    + (warnings.length ? ` (${warnings.length} warnings)` : ''))
}
appendFileSync(out, JSON.stringify({ footer: true, ...memory.store.stats() }) + '\n')
memory.close()
stop()
rmSync(work, { recursive: true, force: true })
