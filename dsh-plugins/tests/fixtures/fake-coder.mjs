// CPU-only stand-in for a coding model, for the working-memory e2e: embeddings as in fake-openai.mjs, and an agent
// that runs a scripted series of bash commands whose outputs fill a small context window, so the harness compacts.
// The first command prints a unique marker; the summary written for the compaction leaves it out, so the marker can
// only come back through working memory. Every request is logged to $FAKE_LOG.
import { appendFileSync } from 'node:fs'
import { createServer } from 'node:http'

const port = Number(process.argv[2])
const log = process.env.FAKE_LOG
const STEPS = Number(process.env.FAKE_STEPS ?? 6)
export const MARKER = 'E4242 widget config is missing the key retry_timeout'

function vector(text) {
  const v = new Array(64).fill(0)
  const body = text.replace(/^task: search result \| query: /, '').replace(/^title: none \| text: /, '')
  for (const w of body.toLowerCase().match(/[a-z0-9_]+/g) ?? []) {
    if (['the', 'a', 'i', 'my', 'is', 'to', 'do', 'now', 'user', 'assistant', 'tool', 'and', 'of', 'in', 'call', 'bash'].includes(w)) continue
    let h = 2166136261
    for (const c of w) h = Math.imul(h ^ c.codePointAt(0), 16777619)
    v[(h >>> 0) % 64] += 1
  }
  const n = Math.hypot(...v) || 1
  return v.map(x => x / n)
}

const record = (kind, body) => { if (log) appendFileSync(log, JSON.stringify({ kind, body }) + '\n') }
const json = (res, value) => { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify(value)) }
const tokens = body => Math.ceil(JSON.stringify(body.messages ?? []).length / 4)

let agentCalls = 0
const commands = [
  // printf assembles the marker, so it appears in the tool output but never in the model's own command text.
  `printf 'E%s%s widget config is missing the key retry_timeout\\n' 42 42; seq 1 1500`,
  ...Array.from({ length: 20 }, (_, i) => `echo "step ${i + 2}"; seq ${i * 1000 + 1} ${i * 1000 + 1500}`),
]

createServer((req, res) => {
  if (req.method === 'GET' && req.url.endsWith('/models')) return json(res, { object: 'list', data: [{ id: 'fake-coder', object: 'model' }] })
  let raw = ''
  req.on('data', c => { raw += c })
  req.on('end', () => {
    const body = raw ? JSON.parse(raw) : {}
    if (req.url.endsWith('/embeddings')) {
      return json(res, { object: 'list', model: body.model, data: body.input.map((t, index) => ({ object: 'embedding', index, embedding: vector(t) })),
        usage: { prompt_tokens: 1, total_tokens: 1 } })
    }
    const text = JSON.stringify(body.messages ?? [])
    const usage = { prompt_tokens: tokens(body), completion_tokens: 20, total_tokens: tokens(body) + 20 }
    const tools = (body.tools ?? []).map(t => t.function ?? t)
    const bash = tools.find(t => /bash/i.test(t.name))
    let content = '', call = null
    const lastText = JSON.stringify(body.messages?.at(-1)?.content ?? '')
    if (!bash || /acting as a compaction engine/.test(lastText)) {
      // The compaction request (dsh keeps the tools on it for caching) or a title. The summary omits the marker.
      record('summary', { tokens: tokens(body), hasMarker: text.includes('E4242') })
      content = 'Summary: the user asked to investigate the widget build. Several shell commands were run and printed long number lists.'
    } else {
      agentCalls++
      record('agent', { call: agentCalls, tokens: tokens(body), hasMemory: text.includes('that were summarized away'),
        hasMarker: text.includes('E4242'), hasSummary: text.includes('Summary: the user asked'), messages: body.messages.length,
        tools: agentCalls === 1 ? tools.map(t => t.name) : undefined })
      if (agentCalls <= STEPS) {
        call = { id: `call_${agentCalls}`, name: bash.name, arguments: JSON.stringify({ command: commands[agentCalls - 1], description: `diagnostics step ${agentCalls}` }) }
      } else {
        content = text.includes('E4242') ? 'The first error was E4242 (retry_timeout).' : 'I no longer know the first error.'
      }
    }
    if (!body.stream) {
      const message = call ? { role: 'assistant', content: null, tool_calls: [{ id: call.id, type: 'function', function: { name: call.name, arguments: call.arguments } }] }
        : { role: 'assistant', content }
      return json(res, { id: 'c', object: 'chat.completion', model: 'fake-coder', choices: [{ index: 0, message, finish_reason: call ? 'tool_calls' : 'stop' }], usage })
    }
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' })
    const chunk = (delta, finish, extra = {}) => res.write(`data: ${JSON.stringify({ id: 'c', object: 'chat.completion.chunk', created: 0, model: 'fake-coder',
      choices: [{ index: 0, delta, finish_reason: finish }], ...extra })}\n\n`)
    if (call) {
      chunk({ role: 'assistant', tool_calls: [{ index: 0, id: call.id, type: 'function', function: { name: call.name, arguments: call.arguments } }] }, null)
      chunk({}, 'tool_calls', { usage })
    } else {
      chunk({ role: 'assistant', content }, null)
      chunk({}, 'stop', { usage })
    }
    res.end('data: [DONE]\n\n')
  })
}).listen(port, '127.0.0.1', () => console.log(`fake-coder on ${port}`))
