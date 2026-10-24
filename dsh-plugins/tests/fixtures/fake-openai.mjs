// CPU-only stand-in for both the embedding server and the chat model (agent + facts + rerank).
// Embeddings: 64-dim bag-of-words vectors after removing the EmbeddingGemma task prefixes.
// Chat: JSON-schema requests named `fact`/`rerank` get structured answers; agent requests (streamed)
// are recorded to $FAKE_LOG as JSON lines and answered with a fixed sentence. A user message saying
// "forget everything" is answered with a memory_forget {all: true} call when that tool is offered.
import { appendFileSync } from 'node:fs'
import { createServer } from 'node:http'

const port = Number(process.argv[2])
const log = process.env.FAKE_LOG

function vector(text) {
  const v = new Array(64).fill(0)
  const body = text.replace(/^task: search result \| query: /, '').replace(/^title: none \| text: /, '')
  for (const w of body.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (['the', 'a', 'i', 'my', 'is', 'to', 'do', 'now', 'user', 'assistant', 'and', 'of', 'in', 'last'].includes(w)) continue
    let h = 2166136261
    for (const c of w) h = Math.imul(h ^ c.codePointAt(0), 16777619)
    v[(h >>> 0) % 64] += 1
  }
  const n = Math.hypot(...v) || 1
  return v.map(x => x / n)
}

function record(kind, body) {
  if (log) appendFileSync(log, JSON.stringify({ kind, body }) + '\n')
}

function json(res, value) {
  res.writeHead(200, { 'content-type': 'application/json' })
  res.end(JSON.stringify(value))
}

createServer((req, res) => {
  if (req.method === 'GET' && req.url.endsWith('/models')) return json(res, { object: 'list', data: [{ id: 'fake-qwen', object: 'model' }] })
  if (req.method === 'GET' && req.url === '/slots') return json(res, [{ id: 0, is_processing: false }])
  let raw = ''
  req.on('data', c => { raw += c })
  req.on('end', () => {
    const body = raw ? JSON.parse(raw) : {}
    if (req.url.endsWith('/embeddings')) {
      record('embeddings', { n: body.input.length })
      return json(res, { object: 'list', model: body.model, data: body.input.map((t, index) => ({ object: 'embedding', index, embedding: vector(t) })),
        usage: { prompt_tokens: 1, total_tokens: 1 } })
    }
    const schema = body.response_format?.json_schema?.name
    if (schema === 'fact') {
      const text = body.messages.at(-1).content
      const facts = [...text.matchAll(/^\[(M\d+)\] user: (.+)$/gm)].map(([, id, said]) => ({ fact: `The user said: ${said}`, ids: [id] }))
      record('fact', { facts: facts.length })
      return json(res, { choices: [{ index: 0, message: { role: 'assistant', content: JSON.stringify({ facts }) }, finish_reason: 'stop' }] })
    }
    if (schema === 'updates') {
      const text = body.messages.at(-1).content
      record('updates', {})
      const value = /\[E1\]/.test(text) && /\[N1\]/.test(text) ? { updates: [{ old: 'E1', new: 'N1' }] } : { updates: [] }
      return json(res, { choices: [{ index: 0, message: { role: 'assistant', content: JSON.stringify(value) }, finish_reason: 'stop' }] })
    }
    if (schema === 'rerank') {
      record('rerank', {})
      return json(res, { choices: [{ index: 0, message: { role: 'assistant', content: '{"ranking": [1]}' }, finish_reason: 'stop' }] })
    }
    const tools = (body.tools ?? []).map(t => (t.function ?? t).name)
    const messages = body.messages ?? []
    // dsh appends runtime-context user messages after the user's own; recall blocks are left out.
    const asked = messages.some(m => m.role === 'user' && !JSON.stringify(m.content).includes('<memory>')
      && /forget everything/i.test(JSON.stringify(m.content)))
    const result = messages.find(m => m.role === 'tool')
    const call = asked && tools.includes('memory_forget') && !result
      ? { id: 'call_forget', name: 'memory_forget', arguments: JSON.stringify({ all: true }) } : null
    record('agent', { messages: body.messages, tools, ...(call ? { call: call.name } : {}) })
    const content = 'Noted.'
    const usage = { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 }
    const message = call ? { role: 'assistant', content: null, tool_calls: [{ id: call.id, type: 'function', function: { name: call.name, arguments: call.arguments } }] }
      : { role: 'assistant', content }
    if (!body.stream) return json(res, { id: 'c', object: 'chat.completion', model: 'fake-qwen',
      choices: [{ index: 0, message, finish_reason: call ? 'tool_calls' : 'stop' }], usage })
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' })
    const chunk = (delta, finish, extra = {}) => res.write(`data: ${JSON.stringify({ id: 'c', object: 'chat.completion.chunk', created: 0, model: 'fake-qwen',
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
}).listen(port, '127.0.0.1', () => console.log(`fake-openai on ${port}`))
