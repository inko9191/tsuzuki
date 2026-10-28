// Stands in for llama.cpp's llama-server in tests: parses -m, --port, --alias (and --embedding), answers /health,
// /v1/models, /v1/embeddings (bag-of-words vectors) and /v1/chat/completions (no facts). Writes its argv to FAKE_LLAMA_LOG.
import { appendFileSync } from 'node:fs'
import { createServer } from 'node:http'

const argv = process.argv.slice(2)
const value = name => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined }
if (process.env.FAKE_LLAMA_LOG) appendFileSync(process.env.FAKE_LLAMA_LOG, JSON.stringify(argv) + '\n')
const alias = value('--alias'), port = Number(value('--port'))
// The EmbeddingGemma task prompts are left out: counted as words, they would pull the cosine of these short texts
// below the default recall gate (recall.minSimilarity).
const vector = text => {
  const v = new Array(64).fill(0)
  const body = text.replace(/^task: search result \| query: /, '').replace(/^title: none \| text: /, '')
  for (const w of body.toLowerCase().match(/[a-z0-9]+/g) ?? []) { let h = 7; for (const c of w) h = (h * 31 + c.charCodeAt(0)) >>> 0; v[h % 64] += 1 }
  const n = Math.hypot(...v) || 1
  return v.map(x => x / n)
}
createServer((req, res) => {
  let body = ''
  req.on('data', d => { body += d })
  req.on('end', () => {
    const send = (status, value) => { res.writeHead(status, { 'content-type': 'application/json' }); res.end(JSON.stringify(value)) }
    if (req.url === '/health') return send(200, { status: 'ok' })
    if (req.url === '/v1/models') return send(200, { data: [{ id: alias }] })
    if (req.url === '/v1/embeddings') return send(200, { data: JSON.parse(body).input.map((t, index) => ({ index, embedding: vector(t) })) })
    if (req.url === '/v1/chat/completions') return send(200, { choices: [{ message: { content: '{"facts": []}' }, finish_reason: 'stop' }] })
    send(404, { error: 'not found' })
  })
}).listen(port, '127.0.0.1')
