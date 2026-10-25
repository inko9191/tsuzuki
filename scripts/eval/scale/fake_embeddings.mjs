// Test double for egemma-server (dry runs of the GPU steps on CPU): POST /v1/embeddings returns deterministic
// pseudo-random vectors and refuses, like egemma-server, the inputs window-tokens.json counts over 2,048 tokens.
//
//   node scripts/eval/scale/fake_embeddings.mjs --port 8396
import { createServer } from 'node:http'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { OUT, sha1, syntheticEmbedder } from './common.mjs'

const port = Number(process.argv[process.argv.indexOf('--port') + 1] || 8396)
const tokens = JSON.parse(readFileSync(join(OUT, 'window-tokens.json'), 'utf8'))
const over = new Set(Object.entries(tokens.documents).filter(([, n]) => n > tokens.limit).map(([h]) => h))
const PREFIX = 'title: none | text: '
const synth = syntheticEmbedder()
createServer((req, res) => {
  if (req.method === 'GET' && req.url === '/health') {
    res.writeHead(200, { 'content-type': 'application/json' })
    return res.end(JSON.stringify({ status: 'ready', model: 'fake' }))
  }
  let raw = ''
  req.on('data', c => { raw += c })
  req.on('end', async () => {
    const b = JSON.parse(raw)
    const input = Array.isArray(b.input) ? b.input : [b.input]
    if (input.some(t => t.startsWith(PREFIX) && over.has(sha1(t.slice(PREFIX.length))))) {
      res.writeHead(400, { 'content-type': 'application/json' })
      return res.end(JSON.stringify({ error: { message: 'context_length_exceeded' } }))
    }
    const vecs = await synth.embed(input)
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ data: vecs.map((v, index) => ({ index, embedding: Array.from(v) })) }))
  })
}).listen(port, '127.0.0.1', () => console.log(JSON.stringify({ event: 'ready', port })))
