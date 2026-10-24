// Stand-in for egemma-server used by the supervisor tests.
// FAKE_MODE: ready | fail | crash (exit after FAKE_CRASH_MS) | hang (health stops answering after ready)
import { createServer } from 'node:http'

const port = Number(process.argv[process.argv.indexOf('--port') + 1])
const mode = process.env.FAKE_MODE ?? 'ready'
if (mode === 'fail') {
  console.log(JSON.stringify({ event: 'failed', stage: 'load', error: 'FileNotFoundError: model' }))
  console.error('traceback: model missing')
  process.exit(1)
}
let healthy = true
const server = createServer((req, res) => {
  if (req.url === '/health') {
    if (!healthy) return // hang: never answer
    res.writeHead(200, { 'Content-Type': 'application/json' })
    return res.end(JSON.stringify({ status: 'ready', model: 'embeddinggemma-2', pid: process.pid }))
  }
  let body = ''
  req.on('data', c => { body += c })
  req.on('end', () => {
    const request = JSON.parse(body)
    const dims = request.dimensions ?? 768
    const data = request.input.map((_, index) => ({ object: 'embedding', index, embedding: Array.from({ length: dims }, (_, i) => i === index % dims ? 1 : 0) }))
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ object: 'list', model: `embeddinggemma-2:${request.task}:${dims}`, data, usage: { prompt_tokens: 3, total_tokens: 3 } }))
  })
})
server.listen(port, '127.0.0.1', () => {
  console.log(JSON.stringify({ event: 'ready', port, pid: process.pid, cold_start: { ready_s: 0.01 } }))
  if (mode === 'crash') setTimeout(() => process.exit(3), Number(process.env.FAKE_CRASH_MS ?? 100))
  if (mode === 'hang') setTimeout(() => { healthy = false }, 50)
})
process.on('SIGTERM', () => server.close(() => process.exit(0)))
