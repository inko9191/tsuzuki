#!/usr/bin/env node
// One-command setup (macOS, Linux, Windows; needs only Node and llama.cpp's llama-server):
//   1. builds the plugins and the sidecar,
//   2. downloads the models after asking (EmbeddingGemma 300m GGUF for search, Gemma 4 E2B GGUF for facts),
//   3. connects the memory to Hermes Agent, or prepares it for MCP clients (Claude Desktop, Claude Code, Cursor,
//      LM Studio...), for chat apps through a proxy in front of a local model server, or for any app over HTTP,
//   4. checks that storing, recalling and fact extraction work (in a throwaway store).
// The memory starts the two llama-servers itself when it needs them; you do not run them.
// Run with --help for the options.
import { spawn, spawnSync } from 'node:child_process'
import { createHash, randomBytes } from 'node:crypto'
import { chmodSync, createReadStream, createWriteStream, existsSync, mkdirSync, openSync, readdirSync, readFileSync, renameSync,
  rmSync, statSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:net'
import { homedir, platform, tmpdir } from 'node:os'
import { delimiter, dirname, join, resolve } from 'node:path'
import { createInterface } from 'node:readline/promises'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const PKG = join(ROOT, 'dsh-plugins/packages/tsuzuki')
const SERVER_JS = join(PKG, 'lib/server.js')
const MCP_JS = join(PKG, 'lib/mcp.js')
const WIN = platform() === 'win32'
const HOME_DIR = join(homedir(), '.tsuzuki')
const CONFIG = join(HOME_DIR, 'config.json')
/** Model servers looked for (in this order) when the proxy's upstream is not given. */
const UPSTREAMS = [['llama.cpp', 'http://127.0.0.1:8080/v1'], ['Ollama', 'http://127.0.0.1:11434/v1'], ['LM Studio', 'http://127.0.0.1:1234/v1']]
const HELP = `Usage: node scripts/setup.mjs [options]

  --for hermes|mcp|chat|app
                         where the memory is used (asked if missing): Hermes Agent; an MCP client (Claude Desktop,
                         Claude Code, Cursor, LM Studio...); chat apps, through a proxy in front of your model
                         server; or another app, over HTTP
  --upstream URL         with --for chat: your model server's OpenAI-compatible URL (default: the first that answers
                         of llama.cpp :8080, Ollama :11434, LM Studio :1234)
  --profile NAME         Hermes profile (default: "default", the Hermes home itself)
  --hermes-home DIR      Hermes home (default: $HERMES_HOME, else ~/.hermes; on Windows %LOCALAPPDATA%\\hermes)
  --models DIR           where the models go (default: ~/.tsuzuki/models)
  --llama-server PATH    llama.cpp's llama-server, if it is not on PATH
  --no-facts             do not download the facts model (memory works on the messages alone)
  --yes                  do not ask: accept the defaults and the model licenses
  --skip-build           do not rebuild the plugins (if they are built)
  --skip-check           do not run the final check
  --start                start the memory server in the background and exit (chat apps: after a restart)
  --stop                 stop the memory server and the llama-servers it started, and exit
  --help                 this text`

// -- options ----------------------------------------------------------------------------------------
const argv = process.argv.slice(2)
const opts = {}
const VALUED = ['for', 'profile', 'hermes-home', 'models', 'llama-server', 'upstream']
const FLAGS = ['no-facts', 'yes', 'skip-build', 'skip-check', 'start', 'stop', 'help', 'accept-gemma-terms']
for (let i = 0; i < argv.length; i++) {
  const m = /^--([a-z-]+)(?:=(.*))?$/.exec(argv[i])
  if (!m || !(VALUED.includes(m[1]) || FLAGS.includes(m[1]))) { console.error(`unknown argument: ${argv[i]}\n\n${HELP}`); process.exit(2) }
  const [, key, inline] = m
  opts[key] = VALUED.includes(key) ? (inline ?? argv[++i]) : true
  if (VALUED.includes(key) && !opts[key]) { console.error(`--${key} needs a value`); process.exit(2) }
}
if (opts.help) { console.log(HELP); process.exit(0) }
const YES = Boolean(opts.yes)

const MODELS = {
  embedding: {
    title: 'EmbeddingGemma 300m (Q8_0 GGUF), for search',
    file: 'embeddinggemma-300M-Q8_0.gguf',
    url: 'https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf',
    sha256: 'b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63',
    size: 333_590_944,
    license: 'the Gemma Terms of Use (https://ai.google.dev/gemma/terms); downloading it means you accept them',
  },
  facts: {
    title: 'Gemma 4 E2B (QAT q4_0 GGUF), for pulling facts out of past conversations',
    file: 'gemma-4-E2B_q4_0-it.gguf',
    url: 'https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf',
    sha256: 'fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634',
    size: 3_349_516_256,
    license: 'Apache 2.0',
  },
}

class SetupError extends Error {}
const say = (...a) => console.log(...a)
const step = title => say(`\n== ${title}`)
const mb = n => `${Math.round(n / 1e6).toLocaleString()} MB`

// One readline per question, closed right after, so Ctrl+C works everywhere else (the build, downloads).
async function ask(question, fallback, what) {
  if (YES) return fallback
  if (!process.stdin.isTTY) throw new SetupError(`no terminal to ask ${what}. Pass the answer as an option (see --help), or --yes.`)
  const rl = createInterface({ input: process.stdin, output: process.stdout })
  rl.on('SIGINT', () => { rl.close(); process.stdout.write('\n'); process.exit(130) })
  try {
    return (await rl.question(`${question} `)).trim() || fallback
  } finally {
    rl.close()
  }
}
async function confirm(question, what, fallback = true) {
  return /^y/i.test(await ask(`${question} ${fallback ? '[Y/n]' : '[y/N]'}`, fallback ? 'y' : 'n', what))
}

// -- 1. Node and the build --------------------------------------------------------------------
function nodeOk() {
  const [major, minor] = process.versions.node.split('.').map(Number)
  return major >= 24 || (major === 22 && minor >= 19)
}

/** Runs a command quietly; its output is shown only when it fails. */
function run(command, args, cwd, label) {
  process.stdout.write(`  ${label}... `)
  const env = { ...process.env, npm_config_update_notifier: 'false' }
  // npx is a .cmd on Windows: run it through cmd.exe with a fixed command line (these arguments contain no spaces).
  const r = WIN
    ? spawnSync('cmd.exe', ['/d', '/s', '/c', [command, ...args].join(' ')], { cwd, env, encoding: 'utf8', maxBuffer: 64 << 20 })
    : spawnSync(command, args, { cwd, env, encoding: 'utf8', maxBuffer: 64 << 20 })
  if (r.status !== 0) {
    say('failed')
    throw new SetupError(`${command} ${args.join(' ')} failed in ${cwd}:\n${`${r.stdout ?? ''}${r.stderr ?? ''}${r.error ?? ''}`.slice(-3000)}`)
  }
  say('ok')
}

function build() {
  step('Building the plugins and the memory server')
  const pnpm = ['-y', 'pnpm@10.26.1']
  const dir = join(ROOT, 'dsh-plugins')
  run('npx', [...pnpm, 'install', '--frozen-lockfile', '--config.confirm-modules-purge=false'], dir, 'installing packages')
  run('npx', [...pnpm, 'run', 'build'], dir, 'building')
  run('npx', [...pnpm, '-r', '--filter', './packages/*', 'pack', '--pack-destination', 'dist'], dir, 'packing (for DeepSeek Harness)')
}

// -- 2. llama-server ------------------------------------------------------------------------------
function findOnPath(name) {
  const exts = WIN ? ['.exe'] : ['']
  const extra = WIN ? [] : ['/opt/homebrew/bin', '/usr/local/bin', '/home/linuxbrew/.linuxbrew/bin', join(homedir(), '.linuxbrew/bin')]
  const dirs = (process.env.PATH || '').split(delimiter).map(d => d.replace(/^"(.*)"$/, '$1')).concat(extra)
  for (const dir of dirs) for (const ext of exts) {
    const p = join(dir, name.endsWith(ext) ? name : name + ext)
    if (dir && existsSync(p)) return p
  }
  return null
}

function llamaServer() {
  step('Looking for llama.cpp (llama-server)')
  const given = opts['llama-server']
  const binary = given ? (existsSync(given) ? resolve(given) : findOnPath(given)) : findOnPath('llama-server')
  if (!binary) {
    const how = platform() === 'darwin' ? 'brew install llama.cpp'
      : WIN ? 'winget install llama.cpp   (or a Windows build from https://github.com/ggml-org/llama.cpp/releases)'
        : 'a Linux build from https://github.com/ggml-org/llama.cpp/releases (or brew install llama.cpp)'
    throw new SetupError(`llama-server was not found${given ? ` at ${given}` : ' on PATH'}. Install llama.cpp first:\n    ${how}\n`
      + '  then run this again (or pass --llama-server <path to llama-server>).')
  }
  const help = spawnSync(binary, ['--help'], { encoding: 'utf8', timeout: 30000 })
  const text = `${help.stdout ?? ''}${help.stderr ?? ''}`
  if (help.error || !/--embedding/.test(text)) {
    throw new SetupError(`${binary} --help did not work as expected (${help.error ? help.error.message : `exit ${help.status}`}):\n`
      + `${text.slice(0, 1500)}\n  Use a recent llama.cpp build (with --embedding), and pass its llama-server with --llama-server.`)
  }
  say(`  ${binary}`)
  // `--reasoning [on|off|auto]` only; older builds have just --reasoning-format / --reasoning-budget.
  return { binary, reasoningFlag: /--reasoning(?![-\w])/.test(text) }
}

// -- 3. models ------------------------------------------------------------------------------------
async function sha256(path) {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path)) hash.update(chunk)
  return hash.digest('hex')
}

const complete = (path, model) => existsSync(path) && statSync(path).size === model.size

async function download(model, dir) {
  const path = join(dir, model.file)
  if (complete(path, model)) {
    say(`  ${model.file}: already downloaded`)
    return path
  }
  const part = `${path}.part`
  let have = existsSync(part) ? statSync(part).size : 0
  if (have < model.size) {
    const controller = new AbortController()
    let stall = setTimeout(() => controller.abort(), 60_000)
    try {
      const response = await fetch(model.url, { headers: have ? { Range: `bytes=${have}-` } : {}, redirect: 'follow', signal: controller.signal })
      if (!(response.ok || response.status === 206)) throw new SetupError(`download failed: HTTP ${response.status} for ${model.url}`)
      if (have && response.status !== 206) have = 0  // the server sent the whole file again
      const out = createWriteStream(part, { flags: have ? 'a' : 'w' })
      let got = have, last = 0
      for await (const chunk of response.body) {
        clearTimeout(stall)
        stall = setTimeout(() => controller.abort(), 60_000)  // no data for a minute: give up (running setup again resumes)
        if (!out.write(chunk)) await new Promise(r => out.once('drain', r))
        got += chunk.length
        if (Date.now() - last > 1000) { process.stdout.write(`\r  ${model.file}: ${mb(got)} / ${mb(model.size)}   `); last = Date.now() }
      }
      await new Promise((r, j) => { out.once('error', j); out.end(r) })
    } catch (error) {
      if (error instanceof SetupError) throw error
      throw new SetupError(`downloading ${model.file} stopped (${error.cause?.code ?? error.message}). Run setup again: it continues where it stopped.`)
    } finally {
      clearTimeout(stall)
    }
  }
  process.stdout.write(`\r  ${model.file}: ${mb(model.size)}, checking...                \n`)
  const digest = await sha256(part)
  if (digest !== model.sha256) {
    rmSync(part)
    throw new SetupError(`${model.file} is not the expected file (sha256 ${digest}); run setup again to download it again`)
  }
  renameSync(part, path)
  return path
}

async function models(dir) {
  step(`Models (in ${dir})`)
  mkdirSync(dir, { recursive: true })
  const e = MODELS.embedding
  if (!complete(join(dir, e.file), e)) {
    say(`  ${e.title}: ${mb(e.size)}.\n  It is under ${e.license}.`)
    if (!opts['accept-gemma-terms'] && !(await confirm('  Download it?', 'whether to download EmbeddingGemma (Gemma Terms of Use)'))) {
      throw new SetupError('search needs this model, so setup stopped here. Run it again when you want to continue.')
    }
  }
  const embedding = await download(e, dir)
  let facts = null
  const f = MODELS.facts
  if (!opts['no-facts']) {
    if (complete(join(dir, f.file), f) || await confirm(`  ${f.title}: ${mb(f.size)}, ${f.license}.\n`
      + '  Recommended: with facts, recall holds up better as conversations pile up. Download it?', 'whether to download Gemma 4 E2B (or pass --no-facts)')) {
      facts = await download(f, dir)
    } else {
      say('  Without it, memory works on the messages alone (facts off). Run setup again later to add it.')
    }
  }
  return { embedding, facts }
}

// -- 4. the host --------------------------------------------------------------------------------
/** The memory's settings for the two llama-servers; their logs go to `logDir`. */
function managedSettings(llama, files, logDir) {
  const embedding = { llamacpp: { model: files.embedding, binary: llama.binary, log: join(logDir, 'llama-embedding.log') } }
  const chat = files.facts ? { llamacpp: { model: files.facts, binary: llama.binary, log: join(logDir, 'llama-chat.log'),
    args: ['-c', '8192', '-np', '1', ...(llama.reasoningFlag ? ['--reasoning', 'off'] : [])] } } : null
  return { embedding, chat }
}

function privateDir(dir) {
  mkdirSync(dir, { recursive: true, mode: 0o700 })
  if (!WIN) chmodSync(dir, 0o700)
}

function writeJson(path, value) {
  privateDir(dirname(path))
  writeFileSync(path, JSON.stringify(value, null, 2) + '\n', { mode: 0o600 })
  if (!WIN) chmodSync(path, 0o600)
}

/** Our settings merged into a memory config (created with a random token when missing). */
function configure(path, dataDir, port, settings) {
  const config = existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : { port, dataDir }
  config.token ||= randomBytes(24).toString('hex')
  config.embedding = { ...(config.embedding ?? {}), ...settings.embedding }
  delete config.embedding.engine
  if (settings.chat) config.chat = { ...(config.chat ?? {}), ...settings.chat }
  config.recall ??= { budgetTokens: 2048 }
  writeJson(path, config)
  return config
}

function hermesHome() {
  if (opts['hermes-home']) return resolve(opts['hermes-home'])
  if (process.env.HERMES_HOME) return resolve(process.env.HERMES_HOME)
  const local = WIN && process.env.LOCALAPPDATA ? join(process.env.LOCALAPPDATA, 'hermes') : null
  return local && existsSync(local) ? local : join(homedir(), '.hermes')
}

/** Hermes' own Python when it has one (no system Python needed), else python3 / python / py. */
function python(home) {
  const tools = join(home, 'tools')
  if (existsSync(tools)) {
    for (const d of readdirSync(tools).filter(n => n.startsWith('python-')).sort().reverse()) {
      for (const p of [join(tools, d, 'bin', 'python3'), join(tools, d, 'python.exe')]) if (existsSync(p)) return p
    }
  }
  for (const p of WIN ? ['py', 'python', 'python3'] : ['python3', 'python']) {
    if (spawnSync(p, ['--version'], { encoding: 'utf8' }).status === 0) return p
  }
  throw new SetupError('Python 3 was not found (the Hermes installer needs it)')
}

async function forHermes(llama, files) {
  step('Connecting to Hermes Agent')
  const home = hermesHome()
  const profile = opts.profile || await ask('  Hermes profile ("default" is the Hermes home itself) [default]:', 'default', 'the Hermes profile (--profile)')
  const profileDir = profile === 'default' ? home : join(home, 'profiles', profile)
  const configYaml = join(profileDir, 'config.yaml')
  if (!existsSync(configYaml)) {
    throw new SetupError(`${configYaml} was not found. Start Hermes once with this profile, or pass --hermes-home <dir>. `
      + '(For another app than Hermes, use --for app.)')
  }
  if (!YES) await ask('  Quit Hermes if it is running, then press Enter.', '', 'that Hermes is not running')
  const r = spawnSync(python(home), [join(ROOT, 'integrations/hermes/install.py'), '--home', home, '--no-chat', '--embedding', 'llamacpp', profile],
    { encoding: 'utf8' })
  // The installer's own notes about embeddings and facts do not apply: this script sets both below.
  for (const l of `${r.stdout ?? ''}`.split('\n')) if (l && !/^ {2}(embeddings|facts):|^Restart Hermes/.test(l)) say(`  ${l.trim()}`)
  if (r.status !== 0) throw new SetupError(`the Hermes installer stopped:\n${r.stderr || r.stdout}`)
  const path = join(profileDir, 'tsuzuki', 'config.json')
  const settings = managedSettings(llama, files, dirname(path))
  const config = configure(path, dirname(path), 8297, settings)
  say(`  memory settings: ${path} (port ${config.port})`)
  say('  search: EmbeddingGemma, started on demand by the memory')
  say(settings.chat ? '  facts: Gemma 4 E2B, started on demand on this machine (even if Hermes itself uses a cloud model)'
    : '  facts: off (run setup again to add the facts model)')
  return { config, configYaml }
}

async function forApp(llama, files, title = 'Preparing the memory server for your app') {
  step(title)
  const config = configure(CONFIG, join(HOME_DIR, 'memory'), 8297, managedSettings(llama, files, HOME_DIR))
  say(`  settings: ${CONFIG} (port ${config.port}; memory in ${config.dataDir})`)
  return { config }
}

async function answers(url) {
  try {
    return (await fetch(`${url}/models`, { signal: AbortSignal.timeout(1500) })).ok
  } catch {
    return false
  }
}

async function forChat(llama, files) {
  step('Your model server')
  let upstream = opts.upstream
  if (!upstream) {
    let found = null
    for (const [name, url] of UPSTREAMS) {
      if (await answers(url)) { found = url; say(`  found ${name} at ${url}`); break }
    }
    upstream = await ask(`\nYour model server's OpenAI-compatible URL [${found ?? UPSTREAMS[0][1]}]:`, found ?? UPSTREAMS[0][1],
      "your model server's URL (--upstream)")
  }
  upstream = upstream.replace(/\/+$/, '')
  let host = ''
  try { host = new URL(upstream).hostname } catch { throw new SetupError(`not a URL: ${upstream}`) }
  if (!['127.0.0.1', 'localhost', '[::1]'].includes(host)) {
    throw new SetupError(`${upstream} is not on this machine: recalled memory would be sent there. Use a model server on this `
      + 'machine (to allow another one anyway, set "proxy.allowRemoteUpstream": true in the config yourself).')
  }
  const { config } = await forApp(llama, files, 'Preparing the memory proxy for your chat app')
  config.proxy = { port: 8293, apiKey: '', ...(config.proxy ?? {}), enabled: true, upstream }
  writeJson(CONFIG, config)
  say(`  proxy: http://127.0.0.1:${config.proxy.port}/v1 in front of ${upstream}`)
  if (!await answers(upstream)) say(`  note: ${upstream} does not answer now; start your model server before you chat`)
  return { config }
}

async function health(config) {
  try {
    const r = await fetch(`http://127.0.0.1:${config.port}/health`, { headers: { authorization: `Bearer ${config.token}` },
      signal: AbortSignal.timeout(2000) })
    return r.ok ? await r.json() : null
  } catch {
    return null
  }
}

/** Starts the memory server of CONFIG in the background (left running), unless it is running already. */
async function startServer() {
  if (!existsSync(CONFIG) || !existsSync(SERVER_JS)) throw new SetupError(`${CONFIG} does not exist: run the setup first`)
  const config = JSON.parse(readFileSync(CONFIG, 'utf8'))
  if (await health(config)) return say(`  the memory server is running (port ${config.port})`)
  const log = openSync(join(HOME_DIR, 'sidecar.log'), 'a', 0o600)
  spawn(process.execPath, [SERVER_JS, '--config', CONFIG], { detached: true, stdio: ['ignore', log, log], windowsHide: true }).unref()
  for (let i = 0; i < 60; i++) {
    if (await health(config)) return say(`  ✓ the memory server is running in the background (port ${config.port})`)
    await new Promise(r => setTimeout(r, 1000))
  }
  throw new SetupError(`the memory server did not start; see ${join(HOME_DIR, 'sidecar.log')}`)
}

/** Stops the memory server of CONFIG (the one --start or an MCP client started). */
async function stopServer() {
  if (!existsSync(CONFIG)) return
  const h = await health(JSON.parse(readFileSync(CONFIG, 'utf8')))
  if (!h?.pid) return say('The memory server was not running.')
  try {
    process.kill(h.pid)
    say('Stopped the memory server.')
  } catch (error) {
    say(`Could not stop the memory server (pid ${h.pid}): ${error.message}`)
  }
}

// -- 5. check -------------------------------------------------------------------------------------
async function freePort() {
  const s = createServer()
  await new Promise(r => s.listen(0, '127.0.0.1', r))
  const { port } = s.address()
  await new Promise(r => s.close(r))
  return port
}

async function check(config, withFacts) {
  step('Checking (a throwaway memory; your own is not touched)')
  const dir = join(tmpdir(), `tsuzuki-check-${process.pid}`)
  rmSync(dir, { recursive: true, force: true })
  const port = await freePort()
  const token = randomBytes(16).toString('hex')
  writeJson(join(dir, 'config.json'), { ...config, port, token, dataDir: join(dir, 'data'), proxy: { ...config.proxy, enabled: false } })
  const child = spawn(process.execPath, [SERVER_JS, '--config', join(dir, 'config.json')], { stdio: ['ignore', 'pipe', 'pipe'] })
  const exited = new Promise(r => child.once('exit', r))
  let output = ''
  child.stdout.on('data', d => { output += d })
  child.stderr.on('data', d => { output += d })
  const call = async (path, body) => {
    const r = await fetch(`http://127.0.0.1:${port}${path}`, { method: body ? 'POST' : 'GET', signal: AbortSignal.timeout(30_000),
      headers: { authorization: `Bearer ${token}`, ...(body ? { 'content-type': 'application/json' } : {}) }, body: body ? JSON.stringify(body) : undefined })
    return r.json()
  }
  const waitFor = async (test, seconds, what) => {
    for (let i = 0; i < seconds; i++) {
      try { if (await test()) return } catch { /* not up yet */ }
      await new Promise(r => setTimeout(r, 1000))
    }
    throw new SetupError(`${what} did not work within ${seconds} s. The memory's output:\n${output.slice(-2000)}`)
  }
  try {
    await waitFor(async () => (await call('/health')).status === 'ok', 30, 'starting the memory')
    say('  starting the models (the first time can take a minute)...')
    await call('/v1/memory/messages', { session: 'setup-a', messages: [
      { role: 'user', text: 'My dog is called Pochi and I live in Sendai.' }, { role: 'assistant', text: 'Pochi in Sendai, noted.' }] })
    await waitFor(async () => (await call('/health')).stats.unindexed === 0, 300, 'storing a message (the search model)')
    const r = await call('/v1/memory/recall', { query: "What is my dog's name?", session: 'setup-b' })
    if (!/Pochi/.test(r.text)) throw new SetupError(`recall did not find the stored message:\n${JSON.stringify(r).slice(0, 600)}`)
    say('  ✓ a message from one conversation was recalled in another')
    if (withFacts) {
      await call('/v1/memory/flush', { session: 'setup-a' })
      await waitFor(async () => (await call('/health')).stats.facts > 0, 300, 'fact extraction (the facts model)')
      say('  ✓ facts are extracted')
    }
  } finally {
    child.kill()
    await Promise.race([exited, new Promise(r => setTimeout(r, 5000))])
    try { rmSync(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 500 }) } catch { /* a leftover temp folder is harmless */ }
  }
}

// -- stop ----------------------------------------------------------------------------------------------
function stopServers() {
  // Only llama-servers this memory started: they carry --alias tsuzuki-embeddinggemma / tsuzuki-chat.
  if (WIN) {
    const r = spawnSync('powershell.exe', ['-NoProfile', '-Command',
      "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '--alias tsuzuki-(embeddinggemma|chat)' } | "
      + 'ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }'], { encoding: 'utf8' })
    say(r.stdout.trim() ? 'Stopped the memory\'s llama-servers.' : 'No llama-server of the memory was running.')
  } else {
    const r = spawnSync('pkill', ['-f', '--', '--alias tsuzuki-(embeddinggemma|chat)'])
    say(r.status === 0 ? 'Stopped the memory\'s llama-servers.' : 'No llama-server of the memory was running.')
  }
}

// -- main ------------------------------------------------------------------------------------------
async function main() {
  if (opts.stop) {
    await stopServer()
    return stopServers()
  }
  if (opts.start) return startServer()
  say('tsuzuki setup  (Ctrl+C stops it at any time; running it again continues)')
  if (!nodeOk()) throw new SetupError(`Node ${process.versions.node} is not supported: install Node 22 (22.19 or later) or 24+ (https://nodejs.org)`)
  let target = opts.for?.toLowerCase()
  if (target && !['hermes', 'mcp', 'chat', 'app'].includes(target)) {
    throw new SetupError(`--for must be "hermes", "mcp", "chat" or "app", not "${opts.for}"`)
  }
  if (!opts['skip-build'] || !existsSync(SERVER_JS)) build()
  const llama = llamaServer()
  privateDir(HOME_DIR)
  const files = await models(resolve(opts.models || join(HOME_DIR, 'models')))
  if (!target) {
    const answer = await ask('\nWhere will you use the memory?\n'
      + '  1) Hermes Agent\n'
      + '  2) an MCP client: Claude Desktop, Claude Code, Cursor, LM Studio, Jan, Cline...\n'
      + '  3) chat apps with your own model server (llama.cpp, Ollama, LM Studio...): memory in every chat\n'
      + '  4) another app (HTTP API on 127.0.0.1)\nChoose [1]:', '1', 'where to use the memory (--for hermes, mcp, chat or app)')
    target = { 2: 'mcp', 3: 'chat', 4: 'app' }[answer.trim()[0]] ?? 'hermes'
  }
  const result = target === 'hermes' ? await forHermes(llama, files)
    : target === 'mcp' ? await forApp(llama, files, 'Preparing the memory for MCP clients')
      : target === 'chat' ? await forChat(llama, files) : await forApp(llama, files)
  if (!opts['skip-check']) await check(result.config, Boolean(files.facts))
  if (target === 'chat') {
    step('Starting the memory server')
    await startServer()
  }
  step('Done')
  const node = `"${process.execPath}"`
  if (target === 'hermes') {
    say('  Start Hermes. The memory and its models start by themselves when Hermes first needs them.')
    say('  Check it with:  hermes memory status')
    say(`  To let the agent delete memory when you ask, add  forget: true  under  memory: > tsuzuki:  in ${result.configYaml}`)
  } else if (target === 'mcp') {
    say('  Add the memory to your MCP client. Claude Code:')
    say(`    claude mcp add --scope user tsuzuki -- ${node} "${MCP_JS}"`)
    say('  Claude Desktop, Cursor, LM Studio, Jan, Cline... (in their MCP settings: claude_desktop_config.json, mcp.json):')
    say(JSON.stringify({ mcpServers: { tsuzuki: { command: process.execPath, args: [MCP_JS] } } }, null, 2).replace(/^/gm, '    '))
    say('  The client starts the memory when it needs it. The model calls recall and remember itself; to let it delete')
    say('  memory when you ask, add "--forget" to args.')
  } else if (target === 'chat') {
    say('  In your chat app, use this OpenAI-compatible connection instead of your model server:')
    say(`    URL:      http://127.0.0.1:${result.config.proxy.port}/v1`)
    say(`    API key:  ${result.config.token}`)
    say(`  Every chat is stored and earlier chats are recalled; ${result.config.proxy.upstream} still answers.`)
    say(`  After restarting the computer, start the memory again:  ${node} "${join(ROOT, 'scripts/setup.mjs')}" --start`)
  } else {
    say(`  Start the memory server (keep it running while your app uses it):\n    ${node} "${SERVER_JS}"`)
    say(`  Your app sends JSON to http://127.0.0.1:${result.config.port} with  Authorization: Bearer <token>  (the token is in ${join(HOME_DIR, 'config.json')}).`)
    say('  Endpoints: POST /v1/memory/messages, /v1/memory/recall, /v1/memory/search (see the README).')
  }
  say('  The models\' llama-servers keep running in the background so the next start is fast (about 0.5 GB of memory,')
  say(`  4 GB more with facts). Stop everything with:  ${node} "${join(ROOT, 'scripts/setup.mjs')}" --stop`)
}

main().catch(error => {
  console.error(`\n✗ ${error instanceof SetupError ? error.message : error?.stack ?? error}`)
  process.exit(1)
})
