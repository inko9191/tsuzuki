#!/usr/bin/env node
// One-command setup (macOS, Linux, Windows; needs only Node and llama.cpp's llama-server):
//   1. builds the plugins and the sidecar,
//   2. downloads the models after asking (EmbeddingGemma 300m GGUF for search, Gemma 4 E2B GGUF for facts),
//   3. connects the memory to Hermes Agent, or prepares the HTTP sidecar for any other app,
//   4. checks that storing, recalling and fact extraction work (in a throwaway store).
// The sidecar starts the two llama-servers itself when it needs them; you do not run them.
//
//   node scripts/setup.mjs                      # asks what to do
//   node scripts/setup.mjs --for hermes [--profile default] [--hermes-home DIR]
//   node scripts/setup.mjs --for app            # the HTTP sidecar (~/.tsuzuki/config.json)
//   options: --yes (accept the defaults and the model licenses), --no-facts, --models DIR, --llama-server PATH,
//            --skip-build, --skip-check
import { spawn, spawnSync } from 'node:child_process'
import { createHash, randomBytes } from 'node:crypto'
import { chmodSync, createReadStream, createWriteStream, existsSync, mkdirSync, readFileSync, renameSync, rmSync, statSync,
  writeFileSync } from 'node:fs'
import { homedir, platform, tmpdir } from 'node:os'
import { delimiter, dirname, join, resolve } from 'node:path'
import { createInterface } from 'node:readline/promises'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const PKG = join(ROOT, 'dsh-plugins/packages/tsuzuki')
const SERVER_JS = join(PKG, 'lib/server.js')
const WIN = platform() === 'win32'
const args = process.argv.slice(2)
const flag = name => args.includes(`--${name}`)
const option = (name, fallback = '') => { const i = args.indexOf(`--${name}`); return i >= 0 && args[i + 1] ? args[i + 1] : fallback }
const YES = flag('yes')

const MODELS = {
  embedding: {
    title: 'EmbeddingGemma 300m (Q8_0 GGUF), for search',
    file: 'embeddinggemma-300M-Q8_0.gguf',
    url: 'https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf',
    sha256: 'b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63',
    size: 333_590_944,
    license: 'the Gemma Terms of Use (https://ai.google.dev/gemma/terms). Downloading it means you accept them.',
  },
  facts: {
    title: 'Gemma 4 E2B (QAT q4_0 GGUF), for pulling facts out of past conversations',
    file: 'gemma-4-E2B_q4_0-it.gguf',
    url: 'https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf',
    sha256: 'fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634',
    size: 3_349_516_256,
    license: 'Apache 2.0.',
  },
}

const say = (...a) => console.log(...a)
const step = title => say(`\n== ${title}`)
const fail = message => { console.error(`\n✗ ${message}`); process.exit(1) }
const mb = n => `${Math.round(n / 1e6).toLocaleString()} MB`

const rl = process.stdin.isTTY ? createInterface({ input: process.stdin, output: process.stdout }) : null
async function ask(question, fallback) {
  if (YES || !rl) return fallback
  const answer = (await rl.question(`${question} `)).trim()
  return answer || fallback
}
async function confirm(question, fallback = true) {
  const answer = await ask(`${question} ${fallback ? '[Y/n]' : '[y/N]'}`, fallback ? 'y' : 'n')
  return /^y/i.test(answer)
}

// -- 1. Node and the build --------------------------------------------------------------------
function nodeOk() {
  const [major, minor] = process.versions.node.split('.').map(Number)
  return major >= 24 || (major === 22 && minor >= 19)
}

function run(command, commandArgs, cwd) {
  const r = spawnSync(command, commandArgs, { cwd, stdio: 'inherit', shell: WIN })
  if (r.status !== 0) fail(`${command} ${commandArgs.join(' ')} failed (in ${cwd})`)
}

function build() {
  step('Building the plugins and the sidecar')
  const pnpm = ['-y', 'pnpm@10.26.1']
  const dir = join(ROOT, 'dsh-plugins')
  run('npx', [...pnpm, 'install', '--frozen-lockfile'], dir)
  run('npx', [...pnpm, 'run', 'build'], dir)
  run('npx', [...pnpm, '-r', '--filter', './packages/*', 'pack', '--pack-destination', 'dist'], dir)
}

// -- 2. llama-server ------------------------------------------------------------------------------
function findOnPath(name) {
  const exts = WIN ? (process.env.PATHEXT || '.EXE;.CMD;.BAT').split(';') : ['']
  const dirs = (process.env.PATH || '').split(delimiter).concat(WIN ? [] : ['/opt/homebrew/bin', '/usr/local/bin'])
  for (const dir of dirs) for (const ext of exts) {
    const p = join(dir, name + ext)
    if (dir && existsSync(p)) return p
  }
  return null
}

function llamaServer() {
  step('Looking for llama.cpp (llama-server)')
  const given = option('llama-server')
  const binary = given || findOnPath('llama-server')
  if (!binary || !existsSync(binary)) {
    const how = platform() === 'darwin' ? 'brew install llama.cpp'
      : WIN ? 'winget install llama.cpp   (or download a build from https://github.com/ggml-org/llama.cpp/releases)'
        : 'download a build from https://github.com/ggml-org/llama.cpp/releases, or your package manager (e.g. brew install llama.cpp)'
    fail(`llama-server was not found. Install llama.cpp first:\n    ${how}\n  then run this again (or pass --llama-server <path>).`)
  }
  const help = spawnSync(binary, ['--help'], { encoding: 'utf8', timeout: 30000 })
  const text = `${help.stdout}${help.stderr}`
  if (!/--embedding/.test(text)) fail(`${binary} does not look like a recent llama-server (no --embedding option); update llama.cpp`)
  say(`  ${binary}`)
  return { binary, reasoningFlag: /--reasoning\b/.test(text) }
}

// -- 3. models ------------------------------------------------------------------------------------
async function sha256(path) {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path)) hash.update(chunk)
  return hash.digest('hex')
}

async function download(model, dir) {
  const path = join(dir, model.file)
  if (existsSync(path)) {
    say(`  ${model.file}: already downloaded`)
    return path
  }
  const part = `${path}.part`
  let have = existsSync(part) ? statSync(part).size : 0
  const response = await fetch(model.url, { headers: have ? { Range: `bytes=${have}-` } : {}, redirect: 'follow' })
  if (!(response.ok || response.status === 206)) fail(`download failed: HTTP ${response.status} for ${model.url}`)
  if (have && response.status !== 206) have = 0  // the server sent the whole file again
  const out = createWriteStream(part, { flags: have ? 'a' : 'w', mode: 0o644 })
  let got = have, last = 0
  for await (const chunk of response.body) {
    if (!out.write(chunk)) await new Promise(r => out.once('drain', r))
    got += chunk.length
    if (Date.now() - last > 1000) { process.stdout.write(`\r  ${model.file}: ${mb(got)} / ${mb(model.size)}   `); last = Date.now() }
  }
  await new Promise(r => out.end(r))
  process.stdout.write(`\r  ${model.file}: ${mb(got)} / ${mb(model.size)}, checking...   \n`)
  const digest = await sha256(part)
  if (digest !== model.sha256) {
    rmSync(part)
    fail(`${model.file} is not the expected file (sha256 ${digest}); run setup again`)
  }
  renameSync(part, path)
  return path
}

async function models(dir) {
  step(`Models (in ${dir})`)
  mkdirSync(dir, { recursive: true })
  const e = MODELS.embedding
  if (!existsSync(join(dir, e.file))) {
    say(`  ${e.title}: ${mb(e.size)}.\n  It is under ${e.license}`)
    if (!flag('accept-gemma-terms') && !(await confirm('  Download it?', true))) fail('search needs this model; nothing was changed')
  }
  const embedding = await download(e, dir)
  let facts = null
  const f = MODELS.facts
  if (!flag('no-facts')) {
    const present = existsSync(join(dir, f.file))
    if (present || await confirm(`  ${f.title}: ${mb(f.size)}, ${f.license}\n  Recommended: with facts, recall holds up better as conversations pile up. Download it?`, true)) {
      facts = await download(f, dir)
    } else {
      say('  Without it, memory works on the messages alone (facts off). Run setup again later to add it.')
    }
  }
  return { embedding, facts }
}

// -- 4. the host --------------------------------------------------------------------------------
function managedSettings(llama, files) {
  const embedding = { llamacpp: { model: files.embedding, binary: llama.binary } }
  const chat = files.facts ? { llamacpp: { model: files.facts, binary: llama.binary,
    args: ['-c', '8192', '-np', '1', ...(llama.reasoningFlag ? ['--reasoning', 'off'] : [])] } } : null
  return { embedding, chat }
}

function writeJson(path, value) {
  mkdirSync(dirname(path), { recursive: true, mode: 0o700 })
  writeFileSync(path, JSON.stringify(value, null, 2) + '\n', { mode: 0o600 })
  chmodSync(path, 0o600)
}

/** Our settings merged into a sidecar config (created with a random token when missing). */
function configure(path, dataDir, port, settings) {
  const config = existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : { port, dataDir, token: randomBytes(24).toString('hex') }
  config.token ||= randomBytes(24).toString('hex')
  config.embedding = { ...(config.embedding ?? {}), ...settings.embedding }
  delete config.embedding.engine
  if (settings.chat) config.chat = { ...(config.chat ?? {}), ...settings.chat }
  config.recall ??= { budgetTokens: 2048 }
  writeJson(path, config)
  return config
}

function python() {
  for (const p of WIN ? ['py', 'python', 'python3'] : ['python3', 'python']) {
    const r = spawnSync(p, ['--version'], { encoding: 'utf8', shell: WIN })
    if (r.status === 0) return p
  }
  fail('Python 3 was not found (Hermes needs it too)')
}

function hermesHome() {
  const given = option('hermes-home')
  if (given) return resolve(given)
  const local = WIN && process.env.LOCALAPPDATA ? join(process.env.LOCALAPPDATA, 'hermes') : null
  return local && existsSync(local) ? local : join(homedir(), '.hermes')
}

async function forHermes(settings) {
  step('Connecting to Hermes Agent')
  const home = hermesHome()
  const profile = option('profile') || await ask('  Hermes profile ("default" is the Hermes home itself) [default]:', 'default')
  say('  Quit Hermes now if it is running (it reads its config only when it starts).')
  const profileDir = profile === 'default' ? home : join(home, 'profiles', profile)
  if (!existsSync(join(profileDir, 'config.yaml'))) fail(`${join(profileDir, 'config.yaml')} not found: start Hermes once with this profile, or pass --hermes-home`)
  const r = spawnSync(python(), [join(ROOT, 'integrations/hermes/install.py'), '--home', home, '--no-chat', '--embedding', 'llamacpp', profile],
    { stdio: 'inherit', shell: WIN })
  if (r.status !== 0) fail('the Hermes installer stopped (see above); nothing else was changed')
  const path = join(profileDir, 'tsuzuki', 'config.json')
  const config = configure(path, dirname(path), 8297, settings)
  say(`  memory settings: ${path} (port ${config.port})`)
  say(settings.chat ? '  facts: on, with Gemma 4 E2B started on demand on this machine (this replaces "facts: off" above)'
    : '  facts: off (run setup again to add the facts model)')
  return config
}

async function forApp(settings) {
  step('Preparing the HTTP sidecar')
  const path = join(homedir(), '.tsuzuki', 'config.json')
  const config = configure(path, dirname(path), 8297, settings)
  say(`  settings: ${path}`)
  return config
}

// -- 5. check -------------------------------------------------------------------------------------
async function check(config, withFacts) {
  step('Checking (a throwaway memory; your own is not touched)')
  const dir = join(tmpdir(), `tsuzuki-check-${process.pid}`)
  rmSync(dir, { recursive: true, force: true })
  const port = 18000 + Math.floor(Math.random() * 1000)
  const token = randomBytes(16).toString('hex')
  writeJson(join(dir, 'config.json'), { ...config, port, token, dataDir: join(dir, 'data'), facts: { intervalSeconds: 2 } })
  const child = spawn(process.execPath, [SERVER_JS, '--config', join(dir, 'config.json')], { stdio: ['ignore', 'pipe', 'pipe'] })
  let output = ''
  child.stdout.on('data', d => { output += d })
  child.stderr.on('data', d => { output += d })
  const call = async (path, body) => {
    const r = await fetch(`http://127.0.0.1:${port}${path}`, { method: body ? 'POST' : 'GET',
      headers: { authorization: `Bearer ${token}`, ...(body ? { 'content-type': 'application/json' } : {}) }, body: body ? JSON.stringify(body) : undefined })
    return r.json()
  }
  const waitFor = async (test, seconds, what) => {
    for (let i = 0; i < seconds; i++) {
      try { if (await test()) return } catch { /* not up yet */ }
      await new Promise(r => setTimeout(r, 1000))
    }
    child.kill()
    fail(`${what} did not work within ${seconds} s. Sidecar output:\n${output.slice(-2000)}`)
  }
  try {
    await waitFor(async () => (await call('/health')).status === 'ok', 30, 'starting the sidecar')
    say('  starting the models (the first time can take a minute)...')
    await call('/v1/memory/messages', { session: 'setup-a', messages: [
      { role: 'user', text: 'My dog is called Pochi and I live in Sendai.' }, { role: 'assistant', text: 'Pochi in Sendai, noted.' }] })
    await waitFor(async () => (await call('/health')).stats.unindexed === 0, 300, 'storing a message (embedding model)')
    const r = await call('/v1/memory/recall', { query: "What is my dog's name?", session: 'setup-b' })
    if (!/Pochi/.test(r.text)) fail(`recall did not find the stored message:\n${JSON.stringify(r).slice(0, 600)}`)
    say('  ✓ a message from one conversation was recalled in another')
    if (withFacts) {
      await call('/v1/memory/flush', { session: 'setup-a' })
      await waitFor(async () => (await call('/health')).stats.facts > 0, 300, 'fact extraction (chat model)')
      say('  ✓ facts are extracted')
    }
  } finally {
    child.kill()
    rmSync(dir, { recursive: true, force: true })
  }
}

// -- main ------------------------------------------------------------------------------------------
say('tsuzuki setup')
if (!nodeOk()) fail(`Node ${process.versions.node} is too old: install Node 22.19 or later (https://nodejs.org)`)
if (!flag('skip-build') || !existsSync(SERVER_JS)) build()
const llama = llamaServer()
const files = await models(resolve(option('models', join(homedir(), '.tsuzuki', 'models'))))
const settings = managedSettings(llama, files)
let target = option('for')
if (!target) {
  const answer = await ask('\nWhere will you use the memory?\n  1) Hermes Agent\n  2) another app (HTTP sidecar on 127.0.0.1)\nChoose [1]:', '1')
  target = answer.startsWith('2') ? 'app' : 'hermes'
}
const config = target === 'hermes' ? await forHermes(settings) : await forApp(settings)
if (!flag('skip-check')) await check(config, Boolean(files.facts))
rl?.close()
step('Done')
if (target === 'hermes') {
  say('  Start Hermes. The memory sidecar and the models start by themselves when Hermes first needs them.')
  say('  To let the agent delete memory on request, add  forget: true  under memory: tsuzuki: in config.yaml.')
} else {
  say(`  Start the sidecar:   node "${SERVER_JS}"`)
  say(`  Then send JSON with  Authorization: Bearer <token>  (the token is in ${join(homedir(), '.tsuzuki', 'config.json')}).`)
  say('  Endpoints: POST /v1/memory/messages, /v1/memory/recall, /v1/memory/search (see the README).')
}
say('  The llama-servers keep running in the background after first use; they are reused next time.')
