#!/usr/bin/env node
// One-command setup (macOS, Linux, Windows; needs only Node and llama.cpp's llama-server):
//   1. asks where the memory is used (chat apps through a proxy in front of a local model server, MCP clients such as
//      Claude Desktop or Cursor, Hermes Agent, or any program over HTTP) and how much to set up (quick or full),
//   2. builds the plugins and the memory server,
//   3. downloads the models after asking (EmbeddingGemma 300m GGUF for search, Gemma 4 E2B GGUF for facts),
//   4. connects or prepares the memory for that use,
//   5. checks that storing, recalling and fact extraction work (in a throwaway store), and shows it working.
// The memory starts the two llama-servers itself when it needs them; you do not run them.
// Also: --report (a test report to paste), --start / --stop, --autostart, --update. Run with --help for the options.
// What it says is in setup-text.mjs (English and Japanese); the parts that can be tested on their own in setup-lib.mjs.
import { spawn, spawnSync } from 'node:child_process'
import { createHash, createHmac, randomBytes, timingSafeEqual } from 'node:crypto'
import { chmodSync, createReadStream, createWriteStream, existsSync, mkdirSync, mkdtempSync, openSync, readdirSync, readFileSync,
  realpathSync, renameSync, rmSync, statfsSync, statSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:net'
import { homedir, platform, tmpdir } from 'node:os'
import { delimiter, dirname, join, resolve } from 'node:path'
import { createInterface } from 'node:readline/promises'
import { fileURLToPath } from 'node:url'
import { ISSUE_URL, REPO_URL, autostartPlan, changesSince, formatReport, labeled, language, listDevices, machine, parseTarget, stableNode } from './setup-lib.mjs'
import { TEXT } from './setup-text.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const PKG = join(ROOT, 'dsh-plugins/packages/tsuzuki')
const SERVER_JS = join(PKG, 'lib/server.js')
const MCP_JS = join(PKG, 'lib/mcp.js')
const WIN = platform() === 'win32'
const HOME_DIR = join(homedir(), '.tsuzuki')
const CONFIG = join(HOME_DIR, 'config.json')
const SETUP = join(ROOT, 'scripts/setup.mjs')
const MEMORY = join(ROOT, 'scripts/memory.mjs')
/** Model servers looked for (in this order) when the proxy's upstream is not given. */
const UPSTREAMS = [['llama.cpp', 'http://127.0.0.1:8080/v1'], ['Ollama', 'http://127.0.0.1:11434/v1'], ['LM Studio', 'http://127.0.0.1:1234/v1']]
const version = () => JSON.parse(readFileSync(join(PKG, 'package.json'), 'utf8')).version
const VERSION = version()

// -- options ----------------------------------------------------------------------------------------
const argv = process.argv.slice(2)
const opts = {}
const VALUED = ['for', 'profile', 'hermes-home', 'models', 'llama-server', 'upstream', 'lang']
const FLAGS = ['quick', 'no-facts', 'yes', 'skip-build', 'skip-check', 'start', 'stop', 'update', 'report', 'version', 'help',
  'accept-gemma-terms']
const early = []  // problems found before the language is known
for (let i = 0; i < argv.length; i++) {
  const m = /^--([a-z-]+)(?:=(.*))?$/.exec(argv[i])
  if (m?.[1] === 'autostart') { opts.autostart = m[2] === 'off' ? 'off' : 'on'; continue }
  if (!m || !(VALUED.includes(m[1]) || FLAGS.includes(m[1]))) { early.push(['unknownArg', argv[i]]); continue }
  const [, key, inline] = m
  opts[key] = VALUED.includes(key) ? (inline ?? argv[++i]) : true
  if (VALUED.includes(key) && !opts[key]) early.push(['needsValue', key])
}
const T = TEXT[language(opts.lang)]
if (early.length) { for (const [k, v] of early) console.error(T[k](v)); process.exit(2) }
if (opts.help) { console.log(T.help(VERSION)); process.exit(0) }
if (opts.version) { console.log(`tsuzuki ${VERSION}`); process.exit(0) }
const YES = Boolean(opts.yes)
const QUICK = Boolean(opts.quick || opts['no-facts'])

/** How to write "node" in commands we print: plain `node` when that is this node (works in cmd and PowerShell). */
function nodeCommand() {
  const onPath = findOnPath('node')
  try {
    if (onPath && realpathSync(onPath) === realpathSync(process.execPath)) return 'node'
  } catch { /* fall through */ }
  return `"${process.execPath}"`
}

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
const seconds = since => (Date.now() - since) / 1000

/** What --report prints: filled in as setup goes, so a failed run still reports how far it got. */
const report = { version: VERSION, llama: '', setup: '', steps: [], devices: [], error: '', machine: null, logs: [] }
const passed = (what, since, ok = true) => report.steps.push({ what, ok, ...(since ? { seconds: seconds(since) } : {}) })

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

// -- 1. where, and how much ---------------------------------------------------------------------
async function chooseTarget() {
  if (opts.for) {
    const t = parseTarget(opts.for)
    if (!t.target || t.target !== String(opts.for).toLowerCase()) throw new SetupError(T.forInvalid(opts.for))
    return t.target
  }
  if (opts.report) return 'app'  // a test report needs no host
  if (YES) throw new SetupError(T.needFor)
  for (;;) {
    const answer = await ask(T.menu, '', T.whatTarget)
    const r = parseTarget(answer)
    if (r.target) return r.target
    say(r.error === 'number' ? T.menuNumber : T.menuUnknown(answer))
  }
}

const complete = (path, model) => existsSync(path) && statSync(path).size === model.size

/** Whether to set up facts too: --quick says no; otherwise the user picks, seeing what each costs. */
async function choosePlan(dir) {
  if (QUICK) return false
  const missing = m => complete(join(dir, m.file), m) ? 0 : m.size
  if (!missing(MODELS.facts)) return true  // nothing more to download
  step(T.planTitle)
  say(T.planQuick(missing(MODELS.embedding) || MODELS.embedding.size))
  say(T.planFull(missing(MODELS.embedding) + missing(MODELS.facts)))
  for (;;) {
    const a = (await ask(T.planQ, 'full', T.whatPlan)).toLowerCase()
    if ('full'.startsWith(a)) return true
    if ('quick'.startsWith(a)) return false
    say(T.planUnknown(a))
  }
}

// -- 2. Node and the build --------------------------------------------------------------------
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
    say('✗')
    throw new SetupError(T.cmdFailed(`${command} ${args.join(' ')}`, cwd, `${r.stdout ?? ''}${r.stderr ?? ''}${r.error ?? ''}`.slice(-3000)))
  }
  say('✓')
  return `${r.stdout ?? ''}`.trim()
}

function build() {
  step(T.buildStep)
  const started = Date.now()
  const pnpm = ['-y', 'pnpm@10.26.1']
  const dir = join(ROOT, 'dsh-plugins')
  run('npx', [...pnpm, 'install', '--frozen-lockfile', '--config.confirm-modules-purge=false'], dir, T.installing)
  run('npx', [...pnpm, 'run', 'build'], dir, T.building)
  run('npx', [...pnpm, '-r', '--filter', './packages/*', 'pack', '--pack-destination', 'dist'], dir, T.packing)
  passed('built the plugins and the memory server', started)
}

// -- 3. llama-server ------------------------------------------------------------------------------
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
  step(T.llamaStep)
  const given = opts['llama-server']
  const binary = given ? (existsSync(given) ? resolve(given) : findOnPath(given)) : findOnPath('llama-server')
  if (!binary) throw new SetupError(T.llamaMissing(T.llamaWhere(given), T.llamaHow[platform()] ?? T.llamaHow.linux))
  const help = spawnSync(binary, ['--help'], { encoding: 'utf8', timeout: 30000 })
  const text = `${help.stdout ?? ''}${help.stderr ?? ''}`
  if (help.error || !/--embedding/.test(text)) {
    throw new SetupError(T.llamaBad(binary, help.error ? help.error.message : `exit ${help.status}`, text.slice(0, 1500)))
  }
  const v = spawnSync(binary, ['--version'], { encoding: 'utf8', timeout: 30000 })
  report.llama = `${v.stdout ?? ''}${v.stderr ?? ''}`.split('\n').filter(l => /version|built with/i.test(l)).join(' · ').trim()
  const devices = spawnSync(binary, ['--list-devices'], { encoding: 'utf8', timeout: 30000 })
  report.devices = listDevices(`${devices.stdout ?? ''}${devices.stderr ?? ''}`)
  say(`  ${binary}`)
  passed('found llama-server')
  // `--reasoning [on|off|auto]` only; older builds have just --reasoning-format / --reasoning-budget.
  return { binary, reasoningFlag: /--reasoning(?![-\w])/.test(text) }
}

// -- 4. models ------------------------------------------------------------------------------------
async function sha256(path) {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path)) hash.update(chunk)
  return hash.digest('hex')
}

function freeBytes(dir) {
  try {
    const s = statfsSync(dir)
    return s.bavail * s.bsize
  } catch {
    return Infinity  // unknown: let the download find out
  }
}

async function download(model, dir) {
  const path = join(dir, model.file)
  if (complete(path, model)) {
    say(T.already(model.file))
    return path
  }
  const part = `${path}.part`
  let have = existsSync(part) ? statSync(part).size : 0
  if (have < model.size) {
    const need = model.size - have + 100e6
    const free = freeBytes(dir)
    if (free < need) throw new SetupError(T.diskFull(dir, need, free))
    const started = Date.now()
    const controller = new AbortController()
    let stall = setTimeout(() => controller.abort(), 60_000)
    try {
      const response = await fetch(model.url, { headers: have ? { Range: `bytes=${have}-` } : {}, redirect: 'follow', signal: controller.signal })
      if (!(response.ok || response.status === 206)) throw new SetupError(T.httpFailed(response.status, model.url))
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
      passed(`downloaded ${model.file}`, started)
    } catch (error) {
      if (error instanceof SetupError) throw error
      throw new SetupError(T.downloadStopped(model.file, error.cause?.code ?? error.message))
    } finally {
      clearTimeout(stall)
    }
  }
  process.stdout.write(`\r${T.checking(model.file, model.size)}                \n`)
  const digest = await sha256(part)
  if (digest !== model.sha256) {
    rmSync(part)
    throw new SetupError(T.badFile(model.file, digest))
  }
  renameSync(part, path)
  return path
}

async function models(dir, withFacts) {
  step(T.modelsStep(dir))
  mkdirSync(dir, { recursive: true })
  const e = MODELS.embedding
  if (!complete(join(dir, e.file), e)) {
    say(T.license(e.title, e.size, e.license))
    if (!opts['accept-gemma-terms'] && !(await confirm(T.downloadQ, T.whatLicense))) throw new SetupError(T.noEmbedding)
  }
  const embedding = await download(e, dir)
  let facts = null
  if (withFacts) facts = await download(MODELS.facts, dir)
  else say(T.quickNote)
  return { embedding, facts }
}

// -- 5. the host --------------------------------------------------------------------------------
/** The memory's settings for the two llama-servers; their logs go to `logDir`. */
function managedSettings(llama, files, logDir) {
  const embedding = { llamacpp: { model: files.embedding, binary: llama.binary, log: join(logDir, 'llama-embedding.log') } }
  const chat = files.facts ? { llamacpp: { model: files.facts, binary: llama.binary, log: join(logDir, 'llama-chat.log'),
    args: ['-c', '8192', '-np', '1', ...(llama.reasoningFlag ? ['--reasoning', 'off'] : [])] } } : null
  report.logs = [embedding.llamacpp.log, chat?.llamacpp.log].filter(Boolean)
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
  throw new SetupError(T.noPython)
}

async function forHermes(llama, files) {
  step(T.hermesStep)
  const home = hermesHome()
  const profile = opts.profile || await ask(T.profileQ, 'default', T.whatProfile)
  const profileDir = profile === 'default' ? home : join(home, 'profiles', profile)
  const configYaml = join(profileDir, 'config.yaml')
  if (!existsSync(configYaml)) throw new SetupError(T.noHermes(configYaml))
  if (!YES) await ask(T.quitHermes, '', T.whatQuit)
  const r = spawnSync(python(home), [join(ROOT, 'integrations/hermes/install.py'), '--home', home, '--no-chat', '--embedding', 'llamacpp', profile],
    { encoding: 'utf8' })
  // The installer's own notes about embeddings and facts do not apply: this script sets both below.
  for (const l of `${r.stdout ?? ''}`.split('\n')) if (l && !/^ {2}(embeddings|facts):|^Restart Hermes/.test(l)) say(`  ${l.trim()}`)
  if (r.status !== 0) throw new SetupError(T.installerStopped(r.stderr || r.stdout))
  const path = join(profileDir, 'tsuzuki', 'config.json')
  const settings = managedSettings(llama, files, dirname(path))
  const config = configure(path, dirname(path), 8297, settings)
  say(T.memorySettings(path, config.port))
  say(T.searchLine)
  say(settings.chat ? T.factsOn : T.factsOff)
  return { config, configYaml }
}

async function forApp(llama, files, title = T.appStep) {
  step(title)
  const config = configure(CONFIG, join(HOME_DIR, 'memory'), 8297, managedSettings(llama, files, HOME_DIR))
  say(T.settings(CONFIG, config.port, config.dataDir))
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
  step(T.upstreamStep)
  let upstream = opts.upstream
  if (!upstream) {
    let found = null
    for (const [name, url] of UPSTREAMS) {
      if (await answers(url)) { found = url; say(T.found(name, url)); break }
    }
    upstream = await ask(T.upstreamQ(found ?? UPSTREAMS[0][1]), found ?? UPSTREAMS[0][1], T.whatUpstream)
  }
  upstream = upstream.replace(/\/+$/, '')
  let host = ''
  try { host = new URL(upstream).hostname } catch { throw new SetupError(T.notUrl(upstream)) }
  if (!['127.0.0.1', 'localhost', '[::1]'].includes(host)) throw new SetupError(T.notLocal(upstream))
  const { config } = await forApp(llama, files, T.chatStep)
  config.proxy = { port: 8293, apiKey: '', ...(config.proxy ?? {}), enabled: true, upstream }
  writeJson(CONFIG, config)
  say(T.proxyLine(config.proxy.port, upstream))
  if (!await answers(upstream)) say(T.upstreamDown(upstream))
  return { config }
}

// -- the memory server: start, stop, at login -------------------------------------------------------
/**
 * Who answers on the memory's port: nothing, our memory server (it proves it knows the token, which is not sent before),
 * a memory server of 0.1 (it does not know the proof yet, and answers 401), or another program.
 */
async function listener(config) {
  const nonce = randomBytes(16).toString('hex')
  let r
  try {
    r = await fetch(`http://127.0.0.1:${config.port}/health?nonce=${nonce}`, { signal: AbortSignal.timeout(2000) })
  } catch {
    return 'none'
  }
  if (!config.token) return r.ok ? 'ours' : 'other'
  if (r.status === 401) return 'legacy'
  const proof = (await r.json().catch(() => ({}))).proof
  const expected = createHmac('sha256', config.token).update(`tsuzuki-health:${nonce}`).digest('hex')
  return typeof proof === 'string' && proof.length === expected.length && timingSafeEqual(Buffer.from(proof), Buffer.from(expected)) ? 'ours' : 'other'
}

/**
 * GET /health of our memory server, or null. `legacy`: also of a 0.1 server, which cannot prove itself; only --stop and
 * --update use that, once, to replace it.
 */
async function health(config, { legacy = false } = {}) {
  const who = await listener(config)
  if (who !== 'ours' && !(legacy && who === 'legacy')) return null
  try {
    const r = await fetch(`http://127.0.0.1:${config.port}/health`, { headers: { authorization: `Bearer ${config.token}` },
      signal: AbortSignal.timeout(2000) })
    return r.ok ? await r.json() : null
  } catch {
    return null
  }
}

const savedConfig = () => {
  if (!existsSync(CONFIG) || !existsSync(SERVER_JS)) throw new SetupError(T.noConfig(CONFIG))
  return JSON.parse(readFileSync(CONFIG, 'utf8'))
}

async function waitUp(config) {
  for (let i = 0; i < 60; i++) {
    if (await health(config)) return true
    await new Promise(r => setTimeout(r, 1000))
  }
  return false
}

/** Starts the memory server of CONFIG in the background (left running), unless it is running already. */
async function startServer() {
  const config = savedConfig()
  if (await health(config)) return say(T.running(config.port))
  if (await listener(config) !== 'none') throw new SetupError(T.portTaken(config.port, CONFIG))
  const log = openSync(join(HOME_DIR, 'sidecar.log'), 'a', 0o600)
  spawn(process.execPath, [SERVER_JS, '--config', CONFIG], { detached: true, stdio: ['ignore', log, log], windowsHide: true }).unref()
  if (!await waitUp(config)) throw new SetupError(T.didNotStart(join(HOME_DIR, 'sidecar.log')))
  say(T.runningBg(config.port))
}

/** Stops the memory server of CONFIG (the one --start, a login item or an MCP client started). */
async function stopServer(quiet = false) {
  if (!existsSync(CONFIG)) return
  const config = JSON.parse(readFileSync(CONFIG, 'utf8'))
  const h = await health(config, { legacy: true })
  if (!h?.pid) return quiet ? undefined : say(T.notRunning)
  try {
    process.kill(h.pid)
    for (let i = 0; i < 20 && await listener(config) !== 'none'; i++) await new Promise(r => setTimeout(r, 250))
    if (!quiet) say(T.stopped)
  } catch (error) {
    say(T.couldNotStop(h.pid, error.message))
  }
}

function stopServers() {
  // Only llama-servers this memory started: they carry --alias tsuzuki-embeddinggemma / tsuzuki-chat.
  if (WIN) {
    const r = spawnSync('powershell.exe', ['-NoProfile', '-Command',
      "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '--alias tsuzuki-(embeddinggemma|chat)' } | "
      + 'ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }'], { encoding: 'utf8' })
    say(r.stdout.trim() ? T.stoppedLlama : T.noLlama)
  } else {
    const r = spawnSync('pkill', ['-f', '--', '--alias tsuzuki-(embeddinggemma|chat)'])
    say(r.status === 0 ? T.stoppedLlama : T.noLlama)
  }
}

const loginPlan = () => autostartPlan({ node: stableNode(), serverJs: SERVER_JS, config: CONFIG, log: join(HOME_DIR, 'sidecar.log') })

/** Starts the memory server at login (`on`, and now) or no longer (`off`). */
async function autostart(mode) {
  const config = savedConfig()
  const plan = loginPlan()
  if (mode === 'off') {
    for (const [cmd, args] of plan.off) spawnSync(cmd, args, { encoding: 'utf8', windowsHide: true })
    rmSync(plan.path, { force: true })
    if (plan.kind === 'systemd') spawnSync('systemctl', ['--user', 'daemon-reload'], { encoding: 'utf8' })
    return say(T.autostartOff(plan.path))
  }
  // The login item starts the server itself; one started by hand would hold the port.
  await stopServer(true)
  mkdirSync(dirname(plan.path), { recursive: true })
  writeFileSync(plan.path, plan.encoding === 'utf16le' ? Buffer.concat([Buffer.from([0xff, 0xfe]), Buffer.from(plan.content, 'utf16le')])
    : plan.content, { mode: 0o644 })
  for (const [cmd, args, optional] of plan.on) {
    const r = spawnSync(cmd, args, { encoding: 'utf8', windowsHide: true })
    if (r.status !== 0 && !optional) throw new SetupError(T.autostartFailed(`${cmd} ${args.join(' ')}`, `${r.stderr || r.stdout || r.error}`.trim(), plan.path))
  }
  if (!await waitUp(config)) throw new SetupError(T.didNotStart(join(HOME_DIR, 'sidecar.log')))
  say(T.autostartOn(plan.kind, plan.path))
}

// -- 6. check, and show it working --------------------------------------------------------------------
async function freePort() {
  const s = createServer()
  await new Promise(r => s.listen(0, '127.0.0.1', r))
  const { port } = s.address()
  await new Promise(r => s.close(r))
  return port
}

async function check(config, withFacts) {
  step(T.checkStep)
  const dir = mkdtempSync(join(tmpdir(), 'tsuzuki-check-'))
  const port = await freePort()
  const token = randomBytes(16).toString('hex')
  // With a facts model, the throwaway memory also gets a proxy in front of it, for the demo at the end.
  const chatUrl = `http://127.0.0.1:${config.chat?.llamacpp?.port ?? 8292}/v1`
  const proxyPort = withFacts ? await freePort() : 0
  writeJson(join(dir, 'config.json'), { ...config, port, token, dataDir: join(dir, 'data'),
    proxy: withFacts ? { enabled: true, port: proxyPort, upstream: chatUrl, apiKey: '' } : { ...config.proxy, enabled: false } })
  const started = Date.now()
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
  const waitFor = async (test, limit, what, hint) => {
    for (let i = 0; i < limit; i++) {
      try { if (await test()) return } catch { /* not up yet */ }
      await new Promise(r => setTimeout(r, 1000))
    }
    throw new SetupError(T.timedOut(what, limit, output.slice(-2000), hint))
  }
  try {
    await waitFor(async () => (await call('/health')).status === 'ok', 30, T.whatStart, T.hintStart)
    passed('memory server started', started)
    say(T.startingModels)
    let t = Date.now()
    await call('/v1/memory/messages', { session: 'setup-a', messages: [
      { role: 'user', text: T.demoQuickFirst }, { role: 'assistant', text: 'OK.' }] })
    await waitFor(async () => (await call('/health')).stats.unindexed === 0, 300, T.whatIndex, T.hintIndex(report.logs[0] ?? ''))
    passed('search model answered (first message embedded)', t)
    t = Date.now()
    const r = await call('/v1/memory/recall', { query: T.demoQuickLater, session: 'setup-b' })
    if (!/Pochi|ポチ/.test(r.text)) throw new SetupError(T.recallMissing(JSON.stringify(r).slice(0, 600)))
    passed('a message from one conversation was recalled in another', t)
    say(T.recalledOk)
    if (withFacts) {
      t = Date.now()
      await call('/v1/memory/flush', { session: 'setup-a' })
      await waitFor(async () => (await call('/health')).stats.facts > 0, 300, T.whatFacts, T.hintFacts(report.logs[1] ?? ''))
      passed('facts were extracted', t)
      say(T.factsOk)
      await demo(`http://127.0.0.1:${proxyPort}/v1`, token, chatUrl, call)
    } else {
      step(T.demoStep)
      say(`  ${T.demoChat1}`)
      say(labeled(T.demoYou, T.demoQuickFirst))
      say(`  ${T.demoNew}`)
      say(labeled(T.demoYou, T.demoQuickLater))
      const line = r.text.split('\n').find(l => /\] user: /.test(l))
      if (line) say(`    ${T.demoRecalled(today())}\n${labeled('', `"${line.split('] user: ')[1]}"`, { indent: 5 })}`)
      say(T.demoQuickNote)
    }
  } finally {
    child.kill()
    await Promise.race([exited, new Promise(r => setTimeout(r, 5000))])
    try { rmSync(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 500 }) } catch { /* a leftover temp folder is harmless */ }
  }
}

const today = () => new Date().toISOString().slice(0, 10)

/** Chat 1, then a new chat through the proxy, then the same question without it: a real run on this machine. */
async function demo(proxy, token, chatUrl, call) {
  step(T.demoStep)
  try {
    const chat = async (url, auth, text) => {
      const r = await fetch(`${url}/chat/completions`, { method: 'POST', signal: AbortSignal.timeout(180_000),
        headers: { 'content-type': 'application/json', ...(auth ? { authorization: `Bearer ${auth}` } : {}) },
        body: JSON.stringify({ model: 'tsuzuki-chat', temperature: 0, seed: 0, max_tokens: 160,
          messages: [{ role: 'system', content: T.demoSystem }, { role: 'user', content: text }] }) })
      const value = await r.json()
      if (!r.ok) throw new Error(value?.error?.message ?? `HTTP ${r.status}`)
      return String(value.choices?.[0]?.message?.content ?? '').trim()
    }
    const first = await chat(proxy, token, T.demoFirst)
    say(`  ${T.demoChat1}`)
    say(labeled(T.demoYou, T.demoFirst))
    say(labeled(T.demoAi, first))
    for (let i = 0; i < 60 && (await call('/health')).stats.unindexed; i++) await new Promise(r => setTimeout(r, 250))
    const t = Date.now()
    const later = await chat(proxy, token, T.demoLater)
    const took = Date.now()
    const recalled = (await call('/v1/memory/recall', { query: T.demoLater, session: 'setup-demo' })).text?.split('\n')
      .find(l => l.includes(T.demoFirst.slice(0, 12)))
    say(`  ${T.demoNew}`)
    say(labeled(T.demoYou, T.demoLater))
    if (recalled) say(`    ${T.demoRecalled(today())}\n${labeled('', `"${recalled.replace(/^\[[^\]]+\] user: /, '')}"`, { indent: 5 })}`)
    say(labeled(T.demoAi, later))
    const without = await chat(chatUrl, '', T.demoLater)
    say(`  ${T.demoWithout}`)
    say(labeled(T.demoAi, without))
    report.steps.push({ what: 'demo: a new chat answered from memory', ok: /shamisen|三味線/i.test(later) && /sato|佐藤/i.test(later),
      seconds: (took - t) / 1000 })
  } catch (error) {
    say(T.demoSkipped(error.message))
    report.steps.push({ what: `demo did not run: ${error.message}`, ok: false })
  }
}

// -- --update -------------------------------------------------------------------------------------------
async function update() {
  step(T.updateStep(VERSION))
  const git = args => spawnSync('git', ['-C', ROOT, ...args], { encoding: 'utf8' })
  if (git(['rev-parse', '--is-inside-work-tree']).status !== 0) throw new SetupError(T.notGit(REPO_URL))
  const dirty = git(['status', '--porcelain', '--untracked-files=no']).stdout.trim()
  if (dirty) throw new SetupError(T.dirty(dirty))
  const before = git(['rev-parse', 'HEAD']).stdout.trim()
  process.stdout.write(`  ${T.pulling}... `)
  const pull = git(['pull', '--ff-only'])
  if (pull.status !== 0) { say('✗'); throw new SetupError(T.cmdFailed('git pull --ff-only', ROOT, `${pull.stdout}${pull.stderr}`.slice(-3000))) }
  say('✓')
  if (git(['rev-parse', 'HEAD']).stdout.trim() === before) return say(T.upToDate(VERSION))
  build()
  const now = version()
  if (existsSync(CONFIG)) {
    const wasRunning = await health(JSON.parse(readFileSync(CONFIG, 'utf8')), { legacy: true })
    if (existsSync(loginPlan().path)) { say(T.restarting); await autostart('on') }
    else if (wasRunning) { say(T.restarting); await stopServer(true); await startServer() }
  }
  say(`\n${T.updated(VERSION, now)}`)
  const log = join(ROOT, 'CHANGELOG.md')
  const changes = existsSync(log) ? changesSince(readFileSync(log, 'utf8'), VERSION) : []
  if (changes.length) say(`\n${T.whatsNew}\n\n${changes.join('\n\n')}`)
  say(`\n${T.restartHosts}`)
}

// -- the report -----------------------------------------------------------------------------------------
function printReport() {
  report.machine = machine()
  const commit = spawnSync('git', ['-C', ROOT, 'rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).stdout?.trim()
  if (commit) report.version = `${VERSION} (${commit})`
  const text = formatReport(report)
  say(`\n----- 8< -----\n${text}\n----- 8< -----`)
  try {
    privateDir(HOME_DIR)
    writeFileSync(join(HOME_DIR, 'report.md'), text + '\n', { mode: 0o600 })
    say(T.reportSaved(join(HOME_DIR, 'report.md')))
  } catch { /* the printed copy is enough */ }
  say(T.reportSend(ISSUE_URL))
}

// -- main ------------------------------------------------------------------------------------------
async function main() {
  if (opts.stop) {
    await stopServer()
    return stopServers()
  }
  if (opts.start) return startServer()
  if (opts.autostart && !opts.for) return autostart(opts.autostart)
  if (opts.update) return update()
  say(T.banner(VERSION))
  if (!nodeOk()) throw new SetupError(T.nodeOld(process.versions.node, T.nodeHow[platform()] ?? T.nodeHow.linux))
  const target = await chooseTarget()
  const modelDir = resolve(opts.models || join(HOME_DIR, 'models'))
  const full = await choosePlan(modelDir)
  report.setup = `--for ${target}, ${full ? 'full' : 'quick'}`
  if (!opts['skip-build'] || !existsSync(SERVER_JS)) build()
  const llama = llamaServer()
  privateDir(HOME_DIR)
  const files = await models(modelDir, full)
  const result = target === 'hermes' ? await forHermes(llama, files)
    : target === 'mcp' ? await forApp(llama, files, T.mcpStep)
      : target === 'chat' ? await forChat(llama, files) : await forApp(llama, files)
  if (!opts['skip-check']) await check(result.config, Boolean(files.facts))
  let atLogin = false
  if (target === 'chat') {
    step(T.startStep)
    atLogin = opts.autostart === 'on' || (!opts.autostart && !YES && await confirm(T.autostartQ, T.whatAutostart))
    if (atLogin) await autostart('on')
    else await startServer()
  }
  step(T.doneStep)
  const node = nodeCommand()
  const setup = `${node} "${SETUP}"`
  if (target === 'hermes') say(T.doneHermes(result.configYaml))
  else if (target === 'mcp') {
    say(T.doneMcp(`claude mcp add --scope user tsuzuki -- "${stableNode()}" "${MCP_JS}"`,
      JSON.stringify({ mcpServers: { tsuzuki: { command: stableNode(), args: [MCP_JS] } } }, null, 2).replace(/^/gm, '    ')))
  } else if (target === 'chat') {
    say(T.doneChat(result.config.proxy.port, result.config.token, result.config.proxy.upstream))
    if (!atLogin) say(T.doneChatStart(setup))
  } else {
    say(T.doneApp(setup, result.config.port, CONFIG))
    say(T.autostartLater(setup))
  }
  if (target !== 'hermes') say(T.doneMemory(`${node} "${MEMORY}"`))
  say(T.doneStop(setup))
  if (opts.report) printReport()
}

main().catch(error => {
  const message = error instanceof SetupError ? error.message : String(error?.stack ?? error)
  console.error(`\n✗ ${message}`)
  if (opts.report) {
    report.error = message
    printReport()
  } else {
    console.error(T.stuck(ISSUE_URL))
  }
  process.exit(1)
})
