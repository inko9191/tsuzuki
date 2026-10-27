// Parts of scripts/setup.mjs that can be tested on their own: its text in English and Japanese, reading the answer to
// "where will you use the memory?", the login-start files for macOS, Linux and Windows, the test report, and the
// CHANGELOG sections shown by --update.
import { spawnSync } from 'node:child_process'
import { existsSync, readFileSync, realpathSync } from 'node:fs'
import { arch, cpus, homedir, platform, release, totalmem, userInfo, version as osVersion } from 'node:os'
import { delimiter, join, posix, win32 } from 'node:path'
import { pathToFileURL } from 'node:url'

export const REPO_URL = 'https://github.com/inko9191/tsuzuki'
export const ISSUE_URL = `${REPO_URL}/issues/new?template=test-report.yml`
export const TARGETS = ['chat', 'mcp', 'hermes', 'app']

/** Japanese when asked (`--lang ja`) or when the system's language is Japanese; else English. */
export function language(asked, env = process.env) {
  if (asked === 'ja' || asked === 'en') return asked
  const system = env.LC_ALL || env.LC_MESSAGES || env.LANG || Intl.DateTimeFormat().resolvedOptions().locale || ''
  return /^ja/i.test(system) ? 'ja' : 'en'
}

/**
 * The route for an answer to the menu: a word or its first letters ("c", "mc"). Numbers are refused on purpose: the
 * menu of 0.1 was numbered, and its numbers mean other things now.
 */
export function parseTarget(answer) {
  const a = String(answer ?? '').trim().toLowerCase()
  if (!a) return { error: 'empty' }
  if (/^\d+$/.test(a)) return { error: 'number' }
  const hits = TARGETS.filter(t => t.startsWith(a))
  return hits.length === 1 ? { target: hits[0] } : { error: 'unknown' }
}

// -- login start ----------------------------------------------------------------------------------------------

const xml = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&apos;')
/** One argument of a systemd ExecStart line: quoted, with \, ", % (specifiers) and $ (variables) escaped. */
export const systemdArg = s => `"${String(s).replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/%/g, '%%').replace(/\$/g, '$$$$')}"`
/** A VBScript string literal. */
const vbs = s => `"${String(s).replace(/"/g, '""')}"`

export const LAUNCH_LABEL = 'io.github.inko9191.tsuzuki'

/**
 * What starts the memory server at login on this OS: the file to write (and its text), and the commands that turn it
 * on and off. The server logs with its own --log, so no shell redirection is needed anywhere.
 */
export function autostartPlan({ os = platform(), node, serverJs, config, log, home = homedir(), appData = process.env.APPDATA, uid = process.getuid?.() }) {
  const args = [node, serverJs, '--config', config, '--log', log]
  // The target OS's own path rules (the plan is also built, and tested, on another OS).
  if (os === 'darwin') {
    const path = posix.join(home, 'Library', 'LaunchAgents', `${LAUNCH_LABEL}.plist`)
    const content = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${LAUNCH_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
${args.map(a => `    <string>${xml(a)}</string>`).join('\n')}
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ProcessType</key><string>Background</string>
</dict>
</plist>
`
    const domain = `gui/${uid}`
    return { kind: 'launchd', path, content,
      on: [['launchctl', ['bootout', `${domain}/${LAUNCH_LABEL}`], true], ['launchctl', ['bootstrap', domain, path]]],
      off: [['launchctl', ['bootout', `${domain}/${LAUNCH_LABEL}`], true]] }
  }
  if (os === 'win32') {
    const path = win32.join(appData ?? win32.join(home, 'AppData', 'Roaming'), 'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup', 'tsuzuki.vbs')
    const line = args.map(a => `"${a}"`).join(' ')
    const content = `' Starts the tsuzuki memory server at login, without a window (written by scripts/setup.mjs --autostart).\r\n`
      + `CreateObject("WScript.Shell").Run ${vbs(line)}, 0, False\r\n`
    // UTF-16 with a byte order mark: Windows Script Host reads other files in the ANSI code page, which breaks
    // non-ASCII folder names (C:\Users\<Japanese name>).
    return { kind: 'startup', path, content, encoding: 'utf16le', on: [['wscript.exe', [path]]], off: [] }
  }
  const path = posix.join(home, '.config', 'systemd', 'user', 'tsuzuki.service')
  const content = `[Unit]
Description=tsuzuki memory server
After=network.target

[Service]
ExecStart=${args.map(systemdArg).join(' ')}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
`
  return { kind: 'systemd', path, content,
    on: [['systemctl', ['--user', 'daemon-reload']], ['systemctl', ['--user', 'enable', '--now', 'tsuzuki.service']]],
    off: [['systemctl', ['--user', 'disable', '--now', 'tsuzuki.service'], true]] }
}

// -- the test report ------------------------------------------------------------------------------------------

const escapeRegExp = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/**
 * `text` with the home folder and the user name taken out (the report is meant to be posted publicly): the folder in
 * every spelling a path or a stack trace may use (\ or /, as a file:// URL, percent-encoded), and the name in any
 * case. A short ASCII name ("al") is left alone where it is not part of a path, or every "al" would go.
 */
export function sanitize(text, home = homedir(), user = safeUser()) {
  let out = String(text)
  const forms = new Set([home, home.replace(/\\/g, '/'), pathToFileURL(home).href, pathToFileURL(home).pathname, decodeURI(pathToFileURL(home).pathname)])
  for (const form of [...forms].filter(f => f && f.length > 1).sort((a, b) => b.length - a.length)) {
    out = out.replace(new RegExp(escapeRegExp(form), 'gi'), '~')
  }
  if (user) {
    const names = [user, encodeURIComponent(user)]
    for (const name of new Set(names)) {
      const free = name.length >= 3 || /[^\x00-\x7f]/.test(user)
      const pattern = free ? escapeRegExp(name) : `(?<=[\\/])${escapeRegExp(name)}(?=[\\/])`
      out = out.replace(new RegExp(pattern, 'gi'), '<user>')
    }
  }
  return out
}

/**
 * The node to put into files that outlive this run (login items, MCP settings): the one on PATH when it is this same
 * node, because process.execPath is resolved (Homebrew's /opt/homebrew/Cellar/node/<version>/bin/node disappears with
 * the next `brew upgrade node`), else process.execPath.
 */
export function stableNode(execPath = process.execPath, path = process.env.PATH ?? '', exts = platform() === 'win32' ? ['.exe'] : ['']) {
  let real
  try { real = realpathSync(execPath) } catch { return execPath }
  for (const dir of path.split(delimiter).map(d => d.replace(/^"(.*)"$/, '$1')).filter(Boolean)) {
    for (const ext of exts) {
      const candidate = join(dir, `node${ext}`)
      try { if (realpathSync(candidate) === real) return candidate } catch { /* not there */ }
    }
  }
  return execPath
}

function safeUser() {
  try { return userInfo().username } catch { return '' }
}

const first = (cmd, args, timeout = 10_000) => {
  try {
    const r = spawnSync(cmd, args, { encoding: 'utf8', timeout, windowsHide: true })
    return r.status === 0 ? `${r.stdout ?? ''}`.trim() : ''
  } catch {
    return ''
  }
}

/** The machine, for the report: OS, CPU, memory, GPU (best effort), Node. */
export function machine() {
  const os = platform()
  let name = `${os} ${release()}`
  if (os === 'darwin') name = `macOS ${first('sw_vers', ['-productVersion']) || release()}`
  else if (os === 'linux' && existsSync('/etc/os-release')) {
    const m = /^PRETTY_NAME="?([^"\n]*)"?/m.exec(readFileSync('/etc/os-release', 'utf8'))
    if (m) name = `${m[1]} (Linux ${release()})`
  } else if (os === 'win32') name = `${osVersion()} (${release()})`
  let gpu = ''
  if (os === 'darwin') {
    gpu = [...first('system_profiler', ['SPDisplaysDataType'], 20_000).matchAll(/Chipset Model: (.+)/g)].map(m => m[1].trim()).join(', ')
  } else if (os === 'linux') {
    gpu = first('nvidia-smi', ['--query-gpu=name,memory.total', '--format=csv,noheader'])
      || first('sh', ['-c', 'lspci 2>/dev/null | grep -Ei "vga|3d|display" | sed "s/^[^:]*: //"'])
  } else if (os === 'win32') {
    gpu = first('powershell.exe', ['-NoProfile', '-Command', '(Get-CimInstance Win32_VideoController).Name -join ", "'])
  }
  const cpu = cpus()
  return { os: `${name}, ${arch()}`, cpu: `${cpu[0]?.model?.trim() ?? 'unknown'} (${cpu.length} threads)`,
    memory: `${Math.round(totalmem() / 2 ** 30)} GB`, gpu: gpu.split('\n').join(', ') || 'not found', node: process.version }
}

/** The devices `llama-server --list-devices` prints (one per line after "Available devices:"). */
export function listDevices(output) {
  const lines = String(output).split('\n')
  const at = lines.findIndex(l => /^Available devices:/.test(l.trim()))
  return at < 0 ? [] : lines.slice(at + 1).map(l => l.trim()).filter(Boolean)
}

/** Columns a string takes in a terminal: CJK and full-width characters take two. */
export function displayWidth(s) {
  let w = 0
  for (const ch of String(s)) {
    const c = ch.codePointAt(0)
    w += (c >= 0x1100 && c <= 0x115f) || (c >= 0x2e80 && c <= 0xa4cf) || (c >= 0xac00 && c <= 0xd7a3) || (c >= 0xf900 && c <= 0xfaff)
      || (c >= 0xfe30 && c <= 0xfe4f) || (c >= 0xff00 && c <= 0xff60) || (c >= 0xffe0 && c <= 0xffe6) || (c >= 0x1f300 && c <= 0x1faff) ? 2 : 1
  }
  return w
}

/**
 * `label text`, wrapped to `width` columns: by words, or by characters where there are no spaces (Japanese); the lines
 * after the first line up under the text.
 */
export function labeled(label, text, { indent = 4, width = 78 } = {}) {
  const head = `${' '.repeat(indent)}${label} `
  const hang = ' '.repeat(displayWidth(head))
  const flat = String(text).replace(/\s+/g, ' ').trim()
  const tokens = /[\u3000-\u30ff\u4e00-\u9fff\uff00-\uffef]/.test(flat) ? [...flat] : flat.split(/(?<= )/)
  const lines = []
  let line = ''
  for (const tok of tokens) {
    if (line && displayWidth(head) + displayWidth(line + tok) > width && !/^[、。」）！？]/.test(tok)) { lines.push(line.trimEnd()); line = tok.trimStart() }
    else line += tok
  }
  if (line) lines.push(line.trimEnd())
  return head + lines.join(`\n${hang}`)
}

/** The report to paste into an issue (Markdown). */
export function formatReport(r) {
  const row = (k, v) => `| ${k} | ${String(v ?? '').replace(/\|/g, '\\|').replace(/\s*[\r\n]+\s*/g, ' ')} |`
  const out = ['### tsuzuki test report', '', '| | |', '|---|---|',
    row('tsuzuki', r.version), row('OS', r.machine.os), row('CPU', r.machine.cpu), row('Memory', r.machine.memory),
    row('GPU', r.machine.gpu), row('Node.js', r.machine.node), row('llama.cpp', r.llama || 'not found'),
    row('Setup', r.setup), row('Result', r.error ? `✗ stopped: ${r.error.split('\n')[0]}` : '✓ every check passed'), '']
  if (r.steps.length) {
    out.push('Steps:')
    for (const s of r.steps) out.push(`- ${s.ok ? '✓' : '✗'} ${s.what}${s.seconds !== undefined ? ` (${s.seconds.toFixed(1)} s)` : ''}`)
    out.push('')
  }
  if (r.devices.length) out.push('llama.cpp devices (`llama-server --list-devices`):', '```', ...r.devices, '```', '')
  if (r.error && r.error.includes('\n')) out.push('Error:', '```', r.error.slice(0, 3000), '```', '')
  return sanitize(out.join('\n'))
}

// -- --update ---------------------------------------------------------------------------------------------------

const parts = v => String(v).split('.').map(n => parseInt(n, 10) || 0)
export function newer(a, b) {
  const [x, y] = [parts(a), parts(b)]
  for (let i = 0; i < 3; i++) if ((x[i] ?? 0) !== (y[i] ?? 0)) return (x[i] ?? 0) > (y[i] ?? 0)
  return false
}

/** The CHANGELOG sections ("## 0.2.0 ...") of versions newer than `since`, newest first. */
export function changesSince(changelog, since) {
  return String(changelog).split(/^(?=## )/m).filter(s => {
    const m = /^## v?(\d+\.\d+\.\d+)/.exec(s)
    return m && newer(m[1], since)
  }).map(s => s.trim())
}
