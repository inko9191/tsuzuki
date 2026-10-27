// The parts of scripts/setup.mjs that can be tested on their own (scripts/setup-lib.mjs, scripts/setup-text.mjs).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { autostartPlan, changesSince, displayWidth, formatReport, labeled, language, listDevices, newer, parseTarget, sanitize, stableNode, systemdArg } from '../../scripts/setup-lib.mjs'
import { TEXT } from '../../scripts/setup-text.mjs'

test('setup text: English and Japanese have the same keys', () => {
  const shape = o => Object.entries(o).map(([k, v]) => v && typeof v === 'object' ? `${k}{${shape(v)}}` : `${k}:${typeof v}`).sort().join(',')
  assert.equal(shape(TEXT.ja), shape(TEXT.en))
  for (const lang of ['en', 'ja']) assert.match(TEXT[lang].help('9.9.9'), /9\.9\.9[\s\S]*--for chat\|mcp\|hermes\|app[\s\S]*--report/)
})

test('setup: the menu takes words or their first letters, never the old numbers', () => {
  assert.deepEqual(parseTarget('chat'), { target: 'chat' })
  assert.deepEqual(parseTarget(' MCP '), { target: 'mcp' })
  assert.deepEqual(parseTarget('h'), { target: 'hermes' })
  assert.deepEqual(parseTarget('a'), { target: 'app' })
  assert.deepEqual(parseTarget('2'), { error: 'number' })
  assert.deepEqual(parseTarget(''), { error: 'empty' })
  assert.deepEqual(parseTarget('chats'), { error: 'unknown' })
})

test('setup: language from --lang, else from the system', () => {
  assert.equal(language('ja', {}), 'ja')
  assert.equal(language(undefined, { LANG: 'ja_JP.UTF-8' }), 'ja')
  assert.equal(language('en', { LANG: 'ja_JP.UTF-8' }), 'en')
  assert.equal(language(undefined, { LANG: 'de_DE.UTF-8' }), 'en')
})

// Paths with spaces, an ampersand, quotes, $ and % must survive every format unchanged.
const odd = { node: '/Users/a b/.node $HOME/bin/node', serverJs: '/Users/a b/tsuzuki & co/lib/server.js', config: "/Users/a b/.tsu'zuki/config.json",
  log: '/Users/a b/100%/sidecar.log' }

test('autostart (macOS): a LaunchAgent with every argument XML-escaped, restarted only after a crash', () => {
  const plan = autostartPlan({ os: 'darwin', home: '/Users/a b', uid: 501, ...odd })
  assert.equal(plan.path, '/Users/a b/Library/LaunchAgents/io.github.inko9191.tsuzuki.plist')
  assert.match(plan.content, /<string>\/Users\/a b\/tsuzuki &amp; co\/lib\/server\.js<\/string>/)
  assert.match(plan.content, /<string>\/Users\/a b\/\.tsu&apos;zuki\/config\.json<\/string>/)
  assert.match(plan.content, /<key>KeepAlive<\/key><dict><key>SuccessfulExit<\/key><false\/><\/dict>/)
  assert.deepEqual(plan.on.at(-1), ['launchctl', ['bootstrap', 'gui/501', plan.path]])
  if (process.platform === 'darwin') {
    const file = join(mkdtempSync(join(tmpdir(), 'plist-')), 'x.plist')
    writeFileSync(file, plan.content)
    const lint = spawnSync('plutil', ['-lint', file], { encoding: 'utf8' })
    assert.equal(lint.status, 0, lint.stdout + lint.stderr)
    const args = JSON.parse(spawnSync('plutil', ['-extract', 'ProgramArguments', 'json', '-o', '-', file], { encoding: 'utf8' }).stdout)
    assert.deepEqual(args, [odd.node, odd.serverJs, '--config', odd.config, '--log', odd.log])
  }
})

test('autostart (Linux): a systemd user unit; quotes, backslashes, % and $ escaped', () => {
  const plan = autostartPlan({ os: 'linux', home: '/home/a b', ...odd })
  assert.equal(plan.path, '/home/a b/.config/systemd/user/tsuzuki.service')
  assert.equal(systemdArg('a "b" \\ 100% $HOME'), '"a \\"b\\" \\\\ 100%% $$HOME"')
  const exec = /^ExecStart=(.*)$/m.exec(plan.content)[1]
  assert.equal(exec, [odd.node, odd.serverJs, '--config', odd.config, '--log', odd.log].map(systemdArg).join(' '))
  assert.match(plan.content, /^Restart=on-failure$/m)
  assert.deepEqual(plan.on.at(-1), ['systemctl', ['--user', 'enable', '--now', 'tsuzuki.service']])
})

test('autostart (Windows): a hidden start from the Startup folder', () => {
  const plan = autostartPlan({ os: 'win32', home: 'C:\\Users\\a b', appData: 'C:\\Users\\a b\\AppData\\Roaming',
    node: 'C:\\Program Files\\nodejs\\node.exe', serverJs: 'C:\\x y\\server.js', config: 'C:\\c\\config.json', log: 'C:\\c\\sidecar.log' })
  assert.match(plan.path, /AppData.Roaming.Microsoft.Windows.Start Menu.Programs.Startup.tsuzuki\.vbs$/)
  assert.equal(plan.encoding, 'utf16le', 'Windows Script Host reads UTF-16 with a BOM, not UTF-8')
  assert.ok(plan.content.includes('.Run """C:\\Program Files\\nodejs\\node.exe"" ""C:\\x y\\server.js"" ""--config"" ""C:\\c\\config.json"" ""--log"" ""C:\\c\\sidecar.log""", 0, False'),
    plan.content)
})

test('report: home folder and user name taken out, table cells escaped, failures reported', () => {
  assert.equal(sanitize('/home/kenji/x and kenji', '/home/kenji', 'kenji'), '~/x and <user>')
  // Windows: the folder in a stack trace's file URL and with forward slashes, the name in another case.
  const win = 'at file:///C:/Users/Taro/tsuzuki/x.js and C:/Users/Taro/y and C:\\Users\\Taro\\z by TARO'
  const cleaned = sanitize(win, 'C:\\Users\\Taro', 'Taro')
  assert.doesNotMatch(cleaned, /taro/i)
  assert.match(cleaned, /~\/tsuzuki\/x\.js and ~\/y and ~\\z by <user>$/)
  // A short Japanese name, also percent-encoded in a URL; a short ASCII name only inside paths.
  assert.equal(sanitize('/Users/平方/a file:///Users/%E5%B9%B3%E6%96%B9/b 平方さん', '/Users/平方', '平方'), '~/a ~/b <user>さん')
  assert.equal(sanitize('/srv/al/x and also al', '/home/al', 'al'), '/srv/<user>/x and also al')
  const text = formatReport({ version: '0.2.0 (abc1234)', llama: 'version: 6912 | built with clang', setup: '--for app, quick',
    machine: { os: 'Ubuntu 24.04', cpu: 'Ryzen (16 threads)', memory: '32 GB', gpu: 'RTX 3060', node: 'v22.19.0' },
    steps: [{ what: 'memory server started', ok: true, seconds: 0.84 }, { what: 'facts', ok: false }], devices: ['using device CUDA0'],
    error: 'fact extraction did not work within 300 s.\nlog tail' })
  assert.match(text, /\| tsuzuki \| 0\.2\.0 \(abc1234\) \|/)
  assert.match(text, /\| llama\.cpp \| version: 6912 \\\| built with clang \|/)
  assert.match(text, /\| Result \| ✗ stopped: fact extraction did not work within 300 s\. \|/)
  assert.match(text, /- ✓ memory server started \(0\.8 s\)\n- ✗ facts/)
  assert.match(text, /```\nusing device CUDA0\n```/)
})

test('report: the devices llama-server lists', () => {
  assert.deepEqual(listDevices('ggml_cuda_init: found 1 CUDA devices\nAvailable devices:\n  CUDA0: NVIDIA GeForce RTX 3060 (12288 MiB, 11000 MiB free)\n  Vulkan0: x\n\n'),
    ['CUDA0: NVIDIA GeForce RTX 3060 (12288 MiB, 11000 MiB free)', 'Vulkan0: x'])
  assert.deepEqual(listDevices('error: invalid argument: --list-devices'), [])
})

test('demo text: wrapped by display width, Japanese by characters, continuation under the text', () => {
  assert.equal(displayWidth('三味線 ab'), 9)
  const en = labeled('AI:', 'You are learning the shamisen, and your teacher is Ms. Sato. '.repeat(2), { width: 40 })
  for (const line of en.split('\n')) assert.ok(displayWidth(line) <= 40, line)
  assert.match(en, /^    AI: You are learning the shamisen,\n        and your teacher/)
  const ja = labeled('あなた:', '最近、三味線を習い始めたんだ。先生は佐藤さん。今度の発表会で弾く曲も決まった。', { width: 40 })
  const lines = ja.split('\n')
  assert.ok(lines.length > 1)
  // A comma or full stop may hang past the width rather than start the next line.
  for (const line of lines) assert.ok(displayWidth(line) <= 40 || (displayWidth(line) <= 42 && /[、。」）！？]$/.test(line)), line)
  assert.ok(lines[1].startsWith(' '.repeat(displayWidth('    あなた: '))))
  assert.ok(!/^\s*[、。]/.test(lines[1].trimStart().slice(0, 1)), 'no line starts with a comma or a full stop')
})

test('update: versions compared by number; the CHANGELOG sections after the installed one', () => {
  assert.equal(newer('0.10.0', '0.9.9'), true)
  assert.equal(newer('0.2.0', '0.2.0'), false)
  const log = '# Changelog\n\n## 0.3.0 (2026-12-01)\n- c\n\n## 0.2.0 (2026-10-28)\n- b\n\n## 0.1.0 (2026-10-25)\n- a\n'
  assert.deepEqual(changesSince(log, '0.1.0'), ['## 0.3.0 (2026-12-01)\n- c', '## 0.2.0 (2026-10-28)\n- b'])
  assert.deepEqual(changesSince(log, '0.3.0'), [])
})

test('one version everywhere: the package, the memory server, the Hermes plugin', async () => {
  const { readFileSync } = await import('node:fs')
  const pkg = JSON.parse(readFileSync(new URL('../packages/tsuzuki/package.json', import.meta.url), 'utf8')).version
  const { VERSION } = await import('../packages/tsuzuki/lib/server.js')
  const hermes = /^version: (\S+)$/m.exec(readFileSync(new URL('../../integrations/hermes/tsuzuki/plugin.yaml', import.meta.url), 'utf8'))[1]
  assert.deepEqual([VERSION, hermes], [pkg, pkg])
})

test('stable node: the PATH entry that is this node (not a versioned Homebrew Cellar path), else the given one', { skip: process.platform === 'win32' && 'symlinks need privileges on Windows' }, () => {
  const dir = mkdtempSync(join(tmpdir(), 'node-path-'))
  symlinkSync(process.execPath, join(dir, 'node'))
  assert.equal(stableNode(process.execPath, `/nonexistent:${dir}`), join(dir, 'node'))
  assert.equal(stableNode(process.execPath, '/nonexistent'), process.execPath)
})
