/**
 * The memory from a terminal: see what it keeps, search it, delete from it, back it up, and import conversations
 * exported from ChatGPT or Claude. It talks to the memory server of a config (started when it is not running), so it
 * works the same for the chat proxy, MCP clients, Hermes profiles and other apps.
 *
 *   node scripts/memory.mjs [--config <path>] [--lang en|ja] <command> ...
 */
import { createInterface } from 'node:readline/promises'
import { existsSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { httpSidecar, type Sidecar } from './mcp.ts'
import { readExport, type ImportedConversation } from './importers.ts'
import { defaultConfigPath } from './server.ts'

type Lang = 'en' | 'ja'

/** Japanese when asked (`--lang ja`) or when the system's language is Japanese. */
export function language(argv: readonly string[], env = process.env): Lang {
  const at = argv.findIndex(a => a === '--lang' || a.startsWith('--lang='))
  const asked = at < 0 ? '' : argv[at]!.includes('=') ? argv[at]!.split('=')[1]! : argv[at + 1] ?? ''
  if (asked === 'ja' || asked === 'en') return asked
  const system = env.LC_ALL || env.LC_MESSAGES || env.LANG || Intl.DateTimeFormat().resolvedOptions().locale || ''
  return /^ja/i.test(system) ? 'ja' : 'en'
}

const TEXT = {
  en: {
    usage: `Usage: node scripts/memory.mjs [--config <path>] <command>

  status                  whether the memory server runs, and what it keeps
  recent [N]              the last N conversations (default 10), newest first
  show <conversation>     the messages of one conversation (its id from "recent")
  search <words...>       what the memory hands your model for these words
  forget <conversation>   delete one conversation (asks first; --yes skips the question)
  forget --before <date>  delete everything stored before a date, e.g. 2026-01-01
  forget --all            delete all memory
  export [file]           every message as JSON Lines, to the file or the screen (import reads it back)
  import <file>           conversations exported from ChatGPT or Claude: the .zip as downloaded,
                          or the conversations.json in it

--config: the memory's config.json (default: ~/.tsuzuki/config.json; Hermes: <profile>/tsuzuki/config.json).`,
    noConfig: (p: string) => `${p} does not exist. Run "node scripts/setup.mjs" first, or give the config with --config.`,
    notRunning: 'The memory server is not running (it starts when an app or this command needs it).',
    running: (pid: unknown) => `The memory server is running (pid ${String(pid)}).`,
    stats: (s: Record<string, number>) => `Stored: ${s.turns} messages, ${s.facts} facts; ${s.unindexed} not searchable yet.`,
    data: (d: unknown) => `Memory folder: ${String(d)}`,
    models: (e: unknown, c: unknown) => `Search model: ${e}. Facts model: ${c}.`,
    ok: 'answers', down: 'does not answer', none: 'none',
    empty: 'Nothing is stored yet.',
    messages: (n: number) => `${n} messages`,
    noMatch: 'Nothing matches.',
    noSuch: (s: string) => `There is no conversation "${s}". "recent" lists their ids.`,
    confirm: (what: string) => `Delete ${what}? This cannot be undone. [y/N] `,
    whatSession: (s: string) => `the conversation ${s}`,
    whatBefore: (d: string) => `every message stored before ${d}`,
    whatAll: 'all memory',
    noTty: 'There is no terminal to ask. Add --yes to delete without asking.',
    kept: 'Nothing was deleted.',
    deleted: (r: Record<string, unknown>) => `Deleted ${String(r.turns ?? 0)} messages and ${String(r.facts ?? 0)} facts.`,
    exported: (n: number, f: string) => `Wrote ${n} messages to ${f}.`,
    found: (n: number, m: number, src: string) => `${n} conversations, ${m} messages (${src}).`,
    importing: (k: number, n: number) => `  stored ${k} / ${n} conversations`,
    imported: (n: number) => `Imported ${n} conversations. They can be recalled as soon as they are indexed (see "status").`,
    factsLater: (min: number) => `Facts are pulled out of them in the background, one conversation at a time: about ${min} minutes with Gemma 4 E2B on a fast machine, longer without a GPU.`,
    unknown: (c: string) => `Unknown command: ${c}`,
    failed: (e: string) => `Failed: ${e}`,
  },
  ja: {
    usage: `使い方: node scripts/memory.mjs [--config <パス>] <コマンド>

  status                  記憶のサーバーが動いているか、何を覚えているか
  recent [N]              最近の会話 N 件（既定 10）、新しい順
  show <会話>             1 つの会話の発言（会話の ID は recent で分かる）
  search <言葉...>        その言葉でモデルに渡される記憶
  forget <会話>           会話を 1 つ消す（確認する。--yes で確認なし）
  forget --before <日付>  その日より前に保存したものを全部消す（例：2026-01-01）
  forget --all            記憶を全部消す
  export [ファイル]       全部の発言を JSON Lines で書き出す。ファイルか画面へ（import で読み戻せる）
  import <ファイル>       ChatGPT か Claude から書き出した会話を取り込む。ダウンロードした .zip のままか、
                          中の conversations.json

--config：記憶の config.json（既定：~/.tsuzuki/config.json。Hermes は <プロファイル>/tsuzuki/config.json）。`,
    noConfig: (p: string) => `${p} がありません。先に「node scripts/setup.mjs」を実行するか、--config で設定ファイルを指定してください。`,
    notRunning: '記憶のサーバーは動いていません（アプリかこのコマンドが必要としたときに起動します）。',
    running: (pid: unknown) => `記憶のサーバーは動いています（pid ${String(pid)}）。`,
    stats: (s: Record<string, number>) => `保存：発言 ${s.turns} 件、事実 ${s.facts} 件。まだ検索できないもの ${s.unindexed} 件。`,
    data: (d: unknown) => `記憶のフォルダ：${String(d)}`,
    models: (e: unknown, c: unknown) => `検索のモデル：${e}。事実のモデル：${c}。`,
    ok: '応答あり', down: '応答なし', none: 'なし',
    empty: 'まだ何も保存されていません。',
    messages: (n: number) => `${n} 件`,
    noMatch: '当てはまるものはありません。',
    noSuch: (s: string) => `会話「${s}」はありません。ID は recent で分かります。`,
    confirm: (what: string) => `${what}を消しますか？ 元に戻せません。 [y/N] `,
    whatSession: (s: string) => `会話 ${s} `,
    whatBefore: (d: string) => `${d} より前に保存した発言をすべて`,
    whatAll: '記憶をすべて',
    noTty: '確認するための端末がありません。確認なしで消すときは --yes を付けてください。',
    kept: '何も消していません。',
    deleted: (r: Record<string, unknown>) => `発言 ${String(r.turns ?? 0)} 件と事実 ${String(r.facts ?? 0)} 件を消しました。`,
    exported: (n: number, f: string) => `${n} 件の発言を ${f} に書き出しました。`,
    found: (n: number, m: number, src: string) => `会話 ${n} 件、発言 ${m} 件（${src}）。`,
    importing: (k: number, n: number) => `  ${k} / ${n} 件の会話を保存`,
    imported: (n: number) => `${n} 件の会話を取り込みました。検索できるようになるまで少しかかります（status で分かります）。`,
    factsLater: (min: number) => `事実は裏で 1 件ずつ抜き出します。速いマシンの Gemma 4 E2B でおよそ ${min} 分、GPU がないともっとかかります。`,
    unknown: (c: string) => `知らないコマンドです：${c}`,
    failed: (e: string) => `失敗しました：${e}`,
  },
}

export interface Io {
  out(line: string): void
  err(line: string): void
  /** The answer to a yes/no question, or null when there is no terminal to ask. */
  ask(question: string): Promise<string | null>
}

const terminal: Io = {
  out: line => process.stdout.write(line + '\n'),
  err: line => process.stderr.write(line + '\n'),
  async ask(question) {
    if (!process.stdin.isTTY) return null
    const rl = createInterface({ input: process.stdin, output: process.stdout })
    try { return await rl.question(question) } finally { rl.close() }
  },
}

const stamp = (at: number) => {
  const d = new Date(at)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
}
/** Stored text as it may be printed: without control characters (escape sequences in an imported chat could change the
 * terminal, its title or the clipboard). Line breaks and tabs stay. */
export const clean = (text: unknown) => String(text).replace(/[\x00-\x08\x0b-\x1f\x7f-\x9f]/g, '')
const oneLine = (text: string, width: number) => {
  const flat = clean(text).replace(/\s+/g, ' ').trim()
  return flat.length > width ? flat.slice(0, width - 1) + '…' : flat
}

/** The recall block without its header and closing tag: the dated excerpts only. */
function excerpts(text: string): string {
  return text.replace(/^<memory>\n/, '').replace(/\n<\/memory>$/, '').replace(/^Current date:[^\n]*\n[^\n]*\n\n/, '').trim()
}

/** Bytes per request when importing (well under the server's 8 MB body limit), and the longest message kept whole. */
const IMPORT_BATCH_BYTES = 4 * 2 ** 20
const IMPORT_MAX_CHARS = 200_000

/** The messages in requests of at most IMPORT_BATCH_BYTES each; a message longer than IMPORT_MAX_CHARS is cut. */
export function batches<T extends { text: string }>(messages: readonly T[]): T[][] {
  const out: T[][] = []
  let batch: T[] = [], bytes = 0
  for (const m of messages) {
    const kept = m.text.length > IMPORT_MAX_CHARS ? { ...m, text: `${m.text.slice(0, IMPORT_MAX_CHARS)} […]` } : m
    const size = Buffer.byteLength(JSON.stringify(kept))
    if (batch.length && bytes + size > IMPORT_BATCH_BYTES) { out.push(batch); batch = []; bytes = 0 }
    batch.push(kept)
    bytes += size
  }
  if (batch.length) out.push(batch)
  return out
}

export async function run(argv: readonly string[], options: { sidecar?: Sidecar, configPath?: string, io?: Io } = {}): Promise<number> {
  const io = options.io ?? terminal
  const T = TEXT[language(argv)]
  const args: string[] = []
  let configPath = options.configPath ?? ''
  let yes = false
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!
    if (a === '--config') configPath = argv[++i] ?? ''
    else if (a.startsWith('--config=')) configPath = a.slice(9)
    else if (a === '--lang') i++
    else if (a.startsWith('--lang=')) continue
    else if (a === '--yes' || a === '-y') yes = true
    else args.push(a)
  }
  const [command, ...rest] = args
  if (!command || command === 'help' || command === '--help' || command === '-h') {
    io.out(T.usage)
    return command ? 0 : 2
  }
  configPath ||= defaultConfigPath()
  let sidecar = options.sidecar
  if (!sidecar) {
    if (!existsSync(configPath)) { io.err(T.noConfig(configPath)); return 1 }
    sidecar = httpSidecar(configPath, join(dirname(fileURLToPath(import.meta.url)), 'server.js'))
  }
  try {
    switch (command) {
      case 'status': {
        const h = await sidecar.health?.()
        if (!h) { io.out(T.notRunning); return 0 }
        const embedding = h.embedding as Record<string, unknown> | undefined
        const chat = h.chat as Record<string, unknown> | null
        io.out(T.running(h.pid))
        io.out(T.stats(h.stats as Record<string, number>))
        io.out(T.data(h.dataDir))
        io.out(T.models(embedding?.ok === false ? T.down : T.ok, chat ? (chat.ok === false ? T.down : T.ok) : T.none))
        return 0
      }
      case 'recent': {
        const limit = Number(rest[0] ?? 10) || 10
        const { sessions } = await sidecar.post('/v1/memory/sessions', { limit }) as { sessions: Array<Record<string, unknown>> }
        if (!sessions.length) { io.out(T.empty); return 0 }
        for (const s of sessions) {
          io.out(`${stamp(Number(s.last))}  ${clean(s.session)}  (${T.messages(Number(s.messages))})`)
          io.out(`    ${oneLine(String(s.preview), 96)}`)
        }
        return 0
      }
      case 'show': {
        if (!rest[0]) { io.err(T.usage); return 2 }
        const { messages } = await sidecar.post('/v1/memory/session', { session: rest[0] }) as { messages: Array<Record<string, unknown>> }
        if (!messages.length) { io.err(T.noSuch(rest[0])); return 1 }
        for (const m of messages) io.out(`${stamp(Number(m.at))}  ${clean(m.role)}: ${clean(m.text)}`)
        return 0
      }
      case 'search': {
        const query = rest.join(' ').trim()
        if (!query) { io.err(T.usage); return 2 }
        const found = await sidecar.post('/v1/memory/search', { query }) as { text?: string }
        io.out(found.text ? clean(excerpts(found.text)) : T.noMatch)
        return 0
      }
      case 'forget': {
        let scope: Record<string, unknown>, what: string
        if (rest[0] === '--all') { scope = { all: true }; what = T.whatAll }
        else if (rest[0] === '--before' && rest[1]) { scope = { before: rest[1] }; what = T.whatBefore(rest[1]) }
        else if (rest[0] && !rest[0].startsWith('--')) {
          const { messages } = await sidecar.post('/v1/memory/session', { session: rest[0] }) as { messages: unknown[] }
          if (!messages.length) { io.err(T.noSuch(rest[0])); return 1 }
          scope = { session: rest[0] }; what = T.whatSession(rest[0])
        } else { io.err(T.usage); return 2 }
        if (!yes) {
          const answer = await io.ask(T.confirm(what))
          if (answer === null) { io.err(T.noTty); return 1 }
          if (!/^y/i.test(answer.trim())) { io.out(T.kept); return 0 }
        }
        io.out(T.deleted(await sidecar.post('/v1/memory/forget', scope)))
        return 0
      }
      case 'export': {
        const { sessions } = await sidecar.post('/v1/memory/sessions', {}) as { sessions: Array<{ session: string, first: number }> }
        const lines: string[] = []
        for (const s of sessions.sort((a, b) => a.first - b.first)) {
          const { messages } = await sidecar.post('/v1/memory/session', { session: s.session }) as { messages: Array<Record<string, unknown>> }
          for (const m of messages) lines.push(JSON.stringify({ session: s.session, id: m.id, role: m.role, text: m.text, at: m.at }))
        }
        if (rest[0]) {
          writeFileSync(rest[0], lines.join('\n') + (lines.length ? '\n' : ''), { mode: 0o600 })
          io.err(T.exported(lines.length, rest[0]))
        } else {
          for (const line of lines) io.out(line)
        }
        return 0
      }
      case 'import': {
        if (!rest[0]) { io.err(T.usage); return 2 }
        const conversations: ImportedConversation[] = readExport(rest[0])
        const label = { chatgpt: 'ChatGPT', claude: 'Claude', backup: 'backup' }
        const sources = [...new Set(conversations.map(c => label[c.source]))].join(', ') || '-'
        const total = conversations.reduce((n, c) => n + c.messages.length, 0)
        io.out(T.found(conversations.length, total, sources))
        let k = 0
        for (const c of conversations) {
          const session = c.source === 'backup' ? c.id : `${c.source}-${c.id}`
          for (const batch of batches(c.messages)) await sidecar.post('/v1/memory/messages', { session, messages: batch })
          await sidecar.post('/v1/memory/flush', { session })
          if (++k % 25 === 0 || k === conversations.length) io.out(T.importing(k, conversations.length))
        }
        io.out(T.imported(conversations.length))
        const h = await sidecar.health?.()
        if (h?.chat) io.out(T.factsLater(Math.max(1, Math.round(conversations.length * 4.2 / 60))))
        return 0
      }
      default:
        io.err(T.unknown(command))
        io.err(T.usage)
        return 2
    }
  } catch (error) {
    io.err(T.failed((error as Error).message))
    return 1
  }
}

export async function main(argv = process.argv.slice(2)): Promise<void> {
  process.exitCode = await run(argv)
}
