#!/usr/bin/env node
// The memory from a terminal: status, recent, show, search, forget, export, and import from ChatGPT or Claude.
// Run with "help" for the commands. The code is dsh-plugins/packages/tsuzuki/lib/cli.js (built by the setup).
import { existsSync } from 'node:fs'

const cli = new URL('../dsh-plugins/packages/tsuzuki/lib/cli.js', import.meta.url)
if (!existsSync(cli)) {
  process.stderr.write('The memory is not built yet. Run "node scripts/setup.mjs" first.\n')
  process.exit(1)
}
// Node calls its built-in SQLite experimental and says so on every run; that means nothing to someone reading memory.
process.removeAllListeners('warning')
process.on('warning', w => { if (w.name !== 'ExperimentalWarning') process.stderr.write(`${w.name}: ${w.message}\n`) })
const { main } = await import(cli.href)
await main()
