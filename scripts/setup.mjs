#!/usr/bin/env node
// One-command setup of the memory: node scripts/setup.mjs [options] (--help lists them; the setup is in setup-main.mjs).
// This file only checks the Node version first, in syntax that old Node versions can read: Node 12 (Ubuntu 22.04's)
// stopped at a syntax error in setup-main.mjs before it could say that Node is too old.
import { TEXT } from './setup-text.mjs'

const [major, minor] = process.versions.node.split('.').map(Number)
if (major >= 24 || (major === 22 && minor >= 19)) {
  import('./setup-main.mjs').catch(error => {
    console.error(error)
    process.exit(1)
  })
} else {
  const argv = process.argv.slice(2)
  const given = argv.find(a => a.startsWith('--lang=')), at = argv.indexOf('--lang')
  const asked = given ? given.slice('--lang='.length) : at >= 0 ? argv[at + 1] : ''
  const env = process.env
  const system = env.LC_ALL || env.LC_MESSAGES || env.LANG || Intl.DateTimeFormat().resolvedOptions().locale || ''
  const T = TEXT[asked === 'ja' || asked === 'en' ? asked : /^ja/i.test(system) ? 'ja' : 'en']
  console.error(`\n✗ ${T.nodeOld(process.versions.node, T.nodeHow[process.platform] || T.nodeHow.linux)}`)
  process.exit(1)
}
