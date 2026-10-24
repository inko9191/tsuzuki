// The dsh end-to-end scripts (readiness test 2026-10-23, R17): they stop with a clear message when the dsh runtime is
// missing, before writing anything, and keep their output out of the committed results/ unless asked.
import { describe, it } from 'node:test'
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const engine = fileURLToPath(new URL('../..', import.meta.url))

describe('dsh e2e scripts', () => {
  for (const [script, label] of [['run-memory.sh', 'memory_e2e'], ['run-working-memory.sh', 'working_memory_e2e']]) {
    it(`${script} fails loudly without the dsh runtime and writes under build/ by default`, () => {
      const path = fileURLToPath(new URL(`./dsh-e2e/${script}`, import.meta.url))
      const version = '0.0.0-not-installed'
      const run = spawnSync('bash', [path, version, '18999'], { encoding: 'utf8', env: { ...process.env, E2E_OUT_DIR: '' } })
      assert.equal(run.status, 2, run.stderr)
      assert.match(run.stderr, /dsh runtime 0\.0\.0-not-installed not found: .*build\/dsh-runtime\/0\.0\.0-not-installed/)
      assert.ok(!existsSync(`${engine}/results/dsh/${label}_${version}`))
      assert.ok(!existsSync(`${engine}/build/dsh-e2e/${label}_${version}`), 'nothing is written before the check')
      const text = readFileSync(path, 'utf8')
      assert.match(text, /out_dir="\$\{E2E_OUT_DIR:-\$engine\/build\/dsh-e2e\}"/)
      assert.doesNotMatch(text, /^out="\$engine\/results/m)
    })
  }
})
