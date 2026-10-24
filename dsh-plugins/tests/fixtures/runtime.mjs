// Minimal SubprocessRuntime over node:child_process for tests (spawn only).
import { spawn } from 'node:child_process'

export class TestRuntime {
  constructor() { this.spawned = [] }
  spawn(spec) {
    const child = spawn(spec.argv[0], spec.argv.slice(1), { cwd: spec.cwd, env: { ...process.env, ...spec.env }, stdio: ['ignore', 'pipe', 'pipe'] })
    let stderr = ''
    child.stderr.on('data', c => { stderr = (stderr + c).slice(-spec.stdio.stderr.maxBytes) })
    const done = new Promise((resolve, reject) => {
      child.on('error', reject)
      child.on('close', (exitCode, signal) => resolve({ exitCode, signal }))
    })
    this.spawned.push(child)
    return {
      stdin: undefined, stdout: child.stdout, stderr: undefined, control: undefined,
      collected: { stderr: { readFrom: () => ({ text: stderr, nextOffset: stderr.length, lossy: false }) } },
      done,
      terminate: () => { if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM') },
      waitForExit: async () => { await done; return true },
    }
  }
}
