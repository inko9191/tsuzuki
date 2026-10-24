// In-dsh measurement of dsh-prime-memory 0.22.1 (scripts/eval/dsh_bench.py). Keyless: no LLM calls.
//
// Runs inside the dsh process next to prime-memory and drives it the way a session would:
//   warmup   memory_add until a stored memory actually gets a vector: the time until embeddings work after
//            the plugin loaded (lazy local model load in v1, daemon start in v2). One attempt per
//            warmupIntervalMs (default 1000);
//            memories written before that are stored without a vector by the plugin (vectorlessWrites).
//            All warm-up memories are retired with memory_delete before the measured phases.
//   capture  the same conversation log is appended turn by turn to a real dsh session
//            (ctx.sessions.create(); turn/start, user/message, step/start, assistant/message, step/end,
//            turn/end), so prime-memory's L0 capture hook receives the host's own `session/event`
//            dispatch; each turn waits until its messages' vectors are stored (vec0 rowid shadow table,
//            polled read-only in memory.db) — save latency per turn. The model never runs: the assistant text is the log's.
//   add      memory_add for each extracted fact (the conversation's LoCoMo observations), sequential.
//   search   memory_search (L1 facts) and conversation_search (L0 turns) for each question, sequential;
//            hits are matched to the question's evidence turns (reference only, limit 5, plugin defaults).
// Every phase records wall time, per-call latency, CPU time of the dsh process and the event-loop delay.
// With config.daemonPort (engine modes) it also records the daemon's CPU time (ps, 10 ms resolution) and the
// engine time from its /stats (batches.engine_seconds: time inside engine calls, GPU work included, an upper
// bound of GPU time); both are read outside the timed window of the phase.
// (perf_hooks.monitorEventLoopDelay, 1 ms resolution: max = the longest stall of the dsh main thread).
// After writing probe.json it waits for the runner to read the process footprint, then exits dsh.
import { writeFileSync, readFileSync, existsSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { monitorEventLoopDelay, performance } from 'node:perf_hooks'
import { setTimeout as delay } from 'node:timers/promises'
import { DatabaseSync } from 'node:sqlite'

export const name = 'prime-memory-bench-probe'
export const inject = ['tools', 'sessions']

const ms = (ns) => ns / 1e6
const quantile = (values, q) => {
  if (values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]
}
const latencyStats = (values) => values.length === 0 ? null : {
  n: values.length,
  mean: values.reduce((a, b) => a + b, 0) / values.length,
  p50: quantile(values, 0.5),
  p95: quantile(values, 0.95),
  max: Math.max(...values),
}

export function apply(ctx, config) {
  ctx.effect(() => {
    const controller = new AbortController()
    const applied = performance.now()
    const report = { ok: false, mode: config.mode, uptimeAtApplyMs: process.uptime() * 1e3, phases: {} }
    const histogram = monitorEventLoopDelay({ resolution: 1 })
    histogram.enable()
    const execute = (tool, args) => ctx.tools.execute({
      callId: `bench-${tool}-${Math.random().toString(36).slice(2)}`, name: tool, arguments: args, signal: controller.signal,
    })
    // Daemon CPU time in ms from `ps -o time=` ("[[hh:]mm:]ss.cc").
    const daemonCpuMs = (pid) => {
      const text = execFileSync('ps', ['-o', 'time=', '-p', String(pid)], { encoding: 'utf8' }).trim()
      return text.split(':').reduce((acc, part) => acc * 60 + Number(part), 0) * 1e3
    }
    let daemonPid = null
    const daemonSample = async () => {
      if (!config.daemonPort) return null
      if (!daemonPid) daemonPid = Number(execFileSync('lsof', ['-t', '-nP', `-iTCP:${config.daemonPort}`, '-sTCP:LISTEN'], { encoding: 'utf8' }).trim().split('\n')[0])
      const stats = await (await fetch(`http://127.0.0.1:${config.daemonPort}/stats`)).json()
      return { cpuMs: daemonCpuMs(daemonPid), engineMs: stats.batches.engine_seconds * 1e3 }
    }
    const phase = async (label, fn) => {
      const daemonBefore = await daemonSample()
      histogram.reset()
      const cpu = process.cpuUsage()
      const elu = performance.eventLoopUtilization()
      const started = performance.now()
      const extra = await fn()
      const used = process.cpuUsage(cpu)
      report.phases[label] = {
        wallMs: performance.now() - started,
        cpuMs: (used.user + used.system) / 1e3,
        eventLoopUtilization: performance.eventLoopUtilization(elu).utilization,
        eventLoopDelayMs: { max: ms(histogram.max), p99: ms(histogram.percentile(99)), p50: ms(histogram.percentile(50)), mean: ms(histogram.mean) },
        ...extra,
      }
      const daemonAfter = await daemonSample()
      if (daemonBefore && daemonAfter) {
        report.phases[label].daemonCpuMs = daemonAfter.cpuMs - daemonBefore.cpuMs
        report.phases[label].daemonEngineMs = daemonAfter.engineMs - daemonBefore.engineMs
      }
    }

    void (async () => {
      try {
        while (!['memory_add', 'memory_search', 'conversation_search'].every((t) => ctx.tools.get(t))) {
          await delay(20, undefined, { signal: controller.signal })
        }
        report.toolsReadyMs = performance.now() - applied
        const data = JSON.parse(readFileSync(config.conversation, 'utf8'))
        const db = new DatabaseSync(`${config.dataDir}/memory.db`, { readOnly: true })
        // vec0 needs the sqlite-vec extension; its rowid shadow table is a plain table and is written in the
        // same transaction as the vector.
        const countL0 = db.prepare('SELECT count(*) AS n FROM l0_conversations')
        const countL0Vec = db.prepare('SELECT count(*) AS n FROM l0_vec_rowids')
        const countL1Vec = db.prepare('SELECT count(*) AS n FROM l1_vec_rowids')

        await phase('warmup', async () => {
          const t = performance.now()
          const ids = []
          for (;;) {
            const before = countL1Vec.get().n
            const tryStarted = performance.now()
            const result = await execute('memory_add', { content: `Bench warm-up ${ids.length + 1}: the benchmark conversation starts now.`, type: 'work_fact', scene: 'bench' })
            if (result.isError || !result.value?.id) throw new Error(`warm-up memory_add failed: ${JSON.stringify(result).slice(0, 300)}`)
            ids.push(result.value.id)
            if (countL1Vec.get().n > before) break
            if (performance.now() - t > 180_000) throw new Error('embeddings did not become available within 180 s')
            await delay(Math.max(0, (config.warmupIntervalMs ?? 1000) - (performance.now() - tryStarted)))
          }
          report.embeddingsReadyMs = performance.now() - applied
          report.embeddingsReadyUptimeMs = process.uptime() * 1e3
          const latencyMs = performance.now() - t
          let retired = 0
          for (let i = 0; i < ids.length; i += 10) {
            const result = await execute('memory_delete', { ids: ids.slice(i, i + 10), limit: 10 })
            retired += result.value?.deleted ?? 0
          }
          return { latencyMs, attempts: ids.length, vectorlessWrites: ids.length - 1, retired }
        })

        // Event-loop noise floor with the embeddings loaded and nothing to do.
        await phase('idle', async () => { await delay(config.idleMs ?? 5000); return {} })

        const session = ctx.sessions.create()
        report.sessionId = String(session.id)
        // Surface messages (user/assistant) are appended at the end of the model-visible surface.
        const emit = (type, eventData) => type.endsWith('/message') ? session.append(type, eventData, { surfaceOp: 'append' }) : session.append(type, eventData)
        await phase('capture', async () => {
          const latencies = []
          let expected = countL0.get().n
          let expectedVec = countL0Vec.get().n
          let timeouts = 0
          for (let i = 0; i < data.turns.length; i++) {
            const turn = i + 1
            const { user, assistant } = data.turns[i]
            emit('turn/start', { turn })
            if (user) emit('user/message', { id: `bench-u${turn}`, role: 'user', content: [{ type: 'text', text: user }], source: { kind: 'user' } })
            emit('step/start', { turn, step: 1 })
            if (assistant) {
              emit('assistant/message', { turn, step: 1, stream: [], message: { id: `bench-a${turn}`, role: 'assistant',
                content: [{ type: 'text', text: assistant }], source: { kind: 'model', provider: 'bench-log', model: 'bench-log' } } })
            }
            emit('step/end', { turn, step: 1 })
            const t = performance.now()
            emit('turn/end', { turn, reason: { kind: 'completed' } })
            expected += (user ? 1 : 0) + (assistant ? 1 : 0)
            expectedVec += (user ? 1 : 0) + (assistant ? 1 : 0)
            // Stored = the turn's L0 vectors are committed (row, FTS and vector in one transaction).
            const deadline = t + 30_000
            while (countL0Vec.get().n < expectedVec) {
              if (performance.now() > deadline) { timeouts++; break }
              await delay(1)
            }
            if (timeouts >= 3) throw new Error(`capture: ${timeouts} turns not stored within 30 s (l0 rows ${countL0.get().n}, vectors ${countL0Vec.get().n}, expected ${expected})`)
            expectedVec = Math.min(expectedVec, countL0Vec.get().n)
            latencies.push(performance.now() - t)
          }
          return {
            turns: data.turns.length,
            messages: expected,
            latencyMs: latencyStats(latencies),
            timeouts,
            l0Rows: countL0.get().n,
            l0Vectors: countL0Vec.get().n,
          }
        })

        await phase('add', async () => {
          const latencies = []
          let failed = 0
          for (const fact of data.facts) {
            const t = performance.now()
            const result = await execute('memory_add', { content: fact.text, type: 'work_fact', scene: 'bench' })
            latencies.push(performance.now() - t)
            if (result.isError || !result.value?.id) failed++
          }
          return { calls: data.facts.length, failed, latencyMs: latencyStats(latencies), l1Vectors: countL1Vec.get().n }
        })

        const normalize = (s) => String(s).replace(/\s+/g, ' ').trim()
        const factEvidence = new Map(data.facts.map((f) => [normalize(f.text), f.evidence]))
        const turnEvidence = new Map(data.messages.map((m) => [normalize(m.text), m.dia_id]))
        for (const [label, tool, lookup] of [['search_l1', 'memory_search', (item) => factEvidence.get(normalize(item.content)) ?? []],
          ['search_l0', 'conversation_search', (item) => { const d = turnEvidence.get(normalize(item.content)); return d ? [d] : [] }]]) {
          await phase(label, async () => {
            const latencies = []
            let failed = 0
            let hit = 0
            let recall = 0
            let returned = 0
            for (const q of data.questions) {
              const t = performance.now()
              const result = await execute(tool, { query: q.text, limit: 5 })
              latencies.push(performance.now() - t)
              if (result.isError) { failed++; continue }
              const items = result.value?.items ?? []
              returned += items.length
              const found = new Set(items.flatMap(lookup).filter((d) => q.evidence.includes(d)))
              if (found.size > 0) hit++
              recall += found.size / q.evidence.length
            }
            const n = data.questions.length
            return { calls: n, failed, latencyMs: latencyStats(latencies), meanReturned: returned / n,
              hitAt5: hit / n, evidenceRecallAt5: recall / n }
          })
        }
        db.close()
        report.ok = true
      } catch (error) {
        report.error = String(error?.stack ?? error)
      }
      histogram.disable()
      writeFileSync(`${config.out}/probe.json`, JSON.stringify(report, null, 2) + '\n')
      // Hold the process (and the daemon) until the runner has read the footprints.
      const deadline = Date.now() + 120_000
      while (!existsSync(config.release) && Date.now() < deadline) await delay(100)
      ctx.get('appExit')?.(report.ok ? 0 : 1)
    })()
    return () => controller.abort()
  })
}
