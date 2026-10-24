#!/usr/bin/env python3
"""dsh-prime-memory inside dsh: egemma daemon (v2) vs the plugin's default local embedding (v1).

Each run gets a fresh, independent DSH_HOME under build/dsh-home-bench/ (never ~/.dsh), copied from a
template profile with dsh-prime-memory 0.22.1 and this repository's provider tarballs installed on
dsh 0.2.0-rc.2, telemetry disabled and no API key anywhere. Modes:
  v1  prime-memory's default local embedding: embedding source "local", model embeddinggemma-300m
      (ONNX q8 on CPU in a worker thread of the dsh process). The pinned runtime and sha256-verified model
      from scripts/stage2/setup.sh are copied into the data directory, so the plugin downloads nothing.
  v2  the text-embeddings-egemma provider starts egemma-server (EmbeddingGemma 2, fp16, MPP) and
      prime-memory's remote embedding points at it (model embeddinggemma-2, 768 dims, no prompt:
      the plugin sends the same model name for documents and queries).
  v1e the same provider serving EmbeddingGemma 300m from this engine (modelDir models/embeddinggemma-300m,
      modelName embeddinggemma-300m, fp32), i.e. v1's model with its real sentence embedding.
Both: capture on, distillation (L1/L2/L3 extraction) off, memoryMutate on, recall strategy hybrid,
everything else at the plugin defaults. The probe plugin (dsh-plugins/tests/dsh-e2e/prime-memory-bench-probe.mjs)
feeds the same LoCoMo conversation and measures save/search latency, dsh CPU time and event-loop stalls;
this runner adds the peak physical footprint of the dsh process (and of the daemon in v2).
Modes alternate (v1, v2, v1, ...) for --reps repetitions; the summary reports the median over runs.
One extra run, v2-burst, repeats v2 with a warm-up write every 100 ms instead of every second: how the
plugin's embedding circuit breaker reacts to requests that arrive while the daemon is still loading.
Output: results/eval/dsh/<run>/ (probe.json, run.json, configs, logs) and results/eval/dsh_bench.json + .md.
Usage: .venv/bin/python scripts/eval/dsh_bench.py [--reps 3] [--modes v1,v2] [--conversation conv-26]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/stage2'))
from common import footprint, listening_pid, read_json, write_json  # noqa: E402

DSH = ROOT / 'build/dsh-runtime/0.2.0-rc.2/node_modules/.bin/dsh'
HOMES = ROOT / 'build/dsh-home-bench'
TEMPLATE = HOMES / 'template'
PROBE = ROOT / 'dsh-plugins/tests/dsh-e2e/prime-memory-bench-probe.mjs'
TARBALLS = [ROOT / f'dsh-plugins/dist/{p}-0.1.0.tgz' for p in ('dsh-text-embeddings', 'dsh-text-embeddings-egemma',
                                                                 'dsh-text-embeddings-openai')]
OUT = ROOT / 'results/eval/dsh'
LOCOMO = ROOT / 'build/eval/raw/locomo/locomo10.json'
ONNX_RUNTIME = ROOT / 'build/stage2/onnx-runtime'
ONNX_MODEL = ROOT / 'build/stage2/models/embeddinggemma-300m'


def conversation(sample_id: str) -> dict:
    """LoCoMo sample -> turns (speaker A as the user, B as the assistant), facts and questions."""
    sample = next(s for s in json.loads(LOCOMO.read_text()) if s['sample_id'] == sample_id)
    conv = sample['conversation']
    a = conv['speaker_a']
    messages = []
    for key in sorted((k for k in conv if re.fullmatch(r'session_\d+', k)), key=lambda k: int(k.split('_')[1])):
        for t in conv[key]:
            text = f"{t['speaker']}: {t['text']}" + (f" [shares an image: {t['blip_caption']}]" if t.get('blip_caption') else '')
            messages.append({'text': text, 'dia_id': t['dia_id'], 'user': t['speaker'] == a})
    turns, current = [], {}
    for m in messages:
        side = 'user' if m['user'] else 'assistant'
        if side in current or (side == 'user' and 'assistant' in current):
            turns.append(current)
            current = {}
        current[side] = m['text']
    if current:
        turns.append(current)
    facts = [{'text': fact, 'evidence': [e for e in re.split(r'[;,\s]+', str(ev)) if e]}
             for per_session in sample['observation'].values() for items in per_session.values() for fact, ev in items]
    dia = {m['dia_id'] for m in messages}
    questions = []
    for qa in sample['qa']:
        evidence = [e for raw in qa.get('evidence', []) for e in re.split(r'[;,\s]+', raw) if e in dia]
        if qa.get('category') != 5 and evidence:
            questions.append({'text': qa['question'], 'evidence': evidence, 'category': qa['category']})
    return {'sample_id': sample_id, 'turns': turns, 'messages': [{'text': m['text'], 'dia_id': m['dia_id']} for m in messages],
            'facts': facts, 'questions': questions}


def ensure_template() -> None:
    if (TEMPLATE / 'profiles/bench/node_modules/dsh-prime-memory/package.json').exists():
        return
    shutil.rmtree(TEMPLATE, ignore_errors=True)
    TEMPLATE.mkdir(parents=True)
    env = dict(os.environ, DSH_HOME=str(TEMPLATE), DSH_TELEMETRY_DISABLED='1')
    subprocess.run([str(DSH), 'plugin', '--profile', 'bench', 'add', 'dsh-prime-memory@0.22.1', *map(str, TARBALLS)],
                   env=env, check=True, stdout=subprocess.DEVNULL, timeout=600)


def profile_patch(mode: str, data_dir: Path, port: int) -> str:
    memory = f"""- id: dsh-memory
  config:
    dataDir: '{data_dir}'
    capture:
      enabled: true
    extract:
      enabled: false
    l2:
      enabled: false
    l3:
      enabled: false
    live:
      capture: true
      distill: false
      memoryMutate: true
    recall:
      strategy: hybrid
"""
    if mode.startswith('v2') or mode == 'v1e':
        model = 'embeddinggemma-300m' if mode == 'v1e' else 'embeddinggemma-2'
        memory += f"""    embedding:
      enabled: true
      baseUrl: http://127.0.0.1:{port}/v1
      apiKey: ''
      model: {model}
      dimensions: 768
      allowLocalModels: false
- id: text-embeddings-egemma
  disabled: false
  config:
    engineRoot: '{ROOT}'
    port: {port}
"""
        if mode == 'v1e':
            memory += f"""    modelDir: '{ROOT / "models/embeddinggemma-300m"}'
    modelName: embeddinggemma-300m
    dtype: fp32
"""
    return memory + """- insert:
    - id: bench-webserver
      name: '@deepseek-ai/dsh-host-webserver'
      config:
        host: 127.0.0.1
        port: 0
    - id: bench-connection
      name: '@deepseek-ai/dsh-client-connection'
"""


def run_once(mode: str, index: int, conv_path: Path, port: int, idle_ms: int) -> dict:
    label = f'{mode}-r{index}'
    out = OUT / label
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    HOMES.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix=f'{label}.', dir=HOMES))
    subprocess.run(['cp', '-cR', str(TEMPLATE / 'profiles'), str(home / 'profiles')], check=True)
    data_dir = home / 'memory'
    data_dir.mkdir()
    if mode == 'v1':
        subprocess.run(['cp', '-cR', str(ONNX_RUNTIME), str(data_dir / 'runtime')], check=True)
        (data_dir / 'models').mkdir()
        subprocess.run(['cp', '-cR', str(ONNX_MODEL), str(data_dir / 'models/embeddinggemma-300m')], check=True)
        (data_dir / 'embedding-source.json').write_text(json.dumps({'source': 'local', 'activeModel': 'embeddinggemma-300m'}))
    patch = profile_patch(mode, data_dir, port)
    (home / 'profiles/bench/cordis.patch.yml').write_text(patch)
    (out / 'profile.patch.yml').write_text(patch)
    release = home / 'release'  # marker for the probe: footprints read, dsh may exit
    probe_patch = out / 'probe.patch.yml'
    probe_patch.write_text(f"""- insert:
    - id: prime-memory-bench-probe
      name: '{PROBE}'
      config:
        mode: {mode}
        out: '{out}'
        conversation: '{conv_path}'
        dataDir: '{data_dir}'
        release: '{release}'
        idleMs: {idle_ms}
        warmupIntervalMs: {100 if mode == 'v2-burst' else 1000}
        daemonPort: {port if mode.startswith('v2') or mode == 'v1e' else 0}
""")
    env = dict(os.environ, DSH_HOME=str(home), DSH_TELEMETRY_DISABLED='1')
    if listening_pid(port):
        raise RuntimeError(f'port {port} is already in use')
    started = time.time()
    with open(out / 'boot.txt', 'w') as boot:
        proc = subprocess.Popen([str(DSH), '--profile', 'bench', '--patch', str(probe_patch)], env=env, stdout=boot,
                                stderr=subprocess.STDOUT, cwd=str(home))
    result: dict = {'mode': mode, 'run': label, 'dshHome': str(home), 'pid': proc.pid, 'apiKeyConfigured': False,
                    'telemetryDisabled': True}
    try:
        while not (out / 'probe.json').exists():
            if proc.poll() is not None:
                raise RuntimeError(f'dsh exited early (code {proc.returncode}); see {out / "boot.txt"}')
            if time.time() - started > 1800:
                raise TimeoutError('probe did not finish within 30 min')
            time.sleep(0.2)
        result['secondsToProbeDone'] = time.time() - started
        result['dshFootprint'] = footprint(proc.pid)
        daemon = listening_pid(port) if mode.startswith('v2') or mode == 'v1e' else None
        if daemon:
            result['daemonPid'] = daemon
            result['daemonFootprint'] = footprint(daemon)
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/stats', timeout=10) as r:
                stats = json.load(r)
            (out / 'daemon-stats.json').write_text(json.dumps(stats, indent=2) + '\n')
            result['daemonRequests'] = stats.get('requests')
            result['daemonErrors'] = stats.get('errors')
        release.touch()
        result['exitCode'] = proc.wait(timeout=120)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    time.sleep(1)
    result['daemonStoppedAfterExit'] = listening_pid(port) is None
    # memory.log quotes the stored texts (LoCoMo, CC BY-NC): it stays in the run's DSH_HOME under build/;
    # only its warning/error lines are kept with the results.
    log = data_dir / 'memory.log'
    if log.exists():
        lines = [line for line in log.read_text(errors='replace').splitlines() if '[warn]' in line or '[error]' in line]
        (out / 'memory-warnings.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
        result['memoryLog'] = str(log)
    result['probe'] = read_json(out / 'probe.json')
    write_json(out / 'run.json', {k: v for k, v in result.items() if k != 'probe'})
    return result


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize(runs: list[dict]) -> dict:
    def pick(run, *path):
        value = run
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        return value
    rows = {}
    for mode in sorted({r['mode'] for r in runs}):
        ok = [r for r in runs if r['mode'] == mode and (r.get('probe') or {}).get('ok')]
        row = {'runs': len(ok)}
        metrics = {
            'tools_ready_ms': ('probe', 'toolsReadyMs'),
            'embeddings_ready_ms': ('probe', 'embeddingsReadyMs'),
            'embeddings_ready_uptime_ms': ('probe', 'embeddingsReadyUptimeMs'),
            'vectorless_writes': ('probe', 'phases', 'warmup', 'vectorlessWrites'),
            'capture_turn_p50_ms': ('probe', 'phases', 'capture', 'latencyMs', 'p50'),
            'capture_turn_p95_ms': ('probe', 'phases', 'capture', 'latencyMs', 'p95'),
            'capture_wall_s': ('probe', 'phases', 'capture', 'wallMs'),
            'capture_cpu_s': ('probe', 'phases', 'capture', 'cpuMs'),
            'capture_daemon_cpu_s': ('probe', 'phases', 'capture', 'daemonCpuMs'),
            'capture_daemon_engine_s': ('probe', 'phases', 'capture', 'daemonEngineMs'),
            'search_l1_cpu_s': ('probe', 'phases', 'search_l1', 'cpuMs'),
            'search_l1_daemon_cpu_s': ('probe', 'phases', 'search_l1', 'daemonCpuMs'),
            'capture_loop_max_ms': ('probe', 'phases', 'capture', 'eventLoopDelayMs', 'max'),
            'capture_loop_p99_ms': ('probe', 'phases', 'capture', 'eventLoopDelayMs', 'p99'),
            'add_p50_ms': ('probe', 'phases', 'add', 'latencyMs', 'p50'),
            'add_p95_ms': ('probe', 'phases', 'add', 'latencyMs', 'p95'),
            'add_loop_max_ms': ('probe', 'phases', 'add', 'eventLoopDelayMs', 'max'),
            'search_l1_p50_ms': ('probe', 'phases', 'search_l1', 'latencyMs', 'p50'),
            'search_l1_p95_ms': ('probe', 'phases', 'search_l1', 'latencyMs', 'p95'),
            'search_l1_loop_max_ms': ('probe', 'phases', 'search_l1', 'eventLoopDelayMs', 'max'),
            'search_l0_p50_ms': ('probe', 'phases', 'search_l0', 'latencyMs', 'p50'),
            'search_l0_p95_ms': ('probe', 'phases', 'search_l0', 'latencyMs', 'p95'),
            'search_l0_loop_max_ms': ('probe', 'phases', 'search_l0', 'eventLoopDelayMs', 'max'),
            'idle_loop_max_ms': ('probe', 'phases', 'idle', 'eventLoopDelayMs', 'max'),
            'search_l1_hit_at5': ('probe', 'phases', 'search_l1', 'hitAt5'),
            'search_l0_hit_at5': ('probe', 'phases', 'search_l0', 'hitAt5'),
            'dsh_peak_gib': ('dshFootprint', 'lifetime_peak_bytes'),
            'daemon_peak_gib': ('daemonFootprint', 'lifetime_peak_bytes'),
        }
        for name, path in metrics.items():
            value = median([pick(r, *path) for r in ok])
            if value is not None and name.endswith('_gib'):
                value /= 2 ** 30
            if value is not None and name.endswith('_s') and not name.endswith('_ms'):
                value /= 1e3
            row[name] = value
        for phase in ('capture', 'search_l1'):
            cpu = row.get(f'{phase}_cpu_s')
            if cpu is not None:
                row[f'{phase}_total_cpu_s'] = cpu + (row.get(f'{phase}_daemon_cpu_s') or 0)
        if row.get('dsh_peak_gib') is not None:
            row['total_peak_gib'] = row['dsh_peak_gib'] + (row.get('daemon_peak_gib') or 0)
        rows[mode] = row
    return rows


def write_markdown(report: dict, path: Path) -> None:
    rows = report['summary']
    modes = [m for m in ('v1', 'v1e', 'v2') if m in rows]
    title = {'v1': 'v1: prime-memory default (ONNX, CPU, in-process)', 'v1e': 'v1 on egemma daemon (Metal, fp32)',
             'v2': 'v2: egemma daemon (Metal)'}
    fmt = lambda v, d=1: '—' if v is None else f'{v:.{d}f}'  # noqa: E731
    items = [('runs (median over)', 'runs', 0), ('tools ready after plugin load (ms)', 'tools_ready_ms', 0),
             ('embeddings usable after plugin load / after dsh start (ms)', ('embeddings_ready_ms', 'embeddings_ready_uptime_ms'), 0),
             ('memories stored without a vector before that (warm-up writes)', 'vectorless_writes', 0),
             ('capture: save per turn p50 / p95 (ms)', ('capture_turn_p50_ms', 'capture_turn_p95_ms'), 1),
             ('capture: whole conversation (s)', 'capture_wall_s', 1), ('capture: dsh CPU time (s)', 'capture_cpu_s', 1),
             ('capture: daemon CPU time (s)', 'capture_daemon_cpu_s', 1),
             ('capture: CPU time, dsh + daemon (s)', 'capture_total_cpu_s', 1),
             ('capture: daemon engine time incl. GPU (s, upper bound of GPU time)', 'capture_daemon_engine_s', 1),
             ('memory_search: CPU time, dsh + daemon (s)', 'search_l1_total_cpu_s', 2),
             ('capture: event-loop stall max / p99 (ms)', ('capture_loop_max_ms', 'capture_loop_p99_ms'), 1),
             ('memory_add p50 / p95 (ms)', ('add_p50_ms', 'add_p95_ms'), 1), ('memory_add: event-loop stall max (ms)', 'add_loop_max_ms', 1),
             ('memory_search p50 / p95 (ms)', ('search_l1_p50_ms', 'search_l1_p95_ms'), 1),
             ('memory_search: event-loop stall max (ms)', 'search_l1_loop_max_ms', 1),
             ('conversation_search p50 / p95 (ms)', ('search_l0_p50_ms', 'search_l0_p95_ms'), 1),
             ('conversation_search: event-loop stall max (ms)', 'search_l0_loop_max_ms', 1),
             ('idle: event-loop stall max (ms, noise floor)', 'idle_loop_max_ms', 1),
             ('peak footprint: dsh process (GiB)', 'dsh_peak_gib', 2), ('peak footprint: egemma daemon (GiB)', 'daemon_peak_gib', 2),
             ('peak footprint: total (GiB)', 'total_peak_gib', 2),
             ('reference: memory_search hit@5 (evidence fact found)', 'search_l1_hit_at5', 3),
             ('reference: conversation_search hit@5 (evidence turn found)', 'search_l0_hit_at5', 3)]
    conv = report['conversation']
    lines = ['# dsh-prime-memory inside dsh: ' + ' vs '.join(title[m] for m in modes), '',
             f'dsh 0.2.0-rc.2, dsh-prime-memory 0.22.1, LoCoMo {conv["sample_id"]}: {conv["turns"]} turns '
             f'({conv["messages"]} messages), {conv["facts"]} facts (memory_add), {conv["questions"]} questions (both searches). '
             'Median over runs; each run is a fresh DSH_HOME.', '',
             '| | ' + ' | '.join(title[m] for m in modes) + ' |', '|---|' + '---:|' * len(modes)]
    for label, key, digits in items:
        if isinstance(key, tuple):
            cells = [f'{fmt(rows[m].get(key[0]), digits)} / {fmt(rows[m].get(key[1]), digits)}' for m in modes]
        else:
            cells = [fmt(rows[m].get(key), digits) for m in modes]
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    if 'v2-burst' in rows:
        b = rows['v2-burst']
        lines += ['', f'v2-burst (warm-up writes every 100 ms while the daemon loads): embeddings usable after '
                  f'{fmt(b.get("embeddings_ready_ms"), 0)} ms, {fmt(b.get("vectorless_writes"), 0)} memories stored without a vector '
                  '(prime-memory opens its embedding circuit breaker for 60 s after consecutive failures; egemma-server holds '
                  'requests while it loads, up to --load-wait 15 s, instead of answering 503).']
    path.write_text('\n'.join(lines) + '\n')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--modes', default='v1,v2')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--conversation', default='conv-26')
    ap.add_argument('--port', type=int, default=8321)
    ap.add_argument('--idle-ms', type=int, default=5000)
    ap.add_argument('--no-burst', dest='burst', action='store_false', help='skip the v2-burst run')
    ap.add_argument('--summary-name', default='dsh_bench', help='results/eval/<name>.json and .md')
    args = ap.parse_args()
    ensure_template()
    data = conversation(args.conversation)
    OUT.mkdir(parents=True, exist_ok=True)
    # The LoCoMo text stays under build/ (CC BY-NC; not committed).
    conv_path = ROOT / 'build/eval/dsh/conversation.json'
    conv_path.parent.mkdir(parents=True, exist_ok=True)
    conv_path.write_text(json.dumps(data, ensure_ascii=False))
    runs = []
    schedule = [(mode, index) for index in range(1, args.reps + 1) for mode in args.modes.split(',')]
    if args.burst:
        schedule.append(('v2-burst', 1))
    for mode, index in schedule:
        print(f'run {mode} #{index}', flush=True)
        try:
            r = run_once(mode, index, conv_path, args.port, args.idle_ms)
        except Exception as error:
            r = {'mode': mode, 'run': f'{mode}-r{index}', 'error': f'{type(error).__name__}: {error}'}
        p = r.get('probe') or {}
        print(f'  ok={p.get("ok")} ' + (p.get('error', '')[:300] if not p.get('ok') else
              f'capture p50 {p["phases"]["capture"]["latencyMs"]["p50"]:.1f} ms, '
              f'search p50 {p["phases"]["search_l1"]["latencyMs"]["p50"]:.1f} ms') + (r.get('error') or ''), flush=True)
        runs.append(r)
        time.sleep(3)
    report = {'dsh': '0.2.0-rc.2', 'primeMemory': '0.22.1',
              'conversation': {'sample_id': data['sample_id'], 'turns': len(data['turns']), 'messages': len(data['messages']),
                               'facts': len(data['facts']), 'questions': len(data['questions'])},
              'runs': [{k: v for k, v in r.items() if k != 'probe'} | {'probe_ok': (r.get('probe') or {}).get('ok')} for r in runs],
              'summary': summarize(runs)}
    write_json(ROOT / f'results/eval/{args.summary_name}.json', report)
    write_markdown(report, ROOT / f'results/eval/{args.summary_name}.md')
    return 0 if all((r.get('probe') or {}).get('ok') for r in runs) else 3


if __name__ == '__main__':
    sys.exit(main())
