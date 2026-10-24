#!/usr/bin/env python3
"""Separate timing/memory audit; never scores or overwrites frozen responses/results."""
from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.error

import numpy as np

import locomo_answer_audit as audit
import locomo_answer_production as production

ae, lm, st, mm = production.ae, production.lm, production.st, production.mm
LOCAL = audit.LOCAL
PLAN = lm.RESULTS / 'answer-audit-plan.json'
PUBLIC = lm.RESULTS / 'answer-audit-measurements.json'
ARMS = ('A', 'B1536', 'C8192-final', 'C-final')


def status(phase, **extra):
    lm.write(LOCAL / 'status.json', {'time': lm.stamp(), 'pid': os.getpid(),
                                  'pgid': os.getpgid(0), 'phase': phase, **extra})


def pick_questions(items):
    groups = defaultdict(set)
    for item in items.values():
        groups[item['id'].split('/')[0]].add(item['id'])
    return {g: sorted(ids, key=lambda q: ae.am.hashed('locomo-time-audit-v1', q))[:2]
            for g, ids in sorted(groups.items())}


def register():
    ae.gate()
    production.checked()
    prepared = lm.read(ae.LOCAL / 'pilot-inputs.json')
    tracked = subprocess.check_output(['git', 'ls-files', 'results/eval/memory',
                                      'scripts/eval', 'docs/MEMORY_ANSWER_EVAL.md'], text=True).splitlines()
    old_files = {p: lm.file_hash(lm.ROOT / p) for p in tracked}
    plan = {'created_at': lm.stamp(), 'source_plan': lm.file_hash(production.PLAN),
        'code_sha256': lm.file_hash(Path(__file__)),
        'audit_code_sha256': lm.file_hash(Path(audit.__file__)),
        'original_files_sha256': old_files,
        'pilot_inputs_sha256': lm.file_hash(ae.LOCAL / 'pilot-inputs.json'),
        'questions': pick_questions(prepared['items']), 'arms': ARMS, 'context': 65536,
        'generation': ae.am.GENERATION,
        'cache_order': ['first question cache=true', 'second question cache=true',
                        'same second question cache=false; require cache_n=0'],
        'responses': 120, 'estimate_seconds': [1200, 2400],
        'embedding': 'fresh same native fp32; document batches32 separately per conversation/set; '
                     'heldout302 original query embedding batch1; no vector cache overwrite',
        'ranking': 'CPU frozen original query vectors; exact original pre-rerank order check; no reselection',
        'memory': 'fresh same Qwen process per arm; macOS proc_pid_rusage v4 every50ms, '
                  'RSS and physical footprint plus kernel lifetime peak; includes loaded KV/cache; '
                  'same65536 window for all; no claim about isolated GPU bytes',
        'total_time': 'per-question sum of recorded heldout answer/rerank HTTP and fresh embedding/ranking; '
                      'component sum measured on different occasions, not a replayed online latency',
        'failure_policy': 'one HTTP attempt; journal fsync each; no regeneration/overwrite; '
                          'unknown inflight or unexpected server stop ends audit',
        'pause': 'SIGINT checkpoints; persistent user-pause.json gates all phases; server remains running',
        'ingestion_generation': 'original overall durations; count-proportional per-conversation estimate, '
                                'no six-hour rw/fact regeneration; user permits explicit estimates',
        'quality': 'no new scoring, generalization, semantic judgement, or condition selection'}
    if PLAN.exists():
        raise RuntimeError('audit already registered; do not overwrite')
    lm.write(PLAN, plan)


def checked():
    ae.gate()
    p = lm.read(PLAN)
    if p['code_sha256'] != lm.file_hash(Path(__file__)) or p['audit_code_sha256'] != lm.file_hash(Path(audit.__file__)):
        raise RuntimeError('audit implementation changed after registration')
    for name, digest in p['original_files_sha256'].items():
        if lm.file_hash(lm.ROOT / name) != digest:
            raise RuntimeError(f'frozen source changed: {name}')
    if lm.file_hash(ae.LOCAL / 'pilot-inputs.json') != p['pilot_inputs_sha256']:
        raise RuntimeError('pilot inputs changed')
    return p


def quiet(embedding_allowed=False):
    ae.gate()
    state = production.environment()
    lines = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True).splitlines()
    for line in lines:
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        command = fields[1]
        if 'python' in Path(command.split()[0]).name and (
            ('-m egemma_server' in command and not embedding_allowed) or
            any(tag in command for tag in ('locomo_answer_eval.py generate', 'locomo_answer_production.py run',
                                           'locomo_memory_stages.py generate', 'locomo_answer_audit_measure.py run'))):
            raise RuntimeError(f'other evaluation/embedding process active: {fields[0]}')
    if any(s.get('is_processing') for s in st.http_json('http://127.0.0.1:8080/slots')):
        raise RuntimeError('Qwen busy; no overlapping GPU work')
    return state


def restart(label, plan):
    """Only an identified healthy idle server is intentionally replaced; never recover a crashed one."""
    now = ae.server(plan)
    state = quiet()
    command = lm.read(lm.RESULTS / 'answer-pilot-server.json')['new_args']
    server = state['qwen_servers'][0]
    if server['command'] != ' '.join(command):
        raise RuntimeError('server argv differs from the registered65536 command')
    ae.gate()
    os.kill(server['pid'], signal.SIGTERM)
    for _ in range(100):
        try:
            reaped, _ = os.waitpid(server['pid'], os.WNOHANG)
            if reaped:
                break
        except ChildProcessError:
            pass
        try:
            os.kill(server['pid'], 0)
        except ProcessLookupError:
            break
        time.sleep(.1)
    else:
        raise RuntimeError('planned stop failed; no duplicate start')
    with (LOCAL / f'qwen-{label}.log').open('a') as log:
        proc = subprocess.Popen(command, cwd=lm.ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(60):
        if proc.poll() is not None:
            raise RuntimeError('planned restart failed; no retry')
        try:
            new = ae.server(plan)
            if new != now:
                raise RuntimeError('restarted identity differs')
            break
        except (urllib.error.URLError, ConnectionError):
            time.sleep(.5)
    else:
        raise RuntimeError('planned restart not ready')
    ae.gate()
    return proc.pid


def memory(pid):
    library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    buffer = ctypes.create_string_buffer(1024)
    if library.proc_pid_rusage(pid, 4, ctypes.byref(buffer)):
        raise RuntimeError(f'proc_pid_rusage failed for {pid}')
    fields = (ctypes.c_uint64 * 35).from_buffer(buffer, 16)
    return {'rss_bytes': fields[6], 'physical_footprint_bytes': fields[7],
            'kernel_lifetime_peak_footprint_bytes': fields[28]}


class MemorySamples:
    def __init__(self, pid):
        self.pid, self.peak, self.samples, self.error = pid, memory(pid), 0, None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            while not self.stop.is_set():
                values = memory(self.pid)
                self.samples += 1
                for k, v in values.items():
                    self.peak[k] = max(self.peak[k], v)
                self.stop.wait(.05)
        except Exception as e:
            self.error = str(e)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        if self.error and args[0] is None:
            raise RuntimeError(self.error)


def embeddings():
    """Timing only: originals cached for ranks; fresh vectors never replace old caches."""
    quiet()
    search = lm.read(ae.LOCAL / 'production/search-inputs.json')
    from egemma_server.tokenizer import PromptTokenizer
    tokenizer = PromptTokenizer(ae.rp.V1_ST_DIR)
    identity = lm.fingerprint(st.VARIANT)
    if identity != lm.read(lm.RESULTS / 'final-config.json')['embedding_artifact_sha256']:
        raise RuntimeError('embedding artifact changed')
    units = search['units']
    vectors = np.zeros((len(units), 768), np.float32)
    texts_by_name = {}
    def missing(_):
        raise RuntimeError('original frozen vector cache missing')
    for name in sorted({u['set'] for u in units}):
        indices = [i for i, u in enumerate(units) if u['set'] == name]
        texts = ae.rp.fit_like_st(tokenizer, [lm.PROTOCOL['prompts']['document'] + units[i]['text'] for i in indices])
        texts_by_name[name] = dict(zip(indices, texts))
        vectors[indices], _ = lm.cached_vectors(st.VARIANT, 'd' if name == 't1' else f'u-{name}',
                                                texts, identity, missing)
    queries = list(search['queries'].values())
    query_texts = ae.rp.fit_like_st(tokenizer, [lm.PROTOCOL['prompts']['query'] + q['text'] for q in queries])
    qvecs, _ = lm.cached_vectors(st.VARIANT, 'q-holdout-answer', query_texts, identity, missing)
    args = argparse.Namespace(port=8398, model_dir=lm.ROOT / 'models/embeddinggemma-2', v1_dir=ae.rp.V1_ST_DIR)
    ingestion, query_times = [], {}
    started = time.perf_counter()
    with lm.start_backend(st.VARIANT, args) as (info, embed):
        startup = time.perf_counter() - started
        for name in ('p2', 'rw', 'fact'):
            for group in sorted({u['group'] for u in units}):
                quiet(embedding_allowed=True)
                indices = [i for i, u in enumerate(units) if u['set'] == name and u['group'] == group]
                total = 0.
                for offset in range(0, len(indices), 32):
                    ae.gate()
                    batch = [texts_by_name[name][i] for i in indices[offset:offset+32]]
                    tick = time.perf_counter()
                    lm.checked_vectors(embed(batch), len(batch))
                    total += time.perf_counter() - tick
                ingestion.append({'set': name, 'group': group, 'units': len(indices), 'seconds': total})
        for q, text in zip(queries, query_texts):
            quiet(embedding_allowed=True)
            tick = time.perf_counter()
            lm.checked_vectors(embed([text]), 1)
            query_times[q['id']] = time.perf_counter() - tick
    tick = time.perf_counter()
    index = mm.Index(units, vectors)
    index_seconds = time.perf_counter() - tick
    cfg = ae.configs()['C-final']
    ranking_times = {}
    for qi, q in enumerate(queries):
        ae.gate()
        tick = time.perf_counter()
        order = mm.distinct([i for i, _ in index.rank(cfg, q['group'], [(qvecs[qi], q['text'])])], units)
        ranking_times[q['id']] = time.perf_counter() - tick
        if order != q['orders']['C-final']:
            raise RuntimeError('reconstructed frozen ranking differs; no reselection')
    result = {'created_at': lm.stamp(), 'startup_seconds': startup, 'backend': info,
              'ingestion': ingestion, 'query_batch': 1, 'query_seconds': audit.distribution(list(query_times.values())),
              'ranking_seconds': audit.distribution(list(ranking_times.values())),
              'index_build_seconds': index_seconds, 'all302_original_orders_match': True}
    rerank = lm.read(ae.LOCAL / 'production/rerank-responses.json')
    rows = lm.read(ae.LOCAL / 'production/scored-rows.json')['C-final']
    per_question = [{ 'id': r['id'], 'answer_seconds': r['wall_seconds'],
        'rerank_seconds': rerank[r['id']]['wall_seconds'], 'query_embedding_seconds': query_times[r['id']],
        'ranking_seconds': ranking_times[r['id']], 'total_component_seconds': r['wall_seconds']
            + rerank[r['id']]['wall_seconds'] + query_times[r['id']] + ranking_times[r['id']]} for r in rows]
    lm.write(LOCAL / 'component-rows.json', per_question)
    result['C_final_total_component_seconds'] = audit.distribution([r['total_component_seconds'] for r in per_question])
    result['C_final_components_per_conversation'] = {g: {k: audit.distribution([r[k] for r in per_question
        if r['id'].split('/')[0] == g]) for k in ('answer_seconds', 'rerank_seconds', 'query_embedding_seconds',
                                                'ranking_seconds', 'total_component_seconds')}
        for g in sorted({q['group'] for q in queries})}
    lm.write(LOCAL / 'embedding-and-search.json', result)
    return result


def run():
    p = checked()
    LOCAL.mkdir(parents=True, exist_ok=True)
    if (LOCAL / 'inflight.json').exists() or (LOCAL / 'received.jsonl').exists():
        raise RuntimeError('previous audit requests exist; no automatic repeat/resume')
    if (LOCAL / 'status.json').exists():
        raise RuntimeError('audit already started; review saved state before any retry')
    plan = production.checked()['plan']['answer']
    now = ae.server(plan)
    state = quiet()
    args = lm.read(lm.RESULTS / 'answer-pilot-server.json')['new_args']
    if lm.file_hash(Path(args[args.index('-m')+1])) != now['model_sha256']:
        raise RuntimeError('fresh model hash differs')
    status('embedding/search measurement', status='running', environment=state)
    record = {'started_at': lm.stamp(), 'plan_sha256': lm.file_hash(PLAN), 'environment': state,
              'server': now, 'memory': {}, 'cache_comparison': {}, 'failures': []}
    try:
        record['embedding_search'] = embeddings()
        lm.write(PUBLIC, record)
        prepared = lm.read(ae.LOCAL / 'pilot-inputs.json')
        received = []
        for arm in ARMS:
            status('cache/memory measurement', status='running', arm=arm, completed=len(received), total=120)
            pid = restart(arm, plan)
            with MemorySamples(pid) as samples:
                for group, (first, second) in p['questions'].items():
                    for qid, cache, role in ((first, True, 'first'), (second, True, 'warm'), (second, False, 'no_cache')):
                        quiet()
                        item = prepared['items'][f'{arm}|{qid}']
                        body = ae.request(plan, item)
                        body['cache_prompt'] = cache
                        key = f'{arm}|{group}|{role}'
                        lm.write(LOCAL / 'inflight.json', {'key': key, 'digest': lm.digest(body), 'time': lm.stamp()})
                        tick = time.perf_counter()
                        response = st.http_json(now['url'] + '/v1/chat/completions', body, timeout=900)
                        wall = time.perf_counter() - tick
                        timings = ae.checked_timings(response)
                        entry = {'key': key, 'id': qid, 'arm': arm, 'group': group, 'role': role,
                            'input_tokens': item['input_tokens'], 'wall_seconds': wall,
                            'prefill_seconds': timings['prompt_ms']/1000,
                            'generation_seconds': timings['predicted_ms']/1000,
                            'cache_tokens': timings['cache_n'], 'processed_prompt_tokens': timings['prompt_n'],
                            'generated_tokens': timings['predicted_n'], 'memory': memory(pid),
                            'finish_reason': response['choices'][0]['finish_reason']}
                        ae.received(LOCAL / 'received.jsonl', key, {'response': response, **entry})
                        received.append(entry)
                        lm.write(LOCAL / 'timing-rows.json', received)
                        (LOCAL / 'inflight.json').unlink()
                        if not cache and timings['cache_n'] != 0:
                            raise RuntimeError('cache=false still reused tokens; stop rather than mislabel it')
                        status('cache/memory measurement', status='running', arm=arm, completed=len(received), total=120)
            record['memory'][arm] = {'pid': pid, 'samples': samples.samples, **samples.peak,
                'context': 65536, 'fresh_process': True, 'requests': 30, 'include': 'model+KV+RAM prefix cache; same context',
                'limits': 'physical footprint/RSS are OS process counters, not GPU-only allocation; 50ms samples'}
            record['cache_comparison'][arm] = {role: {metric: audit.distribution([r[metric] for r in received
                if r['arm'] == arm and r['role'] == role]) for metric in
                ('wall_seconds', 'prefill_seconds', 'generation_seconds', 'cache_tokens',
                 'processed_prompt_tokens', 'input_tokens', 'generated_tokens')}
                for role in ('first', 'warm', 'no_cache')}
            record['failures'] = [{'key': r['key'], 'finish_reason': r['finish_reason']} for r in received if r['finish_reason'] != 'stop']
            lm.write(PUBLIC, record)
        checked()
        record.update(finished_at=lm.stamp(), original_frozen_hashes_match=True, status='complete')
        lm.write(PUBLIC, record)
        status('audit complete', status='complete', completed=len(received), total=120)
    except BaseException as error:
        record.update(stopped_at=lm.stamp(), status='stopped', error=str(error))
        lm.write(PUBLIC, record)
        status('audit stopped', status='stopped', error=str(error))
        raise


def pause():
    lm.write(lm.LOCAL / 'user-pause.json', {'time': lm.stamp(), 'source': 'human audit pause; explicit resume only'})
    processes = []
    for line in subprocess.check_output(['ps', '-axo', 'pid=,pgid=,command='], text=True).splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3 and 'python' in Path(fields[2].split()[0]).name and 'locomo_answer_audit_measure.py run' in fields[2]:
            pid, pgid = map(int, fields[:2])
            if pid == pgid and pid != os.getpid():
                processes.append(pid)
                os.killpg(pgid, signal.SIGINT)
    print(f'pause gate saved; SIGINT sent to audit groups {processes}; Qwen not signalled')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('register', 'run', 'pause'))
    command = parser.parse_args().command
    {'register': register, 'run': run, 'pause': pause}[command]()
