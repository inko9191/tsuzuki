#!/usr/bin/env python3
"""Matched thirty-question, repeated footprint audit. No accuracy scoring."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

import locomo_long_c8k as frozen
import locomo_long_memory_audit as audit
import locomo_long_memory_audit_continue as previous

base, lm, ae = audit.base, audit.lm, audit.ae
SOURCE = frozen.SOURCE
LOCAL = lm.LOCAL / 'long-context-footprint'
PLAN = lm.RESULTS / 'long-context-footprint-plan.json'
CHECKPOINTS = (1, 3, 10, 30)
GIB = 2**30


def schedule():
    cells = [('default', 8192), ('default', 131072), ('host-cache-off', 131072), ('host-cache-off', 8192)]
    return [(r, arm, window) for r, order in ((1, cells), (2, list(reversed(cells)))) for arm, window in order]


def check_sources(p):
    for section in ('source_sha256', 'private_source_sha256'):
        for name, digest in p[section].items():
            if lm.file_hash(lm.ROOT / name) != digest:
                raise RuntimeError('protected source changed: ' + name)


def other_workers():
    lines = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True).splitlines()
    for line in lines:
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or int(parts[0]) == os.getpid():
            continue
        executable = Path(parts[1].split()[0]).name
        if 'python' in executable and re.search(r'locomo_(?:long|answer|memory).*\.py (?:run|bc|resume-a|generate)', parts[1]):
            raise RuntimeError('another evaluation worker exists')


def checked():
    ae.gate()
    p = base.st.frozen(PLAN)
    check_sources(p)
    other_workers()
    return p


@contextmanager
def runtime(arm=None):
    local, audit_local, audit_plan = base.LOCAL, audit.LOCAL, audit.PLAN
    names = ('LLAMA_ARG_LOG_VERBOSITY', 'LLAMA_ARG_CACHE_RAM')
    env = {name: os.environ.get(name) for name in names}
    base.LOCAL = audit.LOCAL = LOCAL
    audit.PLAN = PLAN
    if arm is None:
        for name in names:
            os.environ.pop(name, None)
    else:
        os.environ['LLAMA_ARG_LOG_VERBOSITY'] = '4'
        os.environ['LLAMA_ARG_CACHE_RAM'] = '0' if arm == 'host-cache-off' else '8192'
    try:
        yield
    finally:
        base.LOCAL, audit.LOCAL, audit.PLAN = local, audit_local, audit_plan
        for name, value in env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def event(kind, **data):
    audit.event(kind, **data)


def cache_settings(text, arm, window):
    match = re.search(r'initializing, n_slots = (\d+), n_ctx_slot = (\d+)', text)
    checkpoint = re.search(r'context checkpoints enabled, max = (\d+), min spacing = (\d+)', text)
    enabled = 'prompt cache is enabled, size limit: 8192 MiB' in text
    disabled = 'prompt cache is disabled' in text
    if not match or (int(match[1]), int(match[2])) != (1, window):
        raise RuntimeError('slot count/context log differs')
    if (arm == 'default' and not enabled) or (arm == 'host-cache-off' and not disabled):
        raise RuntimeError('effective host cache setting not verified')
    if not checkpoint or (int(checkpoint[1]), int(checkpoint[2])) != (32, 8192):
        raise RuntimeError('context checkpoint defaults differ')
    return {'host_cache_enabled': enabled, 'host_cache_limit_bytes': 8*GIB if enabled else 0,
            'slots': 1, 'slot_context': window, 'request_cache_prompt': True,
            'context_checkpoint_log': checkpoint.group(0) if checkpoint else None,
            'scope': 'host saved-prompt cache disabled only; active-slot prefix reuse/checkpoints unchanged'}


def cache_observations(text):
    states = [{'prompts': int(n), 'bytes': float(size)*1024**2} for n, size in
              re.findall(r'cache state:\s*(\d+) prompts,\s*([0-9.]+) MiB', text)]
    return {'recorded_updates': len(states), 'last': states[-1] if states else None,
            'largest_logged_bytes': max((v['bytes'] for v in states), default=None),
            'scope': 'logged update snapshots, not continuous allocator peak; no updates is not measured zero'}


def compression():
    text = subprocess.check_output(['/usr/bin/vm_stat'], text=True)
    page = int(re.search(r'page size of (\d+) bytes', text)[1])
    fields = {k.strip(): int(v) for k, v in re.findall(r'^([^:\n]+):\s+(\d+)\.', text, re.M)}
    value = lambda name: fields[name]*page if name in fields else None
    return {'system_compressor_physical_bytes': value('Pages occupied by compressor'),
            'system_uncompressed_bytes_in_compressor': value('Pages stored in compressor'),
            'scope': 'OS-global; not attributable to Qwen; footprint already includes compressed ledger accounting'}


def amount(value):
    match = re.fullmatch(r'([0-9.]+)([KMGT]?)', value)
    if not match:
        raise ValueError('invalid vmmap amount')
    return float(match[1])*1024**(' KMGT'.index(match[2]) if match[2] else 0)


def vmmap_summary(text):
    # Rounded vmmap values are diagnostic, not the primary libproc byte measurement.
    rows = {}
    for line in text.splitlines():
        match = re.match(r'^(TOTAL|Malloc Large|mapped file|shared memory)\s+((?:[0-9.]+[KMGT]?\s+){4})', line)
        if match:
            values = match[2].split()
            rows[match[1]] = dict(zip(('virtual', 'resident', 'dirty', 'swapped'), map(amount, values)))
    return {'regions_bytes_rounded': rows, 'per_process_compressed_bytes': None,
            'compression_limit': 'vmmap SWAPPED is not an exact separate compressed ledger value; per-process compressed bytes unconfirmed'}


def mapping(pid, label):
    value = audit.mapping_snapshot(pid, label)
    if value['recorded']:
        value.update(vmmap_summary((LOCAL / f'vmmap-{label}.txt').read_text()))
    return value


class SeriesWatch(base.resources.Watch):
    """Keep 50ms memory time series and separate current-request/cumulative peaks."""
    def __init__(self, pid, notify):
        super().__init__(pid, notify)
        self.lock = threading.Lock()
        self.series = []
        self.request = 0
        self.request_peak = self.peak.copy()

    def sample(self, current):
        with self.lock:
            self.samples += 1
            self.peak = {k: max(v, self.peak[k]) for k, v in current.items()}
            self.request_peak = {k: max(v, self.request_peak[k]) for k, v in current.items()}
            self.series.append({'monotonic_seconds': time.monotonic(), 'question_number': self.request, **current})

    def begin(self, number):
        current = base.resources.audit.memory(self.pid)
        with self.lock:
            self.request = number
            self.request_peak = current.copy()
        self.sample(current)
        return current

    def boundary(self):
        current = base.resources.audit.memory(self.pid)
        self.sample(current)
        with self.lock:
            return {'after': current, 'request_peak': self.request_peak.copy(), 'cumulative_peak': self.peak.copy()}

    def loop(self):
        next_os = time.monotonic()+1
        try:
            while not self.stop_event.wait(.05):
                current = base.resources.audit.memory(self.pid)
                self.sample(current)
                system = self.system[-1]
                if time.monotonic() >= next_os:
                    system = base.resources.observe_os()
                    self.system.append(system)
                    next_os = time.monotonic()+1
                    if not self.swap_affected and base.resources.swap_increased(self.initial, system):
                        self.swap_affected = True
                        self.notify('swap increase', {'initial': self.initial, 'observed': system})
                issue = base.resources.violation(current, system)
                if issue:
                    raise RuntimeError(issue)
        except BaseException as error:
            self.error = f'{type(error).__name__}: {error}'
            self.notify('resource monitoring failure', {'error': self.error})
            os.kill(os.getpid(), signal.SIGINT)


def curve(rows):
    if len(rows) not in CHECKPOINTS:
        raise RuntimeError('unregistered curve checkpoint')
    return {'questions': len(rows), 'peak': rows[-1]['memory']['cumulative_peak'],
            'current_after': rows[-1]['memory']['after'],
            'HTTP_seconds_per_question': sum(r['timing']['wall_seconds'] for r in rows)/len(rows),
            'nonstop_count': sum(r['timing']['finish_reason'] != 'stop' for r in rows),
            'swap': rows[-1]['OS'], 'cache': rows[-1]['cache'], 'provisional': len(rows) != 30}


def conservative_gap(low, high):
    if len(low) != 2 or len(high) != 2:
        raise RuntimeError('two repeats per condition required')
    variation = max(max(low)-min(low), max(high)-min(high))
    gap = min(high)-max(low)
    return {'conservative_gap_bytes': gap, 'largest_repeat_range_bytes': variation,
            'reduction_demonstrated': gap > variation and gap > 0,
            'rule': 'higher condition min minus lower condition max exceeds both repeat ranges'}


def pair_curves(answer, rerank):
    if set(answer) != set(rerank):
        raise RuntimeError('matched question counts required')
    return {n: {'answer': answer[n], 'rerank': rerank[n],
                'serial_pipeline_peak': {k: max(answer[n]['peak'][k], rerank[n]['peak'][k]) for k in answer[n]['peak']},
                'arithmetic_sum_of_stage_peaks': {k: answer[n]['peak'][k]+rerank[n]['peak'][k] for k in answer[n]['peak']},
                'sum_scope': 'different non-overlapping processes; arithmetic sum is not simultaneous footprint/RSS'} for n in answer}


def measure_stage(p, prepared, items, round_number, arm, answer_window, stage, replies):
    label = f'r{round_number}-{arm}-c{answer_window}-{stage}'
    window = answer_window if stage == 'answer' else 8192
    checked()
    audit.status('planned fresh process', status='running', cell=label, context=window)
    before = previous.swap_check('before fresh process')
    old_pid = base.norw.prod.environment()['qwen_servers'][0]['pid']
    pid, now = base.restart(window, label)
    try:
        os.kill(old_pid, 0)
    except ProcessLookupError:
        old_process_exited = True
    else:
        raise RuntimeError('old Qwen PID still exists; no overlapping requests')
    if now['model_sha256'] != p['model_sha256'] or now['build'] != p['build']:
        raise RuntimeError('model/build differs')
    ready = previous.swap_check('fresh ready', pid)
    log = LOCAL / f'qwen-{label}.log'
    settings = cache_settings(log.read_text(), arm, window)
    allocations = audit.allocation_table(log.read_text())
    if not allocations['required_recorded']:
        raise RuntimeError('required allocation log missing')
    startup = {'cell': label, 'round': round_number, 'arm': arm, 'answer_window': answer_window,
               'stage': stage, 'context': window, 'server_pid': pid, 'settings': settings,
               'old_server_pid': old_pid, 'old_process_exited': old_process_exited,
               'allocations': allocations, 'swap_prechecks': {'before': before, 'ready': ready},
               'heavy_processes': audit.heavy_processes(pid), 'baseline_memory': base.resources.audit.memory(pid),
               'baseline_compression': compression()}
    lm.write(lm.RESULTS / f'long-context-footprint-{label}-allocations.json', startup)
    event('allocation ready', **startup)
    maps = {'0': mapping(pid, label+'-n0')}
    watch = SeriesWatch(pid, lambda kind, data: event(kind, cell=label, stage=stage, **data))
    rows, checkpoints = [], {}
    try:
        with watch:
            for number, qid in enumerate(prepared['schedule'], 1):
                ae.gate()
                key = f'{label}|{qid}'
                original = f'128000|C-final-norw|{qid}'
                item = items[original] if stage == 'answer' else prepared['rerank_jobs'][original]
                if item['input_tokens']+384 > window:
                    raise RuntimeError('input overflow; no truncation/window rescue')
                body = ae.request(p['answer_plan'], item) if stage == 'answer' else item['body']
                if body.get('cache_prompt') is not True:
                    raise RuntimeError('registered active-slot cache request differs')
                audit.status('request', status='running', cell=label, completed=len(rows), total=30)
                start_memory = watch.begin(number)
                reply = audit.send(key, body, now, replies, watch)
                row = {'question_number': number, 'question_key_sha256': lm.digest(qid),
                       'memory': {'before': start_memory, **watch.boundary()},
                       'OS': base.resources.observe_os(), 'compression': compression(),
                       'cache': cache_observations(log.read_text()),
                       'timing': audit.timing_row(key, item, reply)}
                rows.append(row)
                ae.received(LOCAL / 'observations.jsonl', key, row)
                if row['timing']['finish_reason'] != 'stop':
                    event('nonstop finish', cell=label, stage=stage, question_number=number)
                if number in CHECKPOINTS:
                    maps[str(number)] = mapping(pid, f'{label}-n{number}')
                    checkpoints[str(number)] = curve(rows)
                    checkpoint = {'cell': label, 'round': round_number, 'arm': arm,
                                  'answer_window': answer_window, 'stage': stage, 'context': window,
                                  'curve': dict(checkpoints), 'allocations': allocations,
                                  'started_with_large_swap': ready['started_with_large_swap'],
                                  'swap_affected': watch.swap_affected or before['increment_observed'] or ready['increment_observed']}
                    lm.write(lm.RESULTS / f'long-context-footprint-{label}-interim.json', checkpoint)
                    event('question checkpoint', **checkpoint)
    finally:
        lm.write(LOCAL / f'{label}-observations.json', rows)
        lm.write(LOCAL / f'{label}-resources.json', {**watch.record(), 'memory_series50ms': watch.series})
        lm.write(LOCAL / 'responses.json', replies)
    if watch.error or len(rows) != 30:
        raise RuntimeError(watch.error or 'thirty requests incomplete')
    result = {**startup, 'curve': checkpoints, 'mapping': maps, 'peak': watch.record()['peak'],
              'swap_affected': watch.swap_affected or before['increment_observed'] or ready['increment_observed'],
              'initial_OS': watch.initial, 'last_OS': rows[-1]['OS'],
              'requests': 30, 'no_accuracy_scoring': True,
              'timing': audit.reporting.timing([r['timing'] for r in rows]),
              'log_sha256_at_completion': lm.file_hash(log)}
    lm.write(lm.RESULTS / f'long-context-footprint-{label}.json', result)
    event('stage complete', cell=label, summary=result)
    return result


def comparisons(cells):
    result = {}
    for arm in ('default', 'host-cache-off'):
        for endpoint in ('answer', 'serial_pipeline'):
            peaks = {}
            for window in (8192, 131072):
                values = [cells[f'r{r}-{arm}-c{window}']['paired_curve']['30'] for r in (1, 2)]
                peaks[window] = [v['answer']['peak']['physical_footprint_bytes'] if endpoint == 'answer'
                                 else v['serial_pipeline_peak']['physical_footprint_bytes'] for v in values]
            result[f'{arm}|{endpoint}'] = conservative_gap(peaks[8192], peaks[131072])
    for window in (8192, 131072):
        for stage in ('answer', 'rerank'):
            values = {arm: [cells[f'r{r}-{arm}-c{window}'][stage]['curve']['30']['peak']['physical_footprint_bytes']
                            for r in (1, 2)] for arm in ('default', 'host-cache-off')}
            result[f'host-cache-effect|{window}|{stage}'] = conservative_gap(values['host-cache-off'], values['default'])
    return result


def markdown(result):
    lines = ['# C-final-norw：同じ30問での8k/128k footprint時系列監査', '',
             '主指標は通常設定の回答サーバーが30問を処理するまでの累積ピークfootprint。RSSは補助。rerankを含む直列パイプラインのピークも別途示す。',
             '各設定2回・順序反転・各工程fresh PID・1並列。回答30とrerank30は別プロセスで直列。ピーク和は同時稼働値ではない。精度採点なし。', '',
             '|round|設定|回答窓|工程/実窓|問数|peak footprint GiB|peak RSS GiB|KV GiB|swap GiB|HTTP秒/問|',
             '|---:|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for label, cell in result['cells'].items():
        for stage in ('answer', 'rerank'):
            s = cell[stage]
            for n, value in s['curve'].items():
                lines.append(f'|{s["round"]}|{s["arm"]}|{s["answer_window"]}|{stage}/{s["context"]}|{n}|'
                             f'{value["peak"]["physical_footprint_bytes"]/GIB:.6f}|{value["peak"]["rss_bytes"]/GIB:.6f}|'
                             f'{s["allocations"]["bytes"]["KV"]/GIB:.6f}|{value["swap"]["swap_used_bytes"]/GIB:.6f}|'
                             f'{value["HTTP_seconds_per_question"]:.6f}|')
            lines.append(f'\n{label}/{stage}: 開始swap {s["initial_OS"]["swap_used_bytes"]/GIB:.6f}GiB、swap増加={s["swap_affected"]}。増加時の時間は参考値。\n')
    lines += ['', '## 事前登録の判定', '', '|比較|保守的gap GiB|最大反復range GiB|削減確認|', '|---|---:|---:|---|']
    for name, value in result['comparisons'].items():
        lines.append(f'|{name}|{value["conservative_gap_bytes"]/GIB:.6f}|{value["largest_repeat_range_bytes"]/GIB:.6f}|{value["reduction_demonstrated"]}|')
    lines += ['', '## 原因の確認と限界', '',
              '現行buildのhost saved-prompt cache上限8GiB、1slot、context checkpoints32は開始ログで確認。offはcache-ram=0のみで、active-slot prefix cacheとcheckpointsは残す。',
              'cache状態ログはcheckpoints込みの保存状態サイズ。キャッシュ無効化による差は上記の事前登録比較で検証し、offでも残る増加は別原因として未確認。アロケータ保持/Metal driverの詳細内訳は未確認。',
              '各再起動は旧PID消失を確認してから新PIDを作成。異なるPIDのバッファが同一プロセスへ積み上がる設計ではない。OS/driver全体の保持はこの確認だけでは否定できない。',
              'Apple XNUのfootprint定義に圧縮ledger項が含まれる。システム全体のcompressor/swapとvmmapの丸めた領域値を記録するが、Qwen個別の圧縮ledger正確値は未確認。swap/RSS/footprintを加算しない。',
              '3問値と30問値は独立した累積範囲。既存3問監査・精度30問値を今回の反復へ混ぜない。',
              '同じ10会話の既存30開発問・合成妨害記憶の範囲のみ。未知会話/日本語/実長期/意味の正しさへの一般化なし。無効化設定の精度は採点していない。',
              f'ジョブwall {result["seconds"]:.3f}秒。vmmap・hash・起動・保存のCPU時間をHTTP回答時間へ混ぜない。',
              'A128は中断のまま。256k/phase3以降へ進まない。本文・生成はignored build/evalのみ。元結果/hash不変。', '',
              'ソース：[llama.cpp同build](https://github.com/ggml-org/llama.cpp/blob/b29c606e2/tools/server/server-task.h)、'
              '[cache更新処理](https://github.com/ggml-org/llama.cpp/blob/b29c606e2/tools/server/server-task.cpp)、'
              '[Apple XNU footprint定義](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/kern/task.c)。Apple公開mainは実行OSと同一revision保証なし。']
    return '\n'.join(lines)+'\n'


def run():
    p = checked()
    if any((LOCAL / name).exists() for name in ('run-start.json', 'received.jsonl', 'inflight.json')):
        raise RuntimeError('already started; no automatic retry')
    prepared = lm.read(SOURCE / 'prepared.json')
    items = lm.read(SOURCE / '128000-C-final-norw-inputs.json')
    frozen.preflight(items, prepared['schedule'])
    if lm.digest(prepared['schedule']) != p['question_ids_sha256']:
        raise RuntimeError('frozen question order differs')
    LOCAL.mkdir(parents=True, exist_ok=True)
    lm.write(LOCAL / 'run-start.json', {'time': lm.stamp(), 'pid': os.getpid(), 'pgid': os.getpgid(0), 'plan_sha256': lm.file_hash(PLAN)})
    replies, cells = {}, {}
    tick = time.perf_counter()
    with runtime('default'):
        try:
            for r, arm, window in schedule():
                with runtime(arm):
                    answer = measure_stage(p, prepared, items, r, arm, window, 'answer', replies)
                    rerank = measure_stage(p, prepared, items, r, arm, window, 'rerank', replies)
                label = f'r{r}-{arm}-c{window}'
                cell = {'answer': answer, 'rerank': rerank,
                        'paired_curve': pair_curves(answer['curve'], rerank['curve']),
                        'same_process': False, 'simultaneous_processes': False}
                cells[label] = cell
                lm.write(lm.RESULTS / f'long-context-footprint-{label}-paired.json', cell)
                event('condition complete', cell=label, summary=cell)
            checked()
            if len(replies) != 480 or (LOCAL / 'inflight.json').exists():
                raise RuntimeError('registered 480 responses incomplete')
            with runtime():
                base.restart(131072, 'restore-original-128k')
            check_sources(p)
            result = {'time': lm.stamp(), 'plan_sha256': lm.file_hash(PLAN), 'all480_complete': True,
                      'original_hash_match': True, 'cells': cells, 'comparisons': comparisons(cells),
                      'primary': 'default answer-server cumulative peak physical footprint after thirty answers; serial pipeline peak separately',
                      'seconds': time.perf_counter()-tick, 'journal_sha256': lm.file_hash(LOCAL / 'received.jsonl'),
                      'no_accuracy_scoring': True, 'A_stays_paused': True}
            lm.write(lm.RESULTS / 'long-context-footprint-report.json', result)
            (lm.RESULTS / 'long-context-footprint-report.md').write_text(markdown(result))
            ae.received(LOCAL / 'verification.jsonl', 'complete', {'time': lm.stamp(), 'received': 480, 'sources_match': True})
            lm.write(ae.PAUSE, {'time': lm.stamp(), 'runner': 'long-context-footprint', 'reason': 'footprint audit complete; report and wait for human'})
            audit.status('footprint audit complete', status='complete', completed=480)
            event('footprint audit complete', completed=480)
        except BaseException as error:
            lm.write(LOCAL / 'responses.json', replies)
            audit.status('stopped', status='stopped', completed=len(replies), error=f'{type(error).__name__}: {error}')
            event('stopped', error=f'{type(error).__name__}: {error}', completed=len(replies))
            raise


def pause():
    lm.write(ae.PAUSE, {'time': lm.stamp(), 'runner': 'long-context-footprint', 'reason': 'human requested pause'})
    print('persistent gate saved; finish current HTTP/checkpoint, Qwen untouched', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'pause'))
    {'run': run, 'pause': pause}[parser.parse_args().command]()
