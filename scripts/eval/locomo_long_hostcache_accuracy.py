#!/usr/bin/env python3
"""Registered host saved-prompt-cache-only accuracy check; thirty fixed inputs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
import time

import locomo_long_footprint as fp
import locomo_long_report as reporting

base, lm, ae, audit = fp.base, fp.lm, fp.ae, fp.audit
LOCAL = lm.LOCAL / 'long-context-hostcache-accuracy'
PLAN = lm.RESULTS / 'long-context-hostcache-accuracy-plan.json'
PREFIX = 'long-context-hostcache-accuracy'
SOURCE = fp.SOURCE
CONTEXT = 8192


@contextmanager
def runtime(off=True):
    saved = (base.LOCAL, audit.LOCAL, audit.PLAN)
    names = ('LLAMA_ARG_CACHE_RAM', 'LLAMA_ARG_LOG_VERBOSITY')
    env = {name: os.environ.get(name) for name in names}
    base.LOCAL = audit.LOCAL = LOCAL
    audit.PLAN = PLAN
    for name in names:
        os.environ.pop(name, None)
    if off:
        os.environ['LLAMA_ARG_CACHE_RAM'] = '0'
        os.environ['LLAMA_ARG_LOG_VERBOSITY'] = '4'
    try:
        yield
    finally:
        base.LOCAL, audit.LOCAL, audit.PLAN = saved
        for name, value in env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def checked():
    ae.gate()
    p = base.st.frozen(PLAN)
    fp.check_sources(p)
    fp.other_workers()
    return p


def answer_match(new, old):
    a, error_a = base.am.parse(new['response'])
    b, error_b = base.am.parse(old['response'])
    return {
        'answer_exact': not error_a and not error_b and a['answer'] == b['answer'],
        'parsed_payload_exact': not error_a and not error_b and a == b,
        'raw_content_exact': new['response']['choices'][0]['message'].get('content') ==
                             old['response']['choices'][0]['message'].get('content'),
        'new_failure': error_a, 'reference_failure': error_b,
    }


def compare(rows, replies, refs):
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)) or any(set(ids) - set(ref['replies']) for ref in refs.values()):
        raise RuntimeError('comparison identities differ')
    result, differences = {}, []
    new_metrics = reporting.aggregates(rows)
    for name, ref in refs.items():
        old_rows = [ref['rows'][qid] for qid in ids]
        metrics = reporting.aggregates(old_rows)
        by_question = {}
        for row in rows:
            qid = row['id']
            match = answer_match(replies[qid], ref['replies'][qid])
            match['f1_difference'] = row['f1'] - ref['rows'][qid]['f1']
            by_question[qid] = match
            if not match['answer_exact'] or not match['parsed_payload_exact']:
                differences.append({'id': qid, 'reference': name, **match,
                    'new_content': replies[qid]['response']['choices'][0]['message'].get('content'),
                    'reference_content': ref['replies'][qid]['response']['choices'][0]['message'].get('content')})
        result[name] = {
            'question_count': len(ids), 'by_question': by_question,
            'answer_exact_count': sum(m['answer_exact'] for m in by_question.values()),
            'all_answers_exact': all(m['answer_exact'] for m in by_question.values()),
            'all_payloads_exact': all(m['parsed_payload_exact'] for m in by_question.values()),
            'all_raw_contents_exact': all(m['raw_content_exact'] for m in by_question.values()),
            'all_question_F1_differences_zero': all(m['f1_difference'] == 0 for m in by_question.values()),
            'reference_metrics': metrics,
            'F1_differences': {k: new_metrics[k]['f1'] - metrics[k]['f1'] for k in new_metrics},
            'all_stratum_F1_differences_zero': all(new_metrics[k]['f1'] == metrics[k]['f1'] for k in new_metrics),
        }
    return result, differences


def references(ids):
    paths = {'default8k': lm.LOCAL / 'long-context-c8k', 'default128k': SOURCE}
    result = {}
    for name, path in paths.items():
        all_replies = ae.recover(path / 'answer-responses.json', path / 'answer-received.jsonl')
        rows = lm.read(path / 'scored-rows.json') if name == 'default8k' else lm.read(
            path / '128000-C-final-norw-completed.json')['rows']
        index = {r['id']: r for r in rows}
        if len(index) != 30 or set(index) != set(ids):
            raise RuntimeError('reference thirty identities differ')
        result[name] = {'rows': index, 'replies': {
            qid: all_replies[f'128000|C-final-norw|{qid}'] for qid in ids}}
    return result


def memory_comparison(checkpoints, p):
    result = {}
    for n, now in checkpoints.items():
        peaks = [lm.read(lm.ROOT / name)['curve'][n]['peak'] for name in p['memory_reference_files']]
        result[n] = {}
        for metric in ('physical_footprint_bytes', 'rss_bytes'):
            low, high = min(v[metric] for v in peaks), max(v[metric] for v in peaks)
            current = now['peak'][metric]
            result[n][metric] = {'new': current, 'prior_repeat_min': low, 'prior_repeat_max': high,
                'inside_prior_repeat_range': low <= current <= high,
                'difference_from_prior_repeat_max': current - high,
                'scope': 'descriptive prior two-repeat range; no invented tolerance or long-use bound'}
    return result


def markdown(result):
    lines = ['# Host-cache-off 8k：追加精度確認', '',
        '少数の同じ10会話・開発30問・合成他話者妨害。独立holdout/一般化ではない。時間はswapの影響があるため参考値。', '',
        '|条件|種類|問数|語彙F1|棄権率|失敗数|', '|---|---|---:|---:|---:|---:|']
    conditions = {'host-cache-off8k': result['metrics'], **{
        name: value['reference_metrics'] for name, value in result['comparisons'].items()}}
    for name, metrics in conditions.items():
        for kind in reporting.STRATA:
            if kind in metrics:
                m = metrics[kind]
                lines.append(f'|{name}|{kind}|{m["questions"]}|{m["f1"]:.6f}|{m["abstention"]:.6f}|{m["failure"]*m["questions"]:.0f}|')
    lines += ['', '|通常設定|回答完全一致|payload完全一致|問ごとF1差ゼロ|全種類F1差ゼロ|', '|---|---:|---|---|---|']
    for name, value in result['comparisons'].items():
        lines.append(f'|{name}|{value["answer_exact_count"]}/{result["questions"]}|{value["all_payloads_exact"]}|'
                     f'{value["all_question_F1_differences_zero"]}|{value["all_stratum_F1_differences_zero"]}|')
    lines += ['', '回答本文・evidence差分はignored private differences.json。公開JSONは問IDと一致判定だけ。', '',
        '|問数|累積peak footprint GiB|peak RSS GiB|KV GiB|swap GiB|回答HTTP秒/問|',
        '|---:|---:|---:|---:|---:|---:|']
    for n, v in result['curve'].items():
        lines.append(f'|{n}|{v["peak"]["physical_footprint_bytes"]/fp.GIB:.6f}|{v["peak"]["rss_bytes"]/fp.GIB:.6f}|'
                     f'{result["allocations"]["bytes"]["KV"]/fp.GIB:.3f}|{v["swap"]["swap_used_bytes"]/fp.GIB:.6f}|{v["HTTP_seconds_per_question"]:.6f}|')
    lines += ['', '主見出しは30問処理までのピークfootprint、RSS補助。1/3/10問は途中の累積値。',
        '既存監査host-cache-off8kの各問数・2反復範囲との比較はJSON memory_comparison。範囲外を一致と呼ばない。',
        'host保存キャッシュだけを無効化し、cache_prompt=true/active-slot/checkpointsは不変。rerankの保存結果と原入力を再利用し、新rerank/検索/取り込みなし。',
        '元精度・本番・監査結果と凍結コードは不変。測定内の一致のみで、未知会話/日本語/実長期/意味の正しさや全運用への一般化はしない。',
        'A128/256k/新フェーズは開始しない。']
    return '\n'.join(lines) + '\n'


def run():
    p = checked()
    if any((LOCAL / name).exists() for name in ('run-start.json', 'received.jsonl', 'inflight.json')):
        raise RuntimeError('already started; no retry/overwrite')
    prepared = lm.read(SOURCE / 'prepared.json')
    ids = prepared['schedule']
    items = lm.read(SOURCE / '128000-C-final-norw-inputs.json')
    fp.frozen.preflight(items, ids)
    if lm.digest(ids) != p['question_ids_sha256']:
        raise RuntimeError('question order changed')
    refs = references(ids)
    reranks = ae.recover(SOURCE / 'rerank-responses.json', SOURCE / 'rerank-received.jsonl')
    for qid in ids:
        key = f'128000|C-final-norw|{qid}'
        if reranks[key]['digest'] != prepared['rerank_jobs'][key]['digest']:
            raise RuntimeError('saved rerank digest differs')
        if lm.digest(ae.request(p['answer_plan'], items[key])) != p['request_body_sha256'][qid]:
            raise RuntimeError('request body differs')
    LOCAL.mkdir(parents=True, exist_ok=True)
    lm.write(LOCAL / 'run-start.json', {'time': lm.stamp(), 'pid': os.getpid(), 'pgid': os.getpgid(0), 'plan_sha256': lm.file_hash(PLAN)})
    tick = time.perf_counter()
    replies, rows, observations, checkpoints = {}, [], [], {}
    with runtime():
        try:
            audit.status('fresh host-cache-off8k', status='running')
            before = fp.previous.swap_check('before fresh off8k process')
            old_pid = base.norw.prod.environment()['qwen_servers'][0]['pid']
            pid, server = base.restart(CONTEXT, 'host-cache-off8k')
            try:
                os.kill(old_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('old Qwen still exists')
            if server['model_sha256'] != p['model_sha256'] or server['build'] != p['build']:
                raise RuntimeError('model/build differs')
            ready = fp.previous.swap_check('fresh off8k ready', pid)
            log = LOCAL / 'qwen-host-cache-off8k.log'
            settings = fp.cache_settings(log.read_text(), 'host-cache-off', CONTEXT)
            allocations = audit.allocation_table(log.read_text())
            if not allocations['required_recorded']:
                raise RuntimeError('required allocation log missing')
            startup = {'settings': settings, 'allocations': allocations, 'server': server,
                'old_pid_exited': True, 'swap_prechecks': {'before': before, 'ready': ready},
                'heavy_processes': audit.heavy_processes(pid)}
            lm.write(LOCAL / 'startup.json', startup)
            audit.event('allocation ready', **startup)
            audit.mapping_snapshot(pid, 'n0')
            watch = fp.SeriesWatch(pid, lambda kind, data: audit.event(kind, stage='answer', **data))
            try:
                with watch:
                    for number, qid in enumerate(ids, 1):
                        checked()
                        item = items[f'128000|C-final-norw|{qid}']
                        audit.status('answer', status='running', completed=len(rows), total=30)
                        start = watch.begin(number)
                        reply = audit.send(qid, ae.request(p['answer_plan'], item), server, replies, watch)
                        obs = {'question_number': number, 'question_key_sha256': lm.digest(qid),
                            'memory': {'before': start, **watch.boundary()}, 'OS': base.resources.observe_os(),
                            'compression': fp.compression(), 'cache': fp.cache_observations(log.read_text()),
                            'timing': audit.timing_row(qid, item, reply)}
                        observations.append(obs)
                        ae.received(LOCAL / 'observations.jsonl', qid, obs)
                        rows.append(base.main_row(item, reply))
                        if rows[-1]['failure']:
                            audit.event('generation failure', id=qid, failure_kind=rows[-1]['failure_kind'])
                        if number in fp.CHECKPOINTS:
                            audit.mapping_snapshot(pid, f'n{number}')
                            checkpoints[str(number)] = fp.curve(observations)
                            contrasts, differences = compare(rows, replies, refs)
                            lm.write(LOCAL / 'differences.json', differences)
                            result = {'time': lm.stamp(), 'provisional': number != 30, 'questions': number,
                                'metrics': reporting.aggregates(rows), 'comparisons': contrasts, 'curve': dict(checkpoints),
                                'allocations': allocations, 'startup': startup,
                                'memory_comparison': memory_comparison(checkpoints, p), 'time_is_reference': True,
                                'scope': p['scope']}
                            lm.write(lm.RESULTS / f'{PREFIX}-n{number}-interim.json', result)
                            audit.event('question checkpoint', questions=number, public_file=f'{PREFIX}-n{number}-interim.json')
            finally:
                lm.write(LOCAL / 'responses.json', replies)
                lm.write(LOCAL / 'scored-rows.json', rows)
                lm.write(LOCAL / 'resources.json', {**watch.record(), 'memory_series50ms': watch.series})
            if watch.error or len(replies) != 30 or (LOCAL / 'inflight.json').exists():
                raise RuntimeError(watch.error or 'registered responses incomplete')
            checked()
            result.update(provisional=False, all30_complete=True, original_hash_match=True,
                plan_sha256=lm.file_hash(PLAN), peak_memory=watch.record()['peak'],
                swap_affected=watch.swap_affected or before['increment_observed'] or ready['increment_observed'],
                answer_time=reporting.timing(rows), journal_sha256=lm.file_hash(LOCAL / 'received.jsonl'))
            with runtime(off=False):
                base.restart(131072, 'restore-original128k')
            fp.check_sources(p)
            result['wall_seconds'] = time.perf_counter() - tick
            lm.write(lm.RESULTS / f'{PREFIX}-report.json', result)
            (lm.RESULTS / f'{PREFIX}-report.md').write_text(markdown(result))
            audit.status('evaluation complete', status='complete', completed=30)
            audit.event('evaluation complete', completed=30)
            lm.write(ae.PAUSE, {'time': lm.stamp(), 'runner': PREFIX,
                'reason': 'host-cache-off accuracy complete; report then wait for missing second task; A remains paused'})
        except BaseException as error:
            lm.write(LOCAL / 'responses.json', replies)
            audit.status('stopped', status='stopped', completed=len(replies), error=f'{type(error).__name__}: {error}')
            audit.event('stopped', error=f'{type(error).__name__}: {error}')
            raise


def pause():
    lm.write(ae.PAUSE, {'time': lm.stamp(), 'runner': PREFIX, 'reason': 'human requested pause'})
    print('gate saved; current HTTP then checkpoint/stop; Qwen untouched', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'pause'))
    {'run': run, 'pause': pause}[parser.parse_args().command]()
