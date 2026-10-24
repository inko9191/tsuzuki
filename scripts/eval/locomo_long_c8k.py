#!/usr/bin/env python3
"""Separately registered 8k-window precision check of frozen 128k-memory inputs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import time

import locomo_long_eval as base
import locomo_long_report as report
import locomo_long_memory_audit as audit
import locomo_long_memory_audit_continue as memory

lm, ae = base.lm, base.ae
SOURCE = base.LOCAL
LOCAL = lm.LOCAL / 'long-context-c8k'
PLAN = lm.RESULTS / 'long-context-c8k-plan.json'
ARM = 'C-final-norw'
LENGTH = '128000'
CONTEXT = 8192


def preflight(items, ids):
    expected = {f'{LENGTH}|{ARM}|{qid}' for qid in ids}
    if len(ids) != 30 or len(set(ids)) != 30 or set(items) != expected:
        raise RuntimeError('same thirty distinct frozen inputs required')
    if any(item['input_tokens'] + 256 + 128 > CONTEXT for item in items.values()):
        raise RuntimeError('8k input overflow; no larger window or truncation')


def other_workers():
    for line in subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and int(parts[0]) != os.getpid() and 'python' in Path(parts[1].split()[0]).name:
            if any(tag in parts[1] for tag in (
                'locomo_long_eval.py run', 'locomo_long_continue.py bc',
                'locomo_long_continue.py resume-a', 'locomo_long_memory_audit.py run',
                'locomo_long_memory_audit_continue.py run', 'locomo_long_c8k.py run',
            )):
                raise RuntimeError('another evaluation worker exists')


def checked(require_gate_clear=True):
    if require_gate_clear:
        ae.gate()
    p = base.st.frozen(PLAN)
    for name, digest in p['source_sha256'].items():
        if lm.file_hash(lm.ROOT / name) != digest:
            raise RuntimeError('protected public source changed:' + name)
    for name, digest in p['private_source_sha256'].items():
        if lm.file_hash(lm.ROOT / name) != digest:
            raise RuntimeError('protected private source changed:' + name)
    other_workers()
    return p


@contextmanager
def isolated():
    saved = base.LOCAL
    base.LOCAL = LOCAL
    try:
        yield
    finally:
        base.LOCAL = saved


def rows_after_complete(items, replies, ids):
    preflight(items, ids)
    expected = {f'{LENGTH}|{ARM}|{qid}' for qid in ids}
    if set(replies) != expected:
        raise RuntimeError('all thirty received before scoring')
    return [base.main_row(items[f'{LENGTH}|{ARM}|{qid}'], replies[f'{LENGTH}|{ARM}|{qid}']) for qid in ids]


def finish(p, prepared, items, replies, reranks, resource, elapsed, server):
    ids = prepared['schedule']
    rows = rows_after_complete(items, replies, ids)
    for key, reply in replies.items():
        expected = lm.digest({'request': ae.request(p['answer_plan'], items[key]),
                              'plan': lm.file_hash(PLAN), 'server': server})
        if reply['digest'] != expected or reply['attempts'] != 1:
            raise RuntimeError('received request identity or attempt count differs')
        ae.checked_timings(reply)
    reference = lm.read(SOURCE / '128000-C-final-norw-completed.json')
    summary = report.condition_summary(p, prepared, LENGTH, ARM, rows, [], reranks, resource, elapsed)
    summary.update({'provisional': False, 'context_window': CONTEXT,
                    'memory_interpretation': 'single fresh8k precision run; independent two-repeat memory audit remains separate'})
    contrasts = report.paired(rows, reference['rows'])
    checked()
    result = {'time': lm.stamp(), 'plan_sha256': lm.file_hash(PLAN), 'all30_complete': True,
              'original_files_hash_match': True, 'condition': summary,
              'reference_condition': reference['summary'], 'reference_context_window': 131072,
              'contrasts_8k_minus_128k': contrasts,
              'all_four_F1_equivalent': all(contrasts[k]['f1']['equivalent_within_0.03'] for k in report.STRATA),
              'runtime_seconds': elapsed, 'answers_sha256': lm.file_hash(LOCAL / 'answer-responses.json'),
              'journal_sha256': lm.file_hash(LOCAL / 'answer-received.jsonl'),
              'rerank_search': 'same frozen cached rankings/inputs; saved rerank HTTP and CPU search/construction costs, no new calls',
              'scope': 'same thirty development questions/ten conversations, synthetic128k distractor memory; window-only additional check after prior results; no new holdout/generalization/semantic claim'}
    lm.write(LOCAL / 'scored-rows.json', rows)
    lm.write(lm.RESULTS / 'long-context-c8k-report.json', result)
    lines = ['# C-final-norw：128k合成記憶・8k回答窓の追加精度検証', '', result['scope'], '',
             '全30応答が揃った後だけ採点。会話bootstrap10000/seed0、厳密両側符号検定・ゼロ差除外・無補正。区間全体±.03内だけ同等、+.05実用幅不変。', '',
             '|種類|問数|8k語彙F1|128k語彙F1|8k棄権率|128k棄権率|8k失敗数|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for kind in report.STRATA:
        a = summary['metrics'][kind]; b = reference['summary']['metrics'][kind]
        lines.append(f'|{kind}|{a["questions"]}|{a["f1"]:.6f}|{b["f1"]:.6f}|{a["abstention"]:.6f}|{b["abstention"]:.6f}|{a["failure"]*a["questions"]:.0f}|')
    groups = sorted({qid.split('/')[0] for qid in ids})
    for kind in report.STRATA:
        lines += ['', f'## {kind}：8k−128k 語彙F1', '',
                  f'許容差±.03内: {contrasts[kind]["f1"]["equivalent_within_0.03"]}', '',
                  *report.inference.table({'C-final-norw/8192': {'C-final-norw/131072': contrasts[kind]['f1']}}, 'C-final-norw/131072', groups)]
    costs = summary['components_seconds']
    f = lambda v: '未記録' if v is None else f'{v:.6f}'
    lines += ['', '## 時間・資源', '',
              '|新回答HTTP秒/問|prefill|generation|保存済みrerank秒/問|保存済みCPU検索|保存済み入力構築|成分和秒/問|',
              '|---:|---:|---:|---:|---:|---:|---:|',
              '|' + '|'.join(f(v) for v in (costs['answer']['mean'], summary['answer_time']['prefill_seconds']['mean'],
                summary['answer_time']['generation_seconds']['mean'], costs['rerank']['mean'], costs['search']['mean'],
                costs['assembly']['mean'], costs['complete_sum']['mean'])) + '|', '',
              f'新ジョブwall {elapsed:.3f}秒（保存済みrerank/検索/入力構築の時間は含まない）。成分和は別時計の和、fresh embedding/取り込みは未計測。',
              f'RSS {resource["peak"]["rss_bytes"]/2**30:.6f}GiB、footprint {resource["peak"]["physical_footprint_bytes"]/2**30:.6f}GiB、kernel lifetime {resource["peak"]["kernel_lifetime_peak_footprint_bytes"]/2**30:.6f}GiB。',
              f'開始swap {resource["initial_OS"]["swap_used_bytes"]/2**30:.6f}GiB、増加: {resource["swap_affected"]}。precheck増加: {resource.get("swap_precheck_affected")}。増加した時間は参考値。', '',
              '## 確認できたこと・できなかったこと・主張範囲', '',
              '固定30入力・1HTTP試行・原結果hash不変。本文/生成はignored build/evalのみ。新8k回答をメモリ監査や原精度結果へ混ぜない。',
              '少数問・合成他話者妨害で検索が易しくなる可能性、未知会話/日本語/実長期/意味的正答率は未確認。A128は未完了で今回比較しない。',
              '8kと128kの同じC-final-norw入力の比較だけ。結果から条件/質問/窓/判定を変更しない。A再開/256k/新フェーズは自動開始しない。']
    (lm.RESULTS / 'long-context-c8k-report.md').write_text('\n'.join(lines) + '\n')
    return result


def run():
    p = checked()
    if any((LOCAL / n).exists() for n in ('run-start.json', 'answer-received.jsonl', 'inflight.json')):
        raise RuntimeError('already started; no automatic restart or repeat')
    prepared = lm.read(SOURCE / 'prepared.json')
    items = lm.read(SOURCE / '128000-C-final-norw-inputs.json')
    preflight(items, prepared['schedule'])
    reranks = ae.recover(SOURCE / 'rerank-responses.json', SOURCE / 'rerank-received.jsonl')
    for qid in prepared['schedule']:
        key = f'{LENGTH}|{ARM}|{qid}'
        if reranks[key]['digest'] != prepared['rerank_jobs'][key]['digest']:
            raise RuntimeError('saved frozen rerank identity differs')
    LOCAL.mkdir(parents=True, exist_ok=True)
    lm.write(LOCAL / 'run-start.json', {'time': lm.stamp(), 'pid': os.getpid(), 'pgid': os.getpgid(0), 'plan_sha256': lm.file_hash(PLAN)})
    replies = {}; tick = time.perf_counter(); resource = None
    with isolated():
        try:
            base.status('planned8k fresh server', status='running')
            before = memory.swap_check('before planned8k restart')
            pid, now = base.restart(CONTEXT, 'C-final-norw-8k')
            if now['model_sha256'] != p['model_sha256'] or now['build'] != p['build']:
                raise RuntimeError('frozen model/build differs; no requests')
            ready = memory.swap_check('fresh8k before answers', pid)
            lm.write(LOCAL / 'swap-prechecks.json', {'before': before, 'ready': ready})
            lm.write(LOCAL / 'heavy-processes.json', audit.heavy_processes(pid))
            watch = base.resources.Watch(pid, lambda kind, data: base.event(kind, {'condition': ARM, 'context': CONTEXT, 'stage': 'answer', **data}))
            try:
                with watch:
                    for qid in prepared['schedule']:
                        base.status('answer', status='running', completed=len(replies), total=30, context=CONTEXT)
                        key = f'{LENGTH}|{ARM}|{qid}'; item = items[key]
                        body = ae.request(p['answer_plan'], item)
                        digest = lm.digest({'request': body, 'plan': lm.file_hash(PLAN), 'server': now})
                        reply = base.send(key, 'answer', body, digest, item['input_tokens'], now, replies, watch)
                        if len(replies) % 25 == 0:
                            lm.write(LOCAL / 'answer-responses.json', replies)
                        finish_reason = reply['response']['choices'][0].get('finish_reason')
                        if finish_reason != 'stop':
                            base.event('nonstop finish', {'context': CONTEXT, 'stage': 'answer', 'finish_reason': finish_reason})
            finally:
                lm.write(LOCAL / 'answer-responses.json', replies)
                resource = watch.record()
                resource['swap_precheck_affected'] = before['increment_observed'] or ready['increment_observed']
                if resource['swap_precheck_affected']:
                    resource['time_interpretation'] = 'reference only; precheck swap increase'
                lm.write(LOCAL / 'resources.json', resource)
            if watch.error:
                raise RuntimeError(watch.error)
            if len(replies) != 30:
                raise RuntimeError('thirty answers incomplete')
            checked()
            base.status('planned restore original128k', status='running', completed=30)
            base.restart(131072, 'restore-original-128k')
        except BaseException as error:
            lm.write(LOCAL / 'answer-responses.json', replies)
            base.status('stopped', status='stopped', error=f'{type(error).__name__}: {error}', completed=len(replies))
            base.event('stopped', {'error': f'{type(error).__name__}: {error}'})
            raise
    try:
        result = finish(p, prepared, items, replies, reranks, resource, time.perf_counter() - tick, now)
        with isolated():
            base.status('evaluation complete', status='complete', completed=30)
            base.event('evaluation complete', {'completed': 30, 'all_four_F1_equivalent': result['all_four_F1_equivalent']})
        lm.write(ae.PAUSE, {'time': lm.stamp(), 'reason': 'C8k precision complete; report then wait for human; A not automatically resumed', 'runner': 'long-context-c8k'})
    except BaseException as error:
        with isolated():
            base.status('stopped', status='stopped', error=f'{type(error).__name__}: {error}')
            base.event('stopped', {'error': f'{type(error).__name__}: {error}'})
        raise


def pause():
    lm.write(ae.PAUSE, {'time': lm.stamp(), 'reason': 'human requested pause', 'runner': 'long-context-c8k'})
    print('pause gate saved; finish current HTTP then checkpoint/stop, Qwen untouched', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'pause'))
    command = parser.parse_args().command
    {'run': run, 'pause': pause}[command]()
