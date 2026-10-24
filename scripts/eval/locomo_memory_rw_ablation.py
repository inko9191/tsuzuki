#!/usr/bin/env python3
"""Registered development-only rw removal; preserves frozen stages and production."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error

import numpy as np

import locomo_answer_eval as ae
import locomo_answer_production as production
import locomo_memory_reporting as reporting

lm, st, mm, rp = ae.lm, ae.st, ae.mm, ae.rp
LOCAL = lm.LOCAL / 'rw-ablation'
PLAN = lm.RESULTS / 'rw-ablation-plan.json'
DOC = lm.ROOT / 'docs/MEMORY_RW_ABLATION_PREREG.md'
NO = mm.config(['p2', 'fact'])
YES = mm.config(['p2', 'rw', 'fact'])
NO_R = mm.config(NO['units'], rerank='qwen')
YES_R = mm.config(YES['units'], rerank='qwen')
CONFIGS = (NO, NO_R, YES, YES_R)
STRATA = ('type:multiple_evidence', 'overall', 'type:single_fact', 'type:temporal')
LIMIT = .03


def status(phase, **extra):
    lm.write(LOCAL / 'status.json', {'time': lm.stamp(), 'pid': os.getpid(),
        'pgid': os.getpgid(0), 'phase': phase, **extra})


def timing_sample(ids):
    return sorted(ids, key=lambda q: hashlib.sha256(('rw-ablation-timing-v1' + q).encode()).hexdigest())[:302]


def equivalent(ci):
    return ci is not None and -LIMIT <= ci[0] <= ci[1] <= LIMIT


def register():
    ae.gate()
    st.check_plan()
    data, _ = lm.load_dev()
    tracked = subprocess.check_output(['git', 'ls-files', 'results/eval/memory', 'scripts/eval',
        'docs/MEMORY_ANSWER_EVAL.md'], text=True).splitlines()
    p = {'created_at': lm.stamp(), 'scope': 'development1233 only; no holdout or answer evaluation',
         'configs': list(CONFIGS), 'questions': len(data['queries']), 'equivalence_margin': LIMIT,
         'resamples': 10000, 'seed': 0, 'timing302_ids': timing_sample([q['id'] for q in data['queries']]),
         'primary': 'multiple_evidence recall@T512: rw-free reranked minus frozen rw-containing reranked',
         'all_metric_equivalence': 'all24 pointwise intervals must be contained in[-.03,+.03]',
         'generation': st.Qwen.offline().settings, 'context': 8192, 'restore_context': 65536,
         'original_files_sha256': {n: lm.file_hash(lm.ROOT / n) for n in tracked},
         'private_sources_sha256': {n: lm.file_hash(lm.LOCAL / n) for n in
              ('dev.json', 'corpus.json', 'stage2-dev-rows.json', 'stage4-dev-rows.json',
               'stage5-dev-rows.json', 'gen/fact.json', 'gen/rerank.json')},
         'code_sha256': lm.file_hash(Path(__file__)), 'document_sha256': lm.file_hash(DOC),
         'estimate_seconds': [5400, 6600], 'failure_policy': 'one attempt; durable received journal; '
              'same-digest dev reuse only; unknown inflight stops; original parse fallback',
         'pause': 'persistent user-pause gate; SIGINT dedicated group; Qwen remains running',
         'quality': 'retrieval only, no new selection, no frozen result update'}
    if PLAN.exists():
        raise RuntimeError('plan already exists; no overwrite')
    lm.write(PLAN, p)
    print('registered; commit document, code, tests and plan before prepare/run', flush=True)


def checked():
    ae.gate()
    p = st.frozen(PLAN)
    if p['code_sha256'] != lm.file_hash(Path(__file__)) or p['document_sha256'] != lm.file_hash(DOC):
        raise RuntimeError('registered code/document changed')
    for name, expected in p['original_files_sha256'].items():
        if lm.file_hash(lm.ROOT / name) != expected:
            raise RuntimeError(f'original file changed: {name}')
    for name, expected in p['private_sources_sha256'].items():
        if lm.file_hash(lm.LOCAL / name) != expected:
            raise RuntimeError(f'original cache changed: {name}')
    return p


def quiet():
    ae.gate()
    env = production.environment()
    for line in subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or int(parts[0]) == os.getpid():
            continue
        cmd = parts[1]
        if 'python' in Path(cmd.split()[0]).name and any(tag in cmd for tag in (
                'egemma_server', 'locomo_memory_stages.py generate', 'locomo_memory_stages.py run',
                'locomo_answer_eval.py generate', 'locomo_answer_production.py run',
                'locomo_answer_audit_measure.py run', 'locomo_memory_rw_ablation.py run',
                'locomo_memory_resume.py')):
            raise RuntimeError(f'other GPU evaluation/backend process: {parts[0]}')
    if any(s.get('is_processing') for s in st.http_json('http://127.0.0.1:8080/slots')):
        raise RuntimeError('Qwen is busy; stop without overlapping work')
    return env


def server(context=None):
    expected = st.Qwen.offline().settings['server']
    now = st.model_identity(expected['url'], None)
    if any(now[k] != expected[k] for k in ('model_sha256', 'model_id', 'build')):
        raise RuntimeError('Qwen model/build identity mismatch')
    if context is not None and now['n_ctx'] != context:
        raise RuntimeError(f'expected context{context}, got{now["n_ctx"]}')
    return now


def switch(context):
    """Plan-only healthy idle server change; all records stay in the new audit directory."""
    now = server()
    env = quiet()
    if now['n_ctx'] == context:
        return
    if {now['n_ctx'], context} != {8192, 65536}:
        raise RuntimeError('unexpected context transition')
    command = lm.read(lm.RESULTS / 'answer-pilot-server.json')['new_args'].copy()
    expected = command.copy()
    expected[expected.index('-c') + 1] = str(now['n_ctx'])
    instance = env['qwen_servers'][0]
    if instance['command'] != ' '.join(expected):
        raise RuntimeError('cannot identify the exact registered server')
    if lm.file_hash(Path(command[command.index('-m') + 1])) != now['model_sha256']:
        raise RuntimeError('fresh model hash mismatch')
    ae.gate()
    tick = time.perf_counter()
    os.kill(instance['pid'], signal.SIGTERM)
    for _ in range(100):
        try:
            os.waitpid(instance['pid'], os.WNOHANG)
        except ChildProcessError:
            pass
        try:
            os.kill(instance['pid'], 0)
        except ProcessLookupError:
            break
        time.sleep(.1)
    else:
        raise RuntimeError('planned stop failed; no duplicate start')
    command[command.index('-c') + 1] = str(context)
    with (LOCAL / f'qwen-{context}.log').open('a') as stream:
        proc = subprocess.Popen(command, cwd=lm.ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
    for _ in range(60):
        if proc.poll() is not None:
            raise RuntimeError('planned restart failed; no retry')
        try:
            new = server(context)
            break
        except (urllib.error.URLError, ConnectionError):
            time.sleep(1)
    else:
        raise RuntimeError('planned server not ready; no retry')
    path = LOCAL / 'context-switches.json'
    records = lm.read(path) if path.exists() else []
    records.append({'time': lm.stamp(), 'old': now, 'new': new, 'args': command,
        'pid': proc.pid, 'seconds': time.perf_counter() - tick, 'fresh_model_hash_verified': True})
    lm.write(path, records)


def cached(texts, side, identity):
    def missing(_):
        raise RuntimeError('required vector cache missing; stop without recomputing')
    return lm.cached_vectors(st.VARIANT, side, texts, identity, missing)


def row(query, order, units, data, sessions, tokens, headers):
    return {'id': query['id'], 'category': query['category'],
        'type': lm.query_type(query, data['qrels'][query['id']]),
        'ranking': [units[i]['id'] for i in order[:max(mm.KS)]],
        **mm.score(order, units, set(data['qrels'][query['id']]), sessions, tokens, headers, False)}


def require_reproduction(rows, baseline):
    if rows != baseline:
        raise RuntimeError('rw-free dense ranks/metrics differ from stage4; no conditions changed')


def prepare():
    p = checked()
    quiet()
    server()
    path = LOCAL / 'inputs.json'
    if path.exists():
        if lm.read(path)['plan_sha256'] != lm.file_hash(PLAN):
            raise RuntimeError('prepared plan differs')
        return
    tick = time.perf_counter()
    status('preparing development-only inputs', status='running')
    data, manifest = lm.load_dev()
    turns = mm.turn_table(data['corpus'], st.load_raw(manifest))
    qwen = st.Qwen.offline()
    facts, generation = st.generated('fact', data, turns, qwen)
    units = st.build_units(NO['units'], turns, {'fact': facts})
    from egemma_server.tokenizer import PromptTokenizer
    tokenizer = PromptTokenizer(rp.V1_ST_DIR)
    for side in ('query', 'document'):
        if tokenizer.prompt_for(side) != lm.PROTOCOL['prompts'][side]:
            raise RuntimeError('embedding tokenizer prompts differ')
    identity = lm.fingerprint(st.VARIANT)
    if identity != lm.read(lm.RESULTS / 'final-config.json')['embedding_artifact_sha256']:
        raise RuntimeError('embedding artifacts differ')
    vectors = np.zeros((len(units), 768), np.float32)
    cache_info = {}
    for name in NO['units']:
        ids = [i for i, u in enumerate(units) if u['set'] == name]
        texts = rp.fit_like_st(tokenizer, [lm.PROTOCOL['prompts']['document'] + units[i]['text'] for i in ids])
        vectors[ids], cache_info[name] = cached(texts, f'u-{name}', identity)
    texts = rp.fit_like_st(tokenizer, [lm.PROTOCOL['prompts']['query'] + q['text'] for q in data['queries']])
    qvecs, cache_info['orig_query'] = cached(texts, 'q-dev', identity)
    count = lambda texts: [len(e.ids) for e in tokenizer.tokenizer.encode_batch(texts, add_special_tokens=False)]
    tokens = dict(zip([t['id'] for t in turns], count([t['text'] for t in turns])))
    dates = {t['session']: t['date'] for t in turns}
    headers = dict(zip(dates, count([f'Date: {v}' for v in dates.values()])))
    sessions = {t['id']: t['session'] for t in turns}
    index = mm.Index(units, vectors)
    queries, rows = [], []
    for q, v in zip(data['queries'], qvecs):
        ae.gate()
        order = mm.distinct([i for i, _ in index.rank(NO, q['group'], [(v, q['text'])])], units)
        rows.append(row(q, order, units, data, sessions, tokens, headers))
        chat = production.rerank_chat(q, order, units, turns, NO)
        queries.append({'query': q, 'order': order, 'messages': chat,
            'digest': qwen.item_digest('rerank', chat)})
    require_reproduction(rows, lm.read(lm.LOCAL / 'stage4-dev-rows.json')[NO['id']])
    url = qwen.url
    for job in queries:
        ae.gate()
        prompt = st.http_json(url + '/apply-template', {'messages': job['messages'],
            'chat_template_kwargs': qwen.settings['request']['chat_template_kwargs']})['prompt']
        n = len(st.http_json(url + '/tokenize', {'content': prompt, 'add_special': True,
                                              'parse_special': True})['tokens'])
        production.ensure_length(n, mm.PROMPTS['rerank']['max_tokens'], p['context'])
        job['input_tokens'] = n
    old = lm.read(lm.LOCAL / 'gen/rerank.json')
    by_digest = {}
    for entry in old.values():
        if entry['digest'] in by_digest and by_digest[entry['digest']]['reply'] != entry['reply']:
            raise RuntimeError('conflicting old dev replies for an identical request')
        by_digest[entry['digest']] = entry
    shared = {j['query']['id']: {'digest': j['digest'], 'reply': by_digest[j['digest']]['reply'],
        'source': 'identical original dev digest; HTTP timing unrecorded'} for j in queries
        if j['digest'] in by_digest}
    prepared = {'plan_sha256': lm.file_hash(PLAN), 'queries': queries, 'units': units,
        'data': data, 'sessions': sessions, 'tokens': tokens, 'headers': headers, 'shared': shared,
        'unreranked_rows': rows, 'generation': generation, 'cache': cache_info,
        'prepared_at': lm.stamp(), 'prepare_seconds': time.perf_counter() - tick,
        'all1233_unreranked_rows_match': True,
        'max_input_tokens': max(j['input_tokens'] for j in queries)}
    lm.write(path, prepared)
    status('development input preflight complete', status='prepared', questions=len(queries),
        shared=len(shared), max_input_tokens=prepared['max_input_tokens'])
    report(None, prepared)


def replies_for(prepared):
    replies = ae.recover(LOCAL / 'responses.json', LOCAL / 'received.jsonl')
    jobs = {j['query']['id']: j for j in prepared['queries']}
    for key, entry in replies.items():
        if key not in jobs or entry['digest'] != jobs[key]['digest']:
            raise RuntimeError('received reply not from registered request')
    inflight = LOCAL / 'inflight.json'
    if inflight.exists():
        record = lm.read(inflight)
        if record['id'] not in replies:
            raise RuntimeError('unknown inflight; no automatic resend')
        inflight.unlink()
    return replies


def run():
    checked()
    old_status = lm.read(LOCAL / 'status.json') if (LOCAL / 'status.json').exists() else {}
    if old_status.get('status') in ('stopped', 'complete'):
        raise RuntimeError('prior stopped/completed run; no automatic restart')
    prepared = lm.read(LOCAL / 'inputs.json')
    replies = replies_for(prepared)
    for key, entry in prepared['shared'].items():
        if key not in replies:
            replies[key] = entry
            ae.received(LOCAL / 'received.jsonl', key, entry)
    switch(8192)
    qwen = st.Qwen()
    tick = time.perf_counter()
    status('reranking rw-free development queries', status='running', received=len(replies), total=1233)
    try:
        for job in prepared['queries']:
            ae.gate()
            if job['query']['id'] in replies:
                continue
            quiet()
            server(8192)
            key = job['query']['id']
            lm.write(LOCAL / 'inflight.json', {'time': lm.stamp(), 'id': key, 'digest': job['digest']})
            started = time.perf_counter()
            response = st.http_json(qwen.url + '/v1/chat/completions',
                qwen.request('rerank', job['messages']), timeout=900)
            seconds = time.perf_counter() - started
            entry = {'digest': job['digest'], 'response': response, 'wall_seconds': seconds,
                'http_attempts': 1, 'source': 'new development request'}
            replies[key] = entry
            ae.received(LOCAL / 'received.jsonl', key, entry)
            (LOCAL / 'inflight.json').unlink()
            if len(replies) % 25 == 0:
                lm.write(LOCAL / 'responses.json', replies)
            status('reranking rw-free development queries', status='running', received=len(replies),
                total=1233, elapsed_seconds=time.perf_counter() - tick)
            if len(replies) % 100 == 0:
                print(f'{lm.stamp()} received{len(replies)}/1233', flush=True)
        lm.write(LOCAL / 'rerank-runtime.json', {'seconds': time.perf_counter() - tick,
            'finished_at': lm.stamp(), 'environment': quiet()})
        # No quality analysis until the fixed new condition has all responses.
        status('rw-free rerank generation complete', status='generated', received=len(replies), total=1233)
        switch(65536)
        checked()
        report(replies, prepared)
        status('rw removal evaluation complete', status='complete', received=len(replies), total=1233)
    except BaseException as error:
        status('rw removal evaluation stopped', status='stopped', received=len(replies), total=1233,
               error=f'{type(error).__name__}: {error}')
        raise
    finally:
        lm.write(LOCAL / 'responses.json', replies)


def costs(replies, p):
    old = lm.read(lm.RESULTS / 'generation-outputs.json')
    audited = lm.read(lm.RESULTS / 'answer-audit-performance.json')
    embeds = {name: sum(r['seconds'] for r in audited['ingestion_embedding'] if r['set'] == name)
              for name in ('p2', 'rw', 'fact')}
    fact_main = old['fact']['seconds_this_call']
    fact_initial = audited['fact_accounting']['sample_seconds_actual']
    rw = old['rw']['seconds_this_call']
    without = fact_main + fact_initial + embeds['p2'] + embeds['fact']
    with_rw = without + rw + embeds['rw']
    old_stage = lm.read(lm.RESULTS / 'stage5-dev.json')['configs'][YES_R['id']]['seconds']
    result = {'ingestion': {'rw_generation_actual': rw, 'fact_main_generation_actual': fact_main,
        'fact_initial3_actual': fact_initial, 'embedding_actual': embeds,
        'without_rw_known_component_sum': without, 'with_rw_known_component_sum': with_rw,
        'saved_known_component_sum': with_rw - without,
        'note': 'Saved original generation wall + separate embedding audit. Component sum across runs; '
                'startup/text preparation/save unrecorded. Initial fact sample GPU concurrency unrecorded; '
                'conversation generation allocations are estimates.',
        'per_conversation_generation_estimates': audited['fact_accounting']['per_conversation'],
        'per_conversation_embedding_actual': audited['ingestion_embedding']},
        'with_rw_rerank': {'dev1233_stage_seconds_actual': old_stage,
            'stage_amortized_seconds': old_stage / 1233, 'equivalent302_stage_seconds_estimate': old_stage / 1233 * 302,
            'individual_HTTP_seconds': 'unrecorded; stage includes scoring/tokenizer/cache/overhead'}}
    if replies is not None:
        fresh = [e['wall_seconds'] for e in replies.values() if 'wall_seconds' in e]
        sample = [replies[q]['wall_seconds'] for q in p['timing302_ids'] if 'wall_seconds' in replies[q]]
        result['without_rw_rerank'] = {'HTTP_calls_measured': len(fresh),
            'shared_calls_HTTP_time_unrecorded': len(replies) - len(fresh),
            'HTTP_mean_seconds_actual': float(np.mean(fresh)) if fresh else None,
            'HTTP_total_seconds_actual': sum(fresh),
            'HTTP_equivalent302_seconds_estimate': float(np.mean(fresh)) * 302 if fresh else None,
            'fixed_dev302_HTTP_measured': len(sample), 'fixed_dev302_HTTP_total_actual': sum(sample),
            'fixed_dev302_HTTP_mean_actual': float(np.mean(sample)) if sample else None,
            'fixed_dev302_all_times_recorded': len(sample) == 302,
            'stage': lm.read(LOCAL / 'rerank-runtime.json')}
    return result


def report(replies, prepared):
    p = checked()
    candidates = {NO['id']: prepared['unreranked_rows'],
        YES['id']: lm.read(lm.LOCAL / 'stage4-dev-rows.json')[YES['id']],
        YES_R['id']: lm.read(lm.LOCAL / 'stage5-dev-rows.json')[YES_R['id']]}
    failures, finish = 0, Counter()
    if replies is not None:
        if len(replies) != p['questions']:
            raise RuntimeError('refuse to score an incomplete rerank condition')
        output = []
        for job in prepared['queries']:
            entry = replies[job['query']['id']]
            if 'response' in entry:
                choice = entry['response']['choices'][0]
                content = choice['message'].get('content') or ''
                finish[choice.get('finish_reason', 'unknown')] += 1
            else:
                content = entry['reply']
                finish['old_digest_shared_finish_unrecorded'] += 1
            head = job['order'][:mm.RERANK_CANDIDATES]
            value = mm.parse('rerank', content, n_passages=len(head))
            failures += value is None
            order = mm.distinct(production.reranked_order(job['order'], value), prepared['units'])
            output.append(row(job['query'], order, prepared['units'], prepared['data'], prepared['sessions'],
                              prepared['tokens'], prepared['headers']))
        candidates[NO_R['id']] = output
    refs = lm.read(lm.LOCAL / 'stage2-dev-rows.json')
    comparisons = {base: reporting.comparisons(candidates, refs, base) for base in (YES['id'], YES_R['id'])}
    summary = {}
    for label in STRATA:
        summary[label] = {}
        for metric in reporting.METRICS:
            summary[label][metric] = {}
            for arm, base in ((NO['id'], YES['id']), (NO_R['id'], YES_R['id'])):
                if arm not in candidates:
                    continue
                stats = comparisons[base]['strata'][label][metric][arm][base]
                summary[label][metric][arm] = {**stats,
                    'equivalence_within_margin': equivalent(stats['conversation_ci95'])}
    result = {'created_at': lm.stamp(), 'plan_sha256': lm.file_hash(PLAN),
        'registration_commit': subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', str(PLAN)], text=True).strip(),
        'scope': p['scope'], 'reranked_complete': replies is not None, 'frozen_selection_changed': False,
        'holdout_opened': False, 'comparisons': comparisons, 'direct_rw_removal': summary,
        'equivalence_margin': LIMIT, 'parse_fallback': failures, 'finish_reasons': dict(finish),
        'generation': prepared['generation'], 'all1233_unreranked_rows_match': True,
        'max_rerank_input_tokens': prepared['max_input_tokens'], 'costs': costs(replies, p),
        'sources_hashes': p['original_files_sha256'], 'original_files_hashes_match': True}
    stem = 'rw-ablation-dev' if replies is not None else 'rw-ablation-unreranked-dev'
    lm.write(LOCAL / f'{stem}-rows.json', candidates)
    lm.write(lm.RESULTS / f'{stem}.json', result)
    markdown(result, lm.RESULTS / f'{stem}.md')


def markdown(d, path):
    out = ['# rw除去の探索的検索評価（開発用のみ）', '',
        '元の本番結果・設定・選択を変更していない。本番用質問は未使用。差はrwなし−対応するrwあり。',
        '同等は会話単位95%区間全体が[-.03,+.03]内の場合のみ。個別区間、p値は補正なし。',
        f'並べ替えあり新条件の完了：{d["reranked_complete"]}。', '', '## rw除去の直接比較', '',
        '| 種類 | 指標 | rwなし条件 | n | 値 | 差 | 会話単位95%区間 | ±.03内 | 勝/負/同(n; p) |',
        '|---|---|---|---:|---:|---:|---|---|---|']
    for label in STRATA:
        for metric, entries in d['direct_rw_removal'][label].items():
            for arm, s in entries.items():
                lo, hi = s['conversation_ci95']; sign = s['sign_test']
                out.append(f'| {label} | {metric} | `{arm}` | {s["questions"]} | {s["mean"]:.4f} '
                    f'| {s["mean_diff"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] '
                    f'| {"はい" if s["equivalence_within_margin"] else "未確認"} '
                    f'| {sign["wins"]}/{sign["losses"]}/{sign["ties"]} ({sign["n"]}; {sign["two_sided_p"]:.5f}) |')
    out += ['', '## 全条件・会話別差（既存stage4/5と同じ形式）', '']
    for label in STRATA:
        for metric in reporting.METRICS:
            for base, stats in d['comparisons'].items():
                out += [f'### {label} / {metric}', '']
                out += reporting.table(stats['strata'][label][metric], base, stats['groups']) + ['']
    c = d['costs']['ingestion']
    out += ['## 取り込み費用', '', '| 成分 | 秒 | 記録の性質 |', '|---|---:|---|',
        f'| rw生成 | {c["rw_generation_actual"]:.3f} | 保存済みrun実測 |',
        f'| fact本実行 | {c["fact_main_generation_actual"]:.3f} | 保存済みrun実測 |',
        f'| fact初期3件 | {c["fact_initial3_actual"]:.3f} | 別ログ実測、GPU競合状態未記録 |']
    for name, seconds in c['embedding_actual'].items():
        out.append(f'| {name}埋め込み | {seconds:.6f} | 追加監査の実測 |')
    out += [f'| rwなし既知成分合計 | {c["without_rw_known_component_sum"]:.3f} | 別時点の成分合算 |',
        f'| rwあり既知成分合計 | {c["with_rw_known_component_sum"]:.3f} | 別時点の成分合算 |',
        f'| 削減成分合計 | {c["saved_known_component_sum"]:.3f} | rw生成＋rw埋め込み |', '',
        'p2原文埋め込みも必要なのでfact埋め込みだけにしない。起動・本文準備・保存は未記録。'
        '元の7126.977秒を修正せず、最終キャッシュに一致する初期3件75.341秒を別に加えた。'
        '会話別生成は件数比例推定、埋め込みは会話別実測。', '',
        '## 並べ替え時間', '',
        'rwあり元stage全体（1233問、tokenizer/score等込み）：'
        f'{d["costs"]["with_rw_rerank"]["dev1233_stage_seconds_actual"]:.3f}秒。個別HTTP時間は未記録。',
        'rwあり償却平均：'
        f'{d["costs"]["with_rw_rerank"]["stage_amortized_seconds"]:.6f}秒/問。302問相当は'
        f'{d["costs"]["with_rw_rerank"]["equivalent302_stage_seconds_estimate"]:.3f}秒（推定）。']
    if 'without_rw_rerank' in d['costs']:
        a = d['costs']['without_rw_rerank']
        out += [f'rwなし新HTTP実測{a["HTTP_calls_measured"]}件、平均{a["HTTP_mean_seconds_actual"]}秒/問。',
            f'固定dev302問：実測{a["fixed_dev302_HTTP_measured"]}件、平均{a["fixed_dev302_HTTP_mean_actual"]}秒、'
            f'HTTP合計{a["fixed_dev302_HTTP_total_actual"]:.3f}秒。',
            f'全dev平均×302の推定：{a["HTTP_equivalent302_seconds_estimate"]}秒。',
            f'新rerank段階wall：{a["stage"]["seconds"]:.3f}秒。'
            f'既存digest共有でHTTP時間が未記録の要求：{a["shared_calls_HTTP_time_unrecorded"]}件。',
            'HTTPだけの新平均とtokenizer/score込みの旧stage平均を同じ計時範囲として比較しない。']
    out += ['', f'parse順位保持fallback：{d["parse_fallback"]}件。finish_reason：{d["finish_reasons"]}。',
        '', '## 確認できたこと・できなかったこと・範囲', '',
        '確認：開発1233問のrwなしdense順位・指標の完全再現、固定条件の検索比較、保存runと別実測の費用。',
        '未確認：同一時点の完全な取り込みwall、過去の会話別生成秒数、旧個別rerank HTTP時間、'
        '本番rwなし精度、回答F1・意味、未知会話・日本語への一般化。',
        '主張範囲：同じ10会話の開発用検索指標への事後的探索。凍結C-finalは更新しない。'
        '途中得点から候補・判定を変更しない。', '']
    path.write_text('\n'.join(out))


def pause():
    lm.write(lm.LOCAL / 'user-pause.json', {'time': lm.stamp(), 'source': 'human rw-ablation pause; explicit resume only'})
    for line in subprocess.check_output(['ps', '-axo', 'pid=,pgid=,command='], text=True).splitlines():
        fields = line.split(None, 2)
        if len(fields) != 3:
            continue
        pid, pgid = map(int, fields[:2])
        if ('python' in Path(fields[2].split()[0]).name and
                'locomo_memory_rw_ablation.py run' in fields[2] and pid == pgid and pid != os.getpid()):
            os.killpg(pgid, signal.SIGINT)
    print('persistent pause saved; Qwen not signalled', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register', 'prepare', 'run', 'pause'))
    args = parser.parse_args()
    {'register': register, 'prepare': prepare, 'run': run, 'pause': pause}[args.command]()
