#!/usr/bin/env python3
"""One registered rw-free heldout run; all outputs are separate from frozen production."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

import locomo_answer_production as prod
import locomo_memory_rw_ablation as ab
import locomo_memory_reporting as inference

lm, ae, am, st, mm, rp = prod.lm, prod.ae, prod.am, prod.st, prod.mm, prod.ae.rp
LOCAL = lm.LOCAL / 'answer-norw'
PLAN = lm.RESULTS / 'answer-norw-plan.json'
DOC = lm.ROOT / 'docs/MEMORY_ANSWER_NORW_PREREG.md'
ARM = 'C-final-norw'
CFG = mm.config(['p2', 'fact'], rerank='qwen')
STRATA = ('multiple_evidence', 'overall', 'single_fact', 'temporal')
SEARCH_METRICS = ('recall@T512', 'all_evidence@10', 'all_evidence@T512')


def status(phase, **extra):
    lm.write(LOCAL / 'status.json', {'time': lm.stamp(), 'pid': os.getpid(),
        'pgid': os.getpgid(0), 'phase': phase, **extra})


def original_hashes():
    files = subprocess.check_output(['git', 'ls-files', 'results/eval/memory', 'scripts/eval',
        'docs/MEMORY_ANSWER_EVAL.md'], cwd=lm.ROOT, text=True).splitlines()
    files += [str(p.relative_to(lm.ROOT)) for p in prod.LOCAL.rglob('*') if p.is_file()]
    files += [str(p.relative_to(lm.ROOT)) for p in (lm.LOCAL / 'gen').rglob('*') if p.is_file()]
    files += [str(p.relative_to(lm.ROOT)) for p in (lm.LOCAL / 'cache').glob('*.npy')]
    files += [str((lm.LOCAL / name).relative_to(lm.ROOT)) for name in
        ('corpus.json', 'holdout.json', 'split-map.json', 'dev.json')]
    return {n: lm.file_hash(lm.ROOT / n) for n in sorted(set(files)) if n != str(PLAN.relative_to(lm.ROOT))}


def register():
    ae.gate()
    inherited = prod.checked()
    if PLAN.exists():
        raise RuntimeError('registration exists; never overwrite')
    p = {'created_at': lm.stamp(), 'name': ARM, 'config': CFG, 'questions': 302,
        'rerank_calls': 302, 'answer_calls': 302, 'references': ['C-final', 'A'],
        'display_references': ['C-t1', 'C-p2'], 'strata': STRATA,
        'metrics': ['f1', 'abstention', 'failure', *SEARCH_METRICS],
        'equivalence': {'margin': .03, 'comparisons': 'F1 norw minus C-final per four strata',
            'rule': 'entire pointwise95% CI contained in[-.03,+.03]; all four required for all-type claim'},
        'inference': {'resamples': 10000, 'seed': 0, 'exact_two_sided_sign': True,
            'zero_ties_excluded': True, 'multiplicity_adjustment': False},
        'inherited': inherited, 'answer': inherited['plan']['answer'],
        'search_generation': st.Qwen.offline().settings, 'search_context': 8192,
        'answer_context': 65536, 'answer_generation': am.GENERATION,
        'answer_system': am.SYSTEM, 'answer_schema': am.SCHEMA,
        'budgets': {'recent': 1024, 'memory': 512}, 'ordering_seed': 'locomo-answer-v2',
        'estimate_seconds': [2700, 3600], 'HTTP_attempts_per_job': 1,
        'timing': 'fresh batch1 query embedding timing only; frozen cached vectors for rank; '
                  'HTTP/rank/assembly+separatefreshembedding component sum; not continuous online latency',
        'scope': 'post-original-result registered additional heldout302 check within same ten conversations',
        'original_files_sha256': original_hashes(),
        'code_sha256': lm.file_hash(Path(__file__)), 'document_sha256': lm.file_hash(DOC),
        'test_sha256': lm.file_hash(lm.ROOT / 'scripts/eval/tests/test_locomo_answer_norw.py')}
    lm.write(PLAN, p)
    print('registered; commit before run or heldout preparation', flush=True)


def checked():
    ae.gate()
    p = st.frozen(PLAN)
    for path, digest in ((Path(__file__), p['code_sha256']), (DOC, p['document_sha256']),
            (lm.ROOT / 'scripts/eval/tests/test_locomo_answer_norw.py', p['test_sha256'])):
        if lm.file_hash(path) != digest:
            raise RuntimeError(f'registered source changed: {path.name}')
    if p['config'] != CFG:
        raise RuntimeError('registered condition differs')
    verify_originals(p)
    prod.checked()
    return p


def verify_originals(p):
    for name, digest in p['original_files_sha256'].items():
        if lm.file_hash(lm.ROOT / name) != digest:
            raise RuntimeError(f'original changed: {name}')


@contextmanager
def redirected_switch():
    """Only the switch helper's output destination changes; frozen helper file is untouched."""
    previous = ab.LOCAL
    ab.LOCAL = LOCAL
    try:
        yield
    finally:
        ab.LOCAL = previous


def quiet():
    env = ab.quiet()
    for line in subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and int(parts[0]) != os.getpid() and (
                Path(parts[1].split()[0]).name.startswith('python') and
                'locomo_answer_norw.py run' in parts[1]):
            raise RuntimeError('another norw runner is active')
    return env


def schedule(queries):
    return [qid for qid, arm in am.schedule(queries, {a: a for a in am.CONDITIONS}) if arm == 'C-final']


class Tokenizer(ae.Tokenizer):
    def __init__(self, url):
        super().__init__(url)
        self.path = LOCAL / 'token-counts.json'
        if self.path.exists():
            self.counts = lm.read(self.path)


def prepare(p):
    status('preparing search', status='running')
    env = quiet()
    now = ab.server(65536)
    data, manifest = prod.load_heldout(p['inherited'])
    old = lm.read(prod.LOCAL / 'search-inputs.json')
    turns = old['turns']
    qwen = st.Qwen.offline()
    facts = st.generated('fact', data, turns, qwen)[0]
    units = st.build_units(CFG['units'], turns, {'fact': facts})
    from egemma_server.tokenizer import PromptTokenizer
    tokenizer = PromptTokenizer(rp.V1_ST_DIR)
    identity = lm.fingerprint(st.VARIANT)
    if identity != lm.read(lm.RESULTS / 'final-config.json')['embedding_artifact_sha256']:
        raise RuntimeError('embedding artifacts differ')
    vectors = np.zeros((len(units), 768), np.float32)
    for name in CFG['units']:
        ix = [i for i, u in enumerate(units) if u['set'] == name]
        texts = rp.fit_like_st(tokenizer, [lm.PROTOCOL['prompts']['document'] + units[i]['text'] for i in ix])
        vectors[ix], _ = ab.cached(texts, f'u-{name}', identity)
    queries = data['queries']
    texts = rp.fit_like_st(tokenizer, [lm.PROTOCOL['prompts']['query'] + q['text'] for q in queries])
    qvecs, _ = ab.cached(texts, 'q-holdout-answer', identity)
    # The benchmark uses a private log destination, never saves new vectors into original caches.
    args = argparse.Namespace(port=8398, model_dir=lm.ROOT / 'models/embeddinggemma-2', v1_dir=rp.V1_ST_DIR)
    import locomo_answer_audit_measure as audit
    query_times = {}
    tick = time.perf_counter()
    previous_local = lm.LOCAL
    try:
        lm.LOCAL = LOCAL
        with lm.start_backend(st.VARIANT, args) as (backend, embed):
            startup = time.perf_counter() - tick
            for q, text in zip(queries, texts):
                ae.gate()
                audit.quiet(embedding_allowed=True)
                start = time.perf_counter()
                lm.checked_vectors(embed([text]), 1)
                query_times[q['id']] = time.perf_counter() - start
    finally:
        lm.LOCAL = previous_local
    quiet()
    tick = time.perf_counter()
    index = mm.Index(units, vectors)
    index_seconds = time.perf_counter() - tick
    count = Tokenizer(now['url'])
    jobs, parts, ranking_times = {}, {}, {}
    try:
        for q, v in zip(queries, qvecs):
            ae.gate()
            tick = time.perf_counter()
            order = mm.distinct([i for i, _ in index.rank(CFG, q['group'], [(v, q['text'])])], units)
            ranking_times[q['id']] = time.perf_counter() - tick
            chat = prod.rerank_chat(q, order, units, turns, CFG)
            n = count.input_count(chat)
            prod.ensure_length(n, mm.PROMPTS['rerank']['max_tokens'], 8192)
            body = qwen.request('rerank', chat)
            jobs[q['id']] = {'body': body, 'digest': qwen.item_digest('rerank', chat), 'input_tokens': n}
            parts[q['id']] = {**old['queries'][q['id']], 'order': order}
    finally:
        count.save()
    encode = lambda ts: [len(e.ids) for e in tokenizer.tokenizer.encode_batch(ts, add_special_tokens=False)]
    tokens = dict(zip([t['id'] for t in turns], encode([t['text'] for t in turns])))
    dates = {t['session']: t['date'] for t in turns}
    headers = dict(zip(dates, encode([f'Date: {d}' for d in dates.values()])))
    prepared = {'plan_sha256': lm.file_hash(PLAN), 'units': units, 'turns': turns, 'queries': parts,
        'jobs': jobs, 'schedule': schedule(queries), 'data': data, 'prepared_at': lm.stamp(),
        'tokens': tokens, 'headers': headers, 'sessions': {t['id']: t['session'] for t in turns}}
    lm.write(LOCAL / 'search-inputs.json', prepared)
    lm.write(LOCAL / 'search-timings.json', {'query_embedding_seconds': query_times,
        'ranking_seconds': ranking_times, 'embedding_startup_seconds': startup,
        'index_build_seconds': index_seconds, 'backend': backend, 'environment': env,
        'query_vectors': 'existing frozen cache; fresh benchmark vectors discarded'})
    verify_originals(p)
    return prepared


def generate(kind, jobs, order, now):
    if len(jobs) != 302 or len(order) != 302 or set(order) != set(jobs):
        raise RuntimeError('generation must contain exactly302 distinct registered jobs')
    path = LOCAL / f'{kind}-responses.json'
    if path.exists() or (LOCAL / f'{kind}-received.jsonl').exists() or (LOCAL / 'inflight.json').exists():
        raise RuntimeError('generation previously started; no automatic repeat')
    replies = {}
    tick = time.perf_counter()
    try:
        for key in order:
            ae.gate()
            job = jobs[key]
            status(f'{kind} generation', status='running', completed=len(replies), total=302)
            lm.write(LOCAL / 'inflight.json', {'key': key, 'kind': kind, 'digest': job['digest'], 'time': lm.stamp()})
            start = time.perf_counter()
            response = st.http_json(now['url'] + '/v1/chat/completions', job['body'], timeout=900)
            reply = {'digest': job['digest'], 'response': response, 'wall_seconds': time.perf_counter() - start,
                'timings': response.get('timings') or {}, 'input_tokens': job['input_tokens'],
                'finished_at': lm.stamp(), 'attempts': 1}
            replies[key] = reply
            ae.received(LOCAL / f'{kind}-received.jsonl', key, reply)
            (LOCAL / 'inflight.json').unlink()
            ae.checked_timings(reply)
            if len(replies) % 25 == 0:
                lm.write(path, replies)
                print(f'{kind} {len(replies)}/302 saved', flush=True)
    finally:
        lm.write(path, replies)
        lm.write(LOCAL / f'{kind}-runtime.json', {'seconds': time.perf_counter() - tick,
            'received': len(replies), 'total': 302})
    status(f'{kind} complete', status='complete', completed=len(replies), total=302)
    return replies


def orders_for(search, replies):
    if set(replies) != set(search['jobs']):
        raise RuntimeError('all302 reranks required')
    orders, fallback = {}, 0
    for qid, q in search['queries'].items():
        reply = replies[qid]
        if reply['digest'] != search['jobs'][qid]['digest']:
            raise RuntimeError('rerank request differs')
        content = reply['response']['choices'][0]['message'].get('content') or ''
        value = mm.parse('rerank', content, n_passages=len(q['order'][:mm.RERANK_CANDIDATES]))
        fallback += value is None
        orders[qid] = mm.distinct(prod.reranked_order(q['order'], value), search['units'])
    return orders, fallback


def prepare_answers(p, search, replies):
    now = ab.server(65536)
    orders, fallback = orders_for(search, replies)
    by_group = defaultdict(list)
    for t in search['turns']:
        by_group[t['group']].append(t)
    count = Tokenizer(now['url'])
    items, assembly = {}, {}
    try:
        for qid in search['schedule']:
            ae.gate()
            q = search['queries'][qid]
            tick = time.perf_counter()
            ids = list(dict.fromkeys(t for i in orders[qid] for t in search['units'][i]['turns']))
            payload = am.payload('C-final', by_group[q['group']], {'C-final': ids}, count.count)
            messages = am.messages(payload.pop('conversation'), payload.pop('retrieved'), q['text'])
            n = count.input_count(messages)
            prod.ensure_length(n, 256, 65536, 128)
            inserted, evidence = set(payload['recent_ids'] + payload['memory_ids']), set(q['evidence'])
            items[qid] = {'id': qid, 'condition': ARM, 'category': q['category'], 'type': q['type'],
                'messages': messages, 'gold': q['gold'], 'input_tokens': n, **payload,
                'inserted_all': float(evidence <= inserted), 'inserted_recall': len(evidence & inserted)/len(evidence)}
            assembly[qid] = time.perf_counter() - tick
    finally:
        count.save()
    prepared = {'plan_sha256': lm.file_hash(PLAN), 'server': now, 'items': items, 'schedule': search['schedule']}
    lm.write(LOCAL / 'answer-inputs.json', prepared)
    lm.write(LOCAL / 'assembly-timings.json', {'seconds': assembly, 'fallback': fallback,
        'all302_input_preflight_complete': True, 'max_input_tokens': max(i['input_tokens'] for i in items.values())})
    return prepared


def ready(prepared, replies, rerank, state):
    ids = set(prepared['items'])
    if len(ids) != 302 or ids != set(replies) or ids != set(rerank):
        raise RuntimeError('all302 reranks and answers required before scoring')
    if state.get('phase') != 'answer complete' or state.get('status') != 'complete':
        raise RuntimeError('final answer checkpoint required before scoring')


def equivalent(ci):
    return ci is not None and -.03 <= ci[0] <= ci[1] <= .03


def report(p, search, prepared, replies, rerank):
    report_started = time.perf_counter()
    ready(prepared, replies, rerank, lm.read(LOCAL / 'status.json'))
    verify_originals(p)
    orders, fallback = orders_for(search, rerank)
    rows, sr = [], []
    for qid in prepared['schedule']:
        item, reply = prepared['items'][qid], replies[qid]
        body = ae.request(p['answer'], item)
        expected = lm.digest({'request': body, 'plan': lm.file_hash(PLAN), 'server': prepared['server']})
        if reply['digest'] != expected:
            raise RuntimeError('answer request digest differs')
        t = ae.checked_timings(reply)
        rows.append({'id': qid, 'type': item['type'], 'category': item['category'],
            **am.score(reply['response'], item['gold']), 'wall_seconds': reply['wall_seconds'],
            'prefill_seconds': t['prompt_ms']/1000, 'generation_seconds': t['predicted_ms']/1000,
            'inserted_all': item['inserted_all'], 'inserted_recall': item['inserted_recall']})
        sr.append(ab.row(search['queries'][qid], orders[qid], search['units'], search['data'],
            search['sessions'], search['tokens'], search['headers']))
    original_rows = lm.read(prod.LOCAL / 'scored-rows.json')
    display = {a: original_rows[a] for a in ('A', 'C-t1', 'C-p2', 'C-final')}
    display[ARM] = rows
    # Align to the original scoring order, rather than the generation order.
    ids = [r['id'] for r in original_rows['A']]
    display = {a: [dict((r['id'], r) for r in rs)[q] for q in ids] for a, rs in display.items()}
    oldsearch = lm.read(prod.LOCAL / 'search-inputs.json')
    oldrerank = ae.recover(prod.LOCAL / 'rerank-responses.json', prod.LOCAL / 'rerank-received.jsonl')
    retrieval = {ARM: [dict((r['id'], r) for r in sr)[q] for q in ids]}
    for arm in ('C-t1', 'C-p2', 'C-final'):
        part = []
        for qid in ids:
            q = oldsearch['queries'][qid]
            order = q['orders'][arm]
            if arm == 'C-final':
                value = mm.parse('rerank', oldrerank[qid]['response']['choices'][0]['message'].get('content') or '',
                    n_passages=len(order[:mm.RERANK_CANDIDATES]))
                order = mm.distinct(prod.reranked_order(order, value), oldsearch['units'])
            part.append(ab.row(q, order, oldsearch['units'], search['data'], search['sessions'],
                search['tokens'], search['headers']))
        retrieval[arm] = part
    groups = sorted({q.split('/')[0] for q in ids})
    tables, checks = {}, {}
    lines = ['# C-final-norw追加本番302問検証', '',
        '既存本番結果を見た後に事前登録した追加検証。語彙F1であり意味的正答率ではない。'
        '同じ10会話の既存質問のみ。未知会話・日本語への一般化は未確認。', '',
        '会話単位10000回seed0、質問macro比。符号検定は厳密両側・厳密ゼロ差除外、補正なし。'
        '区間全体が±.03内だけ同等性確認。AのT512/上位10検索は対象外。', '']
    for kind in STRATA:
        tables[kind] = {}
        for metric in ('f1', 'abstention', 'failure', *SEARCH_METRICS):
            source = display if metric not in SEARCH_METRICS else retrieval
            subsets = {a: [r for r in rs if kind == 'overall' or r['type'] == kind] for a, rs in source.items()}
            refs = ('C-final', 'A') if metric not in SEARCH_METRICS else ('C-final',)
            entries = {a: {ref: inference.paired(rs, subsets[ref], metric) for ref in refs} for a, rs in subsets.items()}
            tables[kind][metric] = entries
            lines += [f'## {kind} / {metric}', '']
            for ref in refs:
                lines += inference.table(entries, ref, groups) + ['']
        checks[kind] = equivalent(tables[kind]['f1'][ARM]['C-final']['conversation_ci95'])
    timing = lm.read(LOCAL / 'search-timings.json')
    assembly = lm.read(LOCAL / 'assembly-timings.json')
    costs = []
    for qid in ids:
        a, r = replies[qid], rerank[qid]
        record = {'id': qid, 'answer_seconds': a['wall_seconds'], 'rerank_seconds': r['wall_seconds'],
            'fresh_embedding_seconds_separate_actual': timing['query_embedding_seconds'][qid],
            'search_seconds': timing['ranking_seconds'][qid], 'assembly_seconds': assembly['seconds'][qid]}
        record['cached_query_component_seconds'] = record['answer_seconds']+record['rerank_seconds']+record['search_seconds']+record['assembly_seconds']
        record['total_component_seconds'] = record['cached_query_component_seconds']+record['fresh_embedding_seconds_separate_actual']
        for stage, reply in (('answer', a), ('rerank', r)):
            t = ae.checked_timings(reply)
            record[stage+'_prefill_seconds'] = t['prompt_ms']/1000
            record[stage+'_generation_seconds'] = t['predicted_ms']/1000
            record[stage+'_overhead_seconds'] = reply['wall_seconds']-t['prompt_ms']/1000-t['predicted_ms']/1000
        costs.append(record)
    import locomo_answer_audit as audit
    distributions = {k: audit.distribution([r[k] for r in costs]) for k in costs[0] if k != 'id'}
    ingestion = lm.read(lm.RESULTS / 'rw-ablation-dev.json')['costs']['ingestion']
    lines += ['## 時間と取り込み費用', '',
        'HTTP/検索/入力構築の実測＋別実測fresh embedding成分和。連続オンライン遅延ではない。'
        '旧C-final/Aとcache順序・実行日が異なり速度差はrw除去の因果効果ではない。', '',
        '|成分|平均 秒/問|合計 秒|', '|---|---:|---:|']
    for k, d in distributions.items():
        lines.append(f'|{k}|{d["mean"]:.6f}|{sum(r[k] for r in costs):.3f}|')
    lines += ['', '取り込み：元生成wallの保存値＋別監査埋め込み実測の復元成分和。'
        f'rwなし{ingestion["without_rw_known_component_sum"]:.3f}秒、rwあり{ingestion["with_rw_known_component_sum"]:.3f}秒、'
        f'差{ingestion["saved_known_component_sum"]:.3f}秒。'
        '会話別fact生成は件数按分の推定。起動・本文準備・保存と初期factのGPU並行状態は未記録。', '',
        '## 判定', '', str(checks), '',
        '確認済み：登録条件/全応答/digest/hash、検索・回答の追加比較。'
        '未確認：独立未知会話、日本語、意味的正答率、rw除去の因果的速度差。'
        '主張範囲：同じ10会話の既存302問の追加検証。フェーズ2へ進まず人間確認を待つ。', '']
    runtime = lm.read(LOCAL / 'runtime.json')
    old_performance = lm.read(lm.RESULTS / 'answer-audit-performance.json')
    historical_timing = {'C-final': old_performance['C_final_seconds'],
        'A_cached_original302': audit.distribution([r['wall_seconds'] for r in original_rows['A']]),
        'A_cache_audit_development': old_performance['cache_comparison']['A'],
        'note': 'historical restored measurements, distinct cache order/cohorts; not causal rw timing comparison'}
    lines += ['## 保存済み時間の比較', '',
        '|条件/測定|平均 秒/問|対象|', '|---|---:|---|',
        f'|C-final-norw 新成分合算|{distributions["total_component_seconds"]["mean"]:.6f}|302問・別実測embedding加算|',
        f'|C-final 旧回答HTTP|{historical_timing["C-final"]["answer_seconds"]["mean"]:.6f}|既存302問|',
        f'|C-final 旧rerankHTTP|{historical_timing["C-final"]["rerank_seconds"]["mean"]:.6f}|既存302問|',
        f'|A キャッシュあり旧HTTP|{historical_timing["A_cached_original302"]["mean"]:.6f}|既存302問・初回cold含む|', '']
    for label, arm in historical_timing['A_cache_audit_development'].items():
        lines.append(f'|A cache audit {label}|{arm["wall_seconds"]["mean"]:.6f}|開発用監査・本番再生成なし|')
    runtime['report_compute_seconds_before_save'] = time.perf_counter()-report_started
    record = {'created_at': lm.stamp(), 'plan_sha256': lm.file_hash(PLAN), 'questions': 302,
        'answer_responses': len(replies), 'rerank_responses': len(rerank), 'groups': groups,
        'tables': tables, 'equivalence_checks': checks, 'all_four_equivalent': all(checks.values()),
        'failure_counts': dict(Counter(r['failure_kind'] or 'none' for r in rows)), 'rerank_parse_fallback': fallback,
        'timing_distributions': distributions, 'timing_per_conversation': {g: {k: audit.distribution([r[k] for r in costs
            if r['id'].split('/')[0] == g]) for k in distributions} for g in groups},
        'ingestion': ingestion, 'historical_timing': historical_timing, 'runtime': runtime, 'context_switches': lm.read(LOCAL / 'context-switches.json'),
        'original_files_sha256_match': True, 'generation': p['answer_generation'], 'config': CFG,
        'timing_note': p['timing'], 'scope': p['scope']}
    lm.write(LOCAL / 'scored-rows.json', {ARM: rows})
    lm.write(LOCAL / 'component-rows.json', costs)
    lm.write(lm.RESULTS / 'answer-norw-report.json', record)
    (lm.RESULTS / 'answer-norw-report.md').write_text('\n'.join(lines))
    lm.write(lm.RESULTS / 'answer-norw-verification.json', {'time': lm.stamp(),
        'plan_sha256': lm.file_hash(PLAN), 'original_files_sha256_match': True,
        'answer_received_sha256': lm.file_hash(LOCAL / 'answer-received.jsonl'),
        'rerank_received_sha256': lm.file_hash(LOCAL / 'rerank-received.jsonl'),
        'received_counts': {'answers': 302, 'reranks': 302}})
    status('evaluation complete', status='complete', completed=604, total=604)
    print('completed; separate reports saved; await human confirmation', flush=True)


def run():
    p = checked()
    if (LOCAL / 'status.json').exists():
        raise RuntimeError('already started; no automatic resume or duplicate generation')
    LOCAL.mkdir(parents=True, exist_ok=True)
    quiet()
    started = time.perf_counter()
    started_at = lm.stamp()
    search = prepare(p)
    with redirected_switch():
        ab.switch(8192)
    quiet()
    rerank = generate('rerank', search['jobs'], search['schedule'], ab.server(8192))
    verify_originals(p)
    with redirected_switch():
        ab.switch(65536)
    quiet()
    prepared = prepare_answers(p, search, rerank)
    jobs = {qid: {'body': ae.request(p['answer'], item), 'input_tokens': item['input_tokens'],
        'digest': lm.digest({'request': ae.request(p['answer'], item), 'plan': lm.file_hash(PLAN), 'server': prepared['server']})}
        for qid, item in prepared['items'].items()}
    replies = generate('answer', jobs, prepared['schedule'], ab.server(65536))
    lm.write(LOCAL / 'runtime.json', {'started_at': started_at, 'finished_generation_at': lm.stamp(),
        'seconds_through_generation': time.perf_counter()-started,
        'rerank': lm.read(LOCAL / 'rerank-runtime.json'), 'answer': lm.read(LOCAL / 'answer-runtime.json'),
        'embedding_startup_seconds': lm.read(LOCAL / 'search-timings.json')['embedding_startup_seconds'],
        'index_build_seconds': lm.read(LOCAL / 'search-timings.json')['index_build_seconds']})
    report(p, search, prepared, replies, rerank)


def pause():
    lm.write(ae.PAUSE, {'time': lm.stamp(), 'reason': 'human requested pause', 'runner': 'answer-norw'})
    state = lm.read(LOCAL / 'status.json') if (LOCAL / 'status.json').exists() else {}
    pid = state.get('pid')
    if pid:
        cmd = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout.strip()
        if 'locomo_answer_norw.py run' in cmd:
            if os.getpgid(pid) != state['pgid'] or state['pgid'] == os.getpgid(0):
                raise RuntimeError('cannot identify dedicated evaluation group')
            os.killpg(state['pgid'], signal.SIGINT)
    print('persistent pause saved; Qwen untouched', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register', 'run', 'pause'))
    command = parser.parse_args().command
    try:
        {'register': register, 'run': run, 'pause': pause}[command]()
    except BaseException as error:
        if command == 'run' and (not (LOCAL / 'status.json').exists() or
                lm.read(LOCAL / 'status.json').get('pid') == os.getpid()):
            status('stopped', status='stopped', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
