#!/usr/bin/env python3
"""Exploratory probes after U1 (not registered stages; the U1 sample has been seen).

budget: re-assemble the T/V answer inputs from the saved U1 rerank results with memory budgets
        512/1024/2048/4096 (recent 1024 unchanged), answer and judge with the same Qwen settings.
        Budget 512 must reproduce the saved U1 inputs exactly.
speed:  (a) prefill: saved U1 rerank requests with -ub 512/1024/2048; one saved G1 full-context (A)
        input with -ub 512/2048; (b) ingestion: 60 saved fact requests sequential, --parallel 2/4,
        and n-gram speculative decoding. Every output is compared byte for byte with the baseline.

Public outputs hold aggregates and IDs only; texts stay in build/eval/memory/longmemeval-u1/probes/.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/eval'))
import locomo_answer_methods as am  # noqa: E402
import locomo_answer_production as prod  # noqa: E402
import locomo_memory_methods as mm  # noqa: E402
import longmemeval_g1_methods as g1m  # noqa: E402
import longmemeval_u1_methods as um  # noqa: E402

RUN = ROOT/'build/eval/memory/longmemeval-u1/run'
G1 = ROOT/'build/eval/memory/longmemeval-g1'
OUT = ROOT/'build/eval/memory/longmemeval-u1/probes'
RESULTS = ROOT/'results/eval/memory'
PLAN = json.loads((RESULTS/'longmemeval-u1-plan.json').read_text())
JUDGE = json.loads((RESULTS/'longmemeval-g1-judge.json').read_text())
URL = 'http://127.0.0.1:8080'
BUDGETS = (512, 1024, 2048, 4096)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def http(path, body=None, timeout=1800):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL+path, data=data, headers={'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Server:
    def __init__(self, context, extra=(), label='probe'):
        if subprocess.run(['pgrep', '-x', 'llama-server'], capture_output=True).returncode == 0:
            raise SystemExit('a llama-server is already running')
        argv = list(PLAN['server_argv'])
        argv[argv.index('-c')+1] = str(context)
        self.argv = argv + list(extra)
        self.label = label

    def __enter__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, 'LLAMA_ARG_CACHE_RAM': '0'}
        self.log = open(OUT/f'server-{self.label}.log', 'w')
        self.proc = subprocess.Popen(self.argv, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(600):
            if self.proc.poll() is not None:
                raise SystemExit(f'server exited: {self.label}')
            try:
                if http('/health', timeout=2).get('status') == 'ok':
                    break
            except Exception:
                pass
            time.sleep(0.5)
        self.model = http('/v1/models')['data'][0]['id']
        log('server ready', self.label, ' '.join(self.argv[self.argv.index('-c'):]))
        return self

    def __exit__(self, *exc):
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def complete(body):
    tick = time.monotonic()
    r = http('/v1/chat/completions', body)
    choice = r['choices'][0]
    return {'content': choice['message'].get('content') or '', 'finish': choice['finish_reason'],
            'wall': time.monotonic()-tick, 'timings': r.get('timings') or {}}


def received(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        out[rec['key']] = rec['reply']
    return out


# -- budget -----------------------------------------------------------------------------------

def payload(arm, turns, raw, count, table, links, budget):
    lookup = {t['id']: t for t in turns}
    conversation, recent_ids = um.recent(turns, 1024, count)
    order, notes = (raw, None) if arm == 'T' else um.versioned(raw, table, links)
    retrieved, memory_ids = um.memory(order, lookup, recent_ids, budget, count, notes)
    return conversation, retrieved, recent_ids, memory_ids


def coverage(row, ids):
    pos = {i+1 for i, s in enumerate(row['haystack_session_ids']) if s in set(row['answer_session_ids'])}
    present = {int(t.split('/D', 1)[1].split(':')[0]) for t in ids}
    return float(bool(pos) and pos <= present)


def budget_probe():
    prepared = json.loads((RUN/'prepared.json').read_text())
    units = json.loads((RUN/'embedding.json').read_text())['units']
    jobs = json.loads((RUN/'rerank-jobs.json').read_text())
    facts = json.loads((RUN/'facts.json').read_text())
    links = json.loads((RUN/'links.json').read_text())
    saved_inputs = json.loads((RUN/'answer-inputs.json').read_text())
    replies = received(RUN/'received.jsonl')
    counts = {}

    def count(text):
        if not text:
            return 0
        k = hashlib.sha256(text.encode()).hexdigest()
        if k not in counts:
            counts[k] = len(http('/tokenize', {'content': text, 'add_special': False, 'parse_special': False})['tokens'])
        return counts[k]

    rows, private = [], {}
    with Server(8192, label='budget') as srv:
        for qid, row in prepared['questions'].items():
            us, job, ts = units[qid], jobs[qid], prepared['turns'][qid]
            text = replies['rerank|'+qid]['response']['choices'][0]['message'].get('content') or ''
            value = mm.parse('rerank', text, n_passages=min(len(job['order']), mm.RERANK_CANDIDATES))
            order = mm.distinct(prod.reranked_order(job['order'], value), us)
            raw = list(dict.fromkeys(t for i in order for t in us[i]['turns']))
            table = um.fact_table(ts, {s: v for s, v in facts.items() if s.startswith(qid+'#')})
            for arm in ('T', 'V'):
                for budget in BUDGETS:
                    conv, retr, recent_ids, memory_ids = payload(arm, ts, raw, count, table, links[qid], budget)
                    chat = um.messages(conv, retr, row['question'], row['question_date'])
                    if budget == 512 and chat != saved_inputs[arm][qid]['messages']:
                        raise SystemExit(f'budget 512 does not reproduce the saved U1 input: {qid} {arm}')
                    body = {'model': srv.model, 'messages': chat, **am.GENERATION,
                            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': am.SCHEMA}}}
                    reply = complete(body)
                    parsed, failure = am.parse({'choices': [{'finish_reason': reply['finish'], 'message': {'content': reply['content']}}]})
                    answer = parsed['answer'] if parsed else ''
                    kind = 'abstention' if qid.endswith('_abs') else row['question_type']
                    if failure:
                        label = 0
                    else:
                        prompt = JUDGE['templates'][kind].format(question=row['question'], answer=row['answer'], response=answer)
                        j = http('/v1/chat/completions', {'model': srv.model, 'messages': [{'role': 'user', 'content': prompt}], **JUDGE['generation']})
                        label, _ = g1m.judge_label(j)
                    saved_answer = (am.parse(json.loads(json.dumps(replies['answer|'+arm+'|'+qid]['response'])))[0] or {}).get('answer') if budget == 512 else None
                    rows.append({'id': qid, 'arm': arm, 'budget': budget, 'stratum': um.stratum(row), 'correct': label,
                                 'abstain': float(am.abstains(answer)), 'coverage': coverage(row, set(recent_ids) | set(memory_ids)),
                                 'memory_turns': len(memory_ids), 'failure': failure,
                                 'same_as_saved_u1_answer': (answer == saved_answer) if budget == 512 else None})
                    private[f'{qid}|{arm}|{budget}'] = {'answer': answer}
            log('budget', qid, 'done')
    (OUT/'budget-private.json').write_text(json.dumps(private, ensure_ascii=False, indent=1))
    report = summarize_budget(rows)
    (RESULTS/'longmemeval-u1-budget-probe.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    (RESULTS/'longmemeval-u1-budget-probe.md').write_text(budget_markdown(report))
    print(budget_markdown(report))


def summarize_budget(rows):
    out = {'exploratory': True, 'questions': len({r['id'] for r in rows}), 'cells': {}, 'vs512': {}, 'rows': rows}
    for arm in ('T', 'V'):
        base = {r['id']: r for r in rows if r['arm'] == arm and r['budget'] == 512}
        for b in BUDGETS:
            part = [r for r in rows if r['arm'] == arm and r['budget'] == b]
            out['cells'][f'{arm}@{b}'] = {k: float(np.mean([r[k] for r in part])) for k in ('correct', 'abstain', 'coverage', 'memory_turns')}
            if b != 512:
                ids = sorted(base)
                d = [next(r for r in part if r['id'] == q)['correct'] - base[q]['correct'] for q in ids]
                out['vs512'][f'{arm}@{b}'] = {'difference': float(np.mean(d)), 'ci95': g1m.interval(d, ids), 'sign': g1m.sign_test(d)}
        out['cells'][f'{arm}@512']['reproduced_saved_u1_answers'] = float(np.mean([r['same_as_saved_u1_answer'] for r in base.values()]))
    return out


def budget_markdown(r):
    lines = ['# U1 後の探索：記憶の枠（512/1024/2048/4096）', '',
             '登録外の探索（U1 の30問は結果を見た後）。T・V の入力を保存済みの再ランキング結果から組み立て直し、同じ Qwen で回答・採点した。'
             '枠 512 は U1 の実際の入力と完全一致を確認済み。', '',
             '|条件@枠|正答率|棄権率|根拠セッション被覆|記憶の発言数|512との差|95%区間|勝/負/同|', '|---|---:|---:|---:|---:|---:|---|---|']
    for key, c in r['cells'].items():
        v = r['vs512'].get(key)
        diff = f'{v["difference"]:+.3f}|[{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]|{v["sign"]["wins"]}/{v["sign"]["losses"]}/{v["sign"]["ties"]}' if v else '基準|—|—'
        lines.append(f'|{key}|{c["correct"]:.3f}|{c["abstain"]:.3f}|{c["coverage"]:.3f}|{c["memory_turns"]:.1f}|{diff}|')
    for arm in ('T', 'V'):
        lines.append(f'\n{arm}@512 が U1 の保存済み回答と一致した割合：{r["cells"][f"{arm}@512"]["reproduced_saved_u1_answers"]:.3f}')
    return '\n'.join(lines)+'\n'


# -- speed ------------------------------------------------------------------------------------

def run_set(bodies, workers=1):
    tick = time.monotonic()
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(complete, bodies))
    return results, time.monotonic()-tick


def summary(results, wall, reference):
    same = [a['content'] == b for a, b in zip(results, reference)]
    t = [r['timings'] for r in results]
    return {'wall_seconds': wall, 'requests': len(results), 'identical_to_reference': float(np.mean(same)),
            'different_ids': [i for i, s in enumerate(same) if not s],
            'prompt_tokens_per_s': float(np.sum([x.get('prompt_n', 0) for x in t]) / max(1e-9, np.sum([x.get('prompt_ms', 0) for x in t]) / 1000)),
            'generated_tokens_per_s': float(np.sum([x.get('predicted_n', 0) for x in t]) / max(1e-9, np.sum([x.get('predicted_ms', 0) for x in t]) / 1000)),
            'mean_wall_per_request': float(np.mean([r['wall'] for r in results])),
            'draft': {k: int(np.sum([x.get(k, 0) for x in t])) for k in ('draft_n', 'draft_n_accepted')}}


def speed_probe():
    report = {'exploratory': True}
    replies = received(RUN/'received.jsonl')
    jobs = json.loads((RUN/'rerank-jobs.json').read_text())
    ids = list(jobs)
    rerank_bodies = [jobs[q]['body'] for q in ids]
    rerank_saved = [replies['rerank|'+q]['response']['choices'][0]['message'].get('content') or '' for q in ids]
    report['rerank_ub'] = {}
    for ub in (512, 1024, 2048):
        with Server(16384, ['-ub', str(ub)], label=f'rerank-ub{ub}'):
            res, wall = run_set(rerank_bodies)
        report['rerank_ub'][ub] = summary(res, wall, rerank_saved)
        log('rerank ub', ub, report['rerank_ub'][ub]['wall_seconds'], report['rerank_ub'][ub]['identical_to_reference'])

    g1_prepared = json.loads((G1/'run/prepared.json').read_text())
    g1_plan = json.loads((RESULTS/'longmemeval-g1-execution-plan.json').read_text())
    g1_replies = received(G1/'continuation/received.jsonl')
    qid = next(q for q in g1_plan['question_ids'] if g1_prepared['inputs']['A'][q]['fits_context'])
    a_body = {'model': None, 'messages': g1_prepared['inputs']['A'][qid]['messages'], **am.GENERATION,
              'response_format': {'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': am.SCHEMA}}}
    a_saved = g1_replies['answer|A|'+qid]['response']['choices'][0]['message'].get('content') or ''
    report['full_context_ub'] = {'question_id': qid, 'input_tokens': g1_prepared['inputs']['A'][qid]['input_tokens']}
    for ub in (512, 2048):
        with Server(131072, ['-ub', str(ub)], label=f'A-ub{ub}') as srv:
            res, wall = run_set([{**a_body, 'model': srv.model}])
        report['full_context_ub'][ub] = summary(res, wall, [a_saved])
        log('A ub', ub, wall, report['full_context_ub'][ub]['identical_to_reference'])

    prepared = json.loads((RUN/'prepared.json').read_text())
    fact_jobs = [j for j in prepared['fact_jobs'] if j['input_tokens'] + 1500 + 128 <= 8192]
    sample = sorted(fact_jobs, key=lambda j: hashlib.sha256(('u1-speed-v1|'+j['session']).encode()).hexdigest())[:60]
    bodies = [j['body'] for j in sample]
    saved = [replies['fact|'+j['session']]['response']['choices'][0]['message'].get('content') or '' for j in sample]
    report['ingestion'] = {'sessions': len(sample), 'sample_seed': 'u1-speed-v1'}
    with Server(8192, label='fact-p1'):
        base, wall = run_set(bodies)
    base_content = [r['content'] for r in base]
    report['ingestion']['sequential'] = summary(base, wall, base_content)
    report['ingestion']['sequential']['identical_to_saved_u1_ctx32768'] = float(np.mean([a == b for a, b in zip(base_content, saved)]))
    log('fact sequential', wall)
    for n in (2, 4):
        with Server(8192*n, ['--parallel', str(n)], label=f'fact-p{n}'):
            res, wall = run_set(bodies, workers=n)
        report['ingestion'][f'parallel{n}'] = summary(res, wall, base_content)
        log('fact parallel', n, wall, report['ingestion'][f'parallel{n}']['identical_to_reference'])
    with Server(8192, ['--spec-type', 'ngram-simple'], label='fact-ngram'):
        res, wall = run_set(bodies)
    report['ingestion']['ngram_simple'] = summary(res, wall, base_content)
    log('fact ngram', wall, report['ingestion']['ngram_simple']['identical_to_reference'])
    (RESULTS/'longmemeval-u1-speed-probe.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    (RESULTS/'longmemeval-u1-speed-probe.md').write_text(speed_markdown(report))
    print(speed_markdown(report))


def speed_markdown(r):
    lines = ['# U1 後の探索：速さ（出力の一致つき）', '',
             '登録外の探索。同じ依頼を設定だけ変えて流し、出力が基準と1文字残らず一致した割合を並べた。', '',
             '## 読み込み（-ub）', '', '|対象|設定|合計秒|prefill トークン/秒|U1/G1 の保存済み出力と一致|', '|---|---|---:|---:|---:|']
    for ub, s in r['rerank_ub'].items():
        lines.append(f'|再ランキング30件（約9〜16kトークン）|-ub {ub}|{s["wall_seconds"]:.1f}|{s["prompt_tokens_per_s"]:.0f}|{s["identical_to_reference"]:.3f}|')
    f = r['full_context_ub']
    for ub in (512, 2048):
        s = f[ub] if ub in f else f[str(ub)]
        lines.append(f'|全文1件（{f["input_tokens"]}トークン）|-ub {ub}|{s["wall_seconds"]:.1f}|{s["prompt_tokens_per_s"]:.0f}|{s["identical_to_reference"]:.3f}|')
    g = r['ingestion']
    lines += ['', f'## 取り込み（事実抽出 {g["sessions"]} セッション）', '',
              '|設定|合計秒|1件あたり秒|生成トークン/秒|逐次と一致|先読みの採択|', '|---|---:|---:|---:|---:|---|']
    for k in ('sequential', 'parallel2', 'parallel4', 'ngram_simple'):
        s = g[k]
        acc = f'{s["draft"]["draft_n_accepted"]}/{s["draft"]["draft_n"]}' if s['draft']['draft_n'] else '—'
        lines.append(f'|{k}|{s["wall_seconds"]:.1f}|{s["wall_seconds"]/s["requests"]:.2f}|{s["generated_tokens_per_s"]:.1f}|{s["identical_to_reference"]:.3f}|{acc}|')
    lines.append(f'\n逐次（-c 8192）が U1 の保存済み出力（-c 32768）と一致した割合：{g["sequential"]["identical_to_saved_u1_ctx32768"]:.3f}')
    return '\n'.join(lines)+'\n'


# -- dense only (no LLM rerank) -----------------------------------------------------------------

def dense_probe(budgets=(1024, 2048)):
    """Same T/V assembly as the budget probe, but the raw order is the dense (RRF) order without rerank."""
    prepared = json.loads((RUN/'prepared.json').read_text())
    units = json.loads((RUN/'embedding.json').read_text())['units']
    jobs = json.loads((RUN/'rerank-jobs.json').read_text())
    facts = json.loads((RUN/'facts.json').read_text())
    links = json.loads((RUN/'links.json').read_text())
    reranked = json.loads((RESULTS/'longmemeval-u1-budget-probe.json').read_text())['rows']
    ref = {(r['id'], r['arm'], r['budget']): r for r in reranked}
    counts = {}

    def count(text):
        if not text:
            return 0
        k = hashlib.sha256(text.encode()).hexdigest()
        if k not in counts:
            counts[k] = len(http('/tokenize', {'content': text, 'add_special': False, 'parse_special': False})['tokens'])
        return counts[k]

    rows, private = [], {}
    with Server(8192, label='dense') as srv:
        for qid, row in prepared['questions'].items():
            us, ts = units[qid], prepared['turns'][qid]
            raw = list(dict.fromkeys(t for i in jobs[qid]['order'] for t in us[i]['turns']))
            table = um.fact_table(ts, {s: v for s, v in facts.items() if s.startswith(qid+'#')})
            for arm in ('T', 'V'):
                for budget in budgets:
                    conv, retr, recent_ids, memory_ids = payload(arm, ts, raw, count, table, links[qid], budget)
                    chat = um.messages(conv, retr, row['question'], row['question_date'])
                    body = {'model': srv.model, 'messages': chat, **am.GENERATION,
                            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': am.SCHEMA}}}
                    reply = complete(body)
                    parsed, failure = am.parse({'choices': [{'finish_reason': reply['finish'], 'message': {'content': reply['content']}}]})
                    answer = parsed['answer'] if parsed else ''
                    kind = 'abstention' if qid.endswith('_abs') else row['question_type']
                    if failure:
                        label = 0
                    else:
                        prompt = JUDGE['templates'][kind].format(question=row['question'], answer=row['answer'], response=answer)
                        j = http('/v1/chat/completions', {'model': srv.model, 'messages': [{'role': 'user', 'content': prompt}], **JUDGE['generation']})
                        label, _ = g1m.judge_label(j)
                    rows.append({'id': qid, 'arm': arm, 'budget': budget, 'stratum': um.stratum(row), 'correct': label,
                                 'abstain': float(am.abstains(answer)), 'coverage': coverage(row, set(recent_ids) | set(memory_ids)),
                                 'memory_turns': len(memory_ids), 'failure': failure,
                                 'reranked_correct': ref[(qid, arm, budget)]['correct']})
                    private[f'{qid}|{arm}|{budget}'] = {'answer': answer}
            log('dense', qid, 'done')
    (OUT/'dense-private.json').write_text(json.dumps(private, ensure_ascii=False, indent=1))
    report = {'exploratory': True, 'questions': len(prepared['questions']), 'cells': {}, 'dense_minus_rerank': {}, 'rows': rows,
              'rerank_seconds_per_query_measured': json.loads((RESULTS/'longmemeval-u1-speed-probe.json').read_text())['rerank_ub']['512']['wall_seconds'] / 30}
    for arm in ('T', 'V'):
        for b in budgets:
            part = [r for r in rows if r['arm'] == arm and r['budget'] == b]
            key = f'{arm}@{b}'
            report['cells'][key] = {k: float(np.mean([r[k] for r in part])) for k in ('correct', 'abstain', 'coverage', 'memory_turns', 'reranked_correct')}
            ids = [r['id'] for r in part]
            d = [r['correct'] - r['reranked_correct'] for r in part]
            report['dense_minus_rerank'][key] = {'difference': float(np.mean(d)), 'ci95': g1m.interval(d, ids), 'sign': g1m.sign_test(d)}
    (RESULTS/'longmemeval-u1-dense-probe.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    lines = ['# U1 後の探索：Qwen の並べ替えを省く（埋め込みの順位だけ）', '',
             '登録外の探索。保存済みの並べ替え前の順位（p2+fact の RRF）から T・V の入力を組み立て、同じ Qwen で回答・採点した。'
             '比較相手は、同じ条件・同じ枠の「並べ替えあり」（記憶の枠の探索）。', '',
             f'並べ替え1回の実測：約{report["rerank_seconds_per_query_measured"]:.1f}秒（-ub 512、30件の平均）。', '',
             '|条件@枠|並べ替えなし 正答率|並べ替えあり 正答率|差（なし−あり）|95%区間|勝/負/同|棄権率|根拠セッション被覆|',
             '|---|---:|---:|---:|---|---|---:|---:|']
    for key, c in report['cells'].items():
        v = report['dense_minus_rerank'][key]
        lines.append(f'|{key}|{c["correct"]:.3f}|{c["reranked_correct"]:.3f}|{v["difference"]:+.3f}|[{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]|'
                     f'{v["sign"]["wins"]}/{v["sign"]["losses"]}/{v["sign"]["ties"]}|{c["abstain"]:.3f}|{c["coverage"]:.3f}|')
    (RESULTS/'longmemeval-u1-dense-probe.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['budget', 'speed', 'dense'])
    args = parser.parse_args()
    {'budget': budget_probe, 'speed': speed_probe, 'dense': dense_probe}[args.command]()
