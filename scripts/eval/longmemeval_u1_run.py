#!/usr/bin/env python3
"""U1: frozen C vs time-aware input (T) vs write-time update links (V) on a new LongMemEval sample."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import longmemeval_g1_continue as g1c
import longmemeval_g1_methods as g1m
import longmemeval_g1_report as g1r
import longmemeval_g1_run as g1
import longmemeval_u1_methods as um

lm, ae, mm, st, am, fp, base, rp = g1.lm, g1.ae, g1.mm, g1.st, g1.am, g1.fp, g1.base, g1.rp
ROOT = lm.ROOT
WORK = ROOT/'build/eval/memory/longmemeval-u1'
LOCAL = WORK/'run'
PLAN = lm.RESULTS/'longmemeval-u1-plan.json'
PREFIX = 'longmemeval-u1'
G1_PLAN = lm.RESULTS/'longmemeval-g1-execution-plan.json'
JUDGE = lm.RESULTS/'longmemeval-g1-judge.json'
CODE = ['scripts/eval/longmemeval_u1_methods.py', 'scripts/eval/longmemeval_u1_run.py',
        'docs/MEMORY_LONGMEMEVAL_U1_PREREG.md']
FROZEN = ['scripts/eval/longmemeval_g1_run.py', 'scripts/eval/longmemeval_g1_continue.py',
          'scripts/eval/longmemeval_g1_methods.py', 'scripts/eval/longmemeval_g1_report.py',
          'scripts/eval/locomo_answer_methods.py', 'scripts/eval/locomo_memory_methods.py',
          'scripts/eval/locomo_memory_stages.py', 'scripts/eval/locomo_answer_production.py',
          'results/eval/memory/longmemeval-g1-judge.json', 'results/eval/memory/generation.json',
          'results/eval/memory/final-config.json', 'results/eval/memory/answer-pilot-server.json']
RERANK_CONTEXT = 16384
FACT_WINDOWS, RERANK_WINDOWS = (16384, 32768), (16384, 32768, 65536)
ALLOWED = (8192, 16384, 32768, 65536)
INGEST_BUDGET, UPDATE_BUDGET = 12*3600, 4*3600
STALE_GATE = {'runner': 'longmemeval-g1-continuation', 'time': '2026-10-14T11:11:57.078313+09:00'}


# -- registration (metadata only) -------------------------------------------------------

def metadata():
    """IDs, types, session IDs and dates only; question/answer/history text is never read here."""
    raw = lm.read(g1.DATA)
    return [{'question_id': r['question_id'], 'question_type': r['question_type'],
             'sessions': len(r['haystack_session_ids'])} for r in raw]


def register():
    if PLAN.exists():
        raise RuntimeError('U1 plan exists; registration is not overwritten')
    g1p = lm.read(G1_PLAN)
    if lm.file_hash(g1.DATA) != g1p['dataset_sha256']:
        raise RuntimeError('benchmark source hash differs')
    meta = metadata()
    ids = um.sample(meta, set(g1p['question_ids']))
    sessions = {m['question_id']: m['sessions'] for m in meta}
    server = lm.read(lm.RESULTS/'answer-pilot-server.json')['new_args']
    plan = {
        'version': 1, 'created_at': lm.stamp(), 'experiment': 'U1 update-aware memory',
        'authorization': 'user chat 2026-10-22: proposal 1 (dated, update-aware memory) approved',
        'dataset_sha256': g1p['dataset_sha256'], 'model_sha256': g1p['model_sha256'], 'build': g1p['build'],
        'server_argv': server, 'seed': um.SEED, 'quota': um.QUOTA, 'excluded_G1_ids': g1p['question_ids'],
        'question_ids': ids, 'strata': {q: next(um.stratum(m) for m in meta if m['question_id'] == q) for q in ids},
        'sessions_total': sum(sessions[q] for q in ids), 'sessions_by_question': {q: sessions[q] for q in ids},
        'arms': list(um.ARMS), 'contexts': {'fact': 8192, 'update': 8192, 'rerank': RERANK_CONTEXT,
                                            'answer': 8192, 'judge': 8192},
        'selected_rerank_context': RERANK_CONTEXT,
        'prompts': {'system_T': um.SYSTEM_T, 'update': um.UPDATE,
                    'update_limits': {'top_earlier': um.TOP_EARLIER, 'max_earlier': um.MAX_EARLIER,
                                      'max_notes': um.MAX_NOTES}},
        'budgets_seconds': {'ingestion': INGEST_BUDGET, 'update': UPDATE_BUDGET},
        'judge_file': str(JUDGE.relative_to(ROOT)),
        'primary': {'metric': 'Qwen judge correct (official LongMemEval prompts, strict yes/no)',
                    'comparison': 'V - C-final-norw, all 30',
                    'rule': 'paired percentile bootstrap 10000 seed0 over questions: low>0 改善, high<0 悪化, else 判定不能'},
        'secondary': ['T - C-final-norw', 'V - T', 'focus knowledge-update+temporal-reasoning (20)',
                      'per stratum (trend only)', 'abstention rate', 'lexical F1 (reference)',
                      'answer-session coverage of answer inputs (after judging, CPU)'],
        'source_sha256': {f: lm.file_hash(ROOT/f) for f in FROZEN},
        'original_source_sha256': g1p['original_source_sha256'],  # frozen pipeline (G1 runner files: current hashes above)
        'code_sha256': {f: lm.file_hash(ROOT/f) for f in CODE},
        'estimate_hours': {'fact_ingestion': round(sum(sessions[q] for q in ids)*25997.644/1441/3600, 2),
                           'update_links': 1.5, 'rerank_answers_judge': 0.4, 'total': 9.5},
        'push': 'allowed by user 2026-10-22',
    }
    lm.write(PLAN, plan)
    print(json.dumps({'questions': len(ids), 'sessions_total': plan['sessions_total'],
                      'estimate_hours': plan['estimate_hours']}, ensure_ascii=False))


def checked():
    ae.gate()
    p = st.frozen(PLAN)
    g1.verify(p)
    for f, digest in p['code_sha256'].items():
        if lm.file_hash(ROOT/f) != digest:
            raise RuntimeError('registered U1 code changed: '+f)
    return p


# -- runtime --------------------------------------------------------------------------------

@contextmanager
def runtime(local):
    saved = g1.LOCAL, g1.PLAN, g1c.LOCAL
    g1.LOCAL, g1.PLAN, g1c.LOCAL = local, PLAN, local
    try:
        with g1.isolated():
            yield
    finally:
        g1.LOCAL, g1.PLAN, g1c.LOCAL = saved


def servers_running():
    out = subprocess.check_output(['ps', '-axo', 'pid=,comm='], text=True).splitlines()
    return [int(x.split()[0]) for x in out if Path(x.split(None, 1)[1].strip()).name == 'llama-server']


class Servers(g1c.Servers):
    """Fresh registered server per stage; leaves the machine as found (none running before → none after)."""
    def __init__(self, p):
        self.p = p
        self.argv = p['server_argv'].copy()
        self.initial = servers_running()
        self.up = bool(self.initial)

    def start(self, context, label, restore=False):
        if context not in ALLOWED:
            raise RuntimeError('unregistered window')
        self.p = {**self.p, 'selected_rerank_context': context}  # parent allowlist: 8192/this/131072
        return super().start(context, label, restore)

    def finish(self):
        if self.up:
            self.stop()


def journal(local):
    replies = {}
    path = local/'received.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record['key'] in replies:
                raise RuntimeError('duplicate journal key')
            replies[record['key']] = record['reply']
    return replies


def reply_for(key, body, now, replies, watch, tokens):
    """Resume reuses a received reply; it never sends a key twice."""
    if key in replies:
        return replies[key]
    return g1.send(key, body, now, replies, watch, tokens)


def body(now, chat, max_tokens, name, schema):
    return {'model': now['model_id'], 'messages': chat, 'temperature': 0, 'seed': 0, 'stream': False,
            'max_tokens': max_tokens, 'cache_prompt': True, 'chat_template_kwargs': {'enable_thinking': False},
            'response_format': {'type': 'json_schema', 'json_schema': {'name': name, 'schema': schema}}}


def progress(phase, **fields):
    g1.status(phase, status='running', **fields)


# -- stages ---------------------------------------------------------------------------------

def load_questions(p):
    if lm.file_hash(g1.DATA) != p['dataset_sha256']:
        raise RuntimeError('benchmark source hash differs')
    lookup = {r['question_id']: r for r in lm.read(g1.DATA)}
    return {q: {**lookup[q], 'history_group': lm.digest(list(zip(lookup[q]['haystack_session_ids'],
                                                                   lookup[q]['haystack_dates'])))}
            for q in p['question_ids']}


def prepare(p, now, local):
    if (local/'prepared.json').exists():
        prepared = lm.read(local/'prepared.json')
        if lm.file_hash(local/'prepared.json') != lm.read(local/'prepared-hash.json')['sha256']:
            raise RuntimeError('prepared input changed')
        return prepared
    questions = load_questions(p)
    count = ae.Tokenizer(now['url']); qwen = st.Qwen.offline()
    turns, jobs = {}, []
    try:
        for qid, row in questions.items():
            ae.gate()
            ts = turns[qid] = g1m.turns(row)
            for session, chat, extra in st.generation_jobs('fact', {}, ts):
                tokens = count.input_count(chat)
                jobs.append({'session': session, 'group': qid, 'body': qwen.request('fact', chat),
                             'input_tokens': tokens, **extra})
            progress('preflight', checked_sessions=len(jobs), sessions_total=p['sessions_total'])
    finally:
        count.save()
    if len(jobs) != p['sessions_total']:
        raise RuntimeError('registered session count differs')
    longest = max(j['input_tokens'] for j in jobs)
    out = mm.PROMPTS['fact']['max_tokens']
    context = 8192 if g1.fits(longest, out, 8192) else um.window(longest, out, FACT_WINDOWS)
    if context is None:
        raise RuntimeError('fact input exceeds every registered window; no chunking')
    prepared = {'questions': questions, 'turns': turns, 'fact_jobs': jobs, 'fact_context': context}
    lm.write(lm.RESULTS/f'{PREFIX}-preflight.json', {
        'time': lm.stamp(), 'fact_sessions': len(jobs), 'fact_input_max': longest,
        'over_8192': sum(not g1.fits(j['input_tokens'], out, 8192) for j in jobs), 'fact_context': context,
        'rule': 'smallest of 16384/32768 >= ceil(1.10*(max+1500+128)) when any input misses 8192',
        'before_any_generation': True})
    lm.write(local/'prepared.json', prepared)
    lm.write(local/'prepared-hash.json', {'sha256': lm.file_hash(local/'prepared.json')})
    g1.event('preflight complete', questions=len(questions), fact_sessions=len(jobs))
    return prepared


def ingest(prepared, pid, now, replies, local):
    facts, total = {}, len(prepared['fact_jobs'])
    watch = fp.SeriesWatch(pid, lambda kind, data: g1.event(kind, stage='fact', **data))
    tick = time.monotonic(); marks = {round(total*f) for f in (.25, .5, .75, 1.)}
    try:
        with watch:
            for number, job in enumerate(prepared['fact_jobs'], 1):
                ae.gate()
                if time.monotonic()-tick+900 >= INGEST_BUDGET:
                    raise RuntimeError('ingestion budget reserve reached; checkpoint and wait')
                watch.begin(number)
                reply = reply_for('fact|'+job['session'], job['body'], now, replies, watch, job['input_tokens'])
                content = reply['response']['choices'][0]['message'].get('content') or ''
                value = mm.parse('fact', content, valid_ids=job['valid_ids'])
                facts[job['session']] = [{**v, 'turns': [job['group']+'/'+t for t in v['turns']]} for v in value or []]
                if value is None or reply['response']['choices'][0]['finish_reason'] != 'stop':
                    g1.event('generation failure', stage='fact', session_key=job['session'],
                             fallback='frozen parser/fallback, no re-generation')
                watch.boundary()
                progress('fact ingestion', received=number, total=total, phase_elapsed_seconds=time.monotonic()-tick)
                if number in marks:
                    g1.event('ingestion milestone', received=number, total=total,
                             phase_elapsed_seconds=time.monotonic()-tick)
    finally:
        lm.write(local/'facts.json', facts); lm.write(local/'fact-resources.json', watch.record())
    if watch.error:
        raise RuntimeError(watch.error)
    return facts


def embedding(prepared, facts, manager, local):
    if (local/'embedding.json').exists():
        saved = lm.read(local/'embedding.json')
        vectors = {q: np.load(local/f'doc-vectors-{q}.npy') for q in prepared['questions']}
        return saved['units'], vectors, saved['query_vectors'], saved['timings']
    manager.finish()  # serial GPU stages: Qwen is absent during embedding
    from egemma_server.tokenizer import PromptTokenizer
    tk = PromptTokenizer(rp.V1_ST_DIR)
    identity = lm.fingerprint(st.VARIANT)
    if identity != lm.read(lm.RESULTS/'final-config.json')['embedding_artifact_sha256']:
        raise RuntimeError('frozen embedding identity differs')
    args = argparse.Namespace(port=8398, model_dir=ROOT/'models/embeddinggemma-2', v1_dir=rp.V1_ST_DIR)
    vectors, units_by_q, qvectors, timing = {}, {}, {}, {}
    with lm.start_backend(st.VARIANT, args) as (backend, embed):
        def compute(texts):
            out = []
            for start in range(0, len(texts), 32):
                ae.gate(); out.extend(embed(texts[start:start+32]))
            return lm.checked_vectors(np.asarray(out, np.float32), len(texts))
        for number, (qid, ts) in enumerate(prepared['turns'].items(), 1):
            part = {s: v for s, v in facts.items() if s.startswith(qid+'#')}
            units = st.build_units(['p2', 'fact'], ts, {'fact': part})
            tick = time.monotonic()
            vector = compute(rp.fit_like_st(tk, [lm.PROTOCOL['prompts']['document']+u['text'] for u in units]))
            np.save(local/f'doc-vectors-{qid}.npy', vector)
            timing[qid] = {'document_seconds': time.monotonic()-tick, 'document_units': len(units)}
            tick = time.monotonic()
            question = prepared['questions'][qid]['question']
            qvectors[qid] = compute(rp.fit_like_st(tk, [lm.PROTOCOL['prompts']['query']+question]))[0].tolist()
            timing[qid]['query_seconds'] = time.monotonic()-tick
            units_by_q[qid] = units; vectors[qid] = vector
            progress('embedding', completed_questions=number, total_questions=len(prepared['turns']))
    lm.write(local/'embedding.json', {'backend': backend, 'identity': identity, 'timings': timing,
                                      'query_vectors': qvectors, 'units': units_by_q})
    g1.event('ingestion complete', embedding_questions=len(units_by_q))
    return units_by_q, vectors, qvectors, timing


def fact_vectors(units, vectors):
    return {u['id'].split('|', 1)[1]: vectors[i].tolist() for i, u in enumerate(units) if u['set'] == 'fact'}


def updates(prepared, facts, units, vectors, pid, now, replies, local):
    """Write-time links: each later session's facts checked against similar earlier facts (one call per session)."""
    count = ae.Tokenizer(now['url'])
    plans = {}
    for qid, ts in prepared['turns'].items():
        part = {s: v for s, v in facts.items() if s.startswith(qid+'#')}
        table = um.fact_table(ts, part)
        plans[qid] = (table, um.update_jobs(table, fact_vectors(units[qid], vectors[qid])))
    total = sum(len(j) for _, j in plans.values())
    links, stats = {}, {'calls': 0, 'parse_failures': 0, 'overflow': 0, 'links': 0}
    watch = fp.SeriesWatch(pid, lambda kind, data: g1.event(kind, stage='update', **data))
    tick, number = time.monotonic(), 0
    try:
        with watch:
            for qid, (table, jobs) in plans.items():
                lookup = {f['key']: f for f in table}
                links[qid] = []
                for job in jobs:
                    ae.gate(); number += 1
                    if time.monotonic()-tick+900 >= UPDATE_BUDGET:
                        raise RuntimeError('update budget reserve reached; checkpoint and wait')
                    chat = um.update_messages(job, lookup)
                    tokens = count.input_count(chat)
                    key = 'update|'+job['session']
                    if not g1.fits(tokens, um.UPDATE['max_tokens'], 8192):
                        stats['overflow'] += 1  # registered: no links for this session, no truncation
                        g1.event('input overflow', stage='update', session_key=job['session'], input_tokens=tokens)
                        continue
                    watch.begin(number)
                    reply = reply_for(key, body(now, chat, um.UPDATE['max_tokens'], 'updates', um.UPDATE['schema']),
                                      now, replies, watch, tokens)
                    choice = reply['response']['choices'][0]
                    value = um.parse_updates(choice['message'].get('content') or '', job)
                    if value is None or choice['finish_reason'] != 'stop':
                        stats['parse_failures'] += 1; value = value or []
                    links[qid] += value; stats['calls'] += 1; stats['links'] += len(value)
                    watch.boundary()
                    progress('update links', completed=number, total=total,
                             phase_elapsed_seconds=time.monotonic()-tick)
    finally:
        count.save()
        lm.write(local/'links.json', links); lm.write(local/'update-resources.json', watch.record())
    if watch.error:
        raise RuntimeError(watch.error)
    lm.write(local/'update-stats.json', {**stats, 'jobs': total, 'seconds': time.monotonic()-tick})
    g1.event('update links complete', **stats, jobs=total)
    return {q: plans[q][0] for q in plans}, links, stats


def assemble(prepared, qid, raw_ids, table, links, count):
    row, ts = prepared['questions'][qid], prepared['turns'][qid]
    result = {}
    for arm in um.ARMS:
        tick = time.monotonic()
        if arm == 'C-final-norw':
            payload = am.payload(arm, ts, {arm: raw_ids}, count.count)
            chat = am.messages(payload.pop('conversation'), payload.pop('retrieved'), row['question'])
        elif arm == 'S':  # date order only: frozen system prompt and template, no question date
            payload = um.payload(arm, ts, raw_ids, count.count, table, links)
            chat = am.messages(payload.pop('conversation'), payload.pop('retrieved'), row['question'])
        else:
            payload = um.payload(arm, ts, raw_ids, count.count, table, links)
            chat = um.messages(payload.pop('conversation'), payload.pop('retrieved'), row['question'], row['question_date'])
        tokens = count.input_count(chat)
        if not g1.fits(tokens, 256, 8192):
            g1.event('input overflow', stage='answer preflight', condition=arm, question_id=qid, input_tokens=tokens)
            raise RuntimeError('answer input overflow; human review, no window change')
        result[arm] = {'messages': chat, 'input_tokens': tokens, 'assembly_seconds': time.monotonic()-tick, **payload}
    return result


def rerank_jobs(prepared, units, vectors, qvectors, now, local):
    """All rerank inputs built and measured before any rerank call; the window follows the G1 rule."""
    count = ae.Tokenizer(now['url']); qwen = st.Qwen.offline()
    cfg = mm.config(['p2', 'fact'], rerank='qwen')
    jobs = {}
    try:
        for qid, row in prepared['questions'].items():
            ae.gate()
            ts, us = prepared['turns'][qid], units[qid]
            tick = time.monotonic()
            index = mm.Index(us, np.asarray(vectors[qid]))
            order = mm.distinct([i for i, _ in index.rank(cfg, qid, [(np.asarray(qvectors[qid]), row['question'])])], us)
            seconds = time.monotonic()-tick
            chat = base.norw.prod.rerank_chat({'text': row['question']}, order, us, ts, cfg)
            jobs[qid] = {'order': order, 'body': qwen.request('rerank', chat), 'input_tokens': count.input_count(chat),
                         'search_seconds': seconds}
    finally:
        count.save()
    longest = max(j['input_tokens'] for j in jobs.values())
    context = um.window(longest, mm.PROMPTS['rerank']['max_tokens'], RERANK_WINDOWS)
    if context is None:
        raise RuntimeError('rerank input exceeds every registered window; human review')
    lm.write(local/'rerank-jobs.json', jobs)
    g1.event('rerank preflight', questions=len(jobs), rerank_input_max=longest, rerank_context=context)
    return jobs, context


def search(prepared, units, jobs, tables, links, pid, now, replies, local, saved_rerank=None):
    count = ae.Tokenizer(now['url'])
    inputs, timings = {a: {} for a in um.ARMS}, {}
    watch = fp.SeriesWatch(pid, lambda kind, data: g1.event(kind, stage='rerank', **data))
    try:
        with watch:
            for number, qid in enumerate(prepared['questions'], 1):
                ae.gate(); watch.begin(number)
                us, job = units[qid], jobs[qid]
                order = job['order']
                if saved_rerank is not None:
                    reply = saved_rerank[qid]
                else:
                    reply = reply_for('rerank|'+qid, job['body'], now, replies, watch, job['input_tokens'])
                text = reply['response']['choices'][0]['message'].get('content') or ''
                value = mm.parse('rerank', text, n_passages=min(len(order), mm.RERANK_CANDIDATES))
                if value is None or reply['response']['choices'][0]['finish_reason'] != 'stop':
                    g1.event('generation failure', stage='rerank', fallback='frozen order rule; no resend')
                order = mm.distinct(base.norw.prod.reranked_order(order, value), us)
                raw_ids = list(dict.fromkeys(t for i in order for t in us[i]['turns']))
                item = assemble(prepared, qid, raw_ids, tables[qid], links[qid], count)
                for arm in um.ARMS:
                    inputs[arm][qid] = item[arm]
                timings[qid] = {'search_seconds': job['search_seconds'], 'rerank_seconds': reply['wall_seconds']}
                watch.boundary()
                progress('rerank and answer preflight', completed=number, total=len(prepared['questions']))
    finally:
        count.save(); lm.write(local/'rerank-resources.json', watch.record())
    if watch.error:
        raise RuntimeError(watch.error)
    lm.write(local/'answer-inputs.json', inputs); lm.write(local/'search-timings.json', timings)
    g1.event('rerank complete', questions=len(timings))
    return inputs, timings


def answers(prepared, inputs, manager, replies, local, ids=None):
    result = {}
    ids = ids or list(prepared['questions'])
    for arm in um.ARMS:
        pid, now = manager.start(8192, 'answer-'+arm)
        watch = fp.SeriesWatch(pid, lambda kind, data: g1.event(kind, stage='answer', condition=arm, **data))
        result[arm] = {}
        try:
            with watch:
                for number, qid in enumerate(ids, 1):
                    ae.gate(); watch.begin(number)
                    item = inputs[arm][qid]
                    request = {'model': now['model_id'], 'messages': item['messages'], **am.GENERATION,
                               'response_format': {'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': am.SCHEMA}}}
                    result[arm][qid] = reply_for('answer|'+arm+'|'+qid, request, now, replies, watch, item['input_tokens'])
                    watch.boundary()
                    progress('answer', condition=arm, completed=number, total=len(ids))
        finally:
            lm.write(local/f'answer-resources-{arm}.json', watch.record())
        if watch.error:
            raise RuntimeError(watch.error)
        g1.event('condition complete', condition=arm, questions=len(ids))
    lm.write(local/'answers.json', result)
    return result


def judge(prepared, answers_, manager, replies, local, ids=None):
    spec = lm.read(JUDGE)
    pid, now = manager.start(8192, 'judge')
    ids = sorted(ids or prepared['questions'], key=lambda q: hashlib.sha256(('longmemeval-u1-judge-v1'+q).encode()).hexdigest())
    judges = {a: {} for a in um.ARMS}
    count = ae.Tokenizer(now['url'])
    watch = fp.SeriesWatch(pid, lambda kind, data: g1.event(kind, stage='judge', **data))
    try:
        with watch:
            for number, qid in enumerate(ids, 1):
                row = prepared['questions'][qid]
                for arm in sorted(um.ARMS, key=lambda a: lm.digest('longmemeval-u1-judge-v1|'+qid+'|'+a)):
                    ae.gate(); key = 'judge|'+arm+'|'+qid
                    answer = g1r.answer_value(answers_[arm][qid], row['answer'])
                    if answer['failure']:
                        judges[arm][qid] = {'label': 0, 'failure': None, 'generation_failed': True}
                        continue
                    kind = 'abstention' if qid.endswith('_abs') else row['question_type']
                    prompt = spec['templates'][kind].format(question=row['question'], answer=row['answer'],
                                                            response=answer['answer'])
                    chat = [{'role': 'user', 'content': prompt}]
                    tokens = count.input_count(chat)
                    if not g1.fits(tokens, 10, 8192):
                        raise RuntimeError('judge input window overflow; no change/retry')
                    watch.begin(number)
                    reply = reply_for(key, {'model': now['model_id'], 'messages': chat, **spec['generation']},
                                      now, replies, watch, tokens)
                    label, failure = g1m.judge_label(reply['response'])
                    judges[arm][qid] = {'label': label, 'failure': failure}
                    watch.boundary()
                progress('judge', completed_paired_questions=number, total=len(ids))
    finally:
        count.save(); lm.write(local/'judges.json', judges); lm.write(local/'judge-resources.json', watch.record())
    if watch.error:
        raise RuntimeError(watch.error)
    return judges


# -- report ---------------------------------------------------------------------------------

def verdict(ci):
    return '改善' if ci[0] > 0 else '悪化' if ci[1] < 0 else '判定不能'


def compare(rows, ids, left, right, metric='correct'):
    index = {a: {r['id']: r for r in rows[a]} for a in um.ARMS}
    out = {}
    for name, keep in (('overall', lambda r: True),
                       ('focus_update_temporal', lambda r: r['stratum'] in ('knowledge-update', 'temporal-reasoning')),
                       *[(s, (lambda s: lambda r: r['stratum'] == s)(s)) for s in
                         ('knowledge-update', 'temporal-reasoning', 'multi-session', 'single-session', 'abstention')]):
        chosen = [q for q in ids if keep(index[left][q])]
        if not chosen:
            continue
        diffs = [index[left][q][metric]-index[right][q][metric] for q in chosen]
        ci = g1m.interval(diffs, chosen)
        out[name] = {'questions': len(chosen), 'difference': float(np.mean(diffs)), 'ci95': ci,
                     'verdict': verdict(ci) if metric == 'correct' and name == 'overall' else '傾向参考',
                     'sign_test': g1m.sign_test(diffs), 'by_question': dict(zip(chosen, diffs))}
    return out


def coverage(row, item):
    """Answer sessions with at least one message in the answer input (evaluation metadata, after judging)."""
    ids = set(item['recent_ids']) | set(item['memory_ids'])
    positions = {i+1 for i, s in enumerate(row['haystack_session_ids']) if s in set(row['answer_session_ids'])}
    present = {int(t.split('/D', 1)[1].split(':')[0]) for t in ids}
    return float(bool(positions) and positions <= present)


def analyze(prepared, inputs, answers_, judges):
    ids = list(prepared['questions'])
    rows = {a: [] for a in um.ARMS}
    review = []
    for qid in ids:
        q = prepared['questions'][qid]
        item = {'id': qid, 'question': q['question'], 'gold': q['answer'], 'question_type': q['question_type'],
                'conditions': {}}
        for arm in um.ARMS:
            value = g1r.answer_value(answers_[arm][qid], q['answer'])
            rows[arm].append({**{k: v for k, v in value.items() if k != 'answer'}, 'id': qid,
                              'stratum': g1r.stratum(q), 'type': q['question_type'],
                              'correct': judges[arm][qid]['label'],
                              'judge_failure': float(bool(judges[arm][qid].get('failure'))),
                              'input_tokens': inputs[arm][qid]['input_tokens'],
                              'answer_session_coverage': coverage(q, inputs[arm][qid]),
                              'notes_shown': inputs[arm][qid].get('notes_shown', 0)})
            item['conditions'][arm] = {'answer': value['answer'], 'judge': judges[arm][qid]}
        review.append(item)
    public = {'questions': len(ids), 'metrics': {a: g1r.metrics(rows[a]) for a in um.ARMS},
              'coverage': {a: float(np.mean([r['answer_session_coverage'] for r in rows[a]])) for a in um.ARMS},
              'V_notes_shown_questions': sum(r['notes_shown'] > 0 for r in rows['V']),
              'primary_V_minus_C': compare(rows, ids, 'V', 'C-final-norw'),
              'S_minus_C': compare(rows, ids, 'S', 'C-final-norw'),
              'T_minus_S': compare(rows, ids, 'T', 'S'),
              'V_minus_T': compare(rows, ids, 'V', 'T'),
              'F1_V_minus_C_reference': compare(rows, ids, 'V', 'C-final-norw', 'f1'),
              'scope': 'fixed30 new English LongMemEval-S sample; Qwen self judge; contamination unknown'}
    return public, rows, review


def markdown(report, stats):
    m = report['metrics']
    lines = ['# U1：更新を追える記憶（LongMemEval 新標本30問）', '',
             f'更新の判定 {stats["calls"]} 回、リンク {stats["links"]} 件、解析失敗 {stats["parse_failures"]}、窓超過 {stats["overflow"]}。'
             f'V で注記が入った質問 {report["V_notes_shown_questions"]}/30。', '',
             '|条件|種類|問数|正答率|語彙F1（参考）|棄権率|生成失敗|根拠セッション被覆|',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for arm in um.ARMS:
        for kind in g1r.STRATA:
            v = m[arm][kind]
            if v['questions']:
                cov = f'{report["coverage"][arm]:.3f}' if kind == 'overall' else ''
                lines.append(f'|{arm}|{kind}|{v["questions"]}|{v["correct"]:.3f}|{v["f1"]:.3f}|'
                             f'{v["abstention"]:.3f}|{v["failure"]:.3f}|{cov}|')
    lines += ['', '|比較|範囲|問数|点差|95%区間|判定|勝/負/同|符号検定p|', '|---|---|---:|---:|---|---|---|---:|']
    for name in ('primary_V_minus_C', 'S_minus_C', 'T_minus_S', 'V_minus_T'):
        for kind, v in report[name].items():
            s = v['sign_test']
            lines.append(f'|{name}|{kind}|{v["questions"]}|{v["difference"]:+.3f}|'
                         f'[{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]|{v["verdict"]}|'
                         f'{s["wins"]}/{s["losses"]}/{s["ties"]}|{s["p_two_sided"]:.4f}|')
    lines += ['', '主判定は全30問の V−C だけ。種類別・S−C・T−S・V−T は傾向の参考。区間は質問単位の対応ありブートストラップ（10000回、seed 0）。',
              '正誤は同じ Qwen による自己採点（LongMemEval 公式の種類別プロンプト、yes/no の厳密解析）。英語の合成履歴30問に限る。']
    return '\n'.join(lines)+'\n'


# -- commands ---------------------------------------------------------------------------------

def release_stale_gate():
    if not ae.PAUSE.exists():
        return
    gate = lm.read(ae.PAUSE)
    if {k: gate.get(k) for k in STALE_GATE} != STALE_GATE:
        raise RuntimeError('an unexpected pause gate is active; human decision required')
    archived = ae.PAUSE.parent/'user-pause-archived-before-u1.json'
    lm.write(archived, {**gate, 'archived_at': lm.stamp(), 'by': 'U1 start',
                        'authorization': 'user chat 2026-10-22 (proposal 1 approved)'})
    ae.PAUSE.unlink()


def stop_gate(error):
    if not ae.PAUSE.exists():
        lm.write(ae.PAUSE, {'time': lm.stamp(), 'runner': 'longmemeval-u1', 'reason': f'stopped: {error}'})


def run(resume=False):
    if resume:
        gate = lm.read(ae.PAUSE) if ae.PAUSE.exists() else {}
        if gate and gate.get('runner') != 'longmemeval-u1':
            raise RuntimeError('a pause gate from another runner is active')
        if gate:
            lm.write(LOCAL/'resume-gates.jsonl', gate); ae.PAUSE.unlink()
        if (LOCAL/'inflight.json').exists():
            raise RuntimeError('unknown inflight request; human review before resume')
    else:
        release_stale_gate()
        if any((LOCAL/n).exists() for n in ('run-start.json', 'received.jsonl', 'inflight.json')):
            raise RuntimeError('already started; use resume')
    p = checked(); g1.other_workers()
    LOCAL.mkdir(parents=True, exist_ok=True)
    if not resume:
        lm.write(LOCAL/'run-start.json', {'time': lm.stamp(), 'unix_time': time.time(), 'pid': os.getpid(),
                                          'plan_sha256': lm.file_hash(PLAN)})
    replies = journal(LOCAL)
    start = time.monotonic()
    with runtime(LOCAL):
        manager = Servers(p)
        try:
            pid, now = manager.start(8192, 'fact')
            prepared = prepare(p, now, LOCAL)
            if prepared['fact_context'] != 8192:
                pid, now = manager.start(prepared['fact_context'], f'fact-{prepared["fact_context"]}')
            facts = ingest(prepared, pid, now, replies, LOCAL)
            units, vectors, qvectors, etime = embedding(prepared, facts, manager, LOCAL)
            pid, now = manager.start(8192, 'update')
            tables, links, stats = updates(prepared, facts, units, vectors, pid, now, replies, LOCAL)
            jobs, context = rerank_jobs(prepared, units, vectors, qvectors, now, LOCAL)
            pid, now = manager.start(context, 'rerank')
            inputs, timings = search(prepared, units, jobs, tables, links, pid, now, replies, LOCAL)
            answers_ = answers(prepared, inputs, manager, replies, LOCAL)
            judges = judge(prepared, answers_, manager, replies, LOCAL)
            manager.finish()
            checked()
            report, rows, review = analyze(prepared, inputs, answers_, judges)
            report.update(time=lm.stamp(), update_stats=stats, embedding=etime, search_timings=timings,
                          session_wall_seconds=time.monotonic()-start, HTTP_records=len(replies),
                          plan_sha256=lm.file_hash(PLAN))
            lm.write(LOCAL/'scored-rows.json', rows); lm.write(LOCAL/'review.json', review)
            lm.write(lm.RESULTS/f'{PREFIX}-report.json', report)
            (lm.RESULTS/f'{PREFIX}-report.md').write_text(markdown(report, stats))
            g1.status('complete', status='complete', records=len(replies))
            g1.event('evaluation complete', records=len(replies))
        except BaseException as error:
            g1.status('stopped', status='stopped', error=f'{type(error).__name__}: {error}',
                      records=len(replies), unknown_inflight=(LOCAL/'inflight.json').exists())
            g1.event('stopped', error=f'{type(error).__name__}: {error}')
            try:
                manager.finish()  # before the pause gate: stopping the server passes through ae.gate()
            except BaseException as cleanup:
                g1.event('server cleanup failed', error=f'{type(cleanup).__name__}: {cleanup}')
            stop_gate(f'{type(error).__name__}: {error}')
            raise


def smoke(n=2):
    """Code check on saved G1 artifacts (design sample): C must equal G1's C input; T/V/update/answer/judge run."""
    source, cont = ROOT/'build/eval/memory/longmemeval-g1/run', ROOT/'build/eval/memory/longmemeval-g1/continuation'
    local = WORK/'smoke'
    release_stale_gate()
    if local.exists():
        raise RuntimeError('smoke output exists')
    local.mkdir(parents=True)
    g1p = lm.read(G1_PLAN)
    p = {**lm.read(PLAN), 'question_ids': None} if PLAN.exists() else None
    if p is None:
        raise RuntimeError('register first')
    prepared_all = lm.read(source/'prepared.json')
    ids = [q for q in g1p['question_ids'] if prepared_all['questions'][q]['question_type'] == 'knowledge-update'][:n]
    prepared = {'questions': {q: prepared_all['questions'][q] for q in ids},
                'turns': {q: prepared_all['turns'][q] for q in ids}}
    del prepared_all
    facts = {s: v for s, v in lm.read(source/'facts.json').items() if s.split('#')[0] in ids}
    emb = lm.read(source/'embedding.json')
    units = {q: emb['units'][q] for q in ids}
    vectors = {q: np.load(source/f'doc-vectors-{q}.npy') for q in ids}
    qvectors = {q: emb['query_vectors'][q] for q in ids}
    rerank = {r['key'].split('|', 1)[1]: r['reply'] for r in map(json.loads, (cont/'received.jsonl').read_text().splitlines())
              if r['key'].startswith('rerank|') and r['key'].split('|', 1)[1] in ids}
    g1_inputs = lm.read(cont/'answer-inputs.json')['C-final-norw']
    replies = {}
    with runtime(local):
        manager = Servers(p)
        try:
            pid, now = manager.start(8192, 'update')
            tables, links, stats = updates(prepared, facts, units, vectors, pid, now, replies, local)
            jobs, context = rerank_jobs(prepared, units, vectors, qvectors, now, local)
            pid, now = manager.start(context, 'rerank')
            inputs, _ = search(prepared, units, jobs, tables, links, pid, now, replies, local, saved_rerank=rerank)
            same_c = all(inputs['C-final-norw'][q]['messages'] == g1_inputs[q]['messages'] for q in ids)
            answers_ = answers(prepared, inputs, manager, replies, local, ids)
            judges = judge(prepared, answers_, manager, replies, local, ids)
            manager.finish()
        except BaseException:
            try:
                manager.finish()
            except BaseException:
                pass
            raise
    summary = {'questions': len(ids), 'C_input_identical_to_G1': same_c, 'update_stats': stats,
               'links': {q: len(v) for q, v in links.items()},
               'notes_shown': {a: {q: inputs[a][q].get('notes_shown', 0) for q in ids} for a in ('S', 'T', 'V')},
               'input_tokens': {a: {q: inputs[a][q]['input_tokens'] for q in ids} for a in um.ARMS},
               'answer_parse_ok': {a: {q: am.parse(answers_[a][q]['response'])[1] is None for q in ids} for a in um.ARMS},
               'judge_failures': {a: {q: judges[a][q].get('failure') for q in ids} for a in um.ARMS}}
    lm.write(local/'smoke-summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'smoke', 'run', 'resume'])
    args = parser.parse_args()
    {'register': register, 'smoke': smoke, 'run': run, 'resume': lambda: run(resume=True)}[args.command]()
