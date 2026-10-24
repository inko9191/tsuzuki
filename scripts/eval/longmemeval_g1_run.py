#!/usr/bin/env python3
"""One frozen G1 run: preflight, fact ingestion, serial embedding, C/B/A, blind judge."""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.error

import numpy as np
import locomo_long_footprint as fp
import longmemeval_g1_methods as methods
import longmemeval_g1_report as reporting

base, lm, ae, st, mm, am, rp = fp.base, fp.lm, fp.ae, fp.base.st, fp.base.mm, fp.base.am, fp.base.rp
LOCAL = lm.ROOT/'build/eval/memory/longmemeval-g1/run'
DATA = LOCAL.parent/'longmemeval_s_cleaned.json'
PARENT = lm.RESULTS/'longmemeval-g1-plan.json'
PLAN = lm.RESULTS/'longmemeval-g1-execution-plan.json'
PREFIX = 'longmemeval-g1'
EVENT_LOCK = threading.Lock()


def status(phase, **fields):
    start = lm.read(LOCAL/'run-start.json') if (LOCAL/'run-start.json').exists() else {}
    lm.write(LOCAL/'status.json', {'time': lm.stamp(), 'pid': os.getpid(), 'pgid': os.getpgid(0),
             'phase': phase, 'elapsed_seconds': time.time()-start.get('unix_time', time.time()), **fields})


def event(kind, **fields):
    with EVENT_LOCK:
        ae.received(LOCAL/'events.jsonl', kind, {'time': lm.stamp(), 'kind': kind, **fields})
    print(kind, flush=True)


def verify(p):
    for name, digest in {**p['source_sha256'], **p['original_source_sha256']}.items():
        if lm.file_hash(lm.ROOT/name) != digest:
            raise RuntimeError('protected immutable file changed: '+name)
    return True


def checked():
    ae.gate()
    p = st.frozen(PLAN)
    verify(p)
    return p


def other_workers():
    for line in subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or int(parts[0]) == os.getpid():
            continue
        command = parts[1]
        if 'python' in Path(command.split()[0]).name and ('scripts/eval/' in command and ' run' in command
                or any(x in command for x in ('egemma_server', 'ComfyUI', 'mlx_lm'))):
            raise RuntimeError('other evaluation/backend process active: '+parts[0])


@contextmanager
def isolated():
    saved = (lm.LOCAL, ae.LOCAL)
    env = {k: os.environ.get(k) for k in ('LLAMA_ARG_CACHE_RAM', 'LLAMA_ARG_LOG_VERBOSITY')}
    lm.LOCAL = ae.LOCAL = LOCAL
    os.environ['LLAMA_ARG_CACHE_RAM'] = '0'
    os.environ['LLAMA_ARG_LOG_VERBOSITY'] = '4'
    try:
        yield
    finally:
        lm.LOCAL, ae.LOCAL = saved
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def fits(tokens, output, context):
    return tokens+output+128 <= context


def reserve_budget(start):
    if time.monotonic()-start+900 >= 43200:
        raise RuntimeError('ingestion12h budget reserve reached; checkpoint and wait')


class Servers:
    def __init__(self, p):
        self.p = p
        self.argv = lm.read(lm.RESULTS/'answer-pilot-server.json')['new_args'].copy()
        self.up = True  # an absent initial/expected server is an error, never auto-recovered

    def stop(self):
        ae.gate()
        if not self.up:
            raise RuntimeError('cannot stop an expected-absent server')
        now = base.norw.ab.server()
        env = base.norw.ab.quiet()
        instance = env['qwen_servers'][0]
        expected = self.argv.copy()
        expected[expected.index('-c')+1] = str(now['n_ctx'])
        if instance['command'] != ' '.join(expected):
            raise RuntimeError('exact registered server argv differs')
        os.kill(instance['pid'], signal.SIGTERM)
        for _ in range(100):
            try:
                os.waitpid(instance['pid'], os.WNOHANG)
            except ChildProcessError:
                pass
            try:
                os.kill(instance['pid'], 0)
            except ProcessLookupError:
                self.up = False
                event('planned server stopped', pid=instance['pid'])
                return
            time.sleep(.1)
        raise RuntimeError('planned server stop failed; no duplicate start')

    def start(self, context, label, restore=False):
        ae.gate()
        if self.up:
            self.stop()
        other_workers()
        processes = subprocess.check_output(['ps', '-axo', 'comm='], text=True).splitlines()
        if any(Path(x.strip()).name == 'llama-server' for x in processes):
            raise RuntimeError('server exists before planned fresh start')
        if context not in (8192, 131072):
            raise RuntimeError('unregistered window')
        os_state = base.resources.observe_os()
        issue = base.resources.violation({'physical_footprint_bytes': 0}, os_state)
        if issue:
            raise RuntimeError(issue)
        argv = self.argv.copy()
        argv[argv.index('-c')+1] = str(context)
        if lm.file_hash(Path(argv[argv.index('-m')+1])) != self.p['model_sha256']:
            raise RuntimeError('fresh model hash differs')
        env = os.environ.copy()
        if restore:
            env.pop('LLAMA_ARG_CACHE_RAM', None)
            env.pop('LLAMA_ARG_LOG_VERBOSITY', None)
        tick = time.monotonic()
        with (LOCAL/f'qwen-{label}.log').open('a') as log:
            proc = subprocess.Popen(argv, cwd=lm.ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        self.up = True
        for _ in range(60):
            if proc.poll() is not None:
                raise RuntimeError('planned fresh server exited; no retry')
            try:
                now = base.norw.ab.server(context)
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(1)
        else:
            raise RuntimeError('planned server ready timeout; no retry')
        if now['build'] != self.p['build']:
            raise RuntimeError('server build differs')
        record = {'pid': proc.pid, 'context': context, 'label': label, 'server': now,
                  'initial_OS': os_state, 'ready_OS': base.resources.observe_os(),
                  'heavy_processes': fp.audit.heavy_processes(proc.pid),
                  'seconds': time.monotonic()-tick, 'argv': argv, 'restored_original': restore}
        ae.received(LOCAL/'server-starts.jsonl', label, record)
        event('allocation ready', label=label, pid=proc.pid, context=context,
              initial_swap_bytes=os_state['swap_used_bytes'], heavy=record['heavy_processes'])
        return proc.pid, now


def body_for(qwen, kind, messages):
    return qwen.request(kind, messages)


def send(key, body, now, replies, watch, tokens):
    ae.gate()
    if key in replies or (LOCAL/'inflight.json').exists():
        raise RuntimeError('received or unknown inflight; no resend')
    if watch.error:
        raise RuntimeError(watch.error)
    digest = lm.digest({'body': body, 'model': now['model_sha256']})
    lm.write(LOCAL/'inflight.json', {'key': key, 'digest': digest, 'time': lm.stamp()})
    tick = time.monotonic()
    response = st.http_json(now['url']+'/v1/chat/completions', body, timeout=900)
    reply = {'response': response, 'wall_seconds': time.monotonic()-tick,
             'timings': response.get('timings') or {}, 'input_tokens': tokens,
             'attempts': 1, 'digest': digest, 'time': lm.stamp()}
    replies[key] = reply
    ae.received(LOCAL/'received.jsonl', key, reply)
    (LOCAL/'inflight.json').unlink()
    ae.checked_timings(reply)
    return reply


def recorded_failure(key, kind, replies):
    if key in replies or (LOCAL/'inflight.json').exists():
        raise RuntimeError('duplicate failure or unknown inflight')
    value = dict(failure_kind=kind, response={}, attempts=0, wall_seconds=0., input_tokens=None)
    replies[key] = value
    ae.received(LOCAL/'received.jsonl', key, value)
    return value


def load_questions(p):
    if lm.file_hash(DATA) != p['dataset_sha256']:
        raise RuntimeError('benchmark source hash differs')
    raw = lm.read(DATA)
    lookup = {r['question_id']: r for r in raw}
    ids = p['question_ids']
    result = {}
    for qid in ids:
        row = lookup[qid]
        history_group = lm.digest(list(zip(row['haystack_session_ids'], row['haystack_dates'])))
        result[qid] = {**row, 'history_group': history_group}
    return result


def prepare(p, now, budget_start):
    questions = load_questions(p)
    count = ae.Tokenizer(now['url'])
    qwen = st.Qwen.offline()
    per_question, fact_jobs, inputs = {}, [], {'A': {}, 'B1536': {}}
    public = {'A_overflow_ids': [], 'fact_jobs': 0, 'A_tokens': {}, 'B_tokens': {}}
    try:
        for qi, (qid, row) in enumerate(questions.items(), 1):
            ae.gate(); reserve_budget(budget_start)
            ts = methods.turns(row)
            per_question[qid] = ts
            for arm, ctx in (('A', 131072), ('B1536', 8192)):
                tick = time.monotonic()
                payload = am.payload(arm, ts, {}, count.count)
                chat = am.messages(payload.pop('conversation'), payload.pop('retrieved'), row['question'])
                tokens = count.input_count(chat)
                item = {'messages': chat, 'input_tokens': tokens,
                        'fits_context': fits(tokens, 256, ctx),
                        'assembly_seconds': time.monotonic()-tick, **payload}
                inputs[arm][qid] = item
                public[arm.split('1536')[0]+'_tokens'][qid] = tokens
                if not item['fits_context']:
                    if arm != 'A':
                        raise RuntimeError('B input window overflow; no change/retry')
                    public['A_overflow_ids'].append(qid)
            for session, chat, extra in st.generation_jobs('fact', {}, ts):
                ae.gate(); reserve_budget(budget_start)
                tokens = count.input_count(chat)
                if not fits(tokens, mm.PROMPTS['fact']['max_tokens'], 8192):
                    event('input overflow', stage='fact', session_key=session,
                          input_tokens=tokens, output_reservation=1500, context=8192)
                    raise RuntimeError('fact input window overflow; no chunking/window change')
                fact_jobs.append({'session': session, 'group': qid, 'body': body_for(qwen, 'fact', chat),
                                  'input_tokens': tokens, **extra})
                public['fact_jobs'] = len(fact_jobs)
                if len(fact_jobs) % 25 == 0:
                    status('preflight', status='running', checked_sessions=len(fact_jobs),
                           ingestion_sessions_received=0, ingestion_sessions_total=1441, checked_questions=qi)
            lm.write(LOCAL/'preflight-partial.json', public)
    finally:
        count.save()
        lm.write(LOCAL/'preflight-partial.json', public)
    if len(fact_jobs) != 1441 or len(questions) != 30:
        raise RuntimeError('registered cohort/session count differs')
    prepared = {'questions': questions, 'turns': per_question, 'fact_jobs': fact_jobs, 'inputs': inputs}
    lm.write(LOCAL/'prepared.json', prepared)
    lm.write(lm.RESULTS/f'{PREFIX}-preflight.json', {**public,
             'eligible_ids_sha256': lm.digest([q for q in questions if inputs['A'][q]['fits_context']]),
             'prepared_sha256': lm.file_hash(LOCAL/'prepared.json'), 'before_any_generation': True})
    event('preflight complete', questions=30, fact_sessions=1441, A_overflow=len(public['A_overflow_ids']))
    return prepared


def ingest(prepared, pid, now, replies, budget_start):
    facts, reused = {}, {}
    watch = fp.SeriesWatch(pid, lambda kind, data: event(kind, stage='fact', **data))
    tick = time.monotonic()
    try:
        with watch:
            for number, job in enumerate(prepared['fact_jobs'], 1):
                ae.gate(); reserve_budget(budget_start)
                key = 'fact|'+job['session']
                digest = lm.digest(job['body'])
                watch.begin(number)
                if digest in reused:
                    reply = {**reused[digest], 'shared_original_key': reused[digest]['original_key'], 'attempts': 0}
                    replies[key] = reply; ae.received(LOCAL/'received.jsonl', key, reply)
                else:
                    reply = send(key, job['body'], now, replies, watch, job['input_tokens'])
                    reused[digest] = {**reply, 'original_key': key}
                content = reply['response']['choices'][0]['message'].get('content') or ''
                value = mm.parse('fact', content, valid_ids=job['valid_ids'])
                facts[job['session']] = [{**v, 'turns': [job['group']+'/'+tid for tid in v['turns']]}
                                        for v in value or []]
                if value is None or reply['response']['choices'][0]['finish_reason'] != 'stop':
                    event('generation failure', stage='fact', session_key=job['session'],
                          fallback='frozen parser/fallback, no re-generation')
                watch.boundary()
                status('fact ingestion', status='running', ingestion_sessions_received=number,
                       ingestion_sessions_total=1441, phase_elapsed_seconds=time.monotonic()-tick)
                if number % 25 == 0:
                    lm.write(LOCAL/'responses.json', replies); lm.write(LOCAL/'facts.json', facts)
                if number in (361, 721, 1081, 1441):
                    event('ingestion milestone', received_sessions=number, total_sessions=1441,
                          phase_elapsed_seconds=time.monotonic()-tick,
                          elapsed_seconds=time.time()-lm.read(LOCAL/'run-start.json')['unix_time'])
    finally:
        lm.write(LOCAL/'responses.json', replies); lm.write(LOCAL/'facts.json', facts)
        lm.write(LOCAL/'fact-resources.json', watch.record())
    if watch.error:
        raise RuntimeError(watch.error)
    return facts


def embedding(prepared, facts, manager, budget_start):
    ae.gate(); reserve_budget(budget_start)
    manager.stop()  # serial GPU stages: Qwen is deliberately absent during embedding
    from egemma_server.tokenizer import PromptTokenizer
    tk = PromptTokenizer(rp.V1_ST_DIR)
    identity = lm.fingerprint(st.VARIANT)
    if identity != lm.read(lm.RESULTS/'final-config.json')['embedding_artifact_sha256']:
        raise RuntimeError('frozen embedding identity differs')
    args = argparse.Namespace(port=8398, model_dir=lm.ROOT/'models/embeddinggemma-2', v1_dir=rp.V1_ST_DIR)
    vectors, units_by_q, query_vectors, timing = {}, {}, {}, {}
    tick = time.monotonic()
    with lm.start_backend(st.VARIANT, args) as (backend, embed):
        def compute(texts):
            result = []
            for start in range(0, len(texts), 32):
                ae.gate(); reserve_budget(budget_start)
                result.extend(embed(texts[start:start+32]))
            return lm.checked_vectors(np.asarray(result, np.float32), len(texts))
        for number, (qid, ts) in enumerate(prepared['turns'].items(), 1):
            ae.gate(); reserve_budget(budget_start)
            part_facts = {s: v for s, v in facts.items() if s.startswith(qid+'#')}
            units = st.build_units(['p2', 'fact'], ts, {'fact': part_facts})
            start = time.monotonic()
            texts = rp.fit_like_st(tk, [lm.PROTOCOL['prompts']['document']+u['text'] for u in units])
            vector = compute(texts)
            np.save(LOCAL/f'doc-vectors-{qid}.npy', vector)
            timing[qid] = {'document_seconds': time.monotonic()-start, 'document_units': len(units)}
            start = time.monotonic()
            text = rp.fit_like_st(tk, [lm.PROTOCOL['prompts']['query']+prepared['questions'][qid]['question']])
            query_vectors[qid] = compute(text)[0].tolist()
            timing[qid]['query_seconds'] = time.monotonic()-start
            units_by_q[qid] = units; vectors[qid] = vector
            status('embedding', status='running', completed_questions=number, total_questions=30,
                   ingestion_sessions_received=1441, ingestion_sessions_total=1441,
                   phase_elapsed_seconds=time.monotonic()-tick)
    reserve_budget(budget_start)
    lm.write(LOCAL/'embedding.json', {'backend': backend, 'identity': identity, 'timings': timing,
                                    'query_vectors': query_vectors, 'units': units_by_q})
    event('ingestion complete', sessions=1441, embedding_questions=30,
          elapsed_seconds=time.monotonic()-budget_start)
    return units_by_q, vectors, query_vectors, timing


def search(prepared, units, vectors, qvectors, pid, now, replies):
    count = ae.Tokenizer(now['url']); qwen = st.Qwen.offline()
    cfg = mm.config(['p2', 'fact'], rerank='qwen')
    C, timings = {}, {}
    watch = fp.SeriesWatch(pid, lambda kind, data: event(kind, stage='rerank', **data))
    try:
        with watch:
            for number, (qid, row) in enumerate(prepared['questions'].items(), 1):
                ae.gate(); watch.begin(number)
                ts, us = prepared['turns'][qid], units[qid]
                tick = time.monotonic()
                index = mm.Index(us, vectors[qid])
                order = mm.distinct([i for i, _ in index.rank(cfg, qid, [(np.asarray(qvectors[qid]), row['question'])])], us)
                timings[qid] = {'search_seconds': time.monotonic()-tick}
                chat = base.norw.prod.rerank_chat({'text': row['question']}, order, us, ts, cfg)
                tokens = count.input_count(chat)
                if not fits(tokens, mm.PROMPTS['rerank']['max_tokens'], 8192):
                    raise RuntimeError('rerank input window overflow; no change/retry')
                reply = send('rerank|'+qid, qwen.request('rerank', chat), now, replies, watch, tokens)
                text = reply['response']['choices'][0]['message'].get('content') or ''
                value = mm.parse('rerank', text, n_passages=len(order[:mm.RERANK_CANDIDATES]))
                order = mm.distinct(base.norw.prod.reranked_order(order, value), us)
                raw_ids = list(dict.fromkeys(t for i in order for t in us[i]['turns']))
                tick = time.monotonic()
                payload = am.payload('C-final-norw', ts, {'C-final-norw': raw_ids}, count.count)
                chat = am.messages(payload.pop('conversation'), payload.pop('retrieved'), row['question'])
                tokens = count.input_count(chat)
                if not fits(tokens, 256, 8192):
                    raise RuntimeError('C input window overflow; no change/retry')
                C[qid] = {'messages': chat, 'input_tokens': tokens, 'fits_context': True,
                          'assembly_seconds': time.monotonic()-tick, **payload}
                timings[qid]['assembly_seconds'] = time.monotonic()-tick
                timings[qid]['rerank_seconds'] = reply['wall_seconds']
                watch.boundary()
                status('rerank and C preflight', status='running', completed=number, total=30,
                       ingestion_sessions_received=1441, ingestion_sessions_total=1441)
    finally:
        count.save(); lm.write(LOCAL/'responses.json', replies)
        lm.write(LOCAL/'rerank-resources.json', watch.record())
    if watch.error:
        raise RuntimeError(watch.error)
    prepared['inputs']['C-final-norw'] = C
    lm.write(LOCAL/'answer-inputs.json', prepared['inputs']); lm.write(LOCAL/'search-timings.json', timings)
    event('rerank complete', questions=30, C_all_inputs_fit=True)
    return timings


def answers(prepared, manager, replies, p):
    result, memory = {}, {}
    for arm in reporting.ARMS:
        ctx = 131072 if arm == 'A' else 8192
        pid, now = manager.start(ctx, arm)
        watch = fp.SeriesWatch(pid, lambda kind, data: event(kind, stage='answer', condition=arm, **data))
        result[arm] = {}; tick = time.monotonic()
        try:
            with watch:
                for number, qid in enumerate(prepared['questions'], 1):
                    ae.gate(); item = prepared['inputs'][arm][qid]; key = 'answer|'+arm+'|'+qid
                    watch.begin(number)
                    if not item['fits_context']:
                        reply = recorded_failure(key, 'input_overflow', replies)
                    else:
                        body = {'model': now['model_id'], 'messages': item['messages'], **am.GENERATION,
                                'response_format': {'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': am.SCHEMA}}}
                        reply = send(key, body, now, replies, watch, item['input_tokens'])
                    result[arm][qid] = reply
                    obs = {'memory': watch.boundary(), 'OS': base.resources.observe_os()}
                    ae.received(LOCAL/'answer-memory.jsonl', key, obs)
                    status('answer', status='running', condition=arm, completed=number, total=30,
                           ingestion_sessions_received=1441, ingestion_sessions_total=1441,
                           phase_elapsed_seconds=time.monotonic()-tick)
        finally:
            lm.write(LOCAL/'responses.json', replies)
            memory[arm] = watch.record(); lm.write(LOCAL/'answer-resources.json', memory)
        if watch.error:
            raise RuntimeError(watch.error)
        failures = sum(bool(v.get('failure_kind') or am.parse(v.get('response', {}))[1]) for v in result[arm].values())
        interim = {'time': lm.stamp(), 'provisional': True, 'condition': arm, 'questions': 30,
                   'unscored': True, 'failures': failures, 'resources': memory[arm],
                   'HTTP_seconds_per_registered_question': sum(v['wall_seconds'] for v in result[arm].values())/30,
                   'received_HTTP': sum(v['attempts'] for v in result[arm].values()),
                   'elapsed_seconds': time.time()-lm.read(LOCAL/'run-start.json')['unix_time'],
                   'phase_elapsed_seconds': time.monotonic()-tick}
        lm.write(lm.RESULTS/f'{PREFIX}-{arm}-interim.json', interim)
        event('condition complete', **interim)
    lm.write(LOCAL/'answers.json', result)
    return result


def judge(prepared, answers_, manager, replies):
    spec = lm.read(lm.RESULTS/'longmemeval-g1-judge.json')
    pid, now = manager.start(8192, 'judge')
    ids = list(prepared['questions'])
    ids.sort(key=lambda q: hashlib.sha256(('longmemeval-g1-judge-v1'+q).encode()).hexdigest())
    judges = {a: {} for a in reporting.ARMS}
    count = ae.Tokenizer(now['url'])
    watch = fp.SeriesWatch(pid, lambda kind, data: event(kind, stage='judge', **data))
    try:
        with watch:
            for number, qid in enumerate(ids, 1):
                row = prepared['questions'][qid]
                arms = sorted(reporting.ARMS, key=lambda a: lm.digest('longmemeval-g1-judge-v1|'+qid+'|'+a))
                for arm in arms:
                    ae.gate(); key = 'judge|'+arm+'|'+qid
                    answer = reporting.answer_value(answers_[arm][qid], row['answer'])
                    if answer['failure']:
                        recorded_failure(key, 'generator_failure_not_judged', replies)
                        judges[arm][qid] = {'label': 0, 'failure': None, 'generation_failed': True}
                        continue
                    kind = 'abstention' if qid.endswith('_abs') else row['question_type']
                    prompt = spec['templates'][kind].format(question=row['question'], answer=row['answer'], response=answer['answer'])
                    chat = [{'role': 'user', 'content': prompt}]
                    tokens = count.input_count(chat)
                    if not fits(tokens, 10, 8192):
                        raise RuntimeError('judge input window overflow; no change/retry')
                    watch.begin(number)
                    reply = send(key, {'model': now['model_id'], 'messages': chat, **spec['generation']},
                                 now, replies, watch, tokens)
                    label, failure = methods.judge_label(reply['response'])
                    judges[arm][qid] = {'label': label, 'failure': failure}
                    watch.boundary()
                status('judge', status='running', completed_paired_questions=number, total=30,
                       ingestion_sessions_received=1441, ingestion_sessions_total=1441)
                if number in (8, 15, 23, 30):
                    completed = ids[:number]
                    interim_rows = {}
                    for arm in reporting.ARMS:
                        interim_rows[arm] = []
                        for q in completed:
                            r = prepared['questions'][q]
                            value = reporting.answer_value(answers_[arm][q], r['answer'])
                            interim_rows[arm].append({'stratum': reporting.stratum(r), 'type': r['question_type'],
                                **value, 'correct': judges[arm][q]['label'], 'judge_failure': float(bool(judges[arm][q]['failure']))})
                    public = {'time': lm.stamp(), 'provisional': number != 30, 'paired_questions': number,
                              'metrics': {a: reporting.metrics(rows) for a, rows in interim_rows.items()},
                              'no_interim_inference': True, 'elapsed_seconds': time.time()-lm.read(LOCAL/'run-start.json')['unix_time']}
                    lm.write(lm.RESULTS/f'{PREFIX}-judge-{number}-interim.json', public)
                    event('judge milestone', questions=number, public_file=f'{PREFIX}-judge-{number}-interim.json')
    finally:
        count.save(); lm.write(LOCAL/'responses.json', replies); lm.write(LOCAL/'judges.json', judges)
        lm.write(LOCAL/'judge-resources.json', watch.record())
    if watch.error:
        raise RuntimeError(watch.error)
    return judges


def human_review(review):
    import html
    import random
    rng = random.Random('longmemeval-g3-blind-v1')
    pages, key = [], {}
    for item in review:
        arms = list(reporting.ARMS); rng.shuffle(arms)
        key[item['id']] = dict(zip(('X', 'Y', 'Z'), arms))
        pages.append('<section><h2>'+html.escape(item['id'])+'</h2><p>'+html.escape(item['question'])+'</p><p>Gold: '+html.escape(str(item['gold']))+'</p>')
        for alias, arm in key[item['id']].items():
            value = item['conditions'][arm]
            pages.append('<h3>'+alias+'</h3><pre>'+html.escape(value['answer'] or '['+str(value['failure'])+']')+'</pre>')
        pages.append('</section>')
    (LOCAL/'blinded-review.html').write_text('<!doctype html><meta charset="utf-8"><title>G1 blind review</title>'+''.join(pages))
    lm.write(LOCAL/'review-key.json', key); lm.write(LOCAL/'review.json', review)


def pause():
    lm.write(ae.PAUSE, {'time': lm.stamp(), 'runner': 'longmemeval-g1', 'reason': 'human requested pause'})


def run():
    p = checked(); other_workers()
    if any((LOCAL/n).exists() for n in ('run-start.json', 'received.jsonl', 'inflight.json')):
        raise RuntimeError('already started; no auto retry/overwrite')
    LOCAL.mkdir(parents=True, exist_ok=True)
    lm.write(LOCAL/'run-start.json', {'time': lm.stamp(), 'unix_time': time.time(), 'pid': os.getpid(),
                                   'pgid': os.getpgid(0), 'plan_sha256': lm.file_hash(PLAN)})
    replies = {}; start = time.monotonic()
    with isolated():
        manager = Servers(p)
        try:
            status('preflight', status='running', ingestion_sessions_received=0, ingestion_sessions_total=1441)
            pid, now = manager.start(8192, 'fact-preflight')
            prepared = prepare(p, now, start)
            facts = ingest(prepared, pid, now, replies, start)
            units, vectors, qvectors, etime = embedding(prepared, facts, manager, start)
            pid, now = manager.start(8192, 'rerank')
            timings = search(prepared, units, vectors, qvectors, pid, now, replies)
            answer_replies = answers(prepared, manager, replies, p)
            judge_labels = judge(prepared, answer_replies, manager, replies)
            checked()
            report, rows, review = reporting.analyze(prepared['questions'], prepared['inputs'], answer_replies, judge_labels)
            human_review(review)
            lm.write(LOCAL/'scored-rows.json', rows)
            report.update(time=lm.stamp(), embedding=etime, search_timings=timings,
                          answer_resources=lm.read(LOCAL/'answer-resources.json'),
                          wall_seconds=time.monotonic()-start, unique_registered_records=len(replies),
                          HTTP_responses=sum(v['attempts'] for v in replies.values()),
                          original_hash_match=True, execution_plan_sha256=lm.file_hash(PLAN))
            report['answer_timing'] = {}
            for arm, part in answer_replies.items():
                by_question = {}
                for qid, reply in part.items():
                    t = reply.get('timings') or {}
                    components = {'answer_HTTP_seconds': reply['wall_seconds'],
                                  'prefill_seconds': t.get('prompt_ms', 0)/1000,
                                  'generation_seconds': t.get('predicted_ms', 0)/1000,
                                  'assembly_preflight_seconds': prepared['inputs'][arm][qid]['assembly_seconds']}
                    if arm == 'C-final-norw':
                        components.update(timings[qid], query_embedding_seconds=etime[qid]['query_seconds'])
                        components['separately_measured_component_sum_seconds'] = sum(components[k] for k in
                            ('answer_HTTP_seconds', 'rerank_seconds', 'search_seconds', 'assembly_seconds', 'query_embedding_seconds'))
                    by_question[qid] = components
                report['answer_timing'][arm] = {'by_question': by_question,
                    'registered_question_means': {k: float(np.mean([r[k] for r in by_question.values()]))
                                                 for k in next(iter(by_question.values()))},
                    'scope': 'separate staged component sum; preflight construction separate; ingestion excluded; swap marked in resources'}
            expected = 1441+30+90+90
            if len(replies) != expected or (LOCAL/'inflight.json').exists():
                raise RuntimeError('registered records incomplete; no final report')
            manager.start(131072, 'original-restore', restore=True)
            verify(p)
            report['generation_through_scoring_seconds'] = report.pop('wall_seconds')
            report['wall_seconds_including_restore'] = time.monotonic()-start
            lm.write(lm.RESULTS/f'{PREFIX}-report.json', report)
            (lm.RESULTS/f'{PREFIX}-report.md').write_text(reporting.markdown(report))
            lm.write(lm.RESULTS/f'{PREFIX}-verification.json', {'registered_records': len(replies),
                     'HTTP_responses': report['HTTP_responses'], 'no_unknown_inflight': True,
                     'original_hash_match': True, 'source_hash_match': True,
                     'received_journal_sha256': lm.file_hash(LOCAL/'received.jsonl'),
                     'report_sha256': lm.file_hash(lm.RESULTS/f'{PREFIX}-report.json')})
            status('evaluation complete', status='complete', registered_records=len(replies),
                   ingestion_sessions_received=1441, ingestion_sessions_total=1441,
                   wall_seconds=time.monotonic()-start)
            event('evaluation complete', registered_records=len(replies))
            lm.write(ae.PAUSE, {'time': lm.stamp(), 'runner': 'longmemeval-g1',
                              'reason': 'G1 complete; CPU report then human confirmation; A128 remains paused'})
        except BaseException as error:
            status('stopped', status='stopped', error=f'{type(error).__name__}: {error}',
                   registered_records=len(replies), unknown_inflight=(LOCAL/'inflight.json').exists())
            event('stopped', error=f'{type(error).__name__}: {error}', registered_records=len(replies))
            if not ae.PAUSE.exists():
                lm.write(ae.PAUSE, {'time': lm.stamp(), 'runner': 'longmemeval-g1',
                                  'reason': 'stopped; human review required; no auto retry'})
        finally:
            lm.write(LOCAL/'responses.json', replies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'pause'])
    args = parser.parse_args()
    {'run': run, 'pause': pause}[args.command]()
