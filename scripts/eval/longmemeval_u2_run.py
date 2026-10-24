#!/usr/bin/env python3
"""U2: confirmatory LongMemEval run on 40 new questions.

Arms (one shared ingestion): C512 (frozen C-final-norw), P (product memory: V, budget 2048, dense order, no
rerank), V1024R (V, budget 1024, Qwen rerank), T2048 (T, budget 2048, dense), A (full history).
Fact extraction and update checks run with --parallel 4 (registered); answers and judging run one at a time.
Memory arms are answered and judged first (interim report), then A.
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
import threading
import time
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/eval'))
import locomo_answer_methods as am  # noqa: E402
import locomo_answer_production as prod  # noqa: E402
import locomo_memory as lm  # noqa: E402
import locomo_memory_methods as mm  # noqa: E402
import locomo_memory_stages as st  # noqa: E402
import longmemeval_g1_methods as g1m  # noqa: E402
import longmemeval_g1_report as g1r  # noqa: E402
import longmemeval_u1_methods as um  # noqa: E402
import retrieval_public as rp  # noqa: E402

RESULTS = ROOT/'results/eval/memory'
DATA = ROOT/'build/eval/memory/longmemeval-g1/longmemeval_s_cleaned.json'
PLAN = RESULTS/'longmemeval-u2-plan.json'
WORK = ROOT/'build/eval/memory/longmemeval-u2'
JUDGE = json.loads((RESULTS/'longmemeval-g1-judge.json').read_text())
U1_PLAN = json.loads((RESULTS/'longmemeval-u1-plan.json').read_text())
G1_PLAN = json.loads((RESULTS/'longmemeval-g1-execution-plan.json').read_text())
URL = 'http://127.0.0.1:8080'
SEED = 'longmemeval-u2-v1'
QUOTA = {'knowledge-update': 10, 'temporal-reasoning': 10, 'multi-session': 10, 'single-session-user': 2,
         'single-session-assistant': 2, 'single-session-preference': 2, 'abstention': 4}
ARMS = {'C512': {'kind': 'C', 'budget': 512, 'rerank': True}, 'P': {'kind': 'V', 'budget': 2048, 'rerank': False},
        'V1024R': {'kind': 'V', 'budget': 1024, 'rerank': True}, 'T2048': {'kind': 'T', 'budget': 2048, 'rerank': False},
        'A': {'kind': 'A'}}
MEMORY_ARMS = ('C512', 'P', 'V1024R', 'T2048')
PARALLEL = 4
CODE = ['scripts/eval/longmemeval_u2_run.py', 'docs/MEMORY_LONGMEMEVAL_U2_PREREG.md']
MARGIN = 0.03


class Ctx:
    """Run directory, journal and status (thread-safe)."""
    def __init__(self, local: Path):
        self.local = local
        local.mkdir(parents=True, exist_ok=True)
        (local/'inflight').mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.replies = {}
        journal = local/'received.jsonl'
        if journal.exists():
            for line in journal.read_text().splitlines():
                rec = json.loads(line)
                self.replies[rec['key']] = rec['reply']

    def log(self, *a):
        line = time.strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
        print(line, flush=True)
        with self.lock, (self.local/'events.log').open('a') as f:
            f.write(line+'\n')

    def status(self, phase, **fields):
        with self.lock:
            (self.local/'status.json').write_text(json.dumps({'time': lm.stamp(), 'phase': phase, **fields}, indent=1))

    def write(self, name, value):
        (self.local/name).write_text(json.dumps(value, ensure_ascii=False))


def http(path, body=None, timeout=3600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL+path, data=data, headers={'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Server:
    """Registered argv with a chosen per-slot context and slot count; LLAMA_ARG_CACHE_RAM=0 as in G1/U1."""
    def __init__(self, ctx: Ctx):
        self.ctx, self.proc, self.model = ctx, None, None

    def start(self, slot_context, parallel=1, label=''):
        self.stop()
        if subprocess.run(['pgrep', '-x', 'llama-server'], capture_output=True).returncode == 0:
            raise RuntimeError('another llama-server is running')
        argv = list(U1_PLAN['server_argv'])
        argv[argv.index('-c')+1] = str(slot_context*parallel)
        argv[argv.index('--parallel')+1] = str(parallel)
        env = {**os.environ, 'LLAMA_ARG_CACHE_RAM': '0'}
        log = open(self.ctx.local/f'server-{label}.log', 'a')
        self.proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(900):
            if self.proc.poll() is not None:
                raise RuntimeError('server exited: '+label)
            try:
                if http('/health', timeout=2).get('status') == 'ok':
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError('server not ready: '+label)
        self.model = http('/v1/models')['data'][0]['id']
        self.ctx.log('server', label, 'slots', parallel, 'slot ctx', slot_context)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(120)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def send(ctx: Ctx, key, body):
    """One attempt per key; a received reply is reused on resume; an unknown inflight request stops the run."""
    with ctx.lock:
        if key in ctx.replies:
            return ctx.replies[key]
    marker = ctx.local/'inflight'/(hashlib.sha256(key.encode()).hexdigest()+'.json')
    marker.write_text(json.dumps({'key': key, 'time': lm.stamp()}))
    tick = time.monotonic()
    response = http('/v1/chat/completions', body)
    reply = {'response': response, 'wall_seconds': time.monotonic()-tick, 'timings': response.get('timings') or {}, 'time': lm.stamp()}
    with ctx.lock:
        ctx.replies[key] = reply
        with (ctx.local/'received.jsonl').open('a') as f:
            f.write(json.dumps({'key': key, 'reply': reply}, ensure_ascii=False)+'\n')
            f.flush()
            os.fsync(f.fileno())
    marker.unlink()
    return reply


def content(reply):
    return reply['response']['choices'][0]['message'].get('content') or ''


def finished(reply):
    return reply['response']['choices'][0]['finish_reason'] == 'stop'


class Counter_:
    def __init__(self):
        self.cache, self.lock = {}, threading.Lock()

    def count(self, text):
        if not text:
            return 0
        k = hashlib.sha256(text.encode()).hexdigest()
        with self.lock:
            if k in self.cache:
                return self.cache[k]
        n = len(http('/tokenize', {'content': text, 'add_special': False, 'parse_special': False})['tokens'])
        with self.lock:
            self.cache[k] = n
        return n

    @staticmethod
    def chat(messages):
        prompt = http('/apply-template', {'messages': messages, 'chat_template_kwargs': am.GENERATION['chat_template_kwargs']})['prompt']
        return len(http('/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True})['tokens'])


def fits(tokens, output, context):
    return tokens + output + 128 <= context


# -- registration ---------------------------------------------------------------------------------

def layer(meta):
    return 'abstention' if meta['question_id'].endswith('_abs') else meta['question_type']


def register():
    if PLAN.exists():
        raise SystemExit('U2 plan exists; not overwritten')
    raw = json.loads(DATA.read_text())
    if lm.file_hash(DATA) != G1_PLAN['dataset_sha256']:
        raise SystemExit('dataset hash differs')
    used = set(G1_PLAN['question_ids']) | set(U1_PLAN['question_ids'])
    cells = {k: [] for k in QUOTA}
    sessions = {}
    for r in raw:
        sessions[r['question_id']] = len(r['haystack_session_ids'])
        if r['question_id'] not in used and layer(r) in cells:
            cells[layer(r)].append(r['question_id'])
    rank = lambda q: (hashlib.sha256((SEED+'|'+q).encode()).hexdigest(), q)  # noqa: E731
    ids = []
    for kind, n in QUOTA.items():
        pool = sorted(cells[kind], key=rank)
        if len(pool) < n:
            raise SystemExit('short stratum '+kind)
        ids += pool[:n]
    ids = sorted(ids, key=rank)
    plan = {
        'version': 1, 'created_at': lm.stamp(), 'experiment': 'U2 confirmatory: product memory vs frozen C and full context',
        'authorization': 'user chat 2026-10-22: 40 questions, proposed arms, try parallel',
        'dataset_sha256': G1_PLAN['dataset_sha256'], 'model_sha256': U1_PLAN['model_sha256'], 'build': U1_PLAN['build'],
        'server_argv': U1_PLAN['server_argv'], 'seed': SEED, 'quota': QUOTA,
        'excluded': {'G1': G1_PLAN['question_ids'], 'U1': U1_PLAN['question_ids']},
        'question_ids': ids, 'sessions_total': sum(sessions[q] for q in ids), 'arms': ARMS, 'parallel_ingestion': PARALLEL,
        'primary': {
            'P_minus_C512': 'paired bootstrap 10000 seed0 over questions: low>0 改善, high<0 悪化, else 判定不能',
            'P_minus_A': f'A-eligible cohort: low>0 上回る; else low>=-{MARGIN} 同等以上(非劣性); high<0 下回る; else 判定不能',
        },
        'secondary': ['V1024R - C512', 'P - T2048 (update links)', 'P - V1024R (dense 2048 vs rerank 1024)',
                      'all-40 operational P - A (A overflow = 0)', 'per stratum (trend)', 'coverage', 'abstention',
                      'per-question answer time: A vs P'],
        'code_sha256': {f: lm.file_hash(ROOT/f) for f in CODE},
        'exploratory_basis': ['longmemeval-u1-budget-probe.md', 'longmemeval-u1-dense-probe.md', 'longmemeval-u1-speed-probe.md'],
    }
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'questions': len(ids), 'sessions_total': plan['sessions_total']}))


def checked():
    plan = json.loads(PLAN.read_text())
    if not st.committed(PLAN):
        raise SystemExit('commit the U2 plan first')
    for f, h in plan['code_sha256'].items():
        if lm.file_hash(ROOT/f) != h:
            raise SystemExit('registered code changed: '+f)
    return plan


# -- stages ---------------------------------------------------------------------------------------

def prepare(ctx, ids, server, session_cap=None):
    path = ctx.local/'prepared.json'
    if path.exists():
        return json.loads(path.read_text())
    lookup = {r['question_id']: r for r in json.loads(DATA.read_text())}
    server.start(8192, label='preflight')
    count, qwen = Counter_(), st.Qwen.offline()
    questions, turns, jobs, a_inputs = {}, {}, [], {}
    for qid in ids:
        row = lookup[qid]
        if session_cap:
            row = {**row, **{k: row[k][-session_cap:] for k in ('haystack_sessions', 'haystack_dates', 'haystack_session_ids')}}
        questions[qid] = row
        ts = turns[qid] = g1m.turns(row)
        for session, chat, extra in st.generation_jobs('fact', {}, ts):
            jobs.append({'session': session, 'group': qid, 'body': qwen.request('fact', chat), 'input_tokens': Counter_.chat(chat), **extra})
        payload = am.payload('A', ts, {}, count.count)
        chat = am.messages(payload.pop('conversation'), payload.pop('retrieved'), row['question'])
        tokens = Counter_.chat(chat)
        a_inputs[qid] = {'messages': chat, 'input_tokens': tokens, 'fits_context': fits(tokens, 256, 131072)}
        ctx.status('preflight', questions=len(questions), total=len(ids), fact_jobs=len(jobs))
    longest = max(j['input_tokens'] for j in jobs)
    window = 8192 if fits(longest, 1500, 8192) else um.window(longest, 1500, (16384, 32768))
    if window is None:
        raise RuntimeError('fact input exceeds 32768')
    prepared = {'questions': questions, 'turns': turns, 'fact_jobs': jobs, 'A': a_inputs, 'fact_window': window}
    path.write_text(json.dumps(prepared, ensure_ascii=False))
    ctx.write('preflight-public.json', {'fact_jobs': len(jobs), 'fact_input_max': longest, 'fact_window_per_slot': window,
                                        'A_eligible': sum(v['fits_context'] for v in a_inputs.values()),
                                        'A_tokens': {q: v['input_tokens'] for q, v in a_inputs.items()}})
    ctx.log('preflight', len(jobs), 'fact jobs, window', window, 'A eligible', sum(v['fits_context'] for v in a_inputs.values()))
    return prepared


def pooled(ctx, items, fn, workers, phase):
    done, total, lock = [0], len(items), threading.Lock()

    def run(item):
        out = fn(item)
        with lock:
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == total:
                ctx.status(phase, completed=done[0], total=total)
        return out
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(run, items))


def ingest(ctx, prepared, server):
    server.start(prepared['fact_window'], PARALLEL, 'fact')
    tick = time.monotonic()

    def one(job):
        reply = send(ctx, 'fact|'+job['session'], {**job['body'], 'model': server.model})
        value = mm.parse('fact', content(reply), valid_ids=job['valid_ids'])
        if value is None or not finished(reply):
            ctx.log('fact fallback (no facts)', job['session'])
        return job['session'], [{**v, 'turns': [job['group']+'/'+t for t in v['turns']]} for v in value or []]
    facts = dict(pooled(ctx, prepared['fact_jobs'], one, PARALLEL, 'fact ingestion'))
    ctx.write('facts.json', facts)
    ctx.log('ingestion done', len(facts), 'sessions', round(time.monotonic()-tick), 's')
    return facts


def embedding(ctx, prepared, facts, server):
    path = ctx.local/'embedding.json'
    if path.exists():
        saved = json.loads(path.read_text())
        return saved['units'], {q: np.load(ctx.local/f'doc-{q}.npy') for q in prepared['questions']}, saved['query']
    server.stop()
    from egemma_server.tokenizer import PromptTokenizer
    tk = PromptTokenizer(rp.V1_ST_DIR)
    identity = lm.fingerprint(st.VARIANT)
    if identity != json.loads((RESULTS/'final-config.json').read_text())['embedding_artifact_sha256']:
        raise RuntimeError('embedding identity differs')
    args = argparse.Namespace(port=8398, model_dir=ROOT/'models/embeddinggemma-2', v1_dir=rp.V1_ST_DIR)
    units, vectors, query = {}, {}, {}
    with lm.start_backend(st.VARIANT, args) as (_, embed):
        def compute(texts):
            out = []
            for k in range(0, len(texts), 32):
                out.extend(embed(texts[k:k+32]))
            return lm.checked_vectors(np.asarray(out, np.float32), len(texts))
        for n, (qid, ts) in enumerate(prepared['turns'].items(), 1):
            part = {s: v for s, v in facts.items() if s.startswith(qid+'#')}
            us = st.build_units(['p2', 'fact'], ts, {'fact': part})
            vec = compute(rp.fit_like_st(tk, [lm.PROTOCOL['prompts']['document']+u['text'] for u in us]))
            np.save(ctx.local/f'doc-{qid}.npy', vec)
            units[qid], vectors[qid] = us, vec
            query[qid] = compute(rp.fit_like_st(tk, [lm.PROTOCOL['prompts']['query']+prepared['questions'][qid]['question']]))[0].tolist()
            ctx.status('embedding', completed=n, total=len(prepared['turns']))
    path.write_text(json.dumps({'units': units, 'query': query}, ensure_ascii=False))
    ctx.log('embedding done')
    return units, vectors, query


def updates(ctx, prepared, facts, units, vectors, server):
    server.start(8192, PARALLEL, 'update')
    tables, work = {}, []
    for qid, ts in prepared['turns'].items():
        part = {s: v for s, v in facts.items() if s.startswith(qid+'#')}
        tables[qid] = um.fact_table(ts, part)
        fv = {u['id'].split('|', 1)[1]: vectors[qid][i].tolist() for i, u in enumerate(units[qid]) if u['set'] == 'fact'}
        look = {f['key']: f for f in tables[qid]}
        for job in um.update_jobs(tables[qid], fv):
            work.append((qid, job, um.update_messages(job, look)))

    def one(item):
        qid, job, chat = item
        if not fits(Counter_.chat(chat), um.UPDATE['max_tokens'], 8192):
            return qid, []
        body = {'model': server.model, 'messages': chat, 'temperature': 0, 'seed': 0, 'stream': False,
                'max_tokens': um.UPDATE['max_tokens'], 'cache_prompt': True, 'chat_template_kwargs': {'enable_thinking': False},
                'response_format': {'type': 'json_schema', 'json_schema': {'name': 'updates', 'schema': um.UPDATE['schema']}}}
        reply = send(ctx, 'update|'+job['session'], body)
        return qid, (um.parse_updates(content(reply), job) or []) if finished(reply) else []
    links = {q: [] for q in prepared['turns']}
    for qid, found in pooled(ctx, work, one, PARALLEL, 'update links'):
        links[qid] += found
    ctx.write('links.json', links)
    ctx.log('update links done', sum(len(v) for v in links.values()), 'links over', len(work), 'checks')
    return tables, links


def orders(ctx, prepared, units, vectors, query, server):
    cfg = mm.config(['p2', 'fact'], rerank='qwen')
    dense, chats = {}, {}
    for qid, row in prepared['questions'].items():
        us = units[qid]
        index = mm.Index(us, np.asarray(vectors[qid]))
        dense[qid] = mm.distinct([i for i, _ in index.rank(cfg, qid, [(np.asarray(query[qid]), row['question'])])], us)
        chats[qid] = prod.rerank_chat({'text': row['question']}, dense[qid], us, prepared['turns'][qid], cfg)
    server.start(8192, 1, 'rerank-count')
    tokens = {q: Counter_.chat(c) for q, c in chats.items()}
    window = um.window(max(tokens.values()), mm.PROMPTS['rerank']['max_tokens'], (16384, 32768, 65536))
    if window is None:
        raise RuntimeError('rerank input exceeds 65536')
    server.start(window, 1, 'rerank')
    qwen = st.Qwen.offline()
    reranked = {}
    for n, (qid, chat) in enumerate(chats.items(), 1):
        reply = send(ctx, 'rerank|'+qid, {**qwen.request('rerank', chat), 'model': server.model})
        value = mm.parse('rerank', content(reply), n_passages=min(len(dense[qid]), mm.RERANK_CANDIDATES))
        reranked[qid] = mm.distinct(prod.reranked_order(dense[qid], value), units[qid])
        ctx.status('rerank', completed=n, total=len(chats))
    raw = lambda q, o: list(dict.fromkeys(t for i in o for t in units[q][i]['turns']))  # noqa: E731
    out = {q: {'dense': raw(q, dense[q]), 'rerank': raw(q, reranked[q])} for q in chats}
    ctx.write('orders.json', {'orders': out, 'rerank_window': window, 'rerank_tokens': tokens})
    ctx.log('rerank done, window', window)
    return out


def assemble(ctx, prepared, raw, tables, links, server):
    server.start(8192, 1, 'answer')
    count = Counter_()
    inputs = {a: {} for a in MEMORY_ARMS}
    for qid, row in prepared['questions'].items():
        ts = prepared['turns'][qid]
        lookup = {t['id']: t for t in ts}
        for arm in MEMORY_ARMS:
            spec = ARMS[arm]
            order = raw[qid]['rerank' if spec['rerank'] else 'dense']
            if spec['kind'] == 'C':
                payload = am.payload('C-final-norw', ts, {'C-final-norw': order}, count.count)
                chat = am.messages(payload['conversation'], payload['retrieved'], row['question'])
                ids = set(payload['recent_ids']) | set(payload['memory_ids'])
            else:
                conversation, recent_ids = um.recent(ts, 1024, count.count)
                o, notes = (order, None) if spec['kind'] == 'T' else um.versioned(order, tables[qid], links[qid])
                retrieved, memory_ids = um.memory(o, lookup, recent_ids, spec['budget'], count.count, notes)
                chat = um.messages(conversation, retrieved, row['question'], row['question_date'])
                ids = set(recent_ids) | set(memory_ids)
            tokens = Counter_.chat(chat)
            if not fits(tokens, 256, 8192):
                raise RuntimeError(f'memory answer input overflow {arm} {qid}')
            inputs[arm][qid] = {'messages': chat, 'input_tokens': tokens, 'coverage': coverage(row, ids)}
    ctx.write('memory-inputs.json', inputs)
    return inputs


def coverage(row, ids):
    pos = {i+1 for i, s in enumerate(row['haystack_session_ids']) if s in set(row['answer_session_ids'])}
    present = {int(t.split('/D', 1)[1].split(':')[0]) for t in ids}
    return float(bool(pos) and pos <= present)


def answer(ctx, server, arm, qid, item):
    body = {'model': server.model, 'messages': item['messages'], **am.GENERATION,
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': am.SCHEMA}}}
    return send(ctx, f'answer|{arm}|{qid}', body)


def judge(ctx, server, prepared, answers, arms):
    labels = {a: {} for a in arms}
    ids = sorted(prepared['questions'], key=lambda q: hashlib.sha256(('longmemeval-u2-judge-v1'+q).encode()).hexdigest())
    for n, qid in enumerate(ids, 1):
        row = prepared['questions'][qid]
        for arm in sorted(arms, key=lambda a: lm.digest('longmemeval-u2-judge-v1|'+qid+'|'+a)):
            value = g1r.answer_value(answers[arm][qid], row['answer'])
            if value['failure']:
                labels[arm][qid] = {'label': 0, 'failure': value['failure_kind']}
                continue
            kind = 'abstention' if qid.endswith('_abs') else row['question_type']
            prompt = JUDGE['templates'][kind].format(question=row['question'], answer=row['answer'], response=value['answer'])
            reply = send(ctx, f'judge|{arm}|{qid}', {'model': server.model, 'messages': [{'role': 'user', 'content': prompt}], **JUDGE['generation']})
            label, failure = g1m.judge_label(reply['response'])
            labels[arm][qid] = {'label': label, 'failure': failure}
        ctx.status('judge', arms=list(arms), completed=n, total=len(ids))
    return labels


# -- report -----------------------------------------------------------------------------------------

def compare(rows, left, right, ids, mode):
    d = [rows[left][q]['correct'] - rows[right][q]['correct'] for q in ids]
    ci = g1m.interval(d, ids)
    if mode == 'improve':
        verdict = '改善' if ci[0] > 0 else '悪化' if ci[1] < 0 else '判定不能'
    elif mode == 'vs_full':
        verdict = '上回る' if ci[0] > 0 else '同等以上（非劣性）' if ci[0] >= -MARGIN else '下回る' if ci[1] < 0 else '判定不能'
    else:
        verdict = '傾向参考'
    return {'questions': len(ids), 'difference': float(np.mean(d)), 'ci95': ci, 'verdict': verdict, 'sign': g1m.sign_test(d)}


def report(ctx, prepared, inputs, answers, labels, arms, final):
    rows = {a: {} for a in arms}
    for arm in arms:
        for qid, row in prepared['questions'].items():
            value = g1r.answer_value(answers[arm][qid], row['answer'])
            rows[arm][qid] = {'correct': labels[arm][qid]['label'], 'abstain': value['abstention'], 'failure': value['failure'],
                              'stratum': um.stratum(row), 'coverage': inputs.get(arm, {}).get(qid, {}).get('coverage'),
                              'input_tokens': (inputs[arm][qid]['input_tokens'] if arm in inputs else prepared['A'][qid]['input_tokens']),
                              'answer_seconds': answers[arm][qid].get('wall_seconds', 0.)}
    ids = sorted(prepared['questions'])
    out = {'final': final, 'questions': len(ids), 'arms': {}, 'comparisons': {}}
    for arm in arms:
        r = list(rows[arm].values())
        out['arms'][arm] = {k: float(np.mean([x[k] for x in r if x[k] is not None])) if any(x[k] is not None for x in r) else None
                            for k in ('correct', 'abstain', 'coverage', 'input_tokens', 'answer_seconds')}
        out['arms'][arm]['by_stratum'] = {s: float(np.mean([x['correct'] for x in r if x['stratum'] == s]))
                                          for s in sorted({x['stratum'] for x in r})}
    out['comparisons']['primary_P_minus_C512'] = compare(rows, 'P', 'C512', ids, 'improve')
    out['comparisons']['V1024R_minus_C512'] = compare(rows, 'V1024R', 'C512', ids, 'trend')
    out['comparisons']['P_minus_T2048'] = compare(rows, 'P', 'T2048', ids, 'trend')
    out['comparisons']['P_minus_V1024R'] = compare(rows, 'P', 'V1024R', ids, 'trend')
    if 'A' in arms:
        eligible = [q for q in ids if prepared['A'][q]['fits_context']]
        out['A_eligible'] = len(eligible)
        out['comparisons']['primary_P_minus_A'] = compare(rows, 'P', 'A', eligible, 'vs_full')
        out['comparisons']['P_minus_A_all40_operational'] = compare(rows, 'P', 'A', ids, 'trend')
    out['rows'] = {a: rows[a] for a in arms}
    name = 'longmemeval-u2-report' if final else 'longmemeval-u2-interim'
    (RESULTS/f'{name}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (RESULTS/f'{name}.md').write_text(markdown(out))
    ctx.log('report written', name)
    return out


def markdown(r):
    lines = [f'# U2：{"最終" if r["final"] else "途中経過（記憶側のみ）"}', '',
             f'{r["questions"]}問（G1・U1 と重ならない新しい問題）。主判定は Qwen judge。区間は質問単位の対応ありブートストラップ（10000回、seed 0）。', '',
             '|条件|正答率|棄権率|根拠セッション被覆|入力トークン（平均）|回答秒（平均）|', '|---|---:|---:|---:|---:|---:|']
    for arm, m in r['arms'].items():
        cov = '—' if m['coverage'] is None else f'{m["coverage"]:.3f}'
        lines.append(f'|{arm}|{m["correct"]:.3f}|{m["abstain"]:.3f}|{cov}|{m["input_tokens"]:.0f}|{m["answer_seconds"]:.1f}|')
    lines += ['', '|比較|問数|点差|95%区間|判定|勝/負/同|', '|---|---:|---:|---|---|---|']
    for name, c in r['comparisons'].items():
        s = c['sign']
        lines.append(f'|{name}|{c["questions"]}|{c["difference"]:+.3f}|[{c["ci95"][0]:+.3f}, {c["ci95"][1]:+.3f}]|{c["verdict"]}|{s["wins"]}/{s["losses"]}/{s["ties"]}|')
    lines += ['', '|条件|' + '|'.join(sorted(next(iter(r['arms'].values()))['by_stratum'])) + '|',
              '|---|' + '---:|' * len(next(iter(r['arms'].values()))['by_stratum'])]
    for arm, m in r['arms'].items():
        lines.append(f'|{arm}|' + '|'.join(f'{m["by_stratum"][s]:.2f}' for s in sorted(m['by_stratum'])) + '|')
    return '\n'.join(lines)+'\n'


# -- run --------------------------------------------------------------------------------------------

def run(local=WORK/'run', ids=None, session_cap=None, resume=False):
    if ids is None:
        plan = checked()
        ids = plan['question_ids']
    ctx = Ctx(local)
    if any((local/'inflight').iterdir()):
        raise SystemExit('unknown inflight request; human review')
    if not resume and ctx.replies:
        raise SystemExit('already started; use resume')
    server = Server(ctx)
    ctx.log('start', len(ids), 'questions; swap', subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout.strip())
    try:
        prepared = prepare(ctx, ids, server, session_cap)
        facts = json.loads((local/'facts.json').read_text()) if (local/'facts.json').exists() else ingest(ctx, prepared, server)
        units, vectors, query = embedding(ctx, prepared, facts, server)
        tables, links = updates(ctx, prepared, facts, units, vectors, server)
        raw = orders(ctx, prepared, units, vectors, query, server)
        inputs = assemble(ctx, prepared, raw, tables, links, server)
        answers = {a: {} for a in ARMS}
        for arm in MEMORY_ARMS:
            for n, qid in enumerate(prepared['questions'], 1):
                answers[arm][qid] = answer(ctx, server, arm, qid, inputs[arm][qid])
                ctx.status('answer', arm=arm, completed=n, total=len(prepared['questions']))
        labels = judge(ctx, server, prepared, answers, MEMORY_ARMS)
        report(ctx, prepared, inputs, answers, labels, MEMORY_ARMS, final=False)
        server.start(131072, 1, 'A')
        for n, qid in enumerate(prepared['questions'], 1):
            item = prepared['A'][qid]
            if item['fits_context']:
                answers['A'][qid] = answer(ctx, server, 'A', qid, item)
            else:
                answers['A'][qid] = {'failure_kind': 'input_overflow', 'response': {}, 'wall_seconds': 0.}
            ctx.status('answer', arm='A', completed=n, total=len(prepared['questions']))
        server.start(8192, 1, 'judge')
        labels.update(judge(ctx, server, prepared, answers, ('A',)))
        report(ctx, prepared, inputs, answers, labels, tuple(ARMS), final=True)
        ctx.write('answers.json', answers)
        ctx.write('labels.json', labels)
        ctx.status('complete')
        ctx.log('complete')
    except BaseException as error:
        ctx.status('stopped', error=f'{type(error).__name__}: {error}')
        ctx.log('stopped', type(error).__name__, error)
        raise
    finally:
        server.stop()


def smoke():
    """Code path check on one G1 question (design sample), last 10 sessions only."""
    qid = next(q for q in G1_PLAN['question_ids'] if not q.endswith('_abs'))
    run(local=WORK/'smoke', ids=[qid], session_cap=10, resume=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'smoke', 'run', 'resume'])
    a = parser.parse_args()
    {'register': register, 'smoke': smoke, 'run': run, 'resume': lambda: run(resume=True)}[a.command]()
