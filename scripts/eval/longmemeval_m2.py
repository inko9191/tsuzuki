#!/usr/bin/env python3
"""M2 (docs/MEMORY_MISFIRE_M2_PREREG.md): the similarity gate G* (no recall when the best window cosine is below tau*,
tau* set on dev) against the product recall P, on fresh users; W (only the windows at or above tau*) as an exploratory
arm. Gemma 4 E2B answers; Qwen3.8 27B judges what the reply did with the notes (judge v3: none / fits / harmful) and
the questions' answers; memory refusals are found by the pilot's sentence patterns (m2_pilot.refused).

    python3 scripts/eval/longmemeval_m2.py dev          # recall for the dev questions and tau*  (before register)
    python3 scripts/eval/longmemeval_m2.py calibrate    # judge v3 on the pilot items Claude labeled (before register)
    python3 scripts/eval/longmemeval_m2.py register     # the plan: hashes, tau*, judge v3; commit it before run
    python3 scripts/eval/longmemeval_m2.py run --smoke  # two dev users' questions, end to end (separate files)
    python3 scripts/eval/longmemeval_m2.py run          # test recall (P, W), answers N/P/W, judges; resumable
    python3 scripts/eval/longmemeval_m2.py sheet        # blinded items for the second judge (Claude)
    python3 scripts/eval/longmemeval_m2.py report       # results/eval/memory/m2-report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
import longmemeval_g1_methods as g1m  # noqa: E402
import longmemeval_m1 as m1  # noqa: E402
import m2_pilot as pilot  # noqa: E402

RESULTS = ROOT / 'results/eval/memory'
LOCAL = ROOT / 'build/eval/memory/m2'
INPUTS = LOCAL / 'inputs.json'
RECALL_DEV = LOCAL / 'recall-dev.json'
RECALL = LOCAL / 'recall-test.json'
ANSWERS = LOCAL / 'answers.jsonl'
JUDGES = LOCAL / 'judges.jsonl'
TAU = LOCAL / 'tau.json'
PLAN = RESULTS / 'm2-plan.json'
REPORT = RESULTS / 'm2-report.json'
SHEET = LOCAL / 'claude-sheet.json'
SHEET_KEY = LOCAL / 'claude-sheet-key.json'
REFUSAL_SHEET = LOCAL / 'claude-refusals.json'
CLAUDE = RESULTS / 'm2-claude-labels.json'
CODE = ['scripts/eval/longmemeval_m2.py', 'scripts/eval/m2/build_inputs.py', 'scripts/eval/m2/recall.mjs',
        'scripts/eval/m2_pilot.py', 'scripts/eval/longmemeval_m1.py',
        'scripts/eval/j2/evidence_a.py', 'scripts/eval/j2/evidence_b.py', 'scripts/eval/j2/build_dataset.py',
        'scripts/eval/j2/gen_pool.py', 'docs/MEMORY_MISFIRE_M2_PREREG.md'] + [
        f'dsh-plugins/packages/memory-egemma/src/{f}' for f in ('memory.ts', 'format.ts', 'pipeline.ts', 'store.ts',
                                                                 'embedding-http.ts', 'unit-index.ts')]
SLOTS = 4
SERVERS = {
    'embed': [m1.LLAMA, '-m', str(m1.MODELS / 'embeddinggemma-gguf/embeddinggemma-300M-Q8_0.gguf'), '--embedding', '-c', '8192',
              '-np', '4', '-b', '2048', '-ub', '2048', '--host', '127.0.0.1', '--port', '8591', '--alias', 'm2-embeddinggemma'],
    'reader': [m1.LLAMA, '-m', str(m1.MODELS / 'gemma-4-E2B/gemma-4-E2B_q4_0-it.gguf'), '-c', str(8192 * SLOTS), '-np', str(SLOTS),
               '--reasoning', 'off', '--host', '127.0.0.1', '--port', '8592', '--alias', 'm2-e2b'],
    'judge': [m1.LLAMA, '-m', str(m1.QWEN), '-ngl', '99', '-c', str(8192 * SLOTS), '--parallel', str(SLOTS), '--reasoning', 'off',
              '--host', '127.0.0.1', '--port', '8593', '--alias', 'm2-qwen'],
}
GENERATED = ('N', 'P', 'W')
TAU_GRID = [round(0.20 + 0.01 * i, 2) for i in range(31)]
KEEP_ALL, KEEP_EACH = 0.98, 0.95   # tau*: the largest grid value keeping >= 98% of dev questions, and >= 95% per language
MARGIN = 0.05                      # largest accuracy loss accepted for G*
KIND_PROMPT = """You are checking a reply from an AI assistant.

Before the user's new message, the assistant was shown notes: excerpts from the user's earlier conversations, found automatically by searching for the new message. They may have nothing to do with it. Decide what the reply did with the notes, and answer with one label.

none: the reply does not draw on the notes. Everything in it could have been written from the new message alone, even if it is on the same general subject as the notes. Also none when the notes are "none" and the reply does not claim to remember the user.

fits: the reply draws on the notes or on the earlier conversations in a way that serves what the new message asks. For example: it builds on an earlier topic the new message is about; it uses a preference or circumstance the user stated before that matters for this request; it continues something the new message refers back to ("continue", "redo that"); or the user asked the same or nearly the same thing before and the reply gives that earlier answer again or builds on it.

harmful: the reply draws on the notes in a way that does not serve the new message, or the notes make the answer worse. For example: it brings in an earlier topic, detail or piece of advice that has nothing to do with what is asked; it answers a different question, or steers the conversation toward a topic from the notes; it says something from the notes happened "just now", "earlier today" or "先ほど" when it did not; it claims to remember something that is not in the notes; or it declines, limits or hedges the answer because the information is not in the notes, the memory or the earlier conversations.

User's new message:
{message}

Notes shown to the assistant:
{notes}

Assistant's reply:
{reply}

Answer in JSON as {{"kind": "none"}}, {{"kind": "fits"}} or {{"kind": "harmful"}}."""
KIND_VERSION = hashlib.sha256(KIND_PROMPT.encode()).hexdigest()[:8]
KINDS = ('none', 'fits', 'harmful')
KIND_GENERATION = {'temperature': 0, 'seed': 0, 'max_tokens': 20, 'stream': False, 'cache_prompt': True,
                   'chat_template_kwargs': {'enable_thinking': False},
                   'response_format': {'type': 'json_schema', 'json_schema': {'name': 'kind', 'schema': {
                       'type': 'object', 'properties': {'kind': {'type': 'string', 'enum': list(KINDS)}}, 'required': ['kind']}}}}
lock = threading.Lock()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes() -> dict:
    return {p: sha256(ROOT / p) for p in CODE}


def append(path, row):
    with lock, open(path, 'a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


class Server(pilot.Server):
    def __init__(self, name):
        self.port = int(SERVERS[name][SERVERS[name].index('--port') + 1])
        self.alias = SERVERS[name][-1]
        log = open(LOCAL / f'{name}.log', 'a')
        self.proc = subprocess.Popen(SERVERS[name], stdout=log, stderr=log)
        for _ in range(600):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/health', timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            if self.proc.poll() is not None:
                raise SystemExit(f'{name} server exited; see {LOCAL / (name + ".log")}')
            time.sleep(1)
        raise SystemExit(f'{name} server did not start')


def pool(server, jobs, work, label):
    done = [0]

    def one(job):
        work(server, job)
        with lock:
            done[0] += 1
            if done[0] % 100 == 0 or done[0] == len(jobs):
                print(f'{label} {done[0]}/{len(jobs)}', flush=True)
    with ThreadPoolExecutor(SLOTS) as ex:
        list(ex.map(one, jobs))


def recall_run(split, out, tau=None):
    server = Server('embed')
    try:
        cmd = ['node', str(ROOT / 'scripts/eval/m2/recall.mjs'), '--inputs', str(INPUTS), '--split', split, '--out', str(out),
               '--embed', '8591']
        subprocess.run(cmd + (['--tau', str(tau)] if tau is not None else []), check=True)
    finally:
        server.stop()


def choose_tau(recall):
    """tau*: the largest grid value with >= KEEP_ALL of dev questions at or above it, and >= KEEP_EACH in each language."""
    tops = {lang: [r['top'] for r in recall.values() if r['kind'] == 'question' and r['lang'] == lang] for lang in ('en', 'ja')}
    every = tops['en'] + tops['ja']
    ok = [t for t in TAU_GRID if np.mean([x >= t for x in every]) >= KEEP_ALL
          and all(np.mean([x >= t for x in v]) >= KEEP_EACH for v in tops.values())]
    return (max(ok) if ok else min(TAU_GRID)), {lang: len(v) for lang, v in tops.items()}


def dev():
    recall_run('dev', RECALL_DEV)
    recall = json.loads(RECALL_DEV.read_text())
    tau, counts = choose_tau(recall)
    keep = {lang: round(float(np.mean([r['top'] >= tau for r in recall.values() if r['lang'] == lang])), 4) for lang in ('en', 'ja')}
    TAU.write_text(json.dumps({'tau': tau, 'dev_questions': counts, 'kept_at_tau': keep, 'recall_dev_sha256': sha256(RECALL_DEV)}) + '\n')
    print(TAU.read_text())


def judge_kind(s, message, notes, reply):
    out = s.chat({'messages': [{'role': 'user', 'content': KIND_PROMPT.format(message=message, notes=notes, reply=reply)}],
                  **KIND_GENERATION})
    c = out['choices'][0]
    if c['finish_reason'] != 'stop':
        return None, 'unfinished'
    try:
        kind = json.loads(c['message']['content'])['kind']
        return (kind, None) if kind in KINDS else (None, 'parse')
    except (ValueError, KeyError, TypeError):
        return None, 'parse'


def calibration_items():
    """The pilot items with Claude's reading: the 40 sorted P misfires (A, D, E fit; B harmful; X none) and the 120 M1
    items (Claude 0 -> none, or harmful when the reply is a memory refusal; Claude 1 -> used, fits or harmful)."""
    recall = json.loads(m1.RECALL.read_text())
    answers = m1.journal(m1.ANSWERS)
    severity = json.loads((RESULTS / 'm2p-severity.json').read_text())['labels']
    expect = {'A': 'fits', 'D': 'fits', 'E': 'fits', 'B': 'harmful', 'X': 'none'}
    items = {}
    for q, kind in severity.items():
        items[f'P|{q}'] = {'expect': expect[kind], 'source': 'severity'}
    key = json.loads((m1.LOCAL / 'claude-sheet-key.json').read_text())
    claude = json.loads(m1.CLAUDE.read_text())['labels']
    for i, k in key.items():
        if k in items:
            continue
        reply = answers[k]['reply']
        expect_kind = 'used' if int(claude[i]) else ('harmful' if pilot.refused(reply) else 'none')
        items[k] = {'expect': expect_kind, 'source': 'm1'}
    for k, a in answers.items():  # the other P replies the patterns call memory refusals (M1 test probes)
        q = a['query']
        if a['arm'] == 'P' and k not in items and recall[q]['split'] == 'test' and recall[q]['kind'] == 'probe' and pilot.refused(a['reply']):
            items[k] = {'expect': 'harmful', 'source': 'refusal'}
    for k, it in items.items():
        arm, q = k.split('|', 1)
        r = recall[q]
        it.update(message=r['message'], notes=r[arm] if arm != 'N' and r[arm] else 'none', reply=answers[k]['reply'])
    return items


def calibrate():
    LOCAL.mkdir(parents=True, exist_ok=True)
    items = calibration_items()
    path = LOCAL / f'calibrate-{KIND_VERSION}.jsonl'
    have = m1.journal(path)
    todo = [k for k in items if k not in have]
    if todo:
        server = Server('judge')

        def one(s, k):
            it = items[k]
            kind, failure = judge_kind(s, it['message'], it['notes'], it['reply'])
            append(path, {'key': k, 'kind': kind, 'failure': failure})
        try:
            pool(server, todo, one, 'calibrate')
        finally:
            server.stop()
    have = m1.journal(path)
    confusion = {}
    for k, it in items.items():
        got = have[k]['kind'] or 'failed'
        confusion.setdefault(it['expect'], {}).setdefault(got, 0)
        confusion[it['expect']][got] += 1
    harmful = [(int(it['expect'] == 'harmful'), int(have[k]['kind'] == 'harmful')) for k, it in items.items() if it['expect'] != 'used']
    used = [(int(it['expect'] != 'none'), int(have[k]['kind'] not in (None, 'none'))) for k, it in items.items()]
    print('judge v3', KIND_VERSION, 'confusion (expected -> got)', json.dumps(confusion))
    print('harmful vs not (items with a known kind)', json.dumps(pilot.kappa(harmful)))
    print('used vs none (all items)', json.dumps(pilot.kappa(used)))
    for k, it in sorted(items.items()):
        got = have[k]['kind']
        if (it['expect'] == 'used' and got == 'none') or (it['expect'] != 'used' and got != it['expect']):
            print('  differs', k, it['source'], 'expected', it['expect'], 'got', got)


def register():
    tau = json.loads(TAU.read_text())
    plan = {'version': 1, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'experiment': 'M2 similarity gate',
            'inputs_sha256': sha256(INPUTS), 'recall_dev_sha256': sha256(RECALL_DEV), 'code_sha256': code_hashes(),
            'tau': tau['tau'], 'tau_rule': f'largest of {TAU_GRID[0]}..{TAU_GRID[-1]} step 0.01 with >= {KEEP_ALL:.0%} of dev questions '
                                          f'(pooled) and >= {KEEP_EACH:.0%} in each language at or above it', 'tau_dev': tau,
            'models': {'reader': 'gemma-4-E2B_q4_0-it.gguf fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634',
                       'embedding': 'embeddinggemma-300M-Q8_0.gguf b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63',
                       'judge': 'Qwen3.8-27B-Q4_K_M.gguf (the U2 judge)'},
            'servers': {k: [a.replace(str(Path.home()), '~') for a in v] for k, v in SERVERS.items()},
            'arms': {'N': 'no memory', 'P': 'v0.2.0 proxy recall (product header), before the message',
                     'W': 'P from the windows with cosine >= tau* only; none -> N (exploratory)',
                     'G*': 'P when the best window cosine >= tau*, else N; derived',
                     'G*·F': 'G*, and N when that reply is a memory refusal; derived'},
            'reuse': 'a W block equal to the P block takes the P reply, an empty one the N reply (temperature 0)',
            'system': m1.SYSTEM, 'answer_generation': m1.ANSWER, 'kind_prompt': KIND_PROMPT, 'kind_prompt_sha8': KIND_VERSION,
            'kind_generation': KIND_GENERATION, 'kind_failures': 'counted as none, and reported',
            'refusal_patterns': {'en_memory': pilot.EN_MEM.pattern, 'en_negation': pilot.EN_NEG.pattern,
                                 'ja_memory': pilot.JA_MEM.pattern, 'ja_negation': pilot.JA_NEG.pattern},
            'accuracy_generation': m1.JUDGE['generation'], 'accuracy_templates_sha256': m1.JUDGE['templates_sha256'],
            'decisions': {
                'refusal': 'G* - P on test probes (both languages), user-clustered paired bootstrap 10000, seed 0: upper < 0 -> fewer refusals',
                'accuracy': f'G* - P on test questions: lower >= -{MARGIN} -> keeps answers',
                'both': 'propose G* as the product default (the user decides)',
                'secondary': 'harm (refusal or harmful), harmful, fits, by language; G*·F - G*; W - P and W - G*; share recalled; tau sweep',
                'second_judge': 'Claude labels 150 blinded test P probe replies (50 harmful-or-refusal, 50 fits, 50 none by Qwen; '
                                'strata hidden); harmful-vs-not kappa < 0.6 -> harmful and harm reported as unreliable; '
                                'Claude reads every reply the patterns call a refusal (precision)'}}
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(PLAN, 'tau', tau['tau'], 'judge v3', KIND_VERSION)


def check_plan():
    plan = json.loads(PLAN.read_text())
    if plan['inputs_sha256'] != sha256(INPUTS) or plan['code_sha256'] != code_hashes() or plan['kind_prompt_sha8'] != KIND_VERSION:
        raise SystemExit('the plan does not match the inputs, the code or judge v3: register again (before any result) and commit')
    tracked = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', str(PLAN.relative_to(ROOT)), *CODE],
                             capture_output=True, text=True).stdout.strip()
    if tracked:
        raise SystemExit(f'commit the plan and the code first:\n{tracked}')
    return plan


def source(arm, r):
    """The generated reply a W query uses: its own, P's (same block) or N's (no block)."""
    if arm != 'W':
        return arm
    return 'N' if not r['W'] else 'P' if r['W'] == r['P'] else 'W'


def notes_for(arm, r):
    block = '' if arm == 'N' else r[arm]
    return block or 'none'


def run(smoke):
    plan = check_plan()
    tau = plan['tau']
    inputs = json.loads(INPUTS.read_text())
    recall_path, answers_path, judges_path = RECALL, ANSWERS, JUDGES
    if smoke:
        recall_path, answers_path, judges_path = LOCAL / 'smoke-recall.json', LOCAL / 'smoke-answers.jsonl', LOCAL / 'smoke-judges.jsonl'
        dev_users = [u for u in inputs['users'] if u['split'] == 'dev']
        pick = [u for u in dev_users if u['lang'] == 'en'][:1] + [u for u in dev_users if u['lang'] == 'ja'][:1]
        smoke_inputs = LOCAL / 'smoke-inputs.json'
        smoke_inputs.write_text(json.dumps({'users': pick}, ensure_ascii=False))
        if not recall_path.exists():
            server = Server('embed')
            try:
                subprocess.run(['node', str(ROOT / 'scripts/eval/m2/recall.mjs'), '--inputs', str(smoke_inputs), '--split', 'dev',
                                '--tau', str(tau), '--out', str(recall_path), '--embed', '8591'], check=True)
            finally:
                server.stop()
    elif not recall_path.exists() or len(json.loads(recall_path.read_text())) < sum(
            len(u['probes']) + 1 for u in inputs['users'] if u['split'] == 'test'):
        recall_run('test', recall_path, tau)
    recall = json.loads(recall_path.read_text())
    queries = sorted(recall)
    have = m1.journal(answers_path)
    jobs = [(arm, q) for q in queries for arm in GENERATED if source(arm, recall[q]) == arm and f'{arm}|{q}' not in have]
    if jobs:
        server = Server('reader')

        def gen(s, job):
            arm, q = job
            r = recall[q]
            t = time.time()
            c = s.chat({'messages': [{'role': 'system', 'content': m1.SYSTEM},
                                     {'role': 'user', 'content': m1.user_message('' if arm == 'N' else r[arm], r['message'])}],
                        **m1.ANSWER})['choices'][0]
            append(answers_path, {'key': f'{arm}|{q}', 'arm': arm, 'query': q, 'reply': c['message']['content'],
                                  'finish': c['finish_reason'], 'seconds': round(time.time() - t, 3)})
        try:
            pool(server, sorted(jobs, key=lambda j: hashlib.sha256(f'm2-answer|{j[0]}|{j[1]}'.encode()).hexdigest()), gen, 'answers')
        finally:
            server.stop()
    answers = m1.journal(answers_path)
    by_user = {u['user']: u for u in inputs['users']}
    judged = m1.journal(judges_path)
    jobs = [(arm, q) for q in queries for arm in GENERATED if f'{arm}|{q}' in answers and f'{arm}|{q}' not in judged]
    if jobs:
        server = Server('judge')

        def one(s, job):
            arm, q = job
            r, reply = recall[q], answers[f'{arm}|{q}']['reply']
            if r['kind'] == 'probe':
                kind, failure = judge_kind(s, r['message'], notes_for(arm, r), reply)
                append(judges_path, {'key': f'{arm}|{q}', 'kind': kind, 'failure': failure, 'version': KIND_VERSION})
            else:
                u = by_user[r['user']]
                prompt = m1.JUDGE['templates']['abstention' if u['abstention'] else u['question_type']].format(
                    question=u['question'], answer=u['answer'], response=reply)
                label, failure = g1m.judge_label(s.chat({'messages': [{'role': 'user', 'content': prompt}], **m1.JUDGE['generation']}))
                append(judges_path, {'key': f'{arm}|{q}', 'label': label, 'failure': failure})
        try:
            pool(server, sorted(jobs, key=lambda j: hashlib.sha256(f'm2-judge|{j[0]}|{j[1]}'.encode()).hexdigest()), one, 'judged')
        finally:
            server.stop()
    print('done', len(queries), 'queries', len(m1.journal(answers_path)), 'answers', len(m1.journal(judges_path)), 'judged')


def sheet():
    """Claude's sheets: 150 test P probe replies stratified by Qwen's label (strata hidden, shuffled), and every reply
    the refusal patterns flag (arm hidden)."""
    recall = json.loads(RECALL.read_text())
    answers, judged = m1.journal(ANSWERS), m1.journal(JUDGES)
    probes = sorted(q for q, r in recall.items() if r['kind'] == 'probe')
    rng = random.Random(0)
    strata = {'harm': [], 'fits': [], 'none': []}
    for q in probes:
        kind = judged[f'P|{q}']['kind']
        strata['harm' if kind == 'harmful' or pilot.refused(answers[f'P|{q}']['reply']) else 'fits' if kind == 'fits' else 'none'].append(q)
    chosen = rng.sample(strata['harm'], min(50, len(strata['harm']))) + rng.sample(strata['fits'], min(50, len(strata['fits'])))
    rest = [q for q in strata['none']] + [q for q in strata['harm'] + strata['fits'] if q not in chosen]
    chosen += rng.sample(sorted(set(rest) - set(chosen)), 150 - len(chosen))
    rng.shuffle(chosen)
    items = [{'item': i, 'message': recall[q]['message'], 'notes': notes_for('P', recall[q]), 'reply': answers[f'P|{q}']['reply']}
             for i, q in enumerate(chosen)]
    SHEET.write_text(json.dumps(items, ensure_ascii=False, indent=1))
    SHEET_KEY.write_text(json.dumps({i: f'P|{q}' for i, q in enumerate(chosen)}))
    flagged = sorted({k for k, v in answers.items() if recall[v['query']]['kind'] == 'probe' and pilot.refused(v['reply'])})
    rng.shuffle(flagged)
    REFUSAL_SHEET.write_text(json.dumps([{'item': i, 'key_hash': hashlib.sha256(k.encode()).hexdigest()[:12],
                                          'message': recall[answers[k]['query']]['message'], 'reply': answers[k]['reply']}
                                         for i, k in enumerate(flagged)], ensure_ascii=False, indent=1))
    (LOCAL / 'claude-refusals-key.json').write_text(json.dumps({i: k for i, k in enumerate(flagged)}))
    print(SHEET, len(items), {k: len(v) for k, v in strata.items()}, REFUSAL_SHEET, len(flagged))


def report():
    plan = json.loads(PLAN.read_text())
    tau = plan['tau']
    inputs = json.loads(INPUTS.read_text())
    recall = json.loads(RECALL.read_text())
    answers, judged = m1.journal(ANSWERS), m1.journal(JUDGES)
    users = [u for u in inputs['users'] if u['split'] == 'test']
    probes = [p['id'] for u in users for p in u['probes']]
    questions = [f"{u['user']}-q" for u in users]
    owner = lambda q: recall[q]['user']

    def pick(arm, q, t=tau):
        """The generated reply this (possibly derived) arm returns for query q."""
        r = recall[q]
        if arm == 'G':
            return 'P' if r['top'] >= t else 'N'
        if arm.endswith('·F'):
            base = pick(arm[:-2], q, t)
            return 'N' if base != 'N' and pilot.refused(answers[f'{base}|{q}']['reply']) else base
        return source(arm, r)

    refusal = lambda arm, q, t=tau: int(pilot.refused(answers[f'{pick(arm, q, t)}|{q}']['reply']))
    kind = lambda arm, q, t=tau: judged[f'{pick(arm, q, t)}|{q}']['kind'] or 'none'
    harmful = lambda arm, q, t=tau: int(kind(arm, q, t) == 'harmful')
    fits = lambda arm, q, t=tau: int(kind(arm, q, t) == 'fits')
    harm = lambda arm, q, t=tau: int(refusal(arm, q, t) or harmful(arm, q, t))
    correct = lambda arm, q, t=tau: judged[f'{pick(arm, q, t)}|{q}']['label']
    measures = {'refusal': (refusal, probes), 'harm': (harm, probes), 'harmful': (harmful, probes), 'fits': (fits, probes),
                'accuracy': (correct, questions)}

    def rate(f, arm, ids):
        return round(float(np.mean([f(arm, q) for q in ids])), 4)

    def diff(f, a, b, ids):
        d = [f(a, q) - f(b, q) for q in ids]
        return {'difference': round(float(np.mean(d)), 4), 'ci95': [round(x, 4) for x in g1m.interval(d, [owner(q) for q in ids])]}

    arms = ['N', 'P', 'G', 'G·F', 'W', 'W·F']
    out = {'plan_sha256': sha256(PLAN), 'tau': tau, 'kind_prompt_sha8': KIND_VERSION, 'test_users': len(users),
           'probes': len(probes), 'questions': len(questions), 'rates': {}, 'by_language': {}, 'comparisons': {}}
    for arm in arms:
        out['rates'][arm] = {name: rate(f, arm, ids) for name, (f, ids) in measures.items()}
        for lang in ('en', 'ja'):
            out['by_language'].setdefault(lang, {})[arm] = {
                name: rate(f, arm, [q for q in ids if recall[q]['lang'] == lang]) for name, (f, ids) in measures.items()}
    for a, b in (('G', 'P'), ('G·F', 'G'), ('W', 'P'), ('W', 'G'), ('W·F', 'G·F')):
        out['comparisons'][f'{a} - {b}'] = {name: diff(f, a, b, ids) for name, (f, ids) in measures.items()}
        for lang in ('en', 'ja'):
            out['comparisons'][f'{a} - {b}'][f'by_language_{lang}'] = {
                name: diff(f, a, b, [q for q in ids if recall[q]['lang'] == lang]) for name, (f, ids) in measures.items()}
    g = out['comparisons']['G - P']
    out['verdict'] = {'fewer_refusals': g['refusal']['ci95'][1] < 0, 'keeps_answers': g['accuracy']['ci95'][0] >= -MARGIN}
    out['verdict']['propose_gate'] = out['verdict']['fewer_refusals'] and out['verdict']['keeps_answers']
    pos = [recall[q]['top'] for q in questions]
    neg = [recall[q]['top'] for q in probes]
    out['gate'] = {'probes_recalled': round(float(np.mean([t >= tau for t in neg])), 4),
                   'questions_recalled': round(float(np.mean([t >= tau for t in pos])), 4),
                   'by_language': {lang: {'probes_recalled': round(float(np.mean([recall[q]['top'] >= tau for q in probes if recall[q]['lang'] == lang])), 4),
                                          'questions_recalled': round(float(np.mean([recall[q]['top'] >= tau for q in questions if recall[q]['lang'] == lang])), 4)}
                                   for lang in ('en', 'ja')},
                   'auroc_top_cosine': round(float(np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg])), 4),
                   'w_units_median': {'P': float(np.median([recall[q]['units'] for q in probes + questions])),
                                      'W': float(np.median([recall[q]['unitsW'] for q in probes + questions]))},
                   'w_equals_p': round(float(np.mean([recall[q]['W'] == recall[q]['P'] for q in probes + questions])), 4)}
    out['tau_sweep'] = {str(t): {name: rate(lambda a, q, f=f: f(a, q, t), 'G', ids) for name, (f, ids) in measures.items()}
                        for t in [0.25, 0.28, 0.30, 0.32, 0.35, 0.40, 0.45]}
    out['failures'] = {'kind_judge': sum(1 for k, v in judged.items() if 'kind' in v and v['failure']),
                       'accuracy_judge': sum(1 for k, v in judged.items() if 'label' in v and v['failure']),
                       'answers_cut': {arm: sum(1 for v in answers.values() if v['arm'] == arm and v['finish'] != 'stop') for arm in GENERATED},
                       'skipped_probes': inputs['skipped_probes']}
    if CLAUDE.exists():
        claude = json.loads(CLAUDE.read_text())
        key = json.loads(SHEET_KEY.read_text())
        pairs = [(int(claude['kinds'][i] == 'harmful'), int((judged[k]['kind'] or 'none') == 'harmful')) for i, k in key.items() if i in claude['kinds']]
        three = [claude['kinds'][i] == (judged[k]['kind'] or 'none') for i, k in key.items() if i in claude['kinds']]
        rkey = json.loads((LOCAL / 'claude-refusals-key.json').read_text())
        out['second_judge'] = {'harmful_vs_not': pilot.kappa(pairs), 'three_way_agreement': round(float(np.mean(three)), 4),
                               'refusal_precision': round(float(np.mean([int(claude['refusals'][i]) for i in rkey if i in claude['refusals']])), 4)
                               if claude.get('refusals') else None,
                               'refusals_read': len(claude.get('refusals', {}))}
        out['second_judge']['harm_reliable'] = (out['second_judge']['harmful_vs_not']['kappa'] or 0) >= 0.6
    REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('tau', 'rates', 'verdict', 'gate')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['dev', 'calibrate', 'register', 'run', 'sheet', 'report'])
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    {'dev': dev, 'calibrate': calibrate, 'register': register, 'run': lambda: run(a.smoke), 'sheet': sheet, 'report': report}[a.command]()
