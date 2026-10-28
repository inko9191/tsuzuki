#!/usr/bin/env python3
"""M2 pilot (exploratory, docs/MEMORY_MISFIRE_M2_PILOT.md): ways to stop misfires and memory refusals, tried on the
M1 test users. Reuses M1's inputs, recall and N/P/H answers; adds the arms S, H2, SH2 (generated) and the E2B routers
R1/R2, a revised misfire judge (v2) and refusal counts.

    python3 scripts/eval/m2_pilot.py answers     # S, H2, SH2 answers and R1/R2 routes (E2B, 4 slots); resumable
    python3 scripts/eval/m2_pilot.py calibrate   # judge v2 on the 120 items Claude labeled in M1; prints kappa
    python3 scripts/eval/m2_pilot.py judge       # judge v2 on every probe answer, accuracy for the new arms
    python3 scripts/eval/m2_pilot.py report      # writes results/eval/memory/m2p-report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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

LOCAL = ROOT / 'build/eval/memory/m2p'
ANSWERS = LOCAL / 'answers.jsonl'
ROUTES = LOCAL / 'routes.jsonl'
JUDGES = LOCAL / 'judges.jsonl'
REPORT = ROOT / 'results/eval/memory/m2p-report.json'
SLOTS = 4
SERVERS = {
    'reader': [m1.LLAMA, '-m', str(m1.MODELS / 'gemma-4-E2B/gemma-4-E2B_q4_0-it.gguf'), '-c', str(8192 * SLOTS), '-np', str(SLOTS),
               '--reasoning', 'off', '--host', '127.0.0.1', '--port', '8492', '--alias', 'm2p-e2b'],
    'judge': [m1.LLAMA, '-m', str(m1.QWEN), '-ngl', '99', '-c', str(8192 * SLOTS), '--parallel', str(SLOTS), '--reasoning', 'off',
              '--host', '127.0.0.1', '--port', '8493', '--alias', 'm2p-qwen'],
}
H_SENTENCE = re.search(r"export const H_SENTENCE = ((?:'[^']*'\s*\+?\s*)+)", (ROOT / 'scripts/eval/m1/recall.mjs').read_text()).group(1)
H_SENTENCE = ''.join(re.findall(r"'([^']*)'", H_SENTENCE))
H2_SENTENCE = H_SENTENCE.replace(
    'and do not mention them otherwise.',
    'and do not mention them otherwise. Answer everything else from your own knowledge as usual; never say you cannot '
    'answer because something is not in these excerpts.')
ROUTER_1 = """Does the following message from a user need information from the user's earlier conversations with you (something the user or you said or did before) to be answered well? Answer yes or no.

Message:
{message}"""
ROUTER_2 = """Below are a user's new message and excerpts from the user's earlier conversations. Would a good answer to the new message need something from these excerpts? Answer yes only if the message asks about or depends on something in them; answer no if the message can be answered well without them. Answer yes or no.

New message:
{message}

Excerpts:
{excerpts}"""
MISFIRE_V2 = """You are checking a reply from an AI assistant.

The assistant was shown notes from the user's earlier conversations, and then the user's new message. The new message does not need anything from those earlier conversations.

Did the reply draw on the notes or on the earlier conversations? Answer yes if the reply does any of these:
- uses or mentions a detail, preference, plan, name, place or topic that is in the notes but not in the new message;
- repeats or closely follows an answer that is in the notes;
- steers the question toward a topic from the notes;
- says or implies that the assistant remembers the user or talked with them before (for example "as we discussed", "you mentioned before", "以前の会話で").

Answer no if everything in the reply could have been written from the new message alone, even when the reply is on the same general subject as the notes. A reply that only says some information is missing from its memory, without using anything from the notes, is no.

Answer yes or no only.

User's new message:
{message}

Notes shown to the assistant:
{notes}

Assistant's reply:
{reply}"""
V2 = hashlib.sha256(MISFIRE_V2.encode()).hexdigest()[:8]
EN_MEM = re.compile(r'(memory|memories|excerpts?|\bnotes\b|provided (information|context|text|conversation)|conversation history'
                    r'|knowledge base|(past|previous|earlier) conversations?)', re.I)
EN_NEG = re.compile(r"(do not|don't|does not|doesn't|cannot|can't|no (specific |relevant |detailed )?information|not (mentioned|included|contain"
                    r"|available|found|present|provided)|isn't|aren't|unable)", re.I)
JA_MEM = re.compile(r'(提供された|記憶|会話の?履歴|メモ|抜粋|以前の会話)')
JA_NEG = re.compile(r'(ありません|含まれていません|見当たりません|分かりません|わかりません|できません|ないため)')
GENERATED = ('S', 'H2', 'SH2')
lock = threading.Lock()


def refused(text):
    """The reply says something is missing from the memory / excerpts / provided information."""
    for s in re.split(r'(?<=[.!?。！？])\s+|(?<=[。！？])|\n+', text[:1500]):
        if (EN_MEM.search(s) and EN_NEG.search(s)) or (JA_MEM.search(s) and JA_NEG.search(s)):
            return True
    return False


def parts(block):
    """'<memory>\\nCurrent date: X\\n<sentence>\\n\\n<excerpts>\\n</memory>' -> (date line, excerpts)."""
    head, excerpts = block.split('\n\n', 1)
    return head.split('\n')[1], excerpts.rsplit('\n</memory>', 1)[0]


def with_sentence(block, sentence):
    date, excerpts = parts(block)
    return f'<memory>\n{date}\n{sentence}\n\n{excerpts}\n</memory>'


def messages(arm, r):
    if arm == 'S':
        return [{'role': 'system', 'content': m1.SYSTEM + '\n\n' + r['P']}, {'role': 'user', 'content': r['message']}]
    if arm == 'H2':
        return [{'role': 'system', 'content': m1.SYSTEM}, {'role': 'user', 'content': m1.user_message(with_sentence(r['P'], H2_SENTENCE), r['message'])}]
    if arm == 'SH2':
        return [{'role': 'system', 'content': m1.SYSTEM + '\n\n' + with_sentence(r['P'], H2_SENTENCE)}, {'role': 'user', 'content': r['message']}]
    raise ValueError(arm)


class Server:
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

    def chat(self, body):
        req = urllib.request.Request(f'http://127.0.0.1:{self.port}/v1/chat/completions',
                                     data=json.dumps({'model': self.alias, **body}).encode(), headers={'content-type': 'application/json'})
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)

    def stop(self):
        self.proc.terminate()
        self.proc.wait(30)


def append(path, row):
    with lock, open(path, 'a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def pool(server, jobs, work, label):
    done = [0]

    def one(job):
        work(server, job)
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f'{label} {done[0]}/{len(jobs)}', flush=True)
    with ThreadPoolExecutor(SLOTS) as ex:
        list(ex.map(one, jobs))


def load():
    inputs = json.loads(m1.INPUTS.read_text())
    recall = json.loads(m1.RECALL.read_text())
    users = [u for u in inputs['users'] if u['split'] == 'test']
    probes = [p['id'] for u in users for p in u['probes']]
    questions = [f"{u['user']}-q" for u in users]
    return inputs, recall, users, probes, questions


def answers_cmd():
    _, recall, _, probes, questions = load()
    # sanity: the H block rebuilt from P must be the block M1 used
    q0 = probes[0]
    assert with_sentence(recall[q0]['P'], H_SENTENCE) == recall[q0]['H'], 'H rebuild differs from M1'
    queries = [q for q in probes + questions if recall[q]['P']]
    have, routed = m1.journal(ANSWERS), m1.journal(ROUTES)
    todo = [(arm, q) for q in queries for arm in GENERATED if f'{arm}|{q}' not in have]
    routes = [(name, q) for q in queries for name in ('R1', 'R2') if f'{name}|{q}' not in routed]
    if not todo and not routes:
        return
    server = Server('reader')

    def gen(s, job):
        arm, q = job
        t = time.time()
        c = s.chat({'messages': messages(arm, recall[q]), **m1.ANSWER})['choices'][0]
        append(ANSWERS, {'key': f'{arm}|{q}', 'arm': arm, 'query': q, 'reply': c['message']['content'], 'finish': c['finish_reason'],
                         'seconds': round(time.time() - t, 3)})

    def route(s, job):
        name, q = job
        r = recall[q]
        prompt = ROUTER_1.format(message=r['message']) if name == 'R1' else ROUTER_2.format(message=r['message'], excerpts=parts(r['P'])[1])
        t = time.time()
        c = s.chat({'messages': [{'role': 'user', 'content': prompt}], **{**m1.ANSWER, 'max_tokens': 3}})['choices'][0]
        text = c['message']['content'].strip().casefold()
        append(ROUTES, {'key': f'{name}|{q}', 'route': name, 'query': q, 'raw': text, 'yes': 0 if text.startswith('no') else 1,
                        'parsed': text.startswith(('yes', 'no')), 'seconds': round(time.time() - t, 3)})
    try:
        pool(server, routes, route, 'routes')
        pool(server, todo, gen, 'answers')
    finally:
        server.stop()


def judge_misfire(s, message, notes, reply):
    out = s.chat({'messages': [{'role': 'user', 'content': MISFIRE_V2.format(message=message, notes=notes, reply=reply)}],
                  **m1.JUDGE['generation']})
    return g1m.judge_label(out)


def notes_for(arm, r):
    if arm == 'N':
        return 'none'
    return r['P'] if arm == 'S' else with_sentence(r['P'], H2_SENTENCE) if arm in ('H2', 'SH2') else r[arm]


def calibrate():
    recall = json.loads(m1.RECALL.read_text())
    key = json.loads((m1.LOCAL / 'claude-sheet-key.json').read_text())
    claude = json.loads(m1.CLAUDE.read_text())['labels']
    m1_answers = m1.journal(m1.ANSWERS)
    path = LOCAL / f'calibrate-{V2}.jsonl'
    have = m1.journal(path)
    todo = [(i, k) for i, k in key.items() if k not in have]
    if todo:
        server = Server('judge')

        def one(s, job):
            i, k = job
            arm, q = k.split('|', 1)
            r = recall[q]
            label, failure = judge_misfire(s, r['message'], notes_for(arm, r), m1_answers[k]['reply'])
            append(path, {'key': k, 'item': i, 'label': label, 'failure': failure})
        try:
            pool(server, todo, one, 'calibrate')
        finally:
            server.stop()
    have = m1.journal(path)
    v1 = m1.journal(m1.JUDGES)
    for name, labels in (('v1', v1), ('v2', have)):
        pairs = [(int(claude[i]), labels[k]['label']) for i, k in key.items()]
        print(name, json.dumps(kappa(pairs)))
    for i, k in sorted(key.items(), key=lambda kv: int(kv[0])):
        if int(claude[i]) != have[k]['label']:
            print('  differs', i, k, 'claude', claude[i], 'v2', have[k]['label'])


def kappa(pairs):
    agree = float(np.mean([a == b for a, b in pairs]))
    pa, pb = np.mean([a for a, _ in pairs]), np.mean([b for _, b in pairs])
    pe = pa * pb + (1 - pa) * (1 - pb)
    return {'items': len(pairs), 'agreement': round(agree, 4), 'kappa': round(float((agree - pe) / (1 - pe)), 4) if pe < 1 else None,
            'claude_yes_judge_no': sum(1 for a, b in pairs if a and not b), 'claude_no_judge_yes': sum(1 for a, b in pairs if b and not a)}


def all_answers():
    a = {k: v for k, v in m1.journal(m1.ANSWERS).items()}
    a.update(m1.journal(ANSWERS))
    return a


def judge_cmd():
    inputs, recall, _, probes, questions = load()
    answers = all_answers()
    by_user = {u['user']: u for u in inputs['users']}
    have = m1.journal(JUDGES)
    jobs = [('misfire', arm, q) for arm in ('N', 'P', 'H') + GENERATED for q in probes
            if f'{arm}|{q}' in answers and f'v2|{arm}|{q}' not in have]
    jobs += [('accuracy', arm, q) for arm in GENERATED for q in questions if f'{arm}|{q}' in answers and f'acc|{arm}|{q}' not in have]
    if not jobs:
        return
    server = Server('judge')

    def one(s, job):
        kind, arm, q = job
        r, reply = recall[q], answers[f'{arm}|{q}']['reply']
        if kind == 'misfire':
            label, failure = judge_misfire(s, r['message'], notes_for(arm, r), reply)
            append(JUDGES, {'key': f'v2|{arm}|{q}', 'version': V2, 'label': label, 'failure': failure})
        else:
            u = by_user[r['user']]
            prompt = m1.JUDGE['templates']['abstention' if u['abstention'] else u['question_type']].format(
                question=u['question'], answer=u['answer'], response=reply)
            label, failure = g1m.judge_label(s.chat({'messages': [{'role': 'user', 'content': prompt}], **m1.JUDGE['generation']}))
            append(JUDGES, {'key': f'acc|{arm}|{q}', 'label': label, 'failure': failure})
    try:
        pool(server, sorted(jobs, key=lambda j: hashlib.sha256('|'.join(j).encode()).hexdigest()), one, 'judged')
    finally:
        server.stop()


def report():
    inputs, recall, users, probes, questions = load()
    answers = all_answers()
    judged = m1.journal(JUDGES)
    m1_judged = m1.journal(m1.JUDGES)
    routes = m1.journal(ROUTES)
    tau = json.loads(m1.REPORT.read_text())['tau']
    repeats = set()
    for u in users:
        own = {t['text'].strip() for s in u['sessions'] for t in s['turns'] if t['role'] == 'user'}
        repeats |= {p['id'] for p in u['probes'] if p['text'].strip() in own}
    yes = lambda name, q: routes[f'{name}|{q}']['yes'] if recall[q]['P'] else 0

    def pick(arm, q):
        """The generated arm whose reply this (possibly derived) arm returns for query q."""
        if arm == 'G':
            return 'P' if recall[q]['top'] >= tau else 'N'
        if arm.startswith('F·'):
            base = pick(arm[2:], q)
            return 'N' if base != 'N' and refused(answers[f'{base}|{q}']['reply']) else base
        if arm.startswith(('R1·', 'R2·')):
            return arm[3:] if yes(arm[:2], q) else 'N'
        return arm

    def misfire(arm, q):
        return judged[f'v2|{pick(arm, q)}|{q}']['label']

    def correct(arm, q):
        a = pick(arm, q)
        return m1_judged[f'{a}|{q}']['label'] if a in ('N', 'P', 'H') else judged[f'acc|{a}|{q}']['label']

    def refusal(arm, q):
        return int(refused(answers[f'{pick(arm, q)}|{q}']['reply']))

    def diff(f, arm, ids):
        d = [f(arm, q) - f('P', q) for q in ids]
        return {'difference': round(float(np.mean(d)), 4), 'ci95': [round(x, 4) for x in g1m.interval(d, [recall[q]['user'] for q in ids])]}

    arms = ['N', 'P', 'H', 'G', 'S', 'H2', 'SH2', 'F·P', 'F·SH2', 'R1·P', 'R2·P', 'R2·SH2', 'F·R2·SH2']
    en = [q for q in probes if recall[q]['lang'] == 'en']
    kept = [q for q in probes if q not in repeats]
    out = {'note': 'exploratory pilot on M1 test users; see docs/MEMORY_MISFIRE_M2_PILOT.md', 'judge_v2': V2,
           'probes': len(probes), 'questions': len(questions), 'repeated_probes': len(repeats), 'arms': {}}
    for arm in arms:
        out['arms'][arm] = {
            'misfire': round(float(np.mean([misfire(arm, q) for q in probes])), 4),
            'misfire_without_repeats': round(float(np.mean([misfire(arm, q) for q in kept])), 4),
            'refusal_en': round(float(np.mean([refusal(arm, q) for q in en])), 4),
            'refusal_ja': round(float(np.mean([refusal(arm, q) for q in probes if q not in en])), 4),
            'accuracy': round(float(np.mean([correct(arm, q) for q in questions])), 4),
        }
        if arm != 'P':
            out['arms'][arm]['vs_P'] = {'misfire': diff(misfire, arm, probes), 'misfire_without_repeats': diff(misfire, arm, kept),
                                        'refusal': diff(refusal, arm, probes), 'accuracy': diff(correct, arm, questions)}
    out['routes'] = {name: {'probes_yes': round(float(np.mean([yes(name, q) for q in probes])), 4),
                            'questions_yes': round(float(np.mean([yes(name, q) for q in questions])), 4),
                            'unparsed': sum(1 for k, v in routes.items() if k.startswith(name) and not v['parsed']),
                            'median_seconds': float(np.median([v['seconds'] for k, v in routes.items() if k.startswith(name)]))}
                     for name in ('R1', 'R2')}
    out['routes']['R1']['questions_no'] = [q for q in questions if not yes('R1', q)]
    out['routes']['R2']['questions_no'] = [q for q in questions if not yes('R2', q)]
    out['failures'] = {'judge': sum(1 for v in judged.values() if v.get('failure')),
                       'answers_cut': {arm: sum(1 for q in probes + questions if answers.get(f'{arm}|{q}', {}).get('finish', 'stop') != 'stop')
                                       for arm in GENERATED}}
    REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({a: {k: v for k, v in r.items() if k != 'vs_P'} for a, r in out['arms'].items()}, ensure_ascii=False, indent=1))
    print(json.dumps(out['routes'], indent=1))


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['answers', 'calibrate', 'judge', 'report'])
    a = p.parse_args()
    {'answers': answers_cmd, 'calibrate': calibrate, 'judge': judge_cmd, 'report': report}[a.command]()
