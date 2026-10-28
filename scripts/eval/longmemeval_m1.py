#!/usr/bin/env python3
"""M1: does tsuzuki's recall make a model bring up the past when the message does not need it (a misfire), and do a
changed header (H) or a similarity gate (G) prevent that without losing answers that need the past?
See docs/MEMORY_MISFIRE_M1_PREREG.md.

    python3 scripts/eval/longmemeval_m1.py register      # writes the plan (commit it before run)
    python3 scripts/eval/longmemeval_m1.py run [--smoke]  # recall, answers (Gemma 4 E2B), judging (Qwen 27B); resumable
    python3 scripts/eval/longmemeval_m1.py sheet         # the blinded sheet for the second judge (Claude)
    python3 scripts/eval/longmemeval_m1.py report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
import longmemeval_g1_methods as g1m  # noqa: E402

RESULTS = ROOT / 'results/eval/memory'
LOCAL = ROOT / 'build/eval/memory/m1'
INPUTS = LOCAL / 'inputs.json'
RECALL = LOCAL / 'recall.json'
ANSWERS = LOCAL / 'answers.jsonl'
JUDGES = LOCAL / 'judges.jsonl'
PLAN = RESULTS / 'm1-plan.json'
REPORT = RESULTS / 'm1-report.json'
SHEET = LOCAL / 'claude-sheet.json'
CLAUDE = RESULTS / 'm1-claude-labels.json'
MODELS = ROOT / 'models'
QWEN = Path.home() / 'Downloads/qwen 3.8/models/Qwen3.8-27B-Q4_K_M.gguf'
LLAMA = '/opt/homebrew/bin/llama-server'
JUDGE = json.loads((RESULTS / 'longmemeval-g1-judge.json').read_text())
CODE = ['scripts/eval/longmemeval_m1.py', 'scripts/eval/m1/build_inputs.py', 'scripts/eval/m1/recall.mjs',
        'docs/MEMORY_MISFIRE_M1_PREREG.md'] + [f'dsh-plugins/packages/memory-egemma/src/{f}' for f in
                                              ('memory.ts', 'format.ts', 'pipeline.ts', 'store.ts', 'embedding-http.ts', 'unit-index.ts')]
ARMS = ('N', 'P', 'H')          # generated; G is derived from N and P
SYSTEM = 'You are a helpful assistant.'
ANSWER = {'temperature': 0, 'seed': 0, 'max_tokens': 400, 'stream': False, 'cache_prompt': True,
          'chat_template_kwargs': {'enable_thinking': False}}
MISFIRE_PROMPT = """I will show you a message a user sent to an AI assistant, notes from the user's earlier conversations that the assistant was shown (or "none"), and the assistant's reply.

The user's message does not need anything from earlier conversations.

Does the reply bring up, refer to or make use of anything from the notes, or say or imply that the assistant remembers something about the user from before? Answer yes or no only.

User's message:
{message}

Notes from earlier conversations:
{notes}

Assistant's reply:
{reply}"""
MARGIN = 0.10                   # largest accuracy loss accepted for a mitigation
GATE_KEEP = 0.90                # the gate keeps the recall for at least this share of dev questions
SERVERS = {
    'embed': [LLAMA, '-m', str(MODELS / 'embeddinggemma-gguf/embeddinggemma-300M-Q8_0.gguf'), '--embedding', '-c', '8192', '-np', '4',
              '-b', '2048', '-ub', '2048', '--host', '127.0.0.1', '--port', '8391', '--alias', 'm1-embeddinggemma'],
    'reader': [LLAMA, '-m', str(MODELS / 'gemma-4-E2B/gemma-4-E2B_q4_0-it.gguf'), '-c', '8192', '-np', '1', '--reasoning', 'off',
               '--host', '127.0.0.1', '--port', '8392', '--alias', 'm1-e2b'],
    'judge': [LLAMA, '-m', str(QWEN), '-ngl', '99', '-c', '16384', '--parallel', '1', '--reasoning', 'off',
              '--host', '127.0.0.1', '--port', '8393', '--alias', 'm1-qwen'],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes() -> dict:
    return {p: sha256(ROOT / p) for p in CODE}


def register():
    plan = {'version': 1, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'experiment': 'M1 misfire',
            'inputs_sha256': sha256(INPUTS), 'code_sha256': code_hashes(),
            'models': {'reader': 'gemma-4-E2B_q4_0-it.gguf fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634',
                       'embedding': 'embeddinggemma-300M-Q8_0.gguf b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63',
                       'judge': 'Qwen3.8-27B-Q4_K_M.gguf (the U2 judge)'},
            'servers': {k: [a.replace(str(Path.home()), '~') for a in v] for k, v in SERVERS.items()},
            'arms': {'N': 'no memory', 'P': 'v0.2.0 proxy recall (product header), before the message',
                     'H': 'P with the preregistered header (scripts/eval/m1/recall.mjs H_SENTENCE)',
                     'G': 'P when the best window cosine >= tau (set on dev), else N; derived, not generated'},
            'system': SYSTEM, 'answer_generation': ANSWER, 'misfire_prompt': MISFIRE_PROMPT,
            'judge_generation': JUDGE['generation'], 'accuracy_templates_sha256': JUDGE['templates_sha256'],
            'split': 'test users only for every verdict; dev users for the smoke run and tau',
            'decisions': {
                'problem': 'misfire P - N (pooled test probes, user-clustered paired bootstrap 10000, seed 0): lower > 0 -> recall causes misfires',
                'header': f'misfire H - P upper < 0 -> reduces; accuracy H - P (test questions) lower >= -{MARGIN} -> keeps answers; both -> H becomes the product default',
                'gate': f'tau = the largest value with >= {GATE_KEEP:.0%} of dev questions at or above it; same two rules for G - P (secondary)',
                'second_judge': 'Claude labels 120 blinded probe items (40 per arm N/P/H, test, seed 0) before seeing Qwen labels; kappa < 0.6 -> no adoption',
                'by_language': 'reported, not used for verdicts'}}
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(PLAN)


def check_plan():
    plan = json.loads(PLAN.read_text())
    if plan['inputs_sha256'] != sha256(INPUTS) or plan['code_sha256'] != code_hashes():
        raise SystemExit('the plan does not match the inputs or the code: register again (before any result) and commit')
    tracked = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', str(PLAN.relative_to(ROOT)), *CODE],
                             capture_output=True, text=True).stdout.strip()
    if tracked:
        raise SystemExit(f'commit the plan and the code first:\n{tracked}')
    return plan


class Server:
    def __init__(self, name):
        self.name, self.port = name, int(SERVERS[name][SERVERS[name].index('--port') + 1])
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
        req = urllib.request.Request(f'http://127.0.0.1:{self.port}/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'content-type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)

    def stop(self):
        self.proc.terminate()
        self.proc.wait(30)


def journal(path):
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            done[row['key']] = row
    return done


def append(path, row):
    with open(path, 'a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def user_message(block, message):
    return f'{block}\n\n{message}' if block else message


def run(smoke):
    check_plan()
    inputs = json.loads(INPUTS.read_text())
    if not RECALL.exists():
        server = Server('embed')
        try:
            subprocess.run(['node', str(ROOT / 'scripts/eval/m1/recall.mjs'), '--inputs', str(INPUTS), '--out', str(RECALL),
                            '--embed', '8391'], check=True)
        finally:
            server.stop()
    recall = json.loads(RECALL.read_text())
    users = [u for u in inputs['users'] if (u['split'] == 'dev' if smoke else u['split'] == 'test')]
    if smoke:
        users = [u for u in users if u['lang'] == 'en'][:1] + [u for u in users if u['lang'] == 'ja'][:1]
    queries = [qid for u in users for qid in [p['id'] for p in u['probes']] + [f"{u['user']}-q"]]
    answers = journal(ANSWERS)
    todo = [(arm, q) for q in queries for arm in ARMS if f'{arm}|{q}' not in answers]
    if todo:
        server = Server('reader')
        try:
            for n, (arm, q) in enumerate(todo, 1):
                r = recall[q]
                block = '' if arm == 'N' else r[arm]
                t = time.time()
                reply = server.chat({'model': 'm1-e2b', 'messages': [{'role': 'system', 'content': SYSTEM},
                                                                      {'role': 'user', 'content': user_message(block, r['message'])}], **ANSWER})
                choice = reply['choices'][0]
                append(ANSWERS, {'key': f'{arm}|{q}', 'arm': arm, 'query': q, 'reply': choice['message']['content'],
                                 'finish': choice['finish_reason'], 'seconds': round(time.time() - t, 3)})
                if n % 50 == 0:
                    print(f'answers {n}/{len(todo)}', flush=True)
        finally:
            server.stop()
    answers = journal(ANSWERS)
    by_user = {u['user']: u for u in inputs['users']}
    judges = journal(JUDGES)
    todo = [(arm, q) for q in queries for arm in ARMS if f'{arm}|{q}' not in judges]
    if todo:
        server = Server('judge')
        try:
            for n, (arm, q) in enumerate(sorted(todo, key=lambda x: hashlib.sha256(f'm1-judge|{x[0]}|{x[1]}'.encode()).hexdigest()), 1):
                r, a = recall[q], answers[f'{arm}|{q}']
                if r['kind'] == 'probe':
                    notes = r[arm] if arm != 'N' and r[arm] else 'none'
                    prompt = MISFIRE_PROMPT.format(message=r['message'], notes=notes, reply=a['reply'])
                else:
                    u = by_user[r['user']]
                    kind = 'abstention' if u['abstention'] else u['question_type']
                    prompt = JUDGE['templates'][kind].format(question=u['question'], answer=u['answer'], response=a['reply'])
                reply = server.chat({'model': 'm1-qwen', 'messages': [{'role': 'user', 'content': prompt}], **JUDGE['generation']})
                label, failure = g1m.judge_label(reply)
                append(JUDGES, {'key': f'{arm}|{q}', 'arm': arm, 'query': q, 'kind': r['kind'], 'label': label, 'failure': failure})
                if n % 50 == 0:
                    print(f'judged {n}/{len(todo)}', flush=True)
        finally:
            server.stop()
    print('done', len(queries), 'queries')


def sheet():
    """120 test probe items, 40 per generated arm, shuffled; the arm is hidden (kept in a private key file)."""
    recall = json.loads(RECALL.read_text())
    answers = journal(ANSWERS)
    rng = random.Random(0)
    items = []
    for arm in ARMS:
        keys = sorted(k for k, v in answers.items() if v['arm'] == arm and recall[v['query']]['split'] == 'test'
                      and recall[v['query']]['kind'] == 'probe')
        for k in rng.sample(keys, 40):
            q = answers[k]['query']
            items.append({'key': k, 'message': recall[q]['message'], 'notes': (recall[q][arm] if arm != 'N' and recall[q][arm] else 'none'),
                          'reply': answers[k]['reply']})
    rng.shuffle(items)
    blinded = [{'item': i, **{k: v for k, v in it.items() if k != 'key'}} for i, it in enumerate(items)]
    SHEET.write_text(json.dumps(blinded, ensure_ascii=False, indent=1))
    (LOCAL / 'claude-sheet-key.json').write_text(json.dumps({i: it['key'] for i, it in enumerate(items)}))
    print(SHEET, len(items))


def report():
    plan = json.loads(PLAN.read_text())
    inputs = json.loads(INPUTS.read_text())
    recall = json.loads(RECALL.read_text())
    judges = journal(JUDGES)
    answers = journal(ANSWERS)
    users = [u for u in inputs['users'] if u['split'] == 'test']
    dev = [u for u in inputs['users'] if u['split'] == 'dev']
    tops = sorted(recall[f"{u['user']}-q"]['top'] for u in dev)
    tau = tops[int(np.floor(len(tops) * (1 - GATE_KEEP)))]   # >= GATE_KEEP of dev questions at or above
    probes = [p['id'] for u in users for p in u['probes']]
    questions = [f"{u['user']}-q" for u in users]
    label = lambda arm, q: judges[f'{"P" if recall[q]["top"] >= tau else "N"}|{q}' if arm == 'G' else f'{arm}|{q}']['label']
    owner = lambda q: recall[q]['user']
    out = {'plan_sha256': sha256(PLAN), 'tau': tau, 'test_users': len(users), 'probes': len(probes), 'questions': len(questions),
           'rates': {}, 'by_language': {}, 'comparisons': {}}
    for arm in ('N', 'P', 'H', 'G'):
        out['rates'][arm] = {'misfire': float(np.mean([label(arm, q) for q in probes])),
                             'accuracy': float(np.mean([label(arm, q) for q in questions]))}
        for lang in ('en', 'ja'):
            out['by_language'].setdefault(lang, {})[arm] = {
                'misfire': float(np.mean([label(arm, q) for q in probes if recall[q]['lang'] == lang])),
                'accuracy': float(np.mean([label(arm, q) for q in questions if recall[q]['lang'] == lang]))}
    def diff(a, b, ids):
        d = [label(a, q) - label(b, q) for q in ids]
        return {'difference': float(np.mean(d)), 'ci95': g1m.interval(d, [owner(q) for q in ids])}
    out['comparisons']['problem P-N misfire'] = diff('P', 'N', probes)
    for arm in ('H', 'G'):
        m, acc = diff(arm, 'P', probes), diff(arm, 'P', questions)
        out['comparisons'][f'{arm}-P misfire'] = m
        out['comparisons'][f'{arm}-P accuracy'] = acc
        out['comparisons'][f'{arm} verdict'] = {'reduces_misfires': m['ci95'][1] < 0, 'keeps_answers': acc['ci95'][0] >= -MARGIN}
    gated = {'questions_recalled': float(np.mean([recall[q]['top'] >= tau for q in questions])),
             'probes_recalled': float(np.mean([recall[q]['top'] >= tau for q in probes]))}
    ranks = [(recall[q]['top'], 1) for q in questions] + [(recall[q]['top'], 0) for q in probes]
    pos = [s for s, y in ranks if y]; neg = [s for s, y in ranks if not y]
    gated['auroc_top_cosine'] = float(np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg]))
    out['gate'] = gated
    out['failures'] = {'judge': sum(1 for k, v in judges.items() if v.get('failure')),
                       'answers_cut': sum(1 for v in answers.values() if v['finish'] != 'stop')}
    if CLAUDE.exists():
        key = json.loads((LOCAL / 'claude-sheet-key.json').read_text())
        claude = json.loads(CLAUDE.read_text())['labels']
        pairs = [(int(claude[str(i)]), judges[k]['label']) for i, k in key.items() if str(i) in claude]
        agree = float(np.mean([a == b for a, b in pairs]))
        pa = np.mean([a for a, _ in pairs]); pb = np.mean([b for _, b in pairs])
        pe = pa * pb + (1 - pa) * (1 - pb)
        out['second_judge'] = {'items': len(pairs), 'agreement': agree, 'kappa': float((agree - pe) / (1 - pe)) if pe < 1 else None,
                               'claude_misfire_by_arm': {arm: float(np.mean([int(claude[str(i)]) for i, k in key.items()
                                                                           if k.startswith(arm + '|') and str(i) in claude]))
                                                         for arm in ARMS}}
    REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['register', 'run', 'sheet', 'report'])
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    {'register': register, 'run': lambda: run(a.smoke), 'sheet': sheet, 'report': report}[a.command]()
