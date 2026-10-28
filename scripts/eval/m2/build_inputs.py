#!/usr/bin/env python3
"""M2 inputs (docs/MEMORY_MISFIRE_M2_PREREG.md): users, their histories, the question each needs memory for, and for
test users eight probe messages each that need nothing from their history. Deterministic.

dev (only the question's best cosine, to set tau; no replies): the LongMemEval-S questions used in an earlier plan
(U1, U2, U2-AT, E1, G1, S2, S3) and the 40 J1 users. test: 96 of the LongMemEval-S questions used in no plan, and the
40 J2 users (scripts/eval/j2).

A probe is a message from outside the user's history: for English the first user message of a LongMemEval-S session
that is no question's answer session and was no M1 probe; for Japanese one of the first messages E2B wrote for J2
(build/eval/memory/j2/probes.json). 20 to 600 characters; never the same text as, or a near copy of (character
3-gram Jaccard >= 0.8), a user message in the history; no probe goes to two users.

    python3 scripts/eval/m2/build_inputs.py   # writes build/eval/memory/m2/inputs.json
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MEM = ROOT / 'build/eval/memory'
LME = MEM / 'longmemeval-g1/longmemeval_s_cleaned.json'
J1 = MEM / 'j1/j1_dataset.json'
J2 = MEM / 'j2/j2_dataset.json'
J2_PROBES = MEM / 'j2/probes.json'
M1_INPUTS = MEM / 'm1/inputs.json'
PLANS = [ROOT / 'results/eval/memory' / f for f in (
    'longmemeval-u1-plan.json', 'longmemeval-u2-plan.json', 'longmemeval-u2-at-plan.json', 'longmemeval-e1-plan.json',
    'longmemeval-g1-plan.json', 'longmemeval-g1-execution-plan.json', 's2-plan.json', 's3-plan.json')]
OUT = MEM / 'm2/inputs.json'
SEED = 'm2-v1'
EN_TEST = 96
PROBES_PER_USER = 8
NEAR = 0.8


def h(*parts):
    return hashlib.sha256('|'.join([SEED, *parts]).encode()).hexdigest()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_user_turn(session):
    return next((t['content'].strip() for t in session if t['role'] == 'user' and t['content'].strip()), '')


def grams(text):
    t = ''.join(text.split())
    return {t[i:i + 3] for i in range(max(len(t) - 2, 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def user(lang, q, split, source):
    return {'user': f"{lang}-{q['question_id']}", 'lang': lang, 'split': split, 'source': source,
            'question_id': q['question_id'], 'question_type': q['question_type'], 'question': q['question'],
            'answer': q['answer'], 'question_date': q['question_date'], 'abstention': q['question_id'].endswith('_abs'),
            'sessions': [{'id': sid, 'date': date, 'turns': [{'role': t['role'], 'text': t['content']} for t in turns]}
                         for sid, date, turns in zip(q['haystack_session_ids'], q['haystack_dates'], q['haystack_sessions'])]}


def assign(u, pool, taken, skipped):
    """Eight probes for user u from pool {id: text}, in a fixed per-user hash order."""
    own_ids = {s['id'] for s in u['sessions']}
    own = [t['text'].strip() for s in u['sessions'] for t in s['turns'] if t['role'] == 'user']
    own_text, own_grams = set(own), [grams(t) for t in own]
    chosen = []
    for pid, text in sorted(pool.items(), key=lambda kv: h(u['user'], kv[0])):
        if pid in taken or pid in own_ids or not 20 <= len(text) <= 600:
            continue
        if text in own_text:
            skipped['same_text'] += 1
            continue
        g = grams(text)
        if any(jaccard(g, o) >= NEAR for o in own_grams):
            skipped['near_copy'] += 1
            continue
        taken.add(pid)
        chosen.append({'id': f"{u['user']}-p{len(chosen)}", 'session': pid, 'text': text})
        if len(chosen) == PROBES_PER_USER:
            return chosen
    raise SystemExit(f"{u['user']}: not enough probes")


def main():
    lme = json.loads(LME.read_text())
    j1 = json.loads(J1.read_text())
    j2 = json.loads(J2.read_text())
    used_text = ''.join(p.read_text() for p in PLANS)
    used = {q['question_id'] for q in lme if q['question_id'] in used_text}
    fresh = sorted((q for q in lme if q['question_id'] not in used), key=lambda q: h('test', q['question_id']))
    users = ([user('en', q, 'dev', 'longmemeval') for q in lme if q['question_id'] in used]
             + [user('ja', q, 'dev', 'j1') for q in j1]
             + [user('en', q, 'test', 'longmemeval') for q in fresh[:EN_TEST]]
             + [user('ja', q, 'test', 'j2') for q in j2])
    answer_sessions = {s for q in lme + j1 + j2 for s in q['answer_session_ids']}
    m1_probes = {p['session'] for u in json.loads(M1_INPUTS.read_text())['users'] for p in u['probes']}
    en_pool = {}
    for q in lme:
        for sid, turns in zip(q['haystack_session_ids'], q['haystack_sessions']):
            if sid not in answer_sessions and sid not in m1_probes:
                en_pool.setdefault(sid, first_user_turn(turns))
    ja_pool = {k: v['text'].strip() for k, v in json.loads(J2_PROBES.read_text()).items()}
    taken, skipped = set(), {'same_text': 0, 'near_copy': 0}
    for u in sorted(users, key=lambda u: u['user']):
        u['probes'] = assign(u, en_pool if u['lang'] == 'en' else ja_pool, taken, skipped) if u['split'] == 'test' else []
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({'seed': SEED, 'skipped_probes': skipped,
                               'sources': {'longmemeval_s_cleaned': sha256(LME), 'j1_dataset': sha256(J1), 'j2_dataset': sha256(J2),
                                           'j2_probes': sha256(J2_PROBES), 'm1_inputs': sha256(M1_INPUTS),
                                           **{p.name: sha256(p) for p in PLANS}},
                               'users': users}, ensure_ascii=False))
    for split in ('dev', 'test'):
        for lang in ('en', 'ja'):
            us = [u for u in users if u['split'] == split and u['lang'] == lang]
            print(split, lang, 'users', len(us), 'probes', sum(len(u['probes']) for u in us),
                  'abstention', sum(u['abstention'] for u in us))
    print('used', len(used), 'fresh', len(fresh), 'skipped probes', skipped)
    print(OUT, sha256(OUT))


if __name__ == '__main__':
    main()
