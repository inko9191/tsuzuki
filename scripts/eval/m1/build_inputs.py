#!/usr/bin/env python3
"""M1 inputs (docs/MEMORY_MISFIRE_M1_PREREG.md): users, their histories, the question each needs memory for, and
eight probe messages each that need nothing from their history; dev/test split. Deterministic.

English: the 40 users of U2 (LongMemEval-S cleaned). Japanese: the 40 users of J1.
A probe is the first user message of a session that is not in the user's history and is no question's answer
session: from the rest of LongMemEval-S for English, from the other J1 users for Japanese; 20 to 600 characters.

    python3 scripts/eval/m1/build_inputs.py   # writes build/eval/memory/m1/inputs.json
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LME = ROOT / 'build/eval/memory/longmemeval-g1/longmemeval_s_cleaned.json'
J1 = ROOT / 'build/eval/memory/j1/j1_dataset.json'
U2_PLAN = ROOT / 'results/eval/memory/longmemeval-u2-plan.json'
OUT = ROOT / 'build/eval/memory/m1/inputs.json'
SEED = 'm1-v1'
PROBES_PER_USER = 8
DEV_USERS = 8


def h(*parts):
    return hashlib.sha256('|'.join([SEED, *parts]).encode()).hexdigest()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_user_turn(session):
    return next((t['content'].strip() for t in session if t['role'] == 'user' and t['content'].strip()), '')


def user(lang, q):
    return {'user': f"{lang}-{q['question_id']}", 'lang': lang, 'question_id': q['question_id'],
            'question_type': q['question_type'], 'question': q['question'], 'answer': q['answer'],
            'question_date': q['question_date'], 'abstention': q['question_id'].endswith('_abs'),
            'sessions': [{'id': sid, 'date': date, 'turns': [{'role': t['role'], 'text': t['content']} for t in turns]}
                         for sid, date, turns in zip(q['haystack_session_ids'], q['haystack_dates'], q['haystack_sessions'])]}


def probes(u, pool):
    """Eight first messages of sessions outside this user's history, in a fixed per-user hash order."""
    own = {s['id'] for s in u['sessions']}
    chosen = []
    for sid, text in sorted(pool.items(), key=lambda kv: h(u['user'], kv[0])):
        if sid in own or not 20 <= len(text) <= 600:
            continue
        chosen.append({'id': f"{u['user']}-p{len(chosen)}", 'session': sid, 'text': text})
        if len(chosen) == PROBES_PER_USER:
            break
    return chosen


def main():
    lme = json.loads(LME.read_text())
    j1 = json.loads(J1.read_text())
    plan = json.loads(U2_PLAN.read_text())
    answer_sessions = {s for q in lme for s in q['answer_session_ids']} | {s for q in j1 for s in q['answer_session_ids']}
    by_id = {q['question_id']: q for q in lme}
    en_pool, ja_pool = {}, {}
    for q in lme:
        for sid, turns in zip(q['haystack_session_ids'], q['haystack_sessions']):
            if sid not in answer_sessions:
                en_pool.setdefault(sid, first_user_turn(turns))
    for q in j1:
        for sid, turns in zip(q['haystack_session_ids'], q['haystack_sessions']):
            if sid not in answer_sessions:
                ja_pool.setdefault(sid, first_user_turn(turns))
    users = [user('en', by_id[qid]) for qid in plan['question_ids']] + [user('ja', q) for q in j1]
    for lang in ('en', 'ja'):
        ranked = sorted((u for u in users if u['lang'] == lang), key=lambda u: h('split', u['user']))
        for k, u in enumerate(ranked):
            u['split'] = 'dev' if k < DEV_USERS else 'test'
    for u in users:
        u['probes'] = probes(u, en_pool if u['lang'] == 'en' else ja_pool)
        assert len(u['probes']) == PROBES_PER_USER, u['user']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({'seed': SEED, 'sources': {'longmemeval_s_cleaned': sha256(LME), 'j1_dataset': sha256(J1),
                                                          'u2_plan': sha256(U2_PLAN)}, 'users': users}, ensure_ascii=False))
    for lang in ('en', 'ja'):
        us = [u for u in users if u['lang'] == lang]
        print(lang, 'users', len(us), 'dev', sum(u['split'] == 'dev' for u in us), 'probes', sum(len(u['probes']) for u in us),
              'sessions', sum(len(u['sessions']) for u in us))
    print(OUT, sha256(OUT))


if __name__ == '__main__':
    main()
