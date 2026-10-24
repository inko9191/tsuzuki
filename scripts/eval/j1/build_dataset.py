#!/usr/bin/env python3
"""Build the J1 Japanese dataset in LongMemEval-S format from the written evidence and the E2B distractors.

    python3 build_dataset.py   # writes build/eval/memory/j1/j1_dataset.json and prints sizes

Per question: its evidence sessions (answer sessions) + its hard negatives + K distractor sessions sampled from the
pool (seeded by question id) at random times in the 180 days before the question date, all in date order.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import evidence_a  # noqa: E402
import evidence_b  # noqa: E402

POOL = ROOT/'build/eval/memory/j1/distractors.json'
OUT = ROOT/'build/eval/memory/j1/j1_dataset.json'
K = 45
SPAN_DAYS = 180


def fmt(dt: datetime) -> str:
    return dt.strftime('%Y/%m/%d (%a) %H:%M')


def parse(s: str) -> datetime:
    return datetime.strptime(s, '%Y-%m-%d %H:%M')


def chat(turns):
    return [{'role': r, 'content': t} for r, t in turns]


def main() -> None:
    items = evidence_a.ITEMS + evidence_b.ITEMS
    ids = [it['id'] for it in items]
    if len(ids) != 40 or len(set(ids)) != 40:
        raise SystemExit(f'expected 40 unique items, got {len(ids)}')
    pool = json.loads(POOL.read_text())
    keys = sorted(pool)
    if len(keys) < K:
        raise SystemExit(f'pool has {len(keys)} sessions; need at least {K}')
    rows = []
    for it in items:
        qdate = parse(it['qdate'])
        rng = random.Random('j1-haystack-v1|' + it['id'])
        sessions = []
        for k, (d, turns) in enumerate(it['sessions'], 1):
            sessions.append((parse(d), f'{it["id"]}_e{k}', chat(turns), True))
        for k, (d, turns) in enumerate(it.get('near', []), 1):
            sessions.append((parse(d), f'{it["id"]}_n{k}', chat(turns), False))
        taken = {s[0] for s in sessions}
        for key in rng.sample(keys, K):
            while True:
                when = qdate - timedelta(days=rng.randint(1, SPAN_DAYS), hours=rng.randint(0, 12), minutes=rng.randint(0, 59))
                when = when.replace(second=0, microsecond=0)
                if when not in taken:
                    taken.add(when)
                    break
            sessions.append((when, f'{it["id"]}_{key}', [{'role': t['role'], 'content': t['content']} for t in pool[key]['turns']], False))
        sessions.sort(key=lambda s: s[0])
        if any(s[0] >= qdate for s in sessions):
            raise SystemExit(f'{it["id"]}: a session is not before the question date')
        rows.append({
            'question_id': it['id'], 'question_type': it['type'], 'question': it['question'], 'answer': it['answer'],
            'question_date': fmt(qdate),
            'haystack_session_ids': [s[1] for s in sessions], 'haystack_dates': [fmt(s[0]) for s in sessions],
            'haystack_sessions': [s[2] for s in sessions], 'answer_session_ids': [s[1] for s in sessions if s[3]],
        })
    OUT.write_text(json.dumps(rows, ensure_ascii=False))
    chars = [sum(len(m['content']) for s in r['haystack_sessions'] for m in s) for r in rows]
    types = {}
    for r in rows:
        kind = 'abstention' if r['question_id'].endswith('_abs') else r['question_type']
        types[kind] = types.get(kind, 0) + 1
    print(f'{OUT}: {len(rows)} questions, {len(rows[0]["haystack_sessions"])}+ sessions each, '
          f'chars per history min {min(chars)} median {sorted(chars)[len(chars)//2]} max {max(chars)}')
    print('types', types)
    print('sha256', hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
