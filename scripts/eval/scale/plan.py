"""Scale test plan for dsh-memory-egemma: which LongMemEval-S sessions fill the stores, which questions are planted.

CPU only, deterministic. Reads the LongMemEval-S file the G1/U2 runners use and writes

  build/scale/plan.json       fillers (fixed order), held-out ingest sessions, planted questions, latency queries
  build/scale/sessions.jsonl  the content of every session the plan names (role, content, has_answer per message)

Fillers are LongMemEval-S haystack sessions that are not an answer session of ANY question, so a planted
question's evidence is never duplicated or contradicted by another question's evidence. Every store of size S uses
the first S fillers (nested stores), so accuracy drift between sizes is caused only by the added distractors.

    .venv/bin/python -I scripts/eval/scale/plan.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/'build/eval/memory/longmemeval-g1/longmemeval_s_cleaned.json'
OUT = ROOT/'build/scale'
SEED = 20261023
SIZES = [50, 500, 2000, 5000]
MAX_FILLERS = 10000          # perf-only extension beyond the requested sizes (synthetic vectors)
HOLDOUT = 40                 # sessions never in a store: messages for the ingest-at-scale test
PER_TYPE = 10                # planted questions per question type (6 non-abstention types -> 60)
EXTRA_QUERIES = 40           # further question texts for latency only
AGES = {'old': 0.05, 'mid': 0.5, 'recent': 0.95}  # where the first evidence session sits in the timeline


def main() -> None:
    raw = DATA.read_bytes()
    data = json.loads(raw)
    answer_sessions = {s for q in data for s in q['answer_session_ids']}
    content: dict[str, list[dict]] = {}
    for q in data:
        for sid, turns in zip(q['haystack_session_ids'], q['haystack_sessions']):
            content.setdefault(sid, turns)
    pool = sorted(s for s, turns in content.items()
                  if s not in answer_sessions and any(str(t.get('content') or '').strip() for t in turns))
    rng = random.Random(SEED)
    rng.shuffle(pool)
    fillers, holdout = pool[:MAX_FILLERS], pool[MAX_FILLERS:MAX_FILLERS + HOLDOUT]

    eligible = [q for q in data if not q['question_id'].endswith('_abs')]
    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    for q in sorted(eligible, key=lambda q: q['question_id']):
        by_type[q['question_type']].append(q)
    planted = []
    for qtype in sorted(by_type):
        for q in rng.sample(by_type[qtype], PER_TYPE):
            order = {sid: k for k, sid in enumerate(q['haystack_session_ids'])}
            evidence = sorted(q['answer_session_ids'], key=lambda s: order[s])
            answer_turns = {sid: [k for k, t in enumerate(content[sid]) if t.get('has_answer')] for sid in evidence}
            planted.append({'qid': q['question_id'], 'type': qtype, 'question': q['question'], 'answer': str(q['answer']),
                            'question_date': q['question_date'], 'evidence': evidence, 'answer_turns': answer_turns})
    planted_ids = {p['qid'] for p in planted}
    others = [q for q in eligible if q['question_id'] not in planted_ids]
    extra = rng.sample(others, EXTRA_QUERIES)
    queries = [{'qid': p['qid'], 'text': p['question']} for p in planted] + [{'qid': q['question_id'], 'text': q['question']} for q in extra]

    needed = list(dict.fromkeys(fillers + holdout + [s for p in planted for s in p['evidence']]))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'sessions.jsonl').open('w') as f:
        for sid in needed:
            turns = [{'role': t['role'], 'content': str(t.get('content') or ''), 'has_answer': bool(t.get('has_answer'))}
                     for t in content[sid]]
            f.write(json.dumps({'id': sid, 'turns': turns}, ensure_ascii=False) + '\n')
    kinds = collections.Counter(s.split('_')[0] if s.startswith(('sharegpt_', 'ultrachat_')) else 'simulated' for s in fillers[:max(SIZES)])
    plan = {
        'source': str(DATA.relative_to(ROOT)), 'source_sha256': hashlib.sha256(raw).hexdigest(), 'seed': SEED,
        'sizes': SIZES, 'ages': AGES, 'fillers': fillers, 'holdout': holdout, 'planted': planted, 'queries': queries,
        'counts': {'unique_sessions': len(content), 'answer_sessions': len(answer_sessions), 'filler_pool': len(pool),
                   'filler_kinds_5000': dict(kinds), 'planted': len(planted),
                   'planted_evidence_sessions': sum(len(p['evidence']) for p in planted)},
    }
    (OUT/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    print(json.dumps(plan['counts']))


if __name__ == '__main__':
    main()
