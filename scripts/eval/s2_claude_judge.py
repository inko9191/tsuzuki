#!/usr/bin/env python3
"""S2 secondary judge (Claude, blinded): grade a random sample of 90 (arm, question) answers before opening the Qwen
labels or the S2 report (docs/MEMORY_SCALE_S2_PREREG.md).

sheet: from questions.json and run/answers.json only, write a sheet of the sampled answers in a seeded random order
       with no arm names (official per-type rubric); the key goes to a file the grader does not open before grading.
score: read the grader's yes/no labels, unblind, report agreement with the Qwen judge and both judges' accuracy per
       arm on the sample.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/eval'))
import longmemeval_u2_claude_judge as cj  # noqa: E402

SEED = 0
SAMPLE = 90
ARMS = ['A-5000', 'B-5000', 'C30-5000', 'C100-5000', 'D-5000', 'A-50']
WORK = ROOT/'build/eval/memory/s2'
RUN = WORK/'run'
OUT = WORK/'claude-judge'
PUBLIC = ROOT/'results/eval/memory/s2-claude-judge'


def sheet() -> None:
    if (OUT/'labels.json').exists():
        raise SystemExit('labels exist; the sheet is not rebuilt')
    questions = {q['qid']: q for q in json.loads((WORK/'questions.json').read_text())}
    answers = json.loads((RUN/'answers.json').read_text())
    templates = json.loads(cj.JUDGE.read_text())['templates']
    pairs = [(a, q) for a in ARMS for q in sorted(questions)]
    rng = random.Random(SEED)
    chosen = rng.sample(pairs, SAMPLE)
    key, labels = {}, {}
    lines = ['# S2 grading sheet (blinded)', '', 'Label each answer yes/no by the rubric shown with it. '
             'Failed generations are omitted (scored 0).', '']
    for n, (arm, qid) in enumerate(chosen, 1):
        q = questions[qid]
        item = f'S{n:03d}'
        key[item] = {'arm': arm, 'question_id': qid}
        k = q['type']
        text = cj.answer_text(answers[arm][qid])
        lines += [f'## {item} ({k})', '', f'Rubric: {cj.rubric(templates[k])}', '', f'Question: {q["question"]}', '',
                  f'Correct answer: {q["answer"]}', '', f'Response: {text if text is not None else "[no answer: not graded]"}', '']
        if text is not None:
            labels[item] = None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'sheet.md').write_text('\n'.join(lines)+'\n')
    (OUT/'key.json').write_text(json.dumps(key, indent=1))
    (OUT/'labels-template.json').write_text(json.dumps(labels, indent=1))
    print(f'{len(chosen)} answers -> {OUT/"sheet.md"}')


def score() -> None:
    key = json.loads((OUT/'key.json').read_text())
    labels = json.loads((OUT/'labels.json').read_text())
    qwen = json.loads((RUN/'labels.json').read_text())
    rows = []
    for item, entry in key.items():
        value = labels.get(item, 'failed')
        if value not in ('yes', 'no', 'failed', None):
            raise SystemExit(f'{item}: label must be yes/no')
        rows.append((entry['arm'], int(value == 'yes'), int(qwen[entry['arm']][entry['question_id']]['label'])))
    report = {'judge': 'Claude Opus 5.5 (this session, blinded sheet of a random sample, graded before opening the Qwen labels)',
              'sample': len(rows),
              'agreement': {'percent': float(np.mean([c == w for _, c, w in rows])), 'cohen_kappa': cj.kappa([c for _, c, _ in rows], [w for _, _, w in rows])},
              'per_arm_on_sample': {a: {'n': sum(r[0] == a for r in rows),
                                        'claude': float(np.mean([c for r0, c, _ in rows if r0 == a])) if any(r[0] == a for r in rows) else None,
                                        'qwen': float(np.mean([w for r0, _, w in rows if r0 == a])) if any(r[0] == a for r in rows) else None}
                                    for a in ARMS}}
    (OUT/'score.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    PUBLIC.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    md = ['# S2 secondary judge (Claude, 90-answer sample)', '',
          f'- Agreement with Qwen: {report["agreement"]["percent"]:.3f} (kappa {report["agreement"]["cohen_kappa"]:.2f}), {len(rows)} answers',
          *[f'- {a}: n={v["n"]}, Claude {v["claude"]:.3f}, Qwen {v["qwen"]:.3f}' for a, v in report['per_arm_on_sample'].items() if v['n']], '']
    PUBLIC.with_suffix('.md').write_text('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['sheet', 'score'])
    {'sheet': sheet, 'score': score}[parser.parse_args().command]()
