#!/usr/bin/env python3
"""E1 secondary judge (Claude, blinded): grade E2B-P and E2B-AT before opening the Qwen labels or the E1 report.

sheet: from prepared.json and e1/answers.json only, write a sheet where each question's two answers are labeled
       A/B in a seeded random order (official per-type rubric); the arm key goes to a file the grader does not open
       before grading.
score: read the grader's yes/no labels, unblind, report accuracy, the registered comparisons on Claude's labels
       (Qwen-AT from the U2-AT Claude grading) and agreement with the Qwen judge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/eval'))
import longmemeval_g1_methods as g1m  # noqa: E402
import longmemeval_u2_claude_judge as cj  # noqa: E402
import longmemeval_u2_run as u2  # noqa: E402

SEED = 'longmemeval-e1-claude-judge-v1'
ARMS = ('E2B-P', 'E2B-AT')
RUN = ROOT/'build/eval/memory/longmemeval-u2/run'
E1 = ROOT/'build/eval/memory/longmemeval-u2/e1'
OUT = ROOT/'build/eval/memory/longmemeval-u2/claude-judge-e1'
AT_CLAUDE = ROOT/'build/eval/memory/longmemeval-u2/claude-judge-at'
PUBLIC = ROOT/'results/eval/memory/longmemeval-e1-claude-judge'


def sheet() -> None:
    if (OUT/'labels.json').exists():
        raise SystemExit('labels exist; the sheet is not rebuilt')
    prepared = json.loads((RUN/'prepared.json').read_text())
    answers = json.loads((E1/'answers.json').read_text())
    templates = json.loads(cj.JUDGE.read_text())['templates']
    ids = sorted(prepared['questions'], key=lambda q: hashlib.sha256((SEED+'|'+q).encode()).hexdigest())
    rng = random.Random(SEED)
    key, labels = {}, {}
    lines = ['# E1 grading sheet (blinded)', '', 'Label each answer yes/no by the rubric shown with the question. '
             'Failed generations and inputs that did not fit are omitted (scored 0).', '']
    for n, qid in enumerate(ids, 1):
        row = prepared['questions'][qid]
        arms = list(ARMS)
        rng.shuffle(arms)
        item = f'Q{n:02d}'
        key[item] = {'question_id': qid, 'letters': dict(zip('AB', arms))}
        k = cj.kind(qid, row)
        lines += [f'## {item} ({k})', '', f'Rubric: {cj.rubric(templates[k])}', '', f'Question: {row["question"]}', '',
                  f'Correct answer: {row["answer"]}', '']
        labels[item] = {}
        for letter, arm in zip('AB', arms):
            text = cj.answer_text(answers[arm][qid])
            if text is None:
                lines.append(f'- {letter}: [no answer: not graded]')
                continue
            lines.append(f'- {letter}: {text}')
            labels[item][letter] = None
        lines.append('')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'sheet.md').write_text('\n'.join(lines)+'\n')
    (OUT/'key.json').write_text(json.dumps(key, indent=1))
    (OUT/'labels-template.json').write_text(json.dumps(labels, indent=1))
    print(f'{len(ids)} questions -> {OUT/"sheet.md"}')


def score() -> None:
    key = json.loads((OUT/'key.json').read_text())
    labels = json.loads((OUT/'labels.json').read_text())
    prepared = json.loads((RUN/'prepared.json').read_text())
    ids = sorted(prepared['questions'])
    claude = {a: {q: 0 for q in ids} for a in ARMS}
    for item, entry in key.items():
        for letter, arm in entry['letters'].items():
            value = labels[item].get(letter, 'failed')
            if value not in ('yes', 'no', 'failed', None):
                raise SystemExit(f'{item}{letter}: label must be yes/no')
            claude[arm][entry['question_id']] = int(value == 'yes')
    at_key = json.loads((AT_CLAUDE/'key.json').read_text())
    at_labels = json.loads((AT_CLAUDE/'labels.json').read_text())
    claude['Qwen-AT'] = {qid: int(at_labels.get(item) == 'yes') for item, qid in at_key.items()}
    qwen = json.loads((E1/'labels.json').read_text())
    inputs = json.loads((E1/'inputs-public.json').read_text())
    eligible = [q for q in ids if inputs['E2B-AT'][q]['fits_context']]

    def cmp(left, right, chosen):
        d = [claude[left][q]-claude[right][q] for q in chosen]
        ci = g1m.interval(d, chosen)
        v = '上回る' if ci[0] > 0 else '同等以上（非劣性）' if ci[0] >= -u2.MARGIN else '下回る' if ci[1] < 0 else '判定不能'
        return {'questions': len(chosen), 'difference': float(np.mean(d)), 'ci95': ci, 'verdict': v, 'sign_test': g1m.sign_test(d)}
    pairs = [(claude[a][q], int(qwen[a][q]['label'])) for a in ARMS for q in ids]
    report = {'judge': 'Claude Opus 5.5 (this session, blinded A/B sheet, graded before opening the Qwen E1 labels)',
              'accuracy': {a: float(np.mean([claude[a][q] for q in ids])) for a in (*ARMS, 'Qwen-AT')},
              'qwen_accuracy': {a: float(np.mean([int(qwen[a][q]['label']) for q in ids])) for a in ARMS},
              'primary_rule_on_claude_labels': {'E2B_P_minus_Qwen_AT': cmp('E2B-P', 'Qwen-AT', ids),
                                                'E2B_P_minus_E2B_AT': cmp('E2B-P', 'E2B-AT', eligible)},
              'agreement': {'percent': float(np.mean([a == b for a, b in pairs])), 'cohen_kappa': cj.kappa(*zip(*pairs))}}
    (OUT/'score.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    PUBLIC.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    p = report['primary_rule_on_claude_labels']
    md = ['# E1 secondary judge (Claude)', '',
          *[f'- {a}: Claude {report["accuracy"][a]:.3f}' + (f', Qwen {report["qwen_accuracy"][a]:.3f}' if a in ARMS else '') for a in (*ARMS, 'Qwen-AT')],
          *[f'- {k}: {c["difference"]:+.3f} [{c["ci95"][0]:+.3f}, {c["ci95"][1]:+.3f}] → {c["verdict"]} ({c["questions"]} q)' for k, c in p.items()],
          f'- Agreement with Qwen: {report["agreement"]["percent"]:.3f} (kappa {report["agreement"]["cohen_kappa"]:.2f})', '']
    PUBLIC.with_suffix('.md').write_text('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['sheet', 'score'])
    {'sheet': sheet, 'score': score}[parser.parse_args().command]()
