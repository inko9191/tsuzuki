#!/usr/bin/env python3
"""U2-AT secondary judge (Claude): grade the AT answers before seeing the Qwen AT labels or the AT report.

sheet: from prepared.json and at/answers.json only (never at/labels.json or longmemeval-u2-at-report.*), write a
       sheet with the official per-type rubric, the correct answer and the AT answer. AT is the only arm graded,
       so the arm cannot be hidden (a registered limitation).
score: read the grader's yes/no labels, then report Claude's AT accuracy, agreement with the Qwen AT labels, and
       the primary comparison P - AT on Claude's own labels (P from the earlier blinded U2 Claude grading).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/eval'))
import longmemeval_g1_methods as g1m  # noqa: E402
import longmemeval_u1_methods as um  # noqa: E402
import longmemeval_u2_claude_judge as cj  # noqa: E402
import longmemeval_u2_run as u2  # noqa: E402

SEED = 'longmemeval-u2-at-claude-judge-v1'
RUN = ROOT/'build/eval/memory/longmemeval-u2/run'
AT = ROOT/'build/eval/memory/longmemeval-u2/at'
OUT = ROOT/'build/eval/memory/longmemeval-u2/claude-judge-at'
U2_CLAUDE = ROOT/'build/eval/memory/longmemeval-u2/claude-judge'
PUBLIC = ROOT/'results/eval/memory/longmemeval-u2-at-claude-judge'


def sheet() -> None:
    if (OUT/'labels.json').exists():
        raise SystemExit('labels exist; the sheet is not rebuilt')
    prepared = json.loads((RUN/'prepared.json').read_text())
    answers = json.loads((AT/'answers.json').read_text())
    templates = json.loads(cj.JUDGE.read_text())['templates']
    ids = sorted(prepared['questions'], key=lambda q: hashlib.sha256((SEED+'|'+q).encode()).hexdigest())
    key, labels = {}, {}
    lines = ['# U2-AT grading sheet', '', 'Label each answer yes/no by the rubric shown with the question. '
             'Failed or overflowed generations are omitted (scored 0).', '']
    for n, qid in enumerate(ids, 1):
        row = prepared['questions'][qid]
        item = f'Q{n:02d}'
        key[item] = qid
        k = cj.kind(qid, row)
        lines += [f'## {item} ({k})', '', f'Rubric: {cj.rubric(templates[k])}', '', f'Question: {row["question"]}', '',
                  f'Correct answer: {row["answer"]}', '']
        text = cj.answer_text(answers[qid])
        if text is None:
            lines += ['- Answer: [generation failed: not graded]', '']
            continue
        lines += [f'- Answer: {text}', '']
        labels[item] = None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'sheet.md').write_text('\n'.join(lines)+'\n')
    (OUT/'key.json').write_text(json.dumps(key, indent=1))
    (OUT/'labels-template.json').write_text(json.dumps(labels, indent=1))
    print(f'{len(ids)} questions -> {OUT/"sheet.md"}; fill labels.json (yes/no) from labels-template.json')


def score() -> None:
    key = json.loads((OUT/'key.json').read_text())
    labels = json.loads((OUT/'labels.json').read_text())
    prepared = json.loads((RUN/'prepared.json').read_text())
    ids = sorted(prepared['questions'])
    claude_at = {q: 0 for q in ids}
    for item, qid in key.items():
        value = labels.get(item, 'failed')
        if value not in ('yes', 'no', 'failed', None):
            raise SystemExit(f'{item}: label must be yes/no')
        claude_at[qid] = int(value == 'yes')
    # P on Claude's earlier blinded U2 labels.
    u2key = json.loads((U2_CLAUDE/'key.json').read_text())
    u2labels = json.loads((U2_CLAUDE/'labels.json').read_text())
    claude_p = {}
    for item, entry in u2key.items():
        for letter, arm in entry['letters'].items():
            if arm == 'P':
                claude_p[entry['question_id']] = int(u2labels[item].get(letter) == 'yes')
    qwen_at = json.loads((AT/'labels.json').read_text())
    qwen = {q: int(qwen_at[q]['label']) for q in ids}
    inputs = json.loads((AT/'inputs-public.json').read_text())
    eligible = [q for q in ids if inputs[q]['fits_context']]
    d = [claude_p[q]-claude_at[q] for q in eligible]
    ci = g1m.interval(d, eligible)
    verdict = '上回る' if ci[0] > 0 else '同等以上（非劣性）' if ci[0] >= -u2.MARGIN else '下回る' if ci[1] < 0 else '判定不能'
    pairs = [(claude_at[q], qwen[q]) for q in ids]
    strata = {q: um.stratum(prepared['questions'][q]) for q in ids}
    report = {
        'judge': 'Claude Opus 5.5 (this session; AT answers only, graded before opening the Qwen AT labels)',
        'questions': len(ids), 'AT_eligible': len(eligible),
        'accuracy': {'AT': float(np.mean([claude_at[q] for q in ids])), 'P': float(np.mean([claude_p[q] for q in ids]))},
        'qwen_accuracy_AT': float(np.mean([qwen[q] for q in ids])),
        'primary_rule_on_claude_labels': {'P_minus_AT': {'questions': len(eligible), 'difference': float(np.mean(d)),
                                                         'ci95': ci, 'verdict': verdict, 'sign_test': g1m.sign_test(d)}},
        'agreement_AT': {'percent': float(np.mean([a == b for a, b in pairs])), 'cohen_kappa': cj.kappa(*zip(*pairs))},
        'disagreements': [{'question_id': q, 'stratum': strata[q], 'claude': claude_at[q], 'qwen': qwen[q]}
                          for q in ids if claude_at[q] != qwen[q]],
    }
    (OUT/'score.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    PUBLIC.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    p = report['primary_rule_on_claude_labels']['P_minus_AT']
    md = ['# U2-AT secondary judge (Claude)', '', f'- AT accuracy (Claude): {report["accuracy"]["AT"]:.3f}; Qwen: {report["qwen_accuracy_AT"]:.3f}',
          f'- P accuracy (Claude, earlier blinded grading): {report["accuracy"]["P"]:.3f}',
          f'- P − AT on AT-eligible ({p["questions"]}): {p["difference"]:+.3f} [{p["ci95"][0]:+.3f}, {p["ci95"][1]:+.3f}] → {p["verdict"]}',
          f'- Agreement with Qwen on AT: {report["agreement_AT"]["percent"]:.3f} (kappa {report["agreement_AT"]["cohen_kappa"]:.2f})', '']
    PUBLIC.with_suffix('.md').write_text('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['sheet', 'score'])
    {'sheet': sheet, 'score': score}[parser.parse_args().command]()
