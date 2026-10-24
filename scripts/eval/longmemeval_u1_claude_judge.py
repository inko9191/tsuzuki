#!/usr/bin/env python3
"""U1 secondary judge (Claude, blinded): anonymized grading sheet, then scoring and agreement with the Qwen judge.

sheet: from the run's prepared.json/answers.json only (never judges.json or the report), write a sheet where each
       question's answers are labeled A-D in a seeded random order, with the official per-type rubric; the arm key
       goes to a separate file the grader does not open before grading.
score: read the grader's yes/no labels, unblind, and report accuracy, the registered comparisons and agreement.
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
import locomo_answer_methods as am  # noqa: E402
import longmemeval_g1_methods as g1m  # noqa: E402
import longmemeval_u1_methods as um  # noqa: E402

SEED = 'longmemeval-u1-claude-judge-v1'
JUDGE = ROOT/'results/eval/memory/longmemeval-g1-judge.json'
LETTERS = 'ABCD'


def kind(qid, row):
    return 'abstention' if qid.endswith('_abs') else row['question_type']


def rubric(template):
    return template.split('\n\nQuestion:')[0].strip()


def answer_text(reply):
    if reply.get('failure_kind'):
        return None
    parsed, failure = am.parse(reply.get('response') or {})
    return None if failure else parsed['answer']


def sheet(run: Path, out: Path):
    if (out/'labels.json').exists():
        raise SystemExit('labels exist; the sheet is not rebuilt')
    prepared = json.loads((run/'prepared.json').read_text())
    answers = json.loads((run/'answers.json').read_text())
    templates = json.loads(JUDGE.read_text())['templates']
    ids = sorted(prepared['questions'], key=lambda q: hashlib.sha256((SEED+'|'+q).encode()).hexdigest())
    rng = random.Random(SEED)
    key, lines, labels = {}, ['# U1 grading sheet (blinded)', '',
                              'Label each answer yes/no by the rubric shown with the question. '
                              'Failed generations are omitted (scored 0 for every judge).', ''], {}
    for n, qid in enumerate(ids, 1):
        row = prepared['questions'][qid]
        arms = list(um.ARMS)
        rng.shuffle(arms)
        item = f'Q{n:02d}'
        key[item] = {'question_id': qid, 'letters': dict(zip(LETTERS, arms))}
        lines += [f'## {item} ({kind(qid, row)})', '', f'Rubric: {rubric(templates[kind(qid, row)])}', '',
                  f'Question: {row["question"]}', '', f'Correct answer: {row["answer"]}', '']
        labels[item] = {}
        for letter, arm in zip(LETTERS, arms):
            text = answer_text(answers[arm][qid])
            if text is None:
                lines.append(f'- {letter}: [generation failed: not graded]')
                continue
            lines.append(f'- {letter}: {text}')
            labels[item][letter] = None
        lines.append('')
    out.mkdir(parents=True, exist_ok=True)
    (out/'sheet.md').write_text('\n'.join(lines)+'\n')
    (out/'key.json').write_text(json.dumps(key, ensure_ascii=False, indent=1))
    (out/'labels-template.json').write_text(json.dumps(labels, indent=1))
    print(f'{len(ids)} questions -> {out/"sheet.md"}; fill labels.json (yes/no) from labels-template.json')


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + (1-np.mean(a)) * (1-np.mean(b)))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(run: Path, out: Path, public: Path | None):
    key = json.loads((out/'key.json').read_text())
    labels = json.loads((out/'labels.json').read_text())
    qwen = json.loads((run/'judges.json').read_text())
    prepared = json.loads((run/'prepared.json').read_text())
    claude = {a: {} for a in um.ARMS}
    for item, entry in key.items():
        qid = entry['question_id']
        for letter, arm in entry['letters'].items():
            value = labels[item].get(letter, 'failed')
            if value not in ('yes', 'no', 'failed'):
                raise SystemExit(f'{item}{letter}: label must be yes/no')
            claude[arm][qid] = int(value == 'yes')
    ids = sorted(prepared['questions'])
    strata = {q: um.stratum(prepared['questions'][q]) for q in ids}

    def compare(labels_, left, right):
        out_ = {}
        for name, keep in (('overall', lambda q: True),
                           ('focus_update_temporal', lambda q: strata[q] in ('knowledge-update', 'temporal-reasoning'))):
            chosen = [q for q in ids if keep(q)]
            diffs = [labels_[left][q]-labels_[right][q] for q in chosen]
            ci = g1m.interval(diffs, chosen)
            out_[name] = {'questions': len(chosen), 'difference': float(np.mean(diffs)), 'ci95': ci,
                          'verdict': ('改善' if ci[0] > 0 else '悪化' if ci[1] < 0 else '判定不能') if name == 'overall' else '傾向参考',
                          'sign_test': g1m.sign_test(diffs)}
        return out_

    qlabels = {a: {q: qwen[a][q]['label'] for q in ids} for a in um.ARMS}
    pairs = [(claude[a][q], qlabels[a][q]) for a in um.ARMS for q in ids]
    disagreements = [{'question_id': q, 'arm': a, 'stratum': strata[q], 'claude': claude[a][q], 'qwen': qlabels[a][q]}
                     for a in um.ARMS for q in ids if claude[a][q] != qlabels[a][q]]
    report = {
        'judge': 'Claude Opus 5.5 (this session, blinded sheet; the experiment designer)', 'questions': len(ids),
        'accuracy': {a: float(np.mean([claude[a][q] for q in ids])) for a in um.ARMS},
        'qwen_accuracy': {a: float(np.mean([qlabels[a][q] for q in ids])) for a in um.ARMS},
        'primary_rule_on_claude_labels': {'V_minus_C': compare(claude, 'V', 'C-final-norw'),
                                          'S_minus_C': compare(claude, 'S', 'C-final-norw'),
                                          'T_minus_S': compare(claude, 'T', 'S'), 'V_minus_T': compare(claude, 'V', 'T')},
        'agreement': {'percent': float(np.mean([a == b for a, b in pairs])), 'cohen_kappa': kappa(*zip(*pairs)),
                      'by_arm': {a: float(np.mean([claude[a][q] == qlabels[a][q] for q in ids])) for a in um.ARMS}},
        'disagreements': disagreements,
    }
    (out/'score.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    if public:
        public.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
        public.with_suffix('.md').write_text(markdown(report))
    print(markdown(report))


def markdown(r):
    lines = ['# U1 副次の採点（Claude、条件名を伏せたシート）', '',
             f'{r["questions"]}問×4条件。主判定は Qwen のまま。採点者は実験の設計者なので、条件名を伏せて採点した。', '',
             '|条件|Claude 正答率|Qwen 正答率|一致率|', '|---|---:|---:|---:|']
    for a in um.ARMS:
        lines.append(f'|{a}|{r["accuracy"][a]:.3f}|{r["qwen_accuracy"][a]:.3f}|{r["agreement"]["by_arm"][a]:.3f}|')
    ag = r['agreement']
    lines += ['', f'全体の一致率 {ag["percent"]:.3f}、Cohen の κ {ag["cohen_kappa"]:.3f}、不一致 {len(r["disagreements"])} 件。', '',
              '|比較（Claude の採点）|範囲|点差|95%区間|判定|勝/負/同|', '|---|---|---:|---|---|---|']
    for name, part in r['primary_rule_on_claude_labels'].items():
        for scope, v in part.items():
            s = v['sign_test']
            lines.append(f'|{name}|{scope}|{v["difference"]:+.3f}|[{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]|{v["verdict"]}|'
                         f'{s["wins"]}/{s["losses"]}/{s["ties"]}|')
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['sheet', 'score'])
    parser.add_argument('--run', default=str(ROOT/'build/eval/memory/longmemeval-u1/run'))
    parser.add_argument('--out', default=str(ROOT/'build/eval/memory/longmemeval-u1/claude-judge'))
    parser.add_argument('--public', default=str(ROOT/'results/eval/memory/longmemeval-u1-claude-judge'))
    args = parser.parse_args()
    if args.command == 'sheet':
        sheet(Path(args.run), Path(args.out))
    else:
        score(Path(args.run), Path(args.out), Path(args.public) if args.public else None)
