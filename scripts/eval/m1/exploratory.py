#!/usr/bin/env python3
"""M1 exploratory checks, written after the preregistered report (docs/MEMORY_MISFIRE_M1_PREREG.md) and not used for
its verdicts. Same journals, same bootstrap (user-clustered, paired, 10000, seed 0).

1. Probes whose text is exactly a user message already in the same user's history. J1 reuses first messages across
   users, so a probe taken from another J1 user can repeat this user's own earlier question; the input builder only
   excluded the user's sessions by id. The preregistered comparisons are repeated without those probes.
2. The second judge (Claude vs Qwen) without those items.
3. Replies that refuse a general question by pointing at the memory ("not in the provided memory", "in my memory",
   "knowledge base"): counted per arm, since the misfire question does not count them.
4. Questions where H and P disagree on correctness.

    python3 scripts/eval/m1/exploratory.py   # writes results/eval/memory/m1-exploratory.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
import longmemeval_g1_methods as g1m  # noqa: E402

LOCAL = ROOT / 'build/eval/memory/m1'
RESULTS = ROOT / 'results/eval/memory'
REFUSAL = re.compile(r'(provided memory|in my memory|my current knowledge base|information in the memory'
                     r'|提供された会話の中にはありません|提供された情報には含まれていません)', re.I)


def journal(path):
    return {r['key']: r for r in map(json.loads, path.read_text().splitlines())}


def kappa(pairs):
    agree = float(np.mean([a == b for a, b in pairs]))
    pa, pb = np.mean([a for a, _ in pairs]), np.mean([b for _, b in pairs])
    pe = pa * pb + (1 - pa) * (1 - pb)
    return {'items': len(pairs), 'agreement': agree, 'kappa': float((agree - pe) / (1 - pe)) if pe < 1 else None,
            'claude_yes_qwen_no': sum(1 for a, b in pairs if a and not b), 'claude_no_qwen_yes': sum(1 for a, b in pairs if b and not a)}


def main():
    inputs = json.loads((LOCAL / 'inputs.json').read_text())
    recall = json.loads((LOCAL / 'recall.json').read_text())
    report = json.loads((RESULTS / 'm1-report.json').read_text())
    judges, answers = journal(LOCAL / 'judges.jsonl'), journal(LOCAL / 'answers.jsonl')
    tau = report['tau']
    users = [u for u in inputs['users'] if u['split'] == 'test']
    repeats = set()
    for u in users:
        own = {t['text'].strip() for s in u['sessions'] for t in s['turns'] if t['role'] == 'user'}
        repeats |= {p['id'] for p in u['probes'] if p['text'].strip() in own}
    probes = [p['id'] for u in users for p in u['probes']]
    questions = [f"{u['user']}-q" for u in users]
    label = lambda arm, q: judges[f'{"P" if recall[q]["top"] >= tau else "N"}|{q}' if arm == 'G' else f'{arm}|{q}']['label']
    owner = lambda q: recall[q]['user']

    def diff(a, b, ids):
        d = [label(a, q) - label(b, q) for q in ids]
        return {'difference': float(np.mean(d)), 'ci95': g1m.interval(d, [owner(q) for q in ids])}

    out = {'note': 'exploratory, after the preregistered report; not used for its verdicts',
           'repeated_probes': {lang: sum(1 for q in repeats if recall[q]['lang'] == lang) for lang in ('en', 'ja')},
           'without_repeated_probes': {}}
    kept = [q for q in probes if q not in repeats]
    w = out['without_repeated_probes']
    w['probes'] = len(kept)
    w['misfire'] = {arm: float(np.mean([label(arm, q) for q in kept])) for arm in ('N', 'P', 'H', 'G')}
    w['misfire_ja'] = {arm: float(np.mean([label(arm, q) for q in kept if recall[q]['lang'] == 'ja'])) for arm in ('N', 'P', 'H', 'G')}
    w['P-N misfire'] = diff('P', 'N', kept)
    w['H-P misfire'] = diff('H', 'P', kept)
    w['G-P misfire'] = diff('G', 'P', kept)

    key = json.loads((LOCAL / 'claude-sheet-key.json').read_text())
    claude = json.loads((RESULTS / 'm1-claude-labels.json').read_text())['labels']
    pairs = {i: (int(claude[i]), judges[k]['label']) for i, k in key.items()}
    out['second_judge_without_repeated_probes'] = kappa([pairs[i] for i, k in key.items() if k.split('|', 1)[1] not in repeats])
    out['second_judge_repeated_probes_only'] = kappa([pairs[i] for i, k in key.items() if k.split('|', 1)[1] in repeats])

    refused = lambda arm, q: bool(REFUSAL.search(answers[f'{arm}|{q}']['reply']))
    out['memory_refusals'] = {arm: {'probes_en': sum(refused(arm, q) for q in probes if recall[q]['lang'] == 'en'),
                                    'probes_ja': sum(refused(arm, q) for q in probes if recall[q]['lang'] == 'ja'),
                                    'questions': sum(refused(arm, q) for q in questions),
                                    'probes_refused_and_misfire_by_qwen': sum(refused(arm, q) and label(arm, q) for q in probes)}
                              for arm in ('N', 'P', 'H')}
    out['questions_H_vs_P'] = {'P_right_H_wrong': [q for q in questions if label('P', q) and not label('H', q)],
                               'H_right_P_wrong': [q for q in questions if label('H', q) and not label('P', q)]}
    out['questions_G_vs_P'] = {'P_right_G_wrong': [q for q in questions if label('P', q) and not label('G', q)],
                               'G_right_P_wrong': [q for q in questions if label('G', q) and not label('P', q)]}
    out['answers_cut_test'] = {arm: sum(1 for q in probes + questions if answers[f'{arm}|{q}']['finish'] != 'stop')
                               for arm in ('N', 'P', 'H')}
    (RESULTS / 'm1-exploratory.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
