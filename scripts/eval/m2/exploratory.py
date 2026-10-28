#!/usr/bin/env python3
"""M2 checks that are not in the pre-registration (docs/MEMORY_MISFIRE_M2_RESULTS.md); none of them is used for a
verdict. Reads the M2 build files and Claude's labels, writes results/eval/memory/m2-exploratory.json.

- P's probe replies split by whether the gate (tau*) would pass them: refusals, harmful (with and without a refusal), fits.
- The questions the gate stops, with both labels.
- Claude's labels on P, weighted back from the stratified sheet to all P probe replies, split the same way (rough: about
  50 items per stratum).
- Generation cut at 400 tokens, by arm.

    python3 scripts/eval/m2/exploratory.py
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
import m2_pilot as pilot  # noqa: E402

LOCAL = ROOT / 'build/eval/memory/m2'
RESULTS = ROOT / 'results/eval/memory'


def journal(path):
    return {r['key']: r for r in map(json.loads, path.read_text().splitlines())}


def main():
    tau = json.loads((RESULTS / 'm2-plan.json').read_text())['tau']
    recall = json.loads((LOCAL / 'recall-test.json').read_text())
    answers, judged = journal(LOCAL / 'answers.jsonl'), journal(LOCAL / 'judges.jsonl')
    probes = sorted(q for q, r in recall.items() if r['kind'] == 'probe')
    questions = sorted(q for q, r in recall.items() if r['kind'] == 'question')
    side = lambda q: 'passed' if recall[q]['top'] >= tau else 'stopped'
    refused = lambda q: pilot.refused(answers[f'P|{q}']['reply'])
    kind = lambda q: judged[f'P|{q}']['kind'] or 'none'
    out = {'tau': tau, 'p_by_gate': {}}
    for lang in ('en', 'ja'):
        for s in ('stopped', 'passed'):
            ids = [q for q in probes if recall[q]['lang'] == lang and side(q) == s]
            out['p_by_gate'][f'{lang} {s}'] = {
                'probes': len(ids), 'refusal': sum(map(refused, ids)), 'harmful': sum(kind(q) == 'harmful' for q in ids),
                'harmful_not_refusal': sum(kind(q) == 'harmful' and not refused(q) for q in ids),
                'harm': sum(kind(q) == 'harmful' or refused(q) for q in ids), 'fits': sum(kind(q) == 'fits' for q in ids)}
    out['questions_stopped'] = [{'query': q, 'top': round(recall[q]['top'], 4), 'P': judged[f'P|{q}']['label'],
                                 'N': judged[f'N|{q}']['label']} for q in questions if side(q) == 'stopped']

    # Claude's labels on the stratified sheet, weighted back to all P probe replies.
    claude = json.loads((RESULTS / 'm2-claude-labels.json').read_text())
    key = json.loads((LOCAL / 'claude-sheet-key.json').read_text())
    stratum = lambda q: 'harm' if kind(q) == 'harmful' or refused(q) else 'fits' if kind(q) == 'fits' else 'none'
    size = collections.Counter(map(stratum, probes))
    rows = [(stratum(k.split('|', 1)[1]), claude['kinds'][i], k.split('|', 1)[1]) for i, k in key.items()]
    drawn = collections.Counter(r[0] for r in rows)
    weighted = collections.Counter()
    for s, label, q in rows:
        weighted[f"{recall[q]['lang']} {side(q)} {label}"] += size[s] / drawn[s]
    out['claude_on_p'] = {'strata_all': dict(size), 'strata_drawn': dict(drawn),
                          'qwen_to_claude': {f'{a} -> {b}': n for (a, b), n in sorted(collections.Counter(
                              (kind(k.split('|', 1)[1]), claude['kinds'][i]) for i, k in key.items()).items())},
                          'weighted_counts': {k: round(v, 1) for k, v in sorted(weighted.items())}}
    out['answers_cut'] = {arm: f"{sum(1 for v in answers.values() if v['arm'] == arm and v['finish'] != 'stop')}/"
                               f"{sum(1 for v in answers.values() if v['arm'] == arm)}" for arm in ('N', 'P', 'W')}
    path = RESULTS / 'm2-exploratory.json'
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
