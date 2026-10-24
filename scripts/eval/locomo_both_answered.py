#!/usr/bin/env python3
"""CPU-only conditional lexical score summaries from immutable saved rows."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import locomo_memory_reporting as inference

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'build/eval/memory/answer/production/scored-rows.json'
ORIGINAL = ROOT/'results/eval/memory/answer-report.json'
PUBLIC = ROOT/'results/eval/memory/answer-both-answered.json'
PLAN = ROOT/'results/eval/memory/answer-both-answered-plan.json'
DOC = ROOT/'docs/MEMORY_BOTH_ANSWERED_PREREG.md'
PRIVATE = ROOT/'build/eval/memory/answer/both-answered'
STRATA = ('multiple_evidence', 'overall', 'single_fact', 'temporal')
ARMS = ('A', 'B1536', 'C-t1', 'C-p2', 'C-final', 'B8192', 'C8192-final')


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def register():
    if PLAN.exists():
        raise RuntimeError('analysis plan already exists')
    save(PLAN, {'scope': 'posthoc conditional saved lexical scores; CPU only',
        'inputs_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in (SOURCE, ORIGINAL)},
        'code_sha256': file_hash(Path(__file__)), 'document_sha256': file_hash(DOC),
        'masks': ['both_valid_nonempty_nonabstaining', 'both_nonabstaining_sensitivity'],
        'conditions': ARMS, 'resamples': 10000, 'seed': 0, 'generation_executed': False})


def align(rows):
    if set(rows) != set(ARMS):
        raise ValueError('missing/extra saved condition')
    ids = [r['id'] for r in rows['A']]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate question IDs')
    output = {}
    for arm, part in rows.items():
        by_id = {r['id']: r for r in part}
        if len(by_id) != len(part) or set(by_id) != set(ids):
            raise ValueError('all conditions require identical unique question IDs')
        output[arm] = [by_id[q] for q in ids]
        if any((r['type'], r['category']) != (a['type'], a['category'])
               for r, a in zip(output[arm], rows['A'])):
            raise ValueError('strata differ between saved conditions')
    return output


def answered(row, sensitivity=False):
    return not row['abstention'] and (sensitivity or not row['empty_answer'] and not row['failure'])


def collect(rows):
    rows = align(rows)
    summaries, private = {}, {}
    for mask, sensitivity in (('both_valid_nonempty_nonabstaining', False),
                              ('both_nonabstaining_sensitivity', True)):
        keep = [answered(a, sensitivity) and answered(c, sensitivity)
                for a, c in zip(rows['A'], rows['C-final'])]
        private[mask] = [a['id'] for a, yes in zip(rows['A'], keep) if yes]
        retained = {arm: [r for r, yes in zip(part, keep) if yes] for arm, part in rows.items()}
        tables = {}
        for kind in STRATA:
            subsets = {arm: [r for r in part if kind == 'overall' or r['type'] == kind]
                       for arm, part in retained.items()}
            tables[kind] = {arm: {'A': inference.paired(part, subsets['A'], 'f1')}
                            for arm, part in subsets.items()}
        summaries[mask] = {'retained': sum(keep), 'excluded': len(keep)-sum(keep),
            'retained_type_counts': dict(Counter(r['type'] for r in retained['A'])),
            'excluded_flags_per_condition': {arm: {flag: sum(bool(r[flag]) for r, yes in zip(rows[arm], keep) if not yes)
                for flag in ('abstention', 'failure', 'empty_answer')} for arm in ('A', 'C-final')},
            'tables': tables,
            'equivalence_checks': {k: (s['C-final']['A']['conversation_ci95'] is not None and
                s['C-final']['A']['conversation_ci95'][0] >= -.03 and
                s['C-final']['A']['conversation_ci95'][1] <= .03) for k, s in tables.items()}}
    return summaries, private


def run():
    tracked = subprocess.check_output(['git','show',f'HEAD:{PLAN.relative_to(ROOT)}'], cwd=ROOT)
    p = json.loads(tracked)
    if p != json.loads(PLAN.read_text()) or p['code_sha256'] != file_hash(Path(__file__)) or p['document_sha256'] != file_hash(DOC):
        raise RuntimeError('uncommitted or changed analysis registration')
    for name, digest in p['inputs_sha256'].items():
        if file_hash(ROOT/name) != digest:
            raise RuntimeError('immutable saved input changed')
    if PUBLIC.exists():
        raise RuntimeError('refuse to overwrite supplemental results')
    rows = json.loads(SOURCE.read_text())
    if len(rows['A']) != 302:
        raise RuntimeError('expected the completed302 saved rows')
    summaries, private = collect(rows)
    save(PRIVATE/'masks.json', private)
    record = {'scope': p['scope'], 'source_plan_sha256': file_hash(PLAN), 'source_hashes': p['inputs_sha256'],
        'masks': summaries, 'groups': sorted({r['id'].split('/')[0] for r in rows['A']}),
        'original_result_changed': False, 'new_generation_executed': False,
        'note': 'Posthoc outcome-conditioned subset; neither causal abstention adjustment nor full302 performance. '
                'All other conditions shown on the same A/C-final subset, not their own nonabstention mask.'}
    save(PUBLIC, record)
    lines = ['# 両方が回答した質問の語彙F1（追加CPU集計）', '', record['note'], '',
        '元302問全体のF1・凍結設定は変更しない。棄権・失敗の出力に条件付けた探索的部分集合。', '']
    for name, d in summaries.items():
        lines += [f'## {name}', '', f'残った質問{d["retained"]}件、除外{d["excluded"]}件。'
            '除外理由のフラグ件数は重複するため足し合わせない。',
            f'除外フラグ：{d["excluded_flags_per_condition"]}', '']
        for kind in STRATA:
            lines += [f'### {kind}', '']
            lines += inference.table(d['tables'][kind], 'A', record['groups']) + ['']
            lines.append(f'C-final−Aの区間が±.03内：{d["equivalence_checks"][kind]}。')
    lines += ['', '確認：同じ保存行の共通mask上の語彙F1、会話別差・区間・厳密符号検定。',
        '未確認：差を生む棄権の因果寄与、意味的正答、本番全体への補正値、未知会話・日本語一般化。',
        '主張範囲：同じ10会話の元質問の、A/C-finalが両方回答した固定部分集合。', '']
    PUBLIC.with_suffix('.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register','run'))
    command=parser.parse_args().command
    {'register': register, 'run': run}[command]()
