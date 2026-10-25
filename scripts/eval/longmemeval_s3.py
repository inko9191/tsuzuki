#!/usr/bin/env python3
"""S3: with the plugin's facts (setting P: facts extracted and update-checked by Gemma 4 E2B), does recall at 5,000
sessions give E2B better answers than windows only? Same harness and questions as S2 (and the dev questions as a
second set). Stores from scripts/eval/scale/s3_build_run.py; recall by scripts/eval/scale/s3_retrieve.mjs; answers by
E2B with the P template; the U2 Qwen judge. See docs/MEMORY_SCALE_S3_PREREG.md.

    python3 longmemeval_s3.py register   # writes the plan (commit it before run)
    python3 longmemeval_s3.py run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/eval'))
import longmemeval_e1_e2b as e1  # noqa: E402
import longmemeval_g1_methods as g1m  # noqa: E402
import longmemeval_g1_report as g1r  # noqa: E402
import longmemeval_s2 as s2  # noqa: E402
import longmemeval_u1_methods as um  # noqa: E402
import longmemeval_u2_run as u2  # noqa: E402

WORK = ROOT/'build/eval/memory/s3'
LOCAL = WORK/'run'
PLAN = u2.RESULTS/'s3-plan.json'
SETS = {'s2': s2.QUESTIONS, 'dev': ROOT/'build/scale/screen/dev-questions.json'}
RECALLS = [('s2', 5000), ('s2', 50), ('dev', 5000)]
CODE = ['scripts/eval/longmemeval_s3.py', 'scripts/eval/scale/s3_retrieve.mjs', 'scripts/eval/scale/s3_build.mjs',
        'scripts/eval/scale/s3_build_run.py', 'scripts/eval/scale/common.mjs', 'docs/MEMORY_SCALE_S3_PREREG.md']


def arms_of(name):
    return [f'{name}:{kind}-{size}' for n, size in RECALLS if n == name for kind in ('A', 'P')]


def register():
    if PLAN.exists():
        raise SystemExit('plan exists')
    plan = {'created_at': u2.lm.stamp(), 'experiment': 'S3: recall with facts (setting P) at 5,000 sessions, answers by Gemma 4 E2B',
            'authorization': 'user chat 2026-10-24 (facts-at-scale test approved: "OK")',
            'question_sets': {n: {'file': str(f.relative_to(ROOT)), 'sha256': u2.lm.file_hash(f),
                                  'question_ids': sorted(q['qid'] for q in json.loads(f.read_text()))} for n, f in SETS.items()},
            'recalls': RECALLS, 'model_file': e1.MODEL.name,
            'primary': 's2:P-5000 - s2:A-5000 (Qwen judge, 60 questions): low>0 改善; high<0 悪化; else 判定不能',
            'secondary': ['dev:P-5000 - dev:A-5000', 's2:P-50 - s2:A-50', 's2:P-5000 - s2:P-50 (scale loss with facts)',
                          'recall hit per arm', 'facts and links per question', 'per type',
                          'Claude blinded sample of 90 answers (seed 0), graded before the Qwen labels are opened'],
            'code_sha256': {f: u2.lm.file_hash(ROOT/f) for f in CODE}}
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    print('registered')


def complete(db: Path) -> bool:
    if not db.exists():
        return False
    with sqlite3.connect(db) as c:
        return bool(c.execute("SELECT value FROM meta WHERE key = 's3.complete'").fetchone())


def run():
    plan = json.loads(PLAN.read_text())
    if not u2.st.committed(PLAN):
        raise SystemExit('commit the plan first')
    for f, h in plan['code_sha256'].items():
        if u2.lm.file_hash(ROOT/f) != h:
            raise SystemExit('registered code changed: '+f)
    for n, spec in plan['question_sets'].items():
        if u2.lm.file_hash(ROOT/spec['file']) != spec['sha256']:
            raise SystemExit('question set changed: '+n)
    for size in {s for _, s in RECALLS}:
        if not complete(WORK/f'fillers-{size}.sqlite'):
            raise SystemExit(f'filler store {size} not built (scripts/eval/scale/s3_build_run.py)')
    LOCAL.mkdir(parents=True, exist_ok=True)
    ctx = u2.Ctx(LOCAL)
    if any((LOCAL/'inflight').iterdir()):
        raise SystemExit('unknown inflight request')
    small, qwen = e1.E2BServer(ctx), u2.Server(ctx)
    questions = {n: {q['qid']: q for q in json.loads(f.read_text())} for n, f in SETS.items()}
    try:
        small.start(s2.E2B_WINDOW, 1, 'E2B')
        recall = {}
        for name, size in RECALLS:
            out = LOCAL/f'recall-{name}-{size}.json'
            if not out.exists():
                ctx.status('recall', set=name, size=size)
                tmp = out.with_suffix('.tmp')
                subprocess.run([shutil.which('node'), 'scripts/eval/scale/s3_retrieve.mjs', '--questions', str(SETS[name]),
                                '--db', str(WORK/f'fillers-{size}.sqlite'), '--work', str(WORK/f'work-{name}-{size}.sqlite'),
                                '--out', str(tmp), '--chat', 'http://127.0.0.1:8080/v1'], cwd=ROOT, check=True,
                               stderr=open(LOCAL/'recall.log', 'a'))
                tmp.rename(out)
            recall[(name, size)] = {r['qid']: r for r in json.loads(out.read_text())['rows']}
        answers = {}
        for name in SETS:
            for arm in arms_of(name):
                kind, size = arm.split(':')[1].split('-')
                answers[arm] = {}
                for n, qid in enumerate(sorted(questions[name]), 1):
                    q = questions[name][qid]
                    messages = um.messages('', recall[(name, int(size))][qid][kind]['text'], q['question'], q['question_date'])
                    answers[arm][qid] = u2.answer(ctx, small, arm, qid, {'messages': messages})
                    ctx.status('answer', arm=arm, completed=n, total=len(questions[name]))
        small.stop()
        qwen.start(8192, 1, 'judge')
        labels = {}
        for name in SETS:
            prepared = {'questions': {qid: {'question': q['question'], 'answer': q['answer'], 'question_type': q['type']}
                                      for qid, q in questions[name].items()}}
            labels.update(u2.judge(ctx, qwen, prepared, {a: answers[a] for a in arms_of(name)}, arms_of(name)))
        ctx.write('answers.json', answers)
        ctx.write('labels.json', labels)
        report(questions, recall, answers, labels)
        ctx.status('complete')
    except BaseException as error:
        ctx.status('stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        small.stop()
        qwen.stop()


def report(questions, recall, answers, labels):
    s2_rows = {(r['size'], r['qid']): r for r in json.loads(s2.RETRIEVALS.read_text())['rows']}
    same = {size: sum(recall[('s2', size)][q]['A']['ids'] == s2_rows[(size, q)]['A']['ids'] for q in questions['s2']) for size in (5000, 50)}

    def correct(arm):
        name = arm.split(':')[0]
        return {q: labels[arm][q]['label'] for q in sorted(questions[name])}

    def cmp(left, right, mode):
        a, b = correct(left), correct(right)
        ids = sorted(a)
        d = [a[q] - b[q] for q in ids]
        ci = g1m.interval(d, ids)
        v = ('改善' if ci[0] > 0 else '悪化' if ci[1] < 0 else '判定不能') if mode == 'improve' else '傾向参考'
        return {'questions': len(ids), 'difference': float(np.mean(d)), 'ci95': ci, 'verdict': v, 'sign': g1m.sign_test(d)}

    arms = [a for n in SETS for a in arms_of(n)]
    summary = {}
    for arm in arms:
        name, kind_size = arm.split(':')
        kind, size = kind_size.split('-')
        rows = recall[(name, int(size))]
        ids = sorted(questions[name])
        c = correct(arm)
        types = sorted({questions[name][q]['type'] for q in ids})
        summary[arm] = {'correct': float(np.mean(list(c.values()))),
                        'abstain': float(np.mean([g1r.answer_value(answers[arm][q], questions[name][q]['answer'])['abstention'] for q in ids])),
                        'recall_hit': float(np.mean([rows[q][kind]['hit'] for q in ids])),
                        'recall_all_sessions': float(np.mean([rows[q][kind]['allSessions'] for q in ids])),
                        'answer_seconds': float(np.mean([answers[arm][q].get('wall_seconds', 0.) for q in ids])),
                        'by_type': {t: float(np.mean([c[q] for q in ids if questions[name][q]['type'] == t])) for t in types}}
    facts = {f'{n}-{s}': {'evidence_facts': float(np.mean([r['evidenceFacts'] for r in recall[(n, s)].values()])),
                          'evidence_links': float(np.mean([r['evidenceLinks'] for r in recall[(n, s)].values()])),
                          'extract_seconds': float(np.mean([r['extractSeconds'] for r in recall[(n, s)].values()]))} for n, s in RECALLS}
    comparisons = {'primary s2:P-5000 − s2:A-5000': cmp('s2:P-5000', 's2:A-5000', 'improve'),
                   'dev:P-5000 − dev:A-5000': cmp('dev:P-5000', 'dev:A-5000', 'trend'),
                   's2:P-50 − s2:A-50': cmp('s2:P-50', 's2:A-50', 'trend'),
                   's2:P-5000 − s2:P-50': cmp('s2:P-5000', 's2:P-50', 'trend'),
                   's2:A-5000 − s2:A-50': cmp('s2:A-5000', 's2:A-50', 'trend')}
    out = {'arms': summary, 'comparisons': comparisons, 'facts': facts, 'a_matches_s2': same,
           'rows': {a: correct(a) for a in arms}}
    (u2.RESULTS/'s3-report.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    lines = ['# S3：事実あり（設定 P）の想起を 5,000 セッションで（答える役は Gemma 4 E2B）', '',
             f'S2 と同じ 60 問（s2）と、規模試験の dev の 60 問（dev）。窓だけ（A）が S2 の A と同じ想起になった問: '
             f'5,000 で {same[5000]}/60、50 で {same[50]}/60。主判定は Qwen judge。', '',
             '|条件|正答率|棄権率|想起に答え|全根拠|回答秒|', '|---|---:|---:|---:|---:|---:|']
    for a, s in summary.items():
        lines.append(f'|{a}|{s["correct"]:.3f}|{s["abstain"]:.3f}|{s["recall_hit"]:.3f}|{s["recall_all_sessions"]:.3f}|{s["answer_seconds"]:.1f}|')
    lines += ['', '|比較|問数|点差|95%区間|判定|勝/負/同|', '|---|---:|---:|---|---|---|']
    for name, c in comparisons.items():
        s = c['sign']
        lines.append(f'|{name}|{c["questions"]}|{c["difference"]:+.3f}|[{c["ci95"][0]:+.3f}, {c["ci95"][1]:+.3f}]|{c["verdict"]}|{s["wins"]}/{s["losses"]}/{s["ties"]}|')
    lines += ['', '|想起|根拠の会話の事実（平均）|更新のメモ（平均）|抽出と更新の秒（平均）|', '|---|---:|---:|---:|']
    for k, v in facts.items():
        lines.append(f'|{k}|{v["evidence_facts"]:.1f}|{v["evidence_links"]:.1f}|{v["extract_seconds"]:.1f}|')
    types = list(summary['s2:A-5000']['by_type'])
    lines += ['', '|種類|' + '|'.join(types) + '|', '|---|' + '---:|'*len(types)]
    for a in arms:
        lines.append(f'|{a}|' + '|'.join(f'{summary[a]["by_type"][t]:.2f}' for t in types) + '|')
    (u2.RESULTS/'s3-report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'run'])
    {'register': register, 'run': run}[parser.parse_args().command]()
