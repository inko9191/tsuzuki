#!/usr/bin/env python3
"""J1-Q: the J1 Japanese test with Qwen3.8 27B doing everything (memory building and reading), the follow-up
that J1's rule triggered (E2B-P - E2B-AT was 上回る). Same 40 questions and histories as J1.

Qwen-P  = the U2 memory pipeline (Qwen facts and update links, EmbeddingGemma 300m, setting P) read by Qwen.
Qwen-AT = the whole history with the same template, read by Qwen (131072 window).
Judged by the U2 Qwen judge. See docs/MEMORY_JA_J1Q_PREREG.md.

    python3 longmemeval_j1q.py register
    python3 longmemeval_j1q.py run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/eval'))
import longmemeval_g1_methods as g1m  # noqa: E402
import longmemeval_g1_report as g1r  # noqa: E402
import longmemeval_u1_methods as um  # noqa: E402
import longmemeval_u2_run as u2  # noqa: E402

DATASET = ROOT/'build/eval/memory/j1/j1_dataset.json'
LOCAL = ROOT/'build/eval/memory/j1/run-qwen'
PLAN = u2.RESULTS/'j1q-plan.json'
J1_REPORT = u2.RESULTS/'j1-report.json'
CODE = ['scripts/eval/longmemeval_j1q.py', 'docs/MEMORY_JA_J1Q_PREREG.md']
ARMS = ('Qwen-P', 'Qwen-AT')


def register():
    if PLAN.exists():
        raise SystemExit('plan exists')
    j1 = json.loads((u2.RESULTS/'j1-plan.json').read_text())
    if u2.lm.file_hash(DATASET) != j1['dataset_sha256']:
        raise SystemExit('dataset differs from J1')
    report = json.loads(J1_REPORT.read_text())
    if not report['qwen_follow_up']:
        raise SystemExit('J1 did not trigger the follow-up')
    plan = {'created_at': u2.lm.stamp(), 'experiment': 'J1-Q: the J1 Japanese test, all on Qwen3.8 27B',
            'authorization': 'user chat 2026-10-24 (run Qwen if E2B shows the effect or is at least equivalent); J1 verdict '
                             + report['primary_E2B_P_minus_E2B_AT']['verdict'],
            'question_ids': j1['question_ids'], 'dataset_sha256': j1['dataset_sha256'],
            'j1_report_sha256': u2.lm.file_hash(J1_REPORT),
            'primary': f'Qwen-P - Qwen-AT on Qwen-AT-eligible: low>0 上回る; else low>=-{u2.MARGIN} 同等以上; high<0 下回る; else 判定不能',
            'secondary': ['Qwen-P - E2B-P (the big model on both sides of the memory)', 'Qwen-AT - E2B-AT', 'per type',
                          'P memory coverage', 'answer seconds'],
            'code_sha256': {f: u2.lm.file_hash(ROOT/f) for f in CODE}}
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    print('registered', len(plan['question_ids']))


def run():
    plan = json.loads(PLAN.read_text())
    if not u2.st.committed(PLAN):
        raise SystemExit('commit the plan first')
    for f, h in plan['code_sha256'].items():
        if u2.lm.file_hash(ROOT/f) != h:
            raise SystemExit('registered code changed: '+f)
    if u2.lm.file_hash(DATASET) != plan['dataset_sha256']:
        raise SystemExit('dataset changed')
    u2.DATA = DATASET
    ids = plan['question_ids']
    ctx = u2.Ctx(LOCAL)
    if any((LOCAL/'inflight').iterdir()):
        raise SystemExit('unknown inflight request')
    qwen = u2.Server(ctx)
    try:
        prepared = u2.prepare(ctx, ids, qwen)
        facts = json.loads((LOCAL/'facts.json').read_text()) if (LOCAL/'facts.json').exists() else u2.ingest(ctx, prepared, qwen)
        units, vectors, query = u2.embedding(ctx, prepared, facts, qwen)
        tables, links = u2.updates(ctx, prepared, facts, units, vectors, qwen)
        raw = u2.orders(ctx, prepared, units, vectors, query, qwen)
        memory = u2.assemble(ctx, prepared, raw, tables, links, qwen)
        answers = {a: {} for a in ARMS}
        inputs = {a: {} for a in ARMS}
        for n, qid in enumerate(prepared['questions'], 1):  # memory answers on the 8192 'answer' server from assemble
            inputs['Qwen-P'][qid] = memory['P'][qid]
            answers['Qwen-P'][qid] = u2.answer(ctx, qwen, 'Qwen-P', qid, memory['P'][qid])
            ctx.status('answer', arm='Qwen-P', completed=n, total=len(ids))
        qwen.start(131072, 1, 'AT')
        for n, (qid, row) in enumerate(prepared['questions'].items(), 1):
            chat = um.messages(um.render(prepared['turns'][qid]), '', row['question'], row['question_date'])
            tokens = u2.Counter_.chat(chat)
            at = {'messages': chat, 'input_tokens': tokens, 'fits_context': u2.fits(tokens, 256, 131072)}
            inputs['Qwen-AT'][qid] = at
            answers['Qwen-AT'][qid] = (u2.answer(ctx, qwen, 'Qwen-AT', qid, at) if at['fits_context']
                                       else {'failure_kind': 'input_overflow', 'response': {}, 'wall_seconds': 0.})
            ctx.status('answer', arm='Qwen-AT', completed=n, total=len(ids))
        qwen.start(8192, 1, 'judge')
        labels = u2.judge(ctx, qwen, prepared, answers, ARMS)
        ctx.write('answers.json', answers)
        ctx.write('labels.json', labels)
        ctx.write('inputs-public.json', {a: {q: {k: v for k, v in x.items() if k != 'messages'} for q, x in d.items()} for a, d in inputs.items()})
        report(prepared, inputs, answers, labels)
        ctx.status('complete')
        ctx.log('complete')
    except BaseException as error:
        ctx.status('stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        qwen.stop()


def kind(qid, row):
    return 'abstention' if qid.endswith('_abs') else row['question_type']


def report(prepared, inputs, answers, labels):
    ids = sorted(prepared['questions'])
    e2b = json.loads(J1_REPORT.read_text())['rows']
    rows = {a: {q: labels[a][q]['label'] for q in ids} for a in ARMS}
    rows.update({a: {q: e2b[a][q] for q in ids} for a in ('E2B-P', 'E2B-AT')})
    eligible = [q for q in ids if inputs['Qwen-AT'][q]['fits_context']]

    def cmp(left, right, chosen, mode):
        d = [rows[left][q]-rows[right][q] for q in chosen]
        ci = g1m.interval(d, chosen)
        if mode == 'vs_full':
            v = '上回る' if ci[0] > 0 else '同等以上（非劣性）' if ci[0] >= -u2.MARGIN else '下回る' if ci[1] < 0 else '判定不能'
        else:
            v = '傾向参考'
        return {'questions': len(chosen), 'difference': float(np.mean(d)), 'ci95': ci, 'verdict': v, 'sign': g1m.sign_test(d)}

    def summary(arm):
        values = {q: g1r.answer_value(answers[arm][q], prepared['questions'][q]['answer']) for q in ids}
        kinds = sorted({kind(q, prepared['questions'][q]) for q in ids})
        return {'correct': float(np.mean(list(rows[arm].values()))),
                'abstain': float(np.mean([v['abstention'] for v in values.values()])),
                'input_tokens': float(np.mean([inputs[arm][q]['input_tokens'] for q in ids])),
                'answer_seconds': float(np.mean([answers[arm][q].get('wall_seconds', 0.) for q in ids])),
                'by_type': {k: float(np.mean([rows[arm][q] for q in ids if kind(q, prepared['questions'][q]) == k])) for k in kinds}}
    out = {'questions': len(ids), 'Qwen_AT_eligible': len(eligible), 'arms': {a: summary(a) for a in ARMS},
           'P_coverage': float(np.mean([inputs['Qwen-P'][q]['coverage'] for q in ids])),
           'comparisons': {'primary_Qwen_P_minus_Qwen_AT': cmp('Qwen-P', 'Qwen-AT', eligible, 'vs_full'),
                           'Qwen_P_minus_E2B_P': cmp('Qwen-P', 'E2B-P', ids, 'trend'),
                           'Qwen_AT_minus_E2B_AT': cmp('Qwen-AT', 'E2B-AT', ids, 'trend')},
           'rows': {a: rows[a] for a in ARMS}}
    (u2.RESULTS/'j1q-report.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    lines = ['# J1-Q：日本語のテストを Qwen3.8 27B で（記憶づくりも答える役も）', '',
             f'J1 と同じ40問・同じ履歴。Qwen-AT が窓に入った問 {len(eligible)}/40。主判定は Qwen judge。'
             f' P の記憶が答えの会話を全部含んだ割合 {out["P_coverage"]:.2f}。', '',
             '|条件|正答率|棄権率|入力トークン|回答秒|', '|---|---:|---:|---:|---:|']
    for a, s in out['arms'].items():
        lines.append(f'|{a}|{s["correct"]:.3f}|{s["abstain"]:.3f}|{s["input_tokens"]:.0f}|{s["answer_seconds"]:.1f}|')
    lines += ['', '|比較|問数|点差|95%区間|判定|勝/負/同|', '|---|---:|---:|---|---|---|']
    for name, c in out['comparisons'].items():
        s = c['sign']
        lines.append(f'|{name}|{c["questions"]}|{c["difference"]:+.3f}|[{c["ci95"][0]:+.3f}, {c["ci95"][1]:+.3f}]|{c["verdict"]}|{s["wins"]}/{s["losses"]}/{s["ties"]}|')
    lines += ['', '|種類|' + '|'.join(out['arms']['Qwen-P']['by_type']) + '|', '|---|' + '---:|'*len(out['arms']['Qwen-P']['by_type'])]
    for a in ARMS:
        lines.append(f'|{a}|' + '|'.join(f'{v:.2f}' for v in out['arms'][a]['by_type'].values()) + '|')
    (u2.RESULTS/'j1q-report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'run'])
    {'register': register, 'run': run}[parser.parse_args().command]()
