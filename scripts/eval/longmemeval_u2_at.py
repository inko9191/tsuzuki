#!/usr/bin/env python3
"""U2 follow-up: full context with the same date-aware template as P (arm AT), on the same 40 questions.

AT = the whole history in date order with a Date line per session, the question date and the T/V system
rules (um.messages), no retrieved excerpts. P and AT then differ only in what is passed: selected memory
versus everything. Answers and judging as in U2 (one at a time, Qwen judge, 131072 window for AT).
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

RUN = u2.WORK/'run'
LOCAL = u2.WORK/'at'
PLAN = u2.RESULTS/'longmemeval-u2-at-plan.json'
CODE = ['scripts/eval/longmemeval_u2_at.py', 'docs/MEMORY_LONGMEMEVAL_U2_AT_PREREG.md']


def register():
    if PLAN.exists():
        raise SystemExit('plan exists')
    u2plan = json.loads(u2.PLAN.read_text())
    plan = {'created_at': u2.lm.stamp(), 'experiment': 'U2-AT: full context with the P template',
            'authorization': 'user chat 2026-10-23 (step 1 approved)', 'question_ids': u2plan['question_ids'],
            'u2_report_sha256': u2.lm.file_hash(u2.RESULTS/'longmemeval-u2-report.json'),
            'primary': f'P - AT on AT-eligible: low>0 上回る; else low>=-{u2.MARGIN} 同等以上; high<0 下回る; else 判定不能',
            'secondary': ['AT - A (date and rules on full context)', 'C512 - AT', 'per stratum', 'abstention', 'answer seconds'],
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
    prepared = json.loads((RUN/'prepared.json').read_text())
    if list(prepared['questions']) != list(prepared['questions']) or set(prepared['questions']) != set(plan['question_ids']):
        raise SystemExit('U2 question set differs')
    ctx = u2.Ctx(LOCAL)
    if any((LOCAL/'inflight').iterdir()):
        raise SystemExit('unknown inflight request')
    server = u2.Server(ctx)
    try:
        server.start(131072, 1, 'AT')
        inputs, answers = {}, {}
        for n, (qid, row) in enumerate(prepared['questions'].items(), 1):
            ts = prepared['turns'][qid]
            chat = um.messages(um.render(ts), '', row['question'], row['question_date'])
            tokens = u2.Counter_.chat(chat)
            inputs[qid] = {'messages': chat, 'input_tokens': tokens, 'fits_context': u2.fits(tokens, 256, 131072)}
            if inputs[qid]['fits_context']:
                answers[qid] = u2.answer(ctx, server, 'AT', qid, inputs[qid])
            else:
                answers[qid] = {'failure_kind': 'input_overflow', 'response': {}, 'wall_seconds': 0.}
            ctx.status('answer', arm='AT', completed=n, total=len(prepared['questions']))
        server.start(8192, 1, 'judge')
        labels = u2.judge(ctx, server, prepared, {'AT': answers}, ('AT',))['AT']
        ctx.write('answers.json', answers); ctx.write('labels.json', labels)
        ctx.write('inputs-public.json', {q: {'input_tokens': v['input_tokens'], 'fits_context': v['fits_context']} for q, v in inputs.items()})
        report(prepared, inputs, answers, labels)
        ctx.status('complete')
        ctx.log('complete')
    except BaseException as error:
        ctx.status('stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        server.stop()


def report(prepared, inputs, answers, labels):
    u2r = json.loads((u2.RESULTS/'longmemeval-u2-report.json').read_text())['rows']
    ids = sorted(prepared['questions'])
    rows = {a: {q: u2r[a][q]['correct'] for q in ids} for a in ('C512', 'P', 'A')}
    rows['AT'] = {q: labels[q]['label'] for q in ids}
    eligible = [q for q in ids if inputs[q]['fits_context']]

    def cmp(left, right, chosen, mode):
        d = [rows[left][q]-rows[right][q] for q in chosen]
        ci = g1m.interval(d, chosen)
        if mode == 'vs_full':
            v = '上回る' if ci[0] > 0 else '同等以上（非劣性）' if ci[0] >= -u2.MARGIN else '下回る' if ci[1] < 0 else '判定不能'
        else:
            v = '傾向参考'
        return {'questions': len(chosen), 'difference': float(np.mean(d)), 'ci95': ci, 'verdict': v, 'sign': g1m.sign_test(d)}
    values = {q: g1r.answer_value(answers[q], prepared['questions'][q]['answer']) for q in ids}
    out = {'questions': len(ids), 'AT_eligible': len(eligible),
           'AT': {'correct': float(np.mean(list(rows['AT'].values()))), 'abstain': float(np.mean([v['abstention'] for v in values.values()])),
                  'input_tokens': float(np.mean([inputs[q]['input_tokens'] for q in ids])),
                  'answer_seconds': float(np.mean([answers[q].get('wall_seconds', 0.) for q in ids])),
                  'by_stratum': {s: float(np.mean([rows['AT'][q] for q in ids if um.stratum(prepared['questions'][q]) == s]))
                                 for s in sorted({um.stratum(prepared['questions'][q]) for q in ids})}},
           'comparisons': {'primary_P_minus_AT': cmp('P', 'AT', eligible, 'vs_full'), 'AT_minus_A': cmp('AT', 'A', ids, 'trend'),
                           'C512_minus_AT': cmp('C512', 'AT', ids, 'trend')},
           'rows_AT': rows['AT']}
    (u2.RESULTS/'longmemeval-u2-at-report.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    a = out['AT']
    lines = ['# U2-AT：全文に P と同じ書式（質問の日付・規則・日付順）を与えた場合', '',
             f'同じ40問。AT の窓に入った問 {out["AT_eligible"]}/40。主判定は Qwen judge。', '',
             f'AT 正答率 {a["correct"]:.3f}、棄権率 {a["abstain"]:.3f}、入力 {a["input_tokens"]:.0f} トークン、回答 {a["answer_seconds"]:.1f} 秒（平均）。', '',
             '|比較|問数|点差|95%区間|判定|勝/負/同|', '|---|---:|---:|---|---|---|']
    for name, c in out['comparisons'].items():
        s = c['sign']
        lines.append(f'|{name}|{c["questions"]}|{c["difference"]:+.3f}|[{c["ci95"][0]:+.3f}, {c["ci95"][1]:+.3f}]|{c["verdict"]}|{s["wins"]}/{s["losses"]}/{s["ties"]}|')
    lines += ['', '|種類|' + '|'.join(a['by_stratum']) + '|', '|---|' + '---:|'*len(a['by_stratum']),
              '|AT|' + '|'.join(f'{v:.2f}' for v in a['by_stratum'].values()) + '|']
    (u2.RESULTS/'longmemeval-u2-at-report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'run'])
    {'register': register, 'run': run}[parser.parse_args().command]()
