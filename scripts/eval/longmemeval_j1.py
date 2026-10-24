#!/usr/bin/env python3
"""J1: does the memory work in Japanese? Gemma 4 E2B as the reader on a self-made Japanese test (40 questions).

Everything a user's PC would run is Gemma 4 E2B: the memory is built with the U2 pipeline (fact extraction and
update links by E2B, EmbeddingGemma 300m windows and facts, setting P: budget 2048, dense order, date-aware
template), from build/eval/memory/j1/j1_dataset.json instead of LongMemEval-S. E2B-P reads P's memory input;
E2B-AT reads the whole history with the same template. Only the judge is Qwen3.8 27B (the U2 judge).
See docs/MEMORY_JA_J1_PREREG.md.

    python3 longmemeval_j1.py register   # writes the plan (commit it before run)
    python3 longmemeval_j1.py run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/eval'))
import longmemeval_e1_e2b as e1  # noqa: E402
import longmemeval_g1_methods as g1m  # noqa: E402
import longmemeval_g1_report as g1r  # noqa: E402
import longmemeval_u1_methods as um  # noqa: E402
import longmemeval_u2_run as u2  # noqa: E402

DATASET = ROOT/'build/eval/memory/j1/j1_dataset.json'
LOCAL = ROOT/'build/eval/memory/j1/run'
PLAN = u2.RESULTS/'j1-plan.json'
CODE = ['scripts/eval/longmemeval_j1.py', 'scripts/eval/longmemeval_e1_e2b.py', 'docs/MEMORY_JA_J1_PREREG.md',
        'scripts/eval/j1/build_dataset.py', 'scripts/eval/j1/evidence_a.py', 'scripts/eval/j1/evidence_b.py',
        'scripts/eval/j1/gen_distractors.py']
PRODUCT_BAR = 0.50
FLOOR, CEILING = 0.10, 0.90


def register():
    if PLAN.exists():
        raise SystemExit('plan exists')
    rows = json.loads(DATASET.read_text())
    plan = {'created_at': u2.lm.stamp(), 'experiment': 'J1: memory in Japanese, all on Gemma 4 E2B (memory building and reading)',
            'authorization': 'user chat 2026-10-24 (Japanese test approved; E2B first, Qwen only if E2B-P is at least equivalent)',
            'question_ids': [r['question_id'] for r in rows], 'dataset_sha256': u2.lm.file_hash(DATASET),
            'distractors_sha256': u2.lm.file_hash(ROOT/'build/eval/memory/j1/distractors.json'),
            'model_file': e1.MODEL.name, 'context_window': e1.gguf_context_length(e1.MODEL),
            'primary': f'E2B-P - E2B-AT on E2B-AT-eligible: low>0 上回る; else low>=-{u2.MARGIN} 同等以上; high<0 下回る; else 判定不能',
            'follow_up_rule': 'run the Qwen 27B reader on the same inputs only if the primary verdict is 上回る or 同等以上',
            'secondary': [f'product bar: E2B-P >= {PRODUCT_BAR}', f'floor/ceiling flags: an arm below {FLOOR} or above {CEILING}',
                          'per type', 'abstention', 'answer seconds', 'P memory coverage of the answer sessions'],
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
    qwen, small = u2.Server(ctx), e1.E2BServer(ctx)
    try:
        # Memory building on E2B, as on a user's PC (token counts, fact extraction, update links, assembly).
        prepared = u2.prepare(ctx, ids, small)
        facts = json.loads((LOCAL/'facts.json').read_text()) if (LOCAL/'facts.json').exists() else u2.ingest(ctx, prepared, small)
        units, vectors, query = u2.embedding(ctx, prepared, facts, small)
        tables, links = u2.updates(ctx, prepared, facts, units, vectors, small)
        raw = u2.orders(ctx, prepared, units, vectors, query, small)  # P uses the dense order; the rerank arms are unused
        memory = u2.assemble(ctx, prepared, raw, tables, links, small)
        small.stop()
        window = plan['context_window']
        small.start(window, 1, 'E2B')
        inputs = {'E2B-P': {}, 'E2B-AT': {}}
        answers = {'E2B-P': {}, 'E2B-AT': {}}
        for n, (qid, row) in enumerate(prepared['questions'].items(), 1):
            p = {'messages': memory['P'][qid]['messages'], 'coverage': memory['P'][qid]['coverage']}
            p['input_tokens'] = e1.count_tokens(p['messages'])
            inputs['E2B-P'][qid] = p
            answers['E2B-P'][qid] = u2.answer(ctx, small, 'E2B-P', qid, p)
            chat = um.messages(um.render(prepared['turns'][qid]), '', row['question'], row['question_date'])
            tokens = e1.count_tokens(chat)
            at = {'messages': chat, 'input_tokens': tokens, 'fits_context': u2.fits(tokens, 256, window)}
            inputs['E2B-AT'][qid] = at
            answers['E2B-AT'][qid] = (u2.answer(ctx, small, 'E2B-AT', qid, at) if at['fits_context']
                                      else {'failure_kind': 'input_overflow', 'response': {}, 'wall_seconds': 0.})
            ctx.status('answer', completed=n, total=len(ids))
        small.stop()
        qwen.start(8192, 1, 'judge')
        labels = u2.judge(ctx, qwen, prepared, answers, tuple(answers))
        ctx.write('answers.json', answers)
        ctx.write('labels.json', labels)
        ctx.write('inputs-public.json', {a: {q: {k: v for k, v in x.items() if k != 'messages'} for q, x in d.items()} for a, d in inputs.items()})
        report(prepared, inputs, answers, labels, window)
        ctx.status('complete')
        ctx.log('complete')
    except BaseException as error:
        ctx.status('stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        qwen.stop()
        small.stop()


def kind(qid, row):
    return 'abstention' if qid.endswith('_abs') else row['question_type']


def report(prepared, inputs, answers, labels, window):
    ids = sorted(prepared['questions'])
    rows = {a: {q: labels[a][q]['label'] for q in ids} for a in ('E2B-P', 'E2B-AT')}
    eligible = [q for q in ids if inputs['E2B-AT'][q]['fits_context']]

    def cmp(left, right, chosen):
        d = [rows[left][q]-rows[right][q] for q in chosen]
        ci = g1m.interval(d, chosen)
        v = '上回る' if ci[0] > 0 else '同等以上（非劣性）' if ci[0] >= -u2.MARGIN else '下回る' if ci[1] < 0 else '判定不能'
        return {'questions': len(chosen), 'difference': float(np.mean(d)), 'ci95': ci, 'verdict': v, 'sign': g1m.sign_test(d)}

    def summary(arm):
        values = {q: g1r.answer_value(answers[arm][q], prepared['questions'][q]['answer']) for q in ids}
        acc = float(np.mean(list(rows[arm].values())))
        kinds = sorted({kind(q, prepared['questions'][q]) for q in ids})
        return {'correct': acc, 'floor_or_ceiling': acc < FLOOR or acc > CEILING,
                'abstain': float(np.mean([v['abstention'] for v in values.values()])),
                'input_tokens': float(np.mean([inputs[arm][q]['input_tokens'] for q in ids])),
                'answer_seconds': float(np.mean([answers[arm][q].get('wall_seconds', 0.) for q in ids])),
                'by_type': {k: float(np.mean([rows[arm][q] for q in ids if kind(q, prepared['questions'][q]) == k])) for k in kinds}}
    primary = cmp('E2B-P', 'E2B-AT', eligible)
    out = {'questions': len(ids), 'context_window': window, 'E2B_AT_eligible': len(eligible),
           'arms': {a: summary(a) for a in ('E2B-P', 'E2B-AT')},
           'P_coverage': float(np.mean([inputs['E2B-P'][q]['coverage'] for q in ids])),
           'product_bar': {'threshold': PRODUCT_BAR, 'met': float(np.mean(list(rows['E2B-P'].values()))) >= PRODUCT_BAR},
           'primary_E2B_P_minus_E2B_AT': primary,
           'qwen_follow_up': primary['verdict'] in ('上回る', '同等以上（非劣性）'),
           'rows': rows}
    (u2.RESULTS/'j1-report.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    lines = ['# J1：日本語での記憶（答える役は Gemma 4 E2B）', '',
             f'自作の日本語テスト40問。E2B の窓 {window}、E2B-AT が窓に入った問 {len(eligible)}/40。主判定は Qwen judge。'
             f' P の記憶が答えの会話を全部含んだ割合 {out["P_coverage"]:.2f}。', '',
             '|条件|正答率|棄権率|入力トークン|回答秒|', '|---|---:|---:|---:|---:|']
    for a, s in out['arms'].items():
        lines.append(f'|{a}|{s["correct"]:.3f}{" ⚠床/天井" if s["floor_or_ceiling"] else ""}|{s["abstain"]:.3f}|{s["input_tokens"]:.0f}|{s["answer_seconds"]:.1f}|')
    s = primary['sign']
    lines += ['', f'主：E2B-P − E2B-AT {primary["difference"]:+.3f} [{primary["ci95"][0]:+.3f}, {primary["ci95"][1]:+.3f}] → {primary["verdict"]}（勝{s["wins"]}/負{s["losses"]}/同{s["ties"]}）',
              f'製品の目安（E2B-P ≥ {PRODUCT_BAR}）：{"満たす" if out["product_bar"]["met"] else "満たさない"}',
              f'Qwen での追試：{"行う" if out["qwen_follow_up"] else "行わない"}', '',
              '|種類|' + '|'.join(out['arms']['E2B-P']['by_type']) + '|', '|---|' + '---:|'*len(out['arms']['E2B-P']['by_type'])]
    for a in ('E2B-P', 'E2B-AT'):
        lines.append(f'|{a}|' + '|'.join(f'{v:.2f}' for v in out['arms'][a]['by_type'].values()) + '|')
    (u2.RESULTS/'j1-report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'run'])
    {'register': register, 'run': run}[parser.parse_args().command]()
