#!/usr/bin/env python3
"""E1: a small local model as the reader (Gemma 4 E2B, QAT q4_0) on the same 40 U2 questions.

E2B-P  = exactly the messages Qwen's arm P received in U2 (same memory, built by the Qwen pipeline,
         same date-aware template); only the reader changes.
E2B-AT = the whole history with the same template as U2-AT, when it fits E2B's context window.
Answers one at a time (temperature 0 per am.GENERATION, JSON schema), judged by the U2 Qwen judge.
See docs/MEMORY_LONGMEMEVAL_E1_PREREG.md.

    python3 longmemeval_e1_e2b.py register   # writes the plan (commit it before run)
    python3 longmemeval_e1_e2b.py run
"""
from __future__ import annotations
import argparse
import json
import struct
import subprocess
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
LOCAL = u2.WORK/'e1'
PLAN = u2.RESULTS/'longmemeval-e1-plan.json'
MODEL = ROOT/'models/gemma-4-E2B/gemma-4-E2B_q4_0-it.gguf'
CODE = ['scripts/eval/longmemeval_e1_e2b.py', 'docs/MEMORY_LONGMEMEVAL_E1_PREREG.md']
PRODUCT_BAR = 0.50


def gguf_context_length(path: Path) -> int:
    """Read `<arch>.context_length` from the GGUF header (no dependencies)."""
    with open(path, 'rb') as f:
        def u32(): return struct.unpack('<I', f.read(4))[0]
        def u64(): return struct.unpack('<Q', f.read(8))[0]
        def string(): return f.read(u64()).decode('utf-8', 'replace')
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        unpack = {4: '<I', 5: '<i', 10: '<Q', 11: '<q'}

        def value(t):
            if t == 8:
                return string()
            if t == 9:
                inner, n = u32(), u64()
                return [value(inner) for _ in range(n)]
            raw = f.read(sizes[t])
            return struct.unpack(unpack[t], raw)[0] if t in unpack else raw
        if f.read(4) != b'GGUF':
            raise SystemExit('not a GGUF file')
        u32()
        _tensors, kvs = u64(), u64()
        meta = {}
        for _ in range(kvs):
            key = string()
            meta[key] = value(u32())
            if key.endswith('.context_length'):
                return int(meta[key])
    raise SystemExit('no context_length in the GGUF header')


class E2BServer(u2.Server):
    """The registered U1/U2 llama-server argv with the E2B model file."""
    def start(self, slot_context, parallel=1, label=''):
        argv = list(u2.U1_PLAN['server_argv'])
        argv[argv.index('-m')+1] = str(MODEL)
        original = u2.U1_PLAN['server_argv']
        u2.U1_PLAN['server_argv'] = argv
        try:
            super().start(slot_context, parallel, label)
        finally:
            u2.U1_PLAN['server_argv'] = original


def count_tokens(messages) -> int:
    """Prompt tokens as E2B's own chat template and tokenizer count them (llama-server /apply-template + /tokenize)."""
    prompt = u2.http('/apply-template', {'messages': messages})['prompt']
    return len(u2.http('/tokenize', {'content': prompt})['tokens'])


def register():
    if PLAN.exists():
        raise SystemExit('plan exists')
    if not MODEL.exists():
        raise SystemExit(f'model missing: {MODEL}')
    u2plan = json.loads(u2.PLAN.read_text())
    window = gguf_context_length(MODEL)
    plan = {'created_at': u2.lm.stamp(), 'experiment': 'E1: Gemma 4 E2B as the reader on the U2 questions',
            'authorization': 'user chat 2026-10-24 (download and experiment approved)',
            'question_ids': u2plan['question_ids'], 'model_file': MODEL.name,
            'model_sha256': (MODEL.parent/'expected.txt').read_text().split()[0], 'context_window': window,
            'u2_report_sha256': u2.lm.file_hash(u2.RESULTS/'longmemeval-u2-report.json'),
            'u2_at_report_sha256': u2.lm.file_hash(u2.RESULTS/'longmemeval-u2-at-report.json'),
            'memory_inputs_sha256': u2.lm.file_hash(RUN/'memory-inputs.json'),
            'primary': f'E2B-P - Qwen-AT (all 40): low>0 上回る; else low>=-{u2.MARGIN} 同等以上; high<0 下回る; else 判定不能',
            'secondary': ['E2B-P - E2B-AT on E2B-AT-eligible (vs_full rule)', 'E2B-P - Qwen-P (cost of the small reader)',
                          f'product bar: E2B-P accuracy >= {PRODUCT_BAR}', 'per stratum', 'abstention', 'answer seconds'],
            'code_sha256': {f: u2.lm.file_hash(ROOT/f) for f in CODE}}
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    print('registered', len(plan['question_ids']), 'context window', window)


def run():
    plan = json.loads(PLAN.read_text())
    if not u2.st.committed(PLAN):
        raise SystemExit('commit the plan first')
    for f, h in plan['code_sha256'].items():
        if u2.lm.file_hash(ROOT/f) != h:
            raise SystemExit('registered code changed: '+f)
    prepared = json.loads((RUN/'prepared.json').read_text())
    if set(prepared['questions']) != set(plan['question_ids']):
        raise SystemExit('U2 question set differs')
    memory_inputs = json.loads((RUN/'memory-inputs.json').read_text())['P']
    window = plan['context_window']
    ctx = u2.Ctx(LOCAL)
    if any((LOCAL/'inflight').iterdir()):
        raise SystemExit('unknown inflight request')
    server, judge_server = E2BServer(ctx), u2.Server(ctx)
    try:
        server.start(window, 1, 'E2B')
        inputs = {'E2B-P': {}, 'E2B-AT': {}}
        answers = {'E2B-P': {}, 'E2B-AT': {}}
        for n, (qid, row) in enumerate(prepared['questions'].items(), 1):
            p = {'messages': memory_inputs[qid]['messages']}
            p['input_tokens'] = count_tokens(p['messages'])
            inputs['E2B-P'][qid] = p
            answers['E2B-P'][qid] = u2.answer(ctx, server, 'E2B-P', qid, p)
            chat = um.messages(um.render(prepared['turns'][qid]), '', row['question'], row['question_date'])
            tokens = count_tokens(chat)
            at = {'messages': chat, 'input_tokens': tokens, 'fits_context': u2.fits(tokens, 256, window)}
            inputs['E2B-AT'][qid] = at
            answers['E2B-AT'][qid] = (u2.answer(ctx, server, 'E2B-AT', qid, at) if at['fits_context']
                                      else {'failure_kind': 'input_overflow', 'response': {}, 'wall_seconds': 0.})
            ctx.status('answer', completed=n, total=len(prepared['questions']))
        server.stop()
        judge_server.start(8192, 1, 'judge')
        labels = u2.judge(ctx, judge_server, prepared, answers, tuple(answers))
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
        server.stop()
        judge_server.stop()


def report(prepared, inputs, answers, labels, window):
    u2r = json.loads((u2.RESULTS/'longmemeval-u2-report.json').read_text())['rows']
    at = json.loads((u2.RESULTS/'longmemeval-u2-at-report.json').read_text())['rows_AT']
    ids = sorted(prepared['questions'])
    rows = {'Qwen-P': {q: u2r['P'][q]['correct'] for q in ids}, 'Qwen-AT': {q: at[q] for q in ids}}
    for arm in ('E2B-P', 'E2B-AT'):
        rows[arm] = {q: labels[arm][q]['label'] for q in ids}
    eligible = [q for q in ids if inputs['E2B-AT'][q]['fits_context']]

    def cmp(left, right, chosen, mode):
        if not chosen:
            return None
        d = [rows[left][q]-rows[right][q] for q in chosen]
        ci = g1m.interval(d, chosen)
        if mode == 'vs_full':
            v = '上回る' if ci[0] > 0 else '同等以上（非劣性）' if ci[0] >= -u2.MARGIN else '下回る' if ci[1] < 0 else '判定不能'
        else:
            v = '傾向参考'
        return {'questions': len(chosen), 'difference': float(np.mean(d)), 'ci95': ci, 'verdict': v, 'sign': g1m.sign_test(d)}

    def summary(arm):
        values = {q: g1r.answer_value(answers[arm][q], prepared['questions'][q]['answer']) for q in ids}
        return {'correct': float(np.mean(list(rows[arm].values()))),
                'abstain': float(np.mean([v['abstention'] for v in values.values()])),
                'input_tokens': float(np.mean([inputs[arm][q]['input_tokens'] for q in ids])),
                'answer_seconds': float(np.mean([answers[arm][q].get('wall_seconds', 0.) for q in ids])),
                'by_stratum': {s: float(np.mean([rows[arm][q] for q in ids if um.stratum(prepared['questions'][q]) == s]))
                               for s in sorted({um.stratum(prepared['questions'][q]) for q in ids})}}
    out = {'questions': len(ids), 'context_window': window, 'E2B_AT_eligible': len(eligible),
           'arms': {a: summary(a) for a in ('E2B-P', 'E2B-AT')},
           'reference': {'Qwen-P': float(np.mean(list(rows['Qwen-P'].values()))), 'Qwen-AT': float(np.mean(list(rows['Qwen-AT'].values())))},
           'product_bar': {'threshold': PRODUCT_BAR, 'met': float(np.mean(list(rows['E2B-P'].values()))) >= PRODUCT_BAR},
           'comparisons': {'primary_E2B_P_minus_Qwen_AT': cmp('E2B-P', 'Qwen-AT', ids, 'vs_full'),
                           'E2B_P_minus_E2B_AT': cmp('E2B-P', 'E2B-AT', eligible, 'vs_full'),
                           'E2B_P_minus_Qwen_P': cmp('E2B-P', 'Qwen-P', ids, 'trend')},
           'rows': {a: rows[a] for a in ('E2B-P', 'E2B-AT')}}
    (u2.RESULTS/'longmemeval-e1-report.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    lines = ['# E1：小さいモデル（Gemma 4 E2B）を答える役にした場合', '',
             f'U2 と同じ40問。E2B の窓 {window} トークン、E2B-AT が窓に入った問 {len(eligible)}/40。主判定は Qwen judge。', '',
             f'参考：Qwen 27B の P {out["reference"]["Qwen-P"]:.3f}、Qwen 27B の AT {out["reference"]["Qwen-AT"]:.3f}。', '',
             '|条件|正答率|棄権率|入力トークン|回答秒|', '|---|---:|---:|---:|---:|']
    for a, s in out['arms'].items():
        lines.append(f'|{a}|{s["correct"]:.3f}|{s["abstain"]:.3f}|{s["input_tokens"]:.0f}|{s["answer_seconds"]:.1f}|')
    lines += ['', f'製品の目安（E2B-P ≥ {PRODUCT_BAR}）：{"満たす" if out["product_bar"]["met"] else "満たさない"}', '',
              '|比較|問数|点差|95%区間|判定|勝/負/同|', '|---|---:|---:|---|---|---|']
    for name, c in out['comparisons'].items():
        if c:
            s = c['sign']
            lines.append(f'|{name}|{c["questions"]}|{c["difference"]:+.3f}|[{c["ci95"][0]:+.3f}, {c["ci95"][1]:+.3f}]|{c["verdict"]}|{s["wins"]}/{s["losses"]}/{s["ties"]}|')
    (u2.RESULTS/'longmemeval-e1-report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'run'])
    {'register': register, 'run': run}[parser.parse_args().command]()
