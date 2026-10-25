#!/usr/bin/env python3
"""S2: at 5,000 stored sessions, do Gemma 4 E2B's answers get better with a larger recall budget, an LLM rerank by
E2B, or a date filter? 60 fresh LongMemEval-S questions (10 per non-abstention type, none used before), each with
its answer sessions at their own dates among the scale test's filler sessions. Recall by the dsh plugin's Memory
(scripts/eval/scale/s2_retrieve.mjs), answers by E2B with the P template, judged by the U2 Qwen judge.
See docs/MEMORY_SCALE_S2_PREREG.md.

    python3 longmemeval_s2.py register   # writes the plan (commit it before run)
    python3 longmemeval_s2.py run
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import subprocess
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

WORK = ROOT/'build/eval/memory/s2'
LOCAL = WORK/'run'
QUESTIONS = WORK/'questions.json'
RETRIEVALS = LOCAL/'retrievals.json'
PLAN = u2.RESULTS/'s2-plan.json'
SCALE_PLAN = ROOT/'build/scale/plan.json'
VECTORS = ROOT/'build/scale/vectors/documents.json'
CODE = ['scripts/eval/longmemeval_s2.py', 'scripts/eval/scale/s2_retrieve.mjs', 'scripts/eval/scale/common.mjs',
        'docs/MEMORY_SCALE_S2_PREREG.md']
SEED = 20261024
PER_TYPE = 10
ARMS = ['A-5000', 'B-5000', 'C30-5000', 'C100-5000', 'D-5000', 'A-50']
PRIMARY = ['B-5000', 'C30-5000', 'C100-5000', 'D-5000']
E2B_WINDOW = 131072
SHIPPED_RERANK_MS = 15000


def used_ids() -> set[str]:
    used = set()
    for name in ['longmemeval-g1-plan.json', 'longmemeval-u1-plan.json', 'longmemeval-u2-plan.json', 'longmemeval-e1-plan.json']:
        used |= set(json.loads((u2.RESULTS/name).read_text()).get('question_ids', []))
    used |= {q['qid'] for q in json.loads(SCALE_PLAN.read_text())['planted']}
    return used


def select() -> list[str]:
    data = json.loads(u2.DATA.read_text())
    used = used_ids()
    by_type = collections.defaultdict(list)
    for q in sorted(data, key=lambda q: q['question_id']):
        if not q['question_id'].endswith('_abs') and q['question_id'] not in used:
            by_type[q['question_type']].append(q['question_id'])
    rng = random.Random(SEED)
    return [qid for t in sorted(by_type) for qid in rng.sample(by_type[t], PER_TYPE)]


def write_questions(ids: list[str]) -> dict:
    data = {q['question_id']: q for q in json.loads(u2.DATA.read_text())}
    out = []
    for qid in ids:
        q = data[qid]
        content = dict(zip(q['haystack_session_ids'], q['haystack_sessions']))
        dates = dict(zip(q['haystack_session_ids'], q['haystack_dates']))
        order = {s: k for k, s in enumerate(q['haystack_session_ids'])}
        evidence = [{'id': s, 'date': dates[s], 'turns': [{'role': t['role'], 'content': t['content'], 'has_answer': bool(t.get('has_answer'))}
                                                          for t in content[s]]}
                    for s in sorted(q['answer_session_ids'], key=lambda s: order[s])]
        out.append({'qid': qid, 'type': q['question_type'], 'question': q['question'], 'answer': str(q['answer']),
                    'question_date': q['question_date'], 'evidence': evidence})
    WORK.mkdir(parents=True, exist_ok=True)
    QUESTIONS.write_text(json.dumps(out, ensure_ascii=False))
    return {q['qid']: q for q in out}


def register():
    if PLAN.exists():
        raise SystemExit('plan exists')
    ids = select()
    plan = {'created_at': u2.lm.stamp(), 'experiment': 'S2: recall at 5,000 sessions, answers by Gemma 4 E2B',
            'authorization': 'user chat 2026-10-24 (pause J1-Q, try the four recall options first)',
            'question_ids': ids, 'seed': SEED, 'dataset_sha256': u2.lm.file_hash(u2.DATA),
            'scale_plan_sha256': u2.lm.file_hash(SCALE_PLAN), 'cached_vectors_index_sha256': u2.lm.file_hash(VECTORS),
            'model_file': e1.MODEL.name, 'arms': ARMS,
            'primary': 'each of ' + ', '.join(PRIMARY) + ' minus A-5000 (Qwen judge): low>0 改善; high<0 悪化; else 判定不能',
            'secondary': ['A-5000 - A-50', 'D-5000 - A-5000 on questions with a date range', 'recall hit per arm',
                          'rerank wall time (share over 15 s)', 'answer seconds and input tokens', 'per type',
                          'Claude blinded sample of 90 rows (seed 0), graded before Qwen labels are opened'],
            'code_sha256': {f: u2.lm.file_hash(ROOT/f) for f in CODE}}
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    print('registered', len(ids))


def run():
    plan = json.loads(PLAN.read_text())
    if not u2.st.committed(PLAN):
        raise SystemExit('commit the plan first')
    for f, h in plan['code_sha256'].items():
        if u2.lm.file_hash(ROOT/f) != h:
            raise SystemExit('registered code changed: '+f)
    if u2.lm.file_hash(u2.DATA) != plan['dataset_sha256'] or u2.lm.file_hash(SCALE_PLAN) != plan['scale_plan_sha256']:
        raise SystemExit('data changed')
    ids = plan['question_ids']
    LOCAL.mkdir(parents=True, exist_ok=True)
    ctx = u2.Ctx(LOCAL)
    if any((LOCAL/'inflight').iterdir()):
        raise SystemExit('unknown inflight request')
    small, qwen = e1.E2BServer(ctx), u2.Server(ctx)
    try:
        questions = write_questions(ids)
        small.start(E2B_WINDOW, 1, 'E2B')
        if not RETRIEVALS.exists():
            ctx.status('recall')
            tmp = RETRIEVALS.with_suffix('.tmp')
            subprocess.run([shutil.which('node'), 'scripts/eval/scale/s2_retrieve.mjs', '--questions', str(QUESTIONS), '--out', str(tmp),
                            '--chat', 'http://127.0.0.1:8080/v1', '--sizes', '50,5000'], cwd=ROOT, check=True,
                           stderr=open(LOCAL/'recall.log', 'a'))
            tmp.rename(RETRIEVALS)
        recall = json.loads(RETRIEVALS.read_text())
        rows = {(r['size'], r['qid']): r for r in recall['rows']}
        inputs = {a: {} for a in ARMS}
        answers = {a: {} for a in ARMS}
        for arm in ARMS:
            kind, size = arm.split('-')
            for n, qid in enumerate(ids, 1):
                rec, q = rows[(int(size), qid)][kind], questions[qid]
                messages = um.messages('', rec['text'], q['question'], q['question_date'])
                tokens = e1.count_tokens(messages)
                if not u2.fits(tokens, 256, E2B_WINDOW):
                    raise RuntimeError(f'input overflow {arm} {qid}')
                inputs[arm][qid] = {'messages': messages, 'input_tokens': tokens}
                answers[arm][qid] = u2.answer(ctx, small, arm, qid, inputs[arm][qid])
                ctx.status('answer', arm=arm, completed=n, total=len(ids))
        small.stop()
        qwen.start(8192, 1, 'judge')
        prepared = {'questions': {qid: {'question': q['question'], 'answer': q['answer'], 'question_type': q['type']}
                                  for qid, q in questions.items()}}
        labels = u2.judge(ctx, qwen, prepared, answers, ARMS)
        ctx.write('answers.json', answers)
        ctx.write('labels.json', labels)
        ctx.write('inputs-public.json', {a: {q: {'input_tokens': x['input_tokens']} for q, x in d.items()} for a, d in inputs.items()})
        report(questions, rows, inputs, answers, labels)
        ctx.status('complete')
        ctx.log('complete')
    except BaseException as error:
        ctx.status('stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        small.stop()
        qwen.stop()


def report(questions, rows, inputs, answers, labels):
    ids = sorted(questions)
    correct = {a: {q: labels[a][q]['label'] for q in ids} for a in ARMS}

    def cmp(left, right, chosen, mode):
        d = [correct[left][q]-correct[right][q] for q in chosen]
        if not chosen:
            return {'questions': 0}
        ci = g1m.interval(d, chosen)
        v = ('改善' if ci[0] > 0 else '悪化' if ci[1] < 0 else '判定不能') if mode == 'improve' else '傾向参考'
        return {'questions': len(chosen), 'difference': float(np.mean(d)), 'ci95': ci, 'verdict': v, 'sign': g1m.sign_test(d)}

    def rec(arm, q):
        kind, size = arm.split('-')
        return rows[(int(size), q)][kind]

    def summary(arm):
        values = {q: g1r.answer_value(answers[arm][q], questions[q]['answer']) for q in ids}
        types = sorted({questions[q]['type'] for q in ids})
        out = {'correct': float(np.mean(list(correct[arm].values()))),
               'abstain': float(np.mean([v['abstention'] for v in values.values()])),
               'recall_hit': float(np.mean([rec(arm, q)['hit'] for q in ids])),
               'recall_all_sessions': float(np.mean([rec(arm, q)['allSessions'] for q in ids])),
               'input_tokens': float(np.mean([inputs[arm][q]['input_tokens'] for q in ids])),
               'answer_seconds': float(np.mean([answers[arm][q].get('wall_seconds', 0.) for q in ids])),
               'by_type': {t: float(np.mean([correct[arm][q] for q in ids if questions[q]['type'] == t])) for t in types}}
        if arm.startswith('C'):
            ms = [rec(arm, q)['ms'] for q in ids if rec(arm, q).get('ms') is not None]
            out['rerank'] = {'calls': len(ms), 'mean_s': float(np.mean(ms))/1000 if ms else None,
                             'p95_s': float(np.percentile(ms, 95))/1000 if ms else None,
                             'over_15s': float(np.mean([m > SHIPPED_RERANK_MS for m in ms])) if ms else None,
                             'unparsed': sum(not rec(arm, q).get('parsed', False) for q in ids)}
        return out

    dated = [q for q in ids if rec('D-5000', q).get('rule')]
    out = {'questions': len(ids), 'arms': {a: summary(a) for a in ARMS},
           'primary': {a: cmp(a, 'A-5000', ids, 'improve') for a in PRIMARY},
           'secondary': {'A-5000_minus_A-50': cmp('A-5000', 'A-50', ids, 'trend'),
                         'D_minus_A_on_dated': cmp('D-5000', 'A-5000', dated, 'trend'),
                         'dated_questions': len(dated),
                         'dated_answer_in_range': sum(bool(rec('D-5000', q).get('answerInRange')) for q in dated)},
           'rows': correct}
    (u2.RESULTS/'s2-report.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    lines = ['# S2：5,000 セッションでの想起の改善（答える役は Gemma 4 E2B）', '',
             f'新しい LongMemEval-S の 60 問（種類ごとに 10）。主判定は Qwen judge。日付の範囲を読み取れた問 {len(dated)}/60'
             f'（うち答えの会話が範囲内 {out["secondary"]["dated_answer_in_range"]}）。', '',
             '|条件|正答率|棄権率|想起に答え|全根拠|入力トークン|回答秒|', '|---|---:|---:|---:|---:|---:|---:|']
    for a, s in out['arms'].items():
        lines.append(f'|{a}|{s["correct"]:.3f}|{s["abstain"]:.3f}|{s["recall_hit"]:.3f}|{s["recall_all_sessions"]:.3f}|{s["input_tokens"]:.0f}|{s["answer_seconds"]:.1f}|')
    lines += ['', '|比較|問数|点差|95%区間|判定|勝/負/同|', '|---|---:|---:|---|---|---|']
    for name, c in [*[(f'{a} − A-5000', out['primary'][a]) for a in PRIMARY],
                    ('A-5000 − A-50', out['secondary']['A-5000_minus_A-50']), ('D − A（日付の範囲がある問）', out['secondary']['D_minus_A_on_dated'])]:
        if c.get('questions'):
            s = c['sign']
            lines.append(f'|{name}|{c["questions"]}|{c["difference"]:+.3f}|[{c["ci95"][0]:+.3f}, {c["ci95"][1]:+.3f}]|{c["verdict"]}|{s["wins"]}/{s["losses"]}/{s["ties"]}|')
    lines += ['', '|並べ直し|呼び出し|平均秒|p95 秒|15 秒超|読めない返事|', '|---|---:|---:|---:|---:|---:|']
    for a in ('C30-5000', 'C100-5000'):
        r = out['arms'][a]['rerank']
        lines.append(f'|{a}|{r["calls"]}|{r["mean_s"]:.1f}|{r["p95_s"]:.1f}|{r["over_15s"]:.2f}|{r["unparsed"]}|')
    types = list(out['arms']['A-5000']['by_type'])
    lines += ['', '|種類|' + '|'.join(types) + '|', '|---|' + '---:|'*len(types)]
    for a in ARMS:
        lines.append(f'|{a}|' + '|'.join(f'{out["arms"][a]["by_type"][t]:.2f}' for t in types) + '|')
    (u2.RESULTS/'s2-report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'run'])
    {'register': register, 'run': run}[parser.parse_args().command]()
