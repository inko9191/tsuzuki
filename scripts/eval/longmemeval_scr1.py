#!/usr/bin/env python3
"""SCR1 (exploratory screening, dev questions only): LLM-free recall changes at 5,000 sessions, judged at the answer
level as in S2 (Gemma 4 E2B answers with the P template, U2 Qwen judge). The 60 questions are the scale test's planted
ones (used in the scale diagnosis: dev). Recall by scripts/eval/scale/screen.mjs. Not a confirmatory test: no verdicts,
differences against A only, to choose what to test next.

    python3 longmemeval_scr1.py run
"""
from __future__ import annotations

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
import longmemeval_s2 as s2  # noqa: E402
import longmemeval_u1_methods as um  # noqa: E402
import longmemeval_u2_run as u2  # noqa: E402

WORK = ROOT/'build/scale/screen'
LOCAL = WORK/'run'
QUESTIONS = WORK/'dev-questions.json'
PROBES = WORK/'probes.json'
RECALL = LOCAL/'recall.json'
ARMS = ['A', 'A1024', 'CTX80', 'CTR', 'HUB', 'CEN', 'ROLE', 'QA']
PROBE_COUNT, PROBE_SEED = 100, 1


def probes() -> None:
    """Probe questions for HUB: LongMemEval-S questions that are not dev, not S2 and not used by any earlier experiment."""
    used = s2.used_ids() | set(json.loads(s2.PLAN.read_text())['question_ids'])
    pool = sorted(q['question_id'] for q in json.loads(u2.DATA.read_text()) if q['question_id'] not in used)
    text = {q['question_id']: q['question'] for q in json.loads(u2.DATA.read_text())}
    PROBES.write_text(json.dumps([text[q] for q in random.Random(PROBE_SEED).sample(pool, PROBE_COUNT)]))


def run():
    LOCAL.mkdir(parents=True, exist_ok=True)
    ctx = u2.Ctx(LOCAL)
    if any((LOCAL/'inflight').iterdir()):
        raise SystemExit('unknown inflight request')
    questions = {q['qid']: q for q in json.loads(QUESTIONS.read_text())}
    ids = sorted(questions)
    small, qwen = e1.E2BServer(ctx), u2.Server(ctx)
    try:
        if not RECALL.exists():
            if not PROBES.exists():
                probes()
            tmp = RECALL.with_suffix('.tmp')
            subprocess.run([shutil.which('node'), 'scripts/eval/scale/screen.mjs', '--questions', str(QUESTIONS), '--probes', str(PROBES),
                            '--out', str(tmp)], cwd=ROOT, check=True, stderr=open(LOCAL/'recall.log', 'a'))
            tmp.rename(RECALL)
        rows = {r['qid']: r for r in json.loads(RECALL.read_text())['rows']}
        small.start(s2.E2B_WINDOW, 1, 'E2B')
        answers = {a: {} for a in ARMS}
        for arm in ARMS:
            for n, qid in enumerate(ids, 1):
                q = questions[qid]
                messages = um.messages('', rows[qid][arm]['text'], q['question'], q['question_date'])
                answers[arm][qid] = u2.answer(ctx, small, arm, qid, {'messages': messages})
                ctx.status('answer', arm=arm, completed=n, total=len(ids))
        small.stop()
        qwen.start(8192, 1, 'judge')
        prepared = {'questions': {qid: {'question': q['question'], 'answer': q['answer'], 'question_type': q['type']} for qid, q in questions.items()}}
        labels = u2.judge(ctx, qwen, prepared, answers, ARMS)
        ctx.write('answers.json', answers)
        ctx.write('labels.json', labels)
        report(questions, rows, answers, labels)
        ctx.status('complete')
    except BaseException as error:
        ctx.status('stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        small.stop()
        qwen.stop()


def report(questions, rows, answers, labels):
    ids = sorted(questions)
    correct = {a: {q: labels[a][q]['label'] for q in ids} for a in ARMS}
    types = sorted({questions[q]['type'] for q in ids})
    lines = ['# SCR1：規模での想起の改良の選別（探索、dev の問 60、5,000 セッション、答える役は E2B）', '',
             f'自前の選び方が Memory.retrieve と一致: {sum(rows[q]["selectionMatchesRetrieve"] for q in ids)}/60。'
             f'ROLE がアシスタント側に振り分けた問: {sum(rows[q]["routed"] == "assistant" for q in ids)}。', '',
             '|条件|正答率|A との差|95%区間|勝/負|想起に答え|全根拠|根拠の発言の割合|トークン|棄権率|',
             '|---|---:|---:|---|---|---:|---:|---:|---:|---:|']
    out = {}
    for a in ARMS:
        d = [correct[a][q] - correct['A'][q] for q in ids]
        ci = g1m.interval(d, ids)
        s = g1m.sign_test(d)
        abstain = np.mean([g1r.answer_value(answers[a][q], questions[q]['answer'])['abstention'] for q in ids])
        m = {k: float(np.mean([rows[q][a][k] for q in ids])) for k in ('hit', 'allSessions', 'evidenceShare', 'tokens')}
        out[a] = {'correct': float(np.mean(list(correct[a].values()))), 'difference': float(np.mean(d)), 'ci95': ci, 'sign': s, **m,
                  'abstain': float(abstain), 'by_type': {t: float(np.mean([correct[a][q] for q in ids if questions[q]['type'] == t])) for t in types}}
        lines.append(f'|{a}|{out[a]["correct"]:.3f}|{out[a]["difference"]:+.3f}|[{ci[0]:+.3f}, {ci[1]:+.3f}]|{s["wins"]}/{s["losses"]}|'
                     f'{m["hit"]:.3f}|{m["allSessions"]:.3f}|{m["evidenceShare"]:.3f}|{m["tokens"]:.0f}|{abstain:.3f}|')
    lines += ['', '|種類|' + '|'.join(types) + '|', '|---|' + '---:|'*len(types)]
    for a in ARMS:
        lines.append(f'|{a}|' + '|'.join(f'{out[a]["by_type"][t]:.2f}' for t in types) + '|')
    (u2.RESULTS/'scr1-report.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (u2.RESULTS/'scr1-report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    run()
