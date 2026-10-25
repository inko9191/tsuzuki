#!/usr/bin/env python3
"""Tables for scripts/eval/scale/diagnose.mjs (exploratory): per size, the share of (question, age) rows whose recall
holds an answer message, for each variant; paired differences against the shipped recall with a question-level
bootstrap (10000, seed 0; the three ages of a question are averaged first).

    .venv/bin/python -I scripts/eval/scale/diagnose_report.py [build/scale/diagnose-real.json]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'build/scale/diagnose-real.json'
d = json.loads(src.read_text())
rows = d['rows']

VARIANTS = [('base', '現状（予算 2,048）', lambda r: r['base']['hit']),
            ('b4096', '予算 4,096', lambda r: r['budget4096']['hit']),
            ('b8192', '予算 8,192', lambda r: r['budget8192']['hit']),
            ('date1', '日付で絞る ±1 日（正しい範囲を知っている上限）', lambda r: r['date1']['hit']),
            ('date7', '日付で絞る ±7 日（同）', lambda r: r['date7']['hit']),
            ('date30', '日付で絞る ±30 日（同）', lambda r: r['date30']['hit']),
            ('hybrid', 'BM25 と密な検索を RRF で融合', lambda r: r['hybrid']['hit']),
            ('bm25', 'BM25 だけ', lambda r: r['bm25']['hit']),
            ('rerank30', '並べ直しの候補 30 窓に答えがある（完璧な並べ直しの上限）', lambda r: r['inRerankCandidates'])]
RANKS = [10, 20, 50, 100, 200, 500]


def per_question(rs, f):
    by = defaultdict(list)
    for r in rs:
        by[r['qid']].append(float(f(r)))
    qids = sorted(by)
    return qids, np.array([np.mean(by[q]) for q in qids])


def paired(rs, f, g):
    qids, a = per_question(rs, f)
    _, b = per_question(rs, g)
    diff = a - b
    rng = np.random.default_rng(0)
    boots = rng.integers(0, len(diff), (10000, len(diff)))
    means = diff[boots].mean(axis=1)
    return float(diff.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], int((diff > 0).sum()), int((diff < 0).sum())


def median(values):
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


sizes = sorted({r['size'] for r in rows})
out = ['# 規模で落ちる精度の診断（探索、LLM なし）', '',
       f'入力: `{src.relative_to(ROOT) if src.is_relative_to(ROOT) else src}`。規模試験（`results/eval/memory/scale-report.md`）と同じ 60 問 × 3 つの古さ、'
       '同じストアの作り方（窓だけ、事実なし、rerank なし、実ベクトル）。値は「答えの発言が想起に 1 つ以上入った割合」。',
       f'再現の確認: 自前の選び方が Memory.retrieve と一致 {d["checks"]["retrieve"]}/{d["checks"]["rows"]}、'
       f'規模試験の結果（accuracy-real.json）と一致 {d["checks"]["reference"]}/{d["checks"]["rows"]}。', '',
       '## 方法ごとの値', '', '|方法|' + '|'.join(f'{s:,}' for s in sizes) + '|', '|---|' + '---:|' * len(sizes)]
for key, name, f in VARIANTS:
    out.append(f'|{name}|' + '|'.join(f'{np.mean([float(f(r)) for r in rows if r["size"] == s]):.3f}' for s in sizes) + '|')

out += ['', '## 答えの発言の順位（密な検索の全順位）', '', '|セッション|' + '|'.join(f'上位{k}' for k in RANKS) + '|順位の中央値|予算 2,048 に入る発言数|融合後の順位の中央値|',
        '|---:|' + '---:|' * (len(RANKS) + 3)]
for s in sizes:
    rs = [r for r in rows if r['size'] == s]
    out.append(f'|{s:,}|' + '|'.join(f'{np.mean([r["recallAt"][str(k)] for r in rs]):.3f}' for k in RANKS)
               + f'|{median([r["messageRank"] for r in rs])}|{np.mean([r["base"]["selected"] for r in rs]):.1f}|{median([r["hybridMessageRank"] for r in rs])}|')

out += ['', '## 現状との差（質問単位の対応ありブートストラップ、10000 回、seed 0）', '',
        '|セッション|方法|差|95% 区間|良くなった問/悪くなった問|', '|---:|---|---:|---|---|']
for s in sizes:
    rs = [r for r in rows if r['size'] == s]
    for key, name, f in VARIANTS[1:]:
        diff, ci, up, down = paired(rs, f, VARIANTS[0][2])
        out.append(f'|{s:,}|{name}|{diff:+.3f}|[{ci[0]:+.3f}, {ci[1]:+.3f}]|{up}/{down}|')

big = max(sizes)
types = sorted({r['type'] for r in rows})
out += ['', f'## 種類別（{big:,} セッション）', '', '|方法|' + '|'.join(types) + '|', '|---|' + '---:|' * len(types)]
for key, name, f in VARIANTS:
    out.append(f'|{name}|' + '|'.join(f'{np.mean([float(f(r)) for r in rows if r["size"] == big and r["type"] == t]):.2f}' for t in types) + '|')
date_c = {s: {k: np.mean([r[f'date{k}']['candidates'] for r in rows if r['size'] == s]) for k in d['dateDays']} for s in sizes}
out += ['', '日付で絞ったときの候補の窓の数（平均）: ' + '、'.join(f'{s:,} セッション: ' + ' / '.join(f'±{k} 日 {date_c[s][k]:.0f}' for k in d['dateDays']) for s in sizes) + '。']
print('\n'.join(out))
