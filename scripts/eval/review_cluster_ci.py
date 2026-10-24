#!/usr/bin/env python3
"""Review item 3: LoCoMo confidence intervals with questions resampled vs whole conversations resampled.

LoCoMo has 10 conversations; questions of one conversation share its turns and speakers, so they are not
independent. For each paired comparison this reports the mean difference with two 95% percentile bootstrap
intervals (10000 resamples, seed 0): over questions (as reported so far), and over conversations (each resample
draws 10 conversations with replacement and averages over all their questions).

Read-only inputs:
  results/eval/retrieval_public.json         all 1535 questions (pre-split public benchmark)
  build/eval/memory/*-dev-rankings.json      memory stage 1, development split (written by locomo_memory.py)
  build/eval/memory/stage2-dev-rows.json     memory stage 2, development split (written by locomo_memory_stages.py)
Output: results/eval/review/locomo_cluster_ci.json (+ .md). The memory-evaluation files are not modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESAMPLES, SEED = 10000, 0


def conversation(qid: str) -> str:
    return qid.split('/')[0]


def intervals(diff: np.ndarray, groups: list[str]) -> dict:
    rng = np.random.default_rng(SEED)
    q = diff[rng.integers(0, len(diff), (RESAMPLES, len(diff)))].mean(axis=1)
    names = sorted(set(groups))
    sums = np.array([diff[[g == n for g in groups]].sum() for n in names])
    counts = np.array([sum(g == n for g in groups) for n in names])
    rng = np.random.default_rng(SEED)
    pick = rng.integers(0, len(names), (RESAMPLES, len(names)))
    c = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    per_conv = sums / counts
    return {'mean_diff': float(diff.mean()), 'questions': len(diff), 'conversations': len(names),
            'question_ci95': [float(np.percentile(q, 2.5)), float(np.percentile(q, 97.5))],
            'conversation_ci95': [float(np.percentile(c, 2.5)), float(np.percentile(c, 97.5))],
            'conversations_favouring': int((per_conv > 0).sum()), 'conversations_against': int((per_conv < 0).sum())}


def main() -> int:
    out = {'method': __doc__, 'comparisons': []}
    pub = json.loads((ROOT / 'results/eval/retrieval_public.json').read_text())['datasets']['locomo']
    ds = json.loads((ROOT / 'build/eval/data/locomo.json').read_text())
    groups = [conversation(q['id']) for q in ds['queries']]
    pq = pub['per_query']
    for label, a, b in [('v1 prompt − v2 prompt (768)', 'v1-prompt-768', 'v2-prompt-768'),
                        ('v1 none − v2 none (768)', 'v1-none-768', 'v2-none-768'),
                        ('v1 prompt − v1 none (768)', 'v1-prompt-768', 'v1-none-768'),
                        ('v2 prompt − v2 none (768)', 'v2-prompt-768', 'v2-none-768'),
                        ('v1 ST none − prime ONNX q8 none (768)', 'v1-none-768', 'v1onnx-none-768')]:
        for m in ('ndcg@10', 'recall@10'):
            diff = np.asarray(pq[a][m]) - np.asarray(pq[b][m])
            out['comparisons'].append({'source': 'retrieval_public (all 1535)', 'comparison': label, 'metric': m} | intervals(diff, groups))

    stage1 = {}
    for name in ('v1-fp32-prompt', 'v2-fp16-prompt', 'prime-onnx-q8-none'):
        rows = json.loads((ROOT / f'build/eval/memory/{name}-dev-rankings.json').read_text())
        stage1[name] = {r['id']: r for r in rows}
    ids = [r for r in stage1['v1-fp32-prompt']]
    g1 = [conversation(i) for i in ids]
    for label, a, b in [('stage 1: v1 − v2', 'v1-fp32-prompt', 'v2-fp16-prompt'), ('stage 1: v1 − prime', 'v1-fp32-prompt', 'prime-onnx-q8-none')]:
        diff = np.asarray([stage1[a][i]['recall@10'] - stage1[b][i]['recall@10'] for i in ids])
        out['comparisons'].append({'source': 'memory stage 1 (dev 1233)', 'comparison': label, 'metric': 'recall@10'} | intervals(diff, g1))

    rows2 = json.loads((ROOT / 'build/eval/memory/stage2-dev-rows.json').read_text())
    base = {r['id']: r for r in rows2['t1/dense/orig/none']}
    ids = list(base)
    g2 = [conversation(i) for i in ids]
    for cfg in ('p2/dense/orig/none', 'p3/dense/orig/none', 'c3/dense/orig/none'):
        other = {r['id']: r for r in rows2[cfg]}
        diff = np.asarray([other[i]['recall@T512'] - base[i]['recall@T512'] for i in ids])
        out['comparisons'].append({'source': 'memory stage 2 (dev 1233)', 'comparison': f'{cfg.split("/")[0]} − t1 (stage 2 selection metric)',
                                   'metric': 'recall@T512'} | intervals(diff, g2))

    path = ROOT / 'results/eval/review/locomo_cluster_ci.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + '\n')
    fmt = lambda ci: f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'  # noqa: E731
    lines = ['# LoCoMo: question-level vs conversation-level bootstrap 95% CI', '',
             '| source | comparison | metric | mean diff | CI over questions | CI over conversations | conversations + / − |',
             '|---|---|---|---:|---|---|---|']
    for c in out['comparisons']:
        lines.append(f"| {c['source']} | {c['comparison']} | {c['metric']} | {c['mean_diff']:+.4f} | {fmt(c['question_ci95'])} | "
                     f"{fmt(c['conversation_ci95'])} | {c['conversations_favouring']} / {c['conversations_against']} |")
    lines += ['', 'With 10 conversations the conversation-level percentile bootstrap is coarse; it is shown to expose the '
              'dependence between questions, not as a precise interval.']
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
