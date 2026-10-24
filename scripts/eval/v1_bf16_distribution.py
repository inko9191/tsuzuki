#!/usr/bin/env python3
"""bf16 stress test for EmbeddingGemma 300m (v1): cosine distribution over thousands of real inputs.

Inputs: every query and document of LoCoMo and the three JMTEB retrieval sets, with and without the retrieval
prompts, exactly as scripts/eval/retrieval_public.py embeds them (its cache must hold the egemma300m fp32/bf16
and st300m vectors; run it with --backends st,egemma-v1-fp32,egemma-v1-bf16 first).
  engine   egemma-server bf16 (MPP) vs egemma-server fp32, per vector
  st       sentence-transformers bf16 vs sentence-transformers fp32 (MPS): the model's own bf16 behaviour
Statistics per set and overall: n, min, 0.1/1/5 percentiles, median, counts below 0.999 / 0.99.
Criterion (the bf16 acceptance): the engine's min, 1st and 5th percentiles are each no lower than ST bf16's,
i.e. the engine adds no error beyond what bf16 itself costs this model.
Output: results/eval/v1_bf16_distribution.json (+ .md).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
import retrieval_public as rp  # noqa: E402

SETS = ['locomo', 'jmteb-jagovfaqs_22k', 'jmteb-nlp_journal_title_abs', 'jmteb-nlp_journal_abs_intro']
TASKS = {'none': ('none', 'none'), 'prompt': ('query', 'document')}


def load(pattern: str) -> np.ndarray:
    paths = sorted(glob.glob(str(rp.CACHE / pattern)))
    if len(paths) != 1:
        raise FileNotFoundError(f'expected one cache file for {pattern}, found {len(paths)}')
    return np.load(paths[0])


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64); b = b.astype(np.float64)
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def stats(c: np.ndarray) -> dict:
    return {'n': int(len(c)), 'min': float(c.min()), 'p0.1': float(np.percentile(c, 0.1)), 'p1': float(np.percentile(c, 1)),
            'p5': float(np.percentile(c, 5)), 'median': float(np.median(c)),
            'below_0.999': int((c < 0.999).sum()), 'below_0.99': int((c < 0.99).sum())}


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer
    st16 = None
    report = {'reference': {'engine': 'egemma-server fp32', 'st': 'sentence-transformers fp32 MPS'}, 'sets': {}}
    pooled = {'engine': [], 'st': []}
    worst = []
    for name in SETS:
        ds = rp.read_json(rp.DATA / f'{name}.json')
        for variant, (qt, dt) in TASKS.items():
            for role, task, items in (('q', qt, ds['queries']), ('d', dt, ds['corpus'])):
                texts = rp.v1_inputs([x['text'] for x in items], rp.V1_PROMPTS[task])
                e32 = load(f'{name}-egemma300m-fp32-{task}-{role}-*.npy')
                e16 = load(f'{name}-egemma300m-bf16-{task}-{role}-*.npy')
                s32 = load(f'{name}-st300m-{task}-{role}-*.npy')

                def st_bf16(batch: list[str]) -> np.ndarray:
                    nonlocal st16
                    if st16 is None:
                        st16 = SentenceTransformer(str(rp.V1_ST_DIR), device='mps').to(torch.bfloat16)
                    return st16.encode(batch, batch_size=16, normalize_embeddings=True).astype(np.float32)
                s16 = rp.cached(f'{name}-st300m-bf16-{task}-{role}', texts, st_bf16)
                ce, cs = cos(e16, e32), cos(s16, s32)
                key = f'{name} {variant} {"queries" if role == "q" else "documents"}'
                report['sets'][key] = {'engine': stats(ce), 'st': stats(cs)}
                pooled['engine'].append(ce); pooled['st'].append(cs)
                for i in np.argsort(ce)[:3]:
                    worst.append({'set': key, 'index': int(i), 'id': items[i]['id'], 'chars': len(texts[i]),
                                  'engine_cos': float(ce[i]), 'st_bf16_cos': float(cs[i])})
                print(f'{key:52s} engine min {ce.min():.5f} p1 {np.percentile(ce, 1):.6f} | st bf16 min {cs.min():.5f} '
                      f'p1 {np.percentile(cs, 1):.6f}', flush=True)
    overall = {k: stats(np.concatenate(v)) for k, v in pooled.items()}
    report['overall'] = overall
    report['worst_engine'] = sorted(worst, key=lambda w: w['engine_cos'])[:10]
    checks = {q: overall['engine'][q] >= overall['st'][q] for q in ('min', 'p1', 'p5')}
    report['criterion'] = {'rule': 'engine bf16 min/p1/p5 >= sentence-transformers bf16 min/p1/p5 (pooled)', 'checks': checks,
                           'pass': all(checks.values())}
    out = ROOT / 'results/eval/v1_bf16_distribution.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# v1 bf16 stress test: cosine to fp32 over LoCoMo and JMTEB (queries and documents, with and without prompts)', '',
             '| set | engine bf16 vs fp32: min / p1 / p5 / median | <0.999 | ST bf16 vs fp32: min / p1 / p5 / median | <0.999 |',
             '|---|---|---:|---|---:|']
    fmt = lambda s: f'{s["min"]:.5f} / {s["p1"]:.6f} / {s["p5"]:.6f} / {s["median"]:.6f}'  # noqa: E731
    for key, s in list(report['sets'].items()) + [('**all**', overall)]:
        lines.append(f'| {key} ({s["engine"]["n"]}) | {fmt(s["engine"])} | {s["engine"]["below_0.999"]} | {fmt(s["st"])} | {s["st"]["below_0.999"]} |')
    lines += ['', f'Criterion (engine no worse than the model\'s own bf16 at min, p1, p5): {checks} -> '
              f'{"PASS" if report["criterion"]["pass"] else "FAIL"}']
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'overall': overall, 'criterion': report['criterion']}, indent=1))
    return 0 if report['criterion']['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
