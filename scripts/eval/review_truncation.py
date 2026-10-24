#!/usr/bin/env python3
"""Review item 2: does the asymmetric cut (v1 at 5000 characters, v2 at 2048 tokens) move the results?

Fixed before measuring (2026-10-12). Datasets with long documents: JMTEB NLP Journal abs->intro and JaGovFaqs.
Prompts as in the main table (generic retrieval prompts, condition A), 768 dims, v1 with sentence-transformers fp32
MPS, v2 with egemma-server fp16.
  current   v1: prompt + text cut at 5000 characters, then ST's 2048 tokens; v2: 2048 tokens   (results/eval/retrieval_public)
  tok2048   both models: each model's own tokenizer, BOS + first 2046 tokens + EOS (no character cut)
  tok512    both models: BOS + first 510 tokens + EOS (a common small budget, sensitivity check)
Reported per dataset and model: nDCG@10, Recall@10, the number of documents each cut changes, and the paired
difference to "current" (question bootstrap 95% CI); and v1 - v2 under each condition.
Output: results/eval/review/truncation.json (+ .md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
import retrieval_public as rp  # noqa: E402

SETS = ['jmteb-nlp_journal_abs_intro', 'jmteb-jagovfaqs_22k']
BUDGETS = {'tok2048': 2048, 'tok512': 512}


def cut(tokenizer, texts: list[str], budget: int) -> tuple[list[str], int]:
    """BOS + the first budget-2 tokens + EOS, cut at a token boundary of the text (as ST truncates)."""
    out, changed = [], 0
    for text in texts:
        if len(tokenizer.encode([text], None)[0]) <= budget:
            out.append(text)
            continue
        offsets = tokenizer.tokenizer.encode(text, add_special_tokens=False).offsets
        keep = budget - 2
        while len(tokenizer.encode([text[:offsets[keep - 1][1]]], None)[0]) > budget:
            keep -= 1
        out.append(text[:offsets[keep - 1][1]])
        changed += 1
    return out, changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=rp.EGEMMA_RETRIEVAL_PORT)
    args = ap.parse_args()
    from egemma_server.tokenizer import PromptTokenizer
    tok = {'v1': PromptTokenizer(rp.V1_ST_DIR), 'v2': PromptTokenizer(ROOT / 'models/embeddinggemma-2')}
    prompts = {'q': 'task: search result | query: ', 'd': 'title: none | text: '}
    sets = {n: rp.read_json(rp.DATA / f'{n}.json') for n in SETS}
    raw = {n: {'q': [prompts['q'] + q['text'] for q in sets[n]['queries']], 'd': [prompts['d'] + d['text'] for d in sets[n]['corpus']]} for n in SETS}
    inputs, changed = {}, {}
    for n in SETS:
        for model in ('v1', 'v2'):
            for cond, budget in BUDGETS.items():
                for side in ('q', 'd'):
                    texts, k = cut(tok[model], raw[n][side], budget)
                    inputs[(n, model, cond, side)] = texts
                    changed[(n, model, cond, side)] = k
            # "current": v1 5000 characters then ST's own 2048 tokens; v2 2048 tokens (same strings as the main table)
            for side in ('q', 'd'):
                if model == 'v1':
                    inputs[(n, model, 'current', side)] = [t[:rp.V1_MAX_CHARS] for t in raw[n][side]]
                    changed[(n, model, 'current', side)] = sum(len(t) > rp.V1_MAX_CHARS for t in raw[n][side]) + \
                        sum(len(tok['v1'].encode([t[:rp.V1_MAX_CHARS]], None)[0]) > 2048 for t in raw[n][side])
                else:
                    inputs[(n, model, 'current', side)] = rp.fit_to_limit(tok['v2'], [x['text'] for x in (sets[n]['queries'] if side == 'q' else sets[n]['corpus'])],
                                                                          prompts[side])[0]
                    changed[(n, model, 'current', side)] = sum(len(tok['v2'].encode([t], None)[0]) > 2048 for t in raw[n][side])
    vec = {}
    st = rp.SentenceTransformersV1(rp.V1_ST_DIR)
    for key, texts in inputs.items():
        n, model, cond, side = key
        if model == 'v1':
            # "current" uses the main table's cache names, so it is the very same vectors.
            name = f'{n}-st300m-{"query" if side == "q" else "document"}-{side}' if cond == 'current' else f'review-trunc-{n}-v1st-{cond}-{side}'
            vec[key] = rp.cached(name, texts, st.embed)
    del st
    eg = rp.Egemma(argparse.Namespace(port=args.port, model_dir=ROOT / 'models/embeddinggemma-2', v1_dir=rp.V1_ST_DIR))
    try:
        for key, texts in inputs.items():
            n, model, cond, side = key
            if model == 'v2':
                if cond == 'current':  # the main table's strings: prompt applied by the server (task field)
                    plain = rp.fit_to_limit(tok['v2'], [x['text'] for x in (sets[n]['queries'] if side == 'q' else sets[n]['corpus'])], prompts[side])[0]
                    task = 'query' if side == 'q' else 'document'
                    vec[key] = rp.cached(f'{n}-egemma-{task}-{side}', plain, lambda t, task=task: eg.embed(t, task))
                else:
                    vec[key] = rp.cached(f'review-trunc-{n}-v2eg-{cond}-{side}', texts, lambda t: eg.embed(t, 'none'))
    finally:
        eg.stop()

    report = {'protocol': __doc__, 'datasets': {}}
    lines = ['# Review 2: truncation conditions (generic prompts, 768 dims, nDCG@10)', '',
             '| dataset | model | condition | docs changed by the cut | nDCG@10 | Recall@10 | Δ nDCG@10 vs current [95% CI] |',
             '|---|---|---|---:|---:|---:|---|']
    for n in SETS:
        entry, per = {}, {}
        for model in ('v1', 'v2'):
            for cond in ('current', 'tok2048', 'tok512'):
                rows = rp.evaluate(sets[n], vec[(n, model, cond, 'q')], vec[(n, model, cond, 'd')])
                per[(model, cond)] = rows
                entry[f'{model}-{cond}'] = {m: float(np.mean([r[m] for r in rows])) for m in rp.METRICS} | \
                    {'docs_changed': changed[(n, model, cond, 'd')], 'queries_changed': changed[(n, model, cond, 'q')]}
            for cond in ('tok2048', 'tok512'):
                entry[f'{model}-{cond}']['vs_current'] = {m: rp.paired([r[m] for r in per[(model, 'current')]], [r[m] for r in per[(model, cond)]])
                                                         for m in rp.METRICS}
        for cond in ('current', 'tok2048', 'tok512'):
            entry[f'v1_minus_v2-{cond}'] = {m: rp.paired([r[m] for r in per[('v2', cond)]], [r[m] for r in per[('v1', cond)]]) for m in rp.METRICS}
        report['datasets'][n] = entry
        for model in ('v1', 'v2'):
            for cond in ('current', 'tok2048', 'tok512'):
                e = entry[f'{model}-{cond}']
                d = e.get('vs_current', {}).get('ndcg@10')
                lines.append(f"| {n} | {model} | {cond} | {e['docs_changed']} | {e['ndcg@10']:.4f} | {e['recall@10']:.4f} | "
                             + (f"{d['mean_diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d else '—') + ' |')
    lines += ['', '| dataset | condition | v1 − v2 nDCG@10 [95% CI] |', '|---|---|---|']
    for n, entry in report['datasets'].items():
        for cond in ('current', 'tok2048', 'tok512'):
            d = entry[f'v1_minus_v2-{cond}']['ndcg@10']
            lines.append(f"| {n} | {cond} | {d['mean_diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    out = ROOT / 'results/eval/review/truncation.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    out.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
