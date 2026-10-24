#!/usr/bin/env python3
"""Review 3a: the v1 vs v2 conclusion over every model-card query prompt (no per-task prompt chosen after the fact).

Fixed and committed before measuring (2026-10-12, second review). Neither model card nor the EmbeddingGemma paper
(arXiv 2509.20354 section 4.2: "using the prompt instructions detailed in the model card", bfloat16 weights, 512-token
context for most tasks) states which prompt was used per dataset, and the fact-checking prompt was already seen to help
ArguAna (exploratory diagnostic). So instead of choosing a prompt per task, every retrieval-type query prompt of the
model cards is measured for both models on all 7 datasets:
  queries    task: search result | query:   /   task: question answering | query:   /   task: fact checking | query:
  documents  model-card form, as in condition B: "title: {title} | text: {text}" (title: none when absent)
Grid G (main): v1 and v2 with sentence-transformers fp32 MPS, 2048 tokens (documents are condition B's, reused).
  Reported per dataset and model for each prompt; and v1 - v2 (paired bootstrap 95% CI) for
    B       the mapping declared before (official_conditions.py)
    same    the same prompt for both models, for each of the three prompts
    best    each model at its own best prompt on that dataset (selection favours both models alike)
    worst   v1 at its worst prompt vs v2 at its best (the most v2-favourable comparison)
Paper protocol P (v1 only): sentence-transformers with bfloat16 weights on MPS, 512 tokens, the three prompts;
  compared with the published v1 scores (JaGovFaqs, NLP Journal x2, ArguAna). Datasets without a published score
  are reported too.
Same metrics as before (nDCG@10, Recall@10; ArguAna excludes the query's own document; LoCoMo per conversation).
Output: results/eval/review/official_prompts_grid.json (+ .md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
import official_conditions as oc  # noqa: E402
import retrieval_public as rp  # noqa: E402

PROMPTS = {'search': oc.SEARCH, 'qa': oc.QA, 'fact': oc.FACT}
OUT = ROOT / 'results/eval/review/official_prompts_grid.json'
B_KEY = {oc.SEARCH: 'search', oc.QA: 'qa', oc.FACT: 'fact'}


def st_v1(dtype: str, max_len: int):
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(rp.V1_ST_DIR), device='mps')
    model = model.to(torch.bfloat16) if dtype == 'bf16' else model.float()
    model.max_seq_length = max_len

    def embed(texts):
        out = []
        for start in range(0, len(texts), 512):
            out.append(model.encode(texts[start:start + 512], batch_size=16, normalize_embeddings=True).astype(np.float32))
            torch.mps.empty_cache()
        v = np.concatenate(out)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return model, embed


def main() -> int:
    import gc
    import torch
    names = rp.DATASETS
    sets = {n: rp.read_json(rp.DATA / f'{n}.json') for n in names}
    docs = {n: oc.titled_corpus(n, sets[n]) for n in names}
    queries = {n: {k: [p + q['text'] for q in sets[n]['queries']] for k, p in PROMPTS.items()} for n in names}
    vec: dict = {n: {} for n in names}

    # Grid G: documents are condition B's (same strings, same caches); new queries only.
    model, embed = st_v1('fp32', 2048)
    for n in names:
        vec[n]['v1-d'] = rp.cached(f'review-{n}-v1st-B-d', docs[n], embed)
        for k in PROMPTS:
            vec[n][f'v1-{k}-q'] = rp.cached(f'review-grid-{n}-v1st-{k}-q', queries[n][k], embed)
    del model, embed; gc.collect(); torch.mps.empty_cache()
    v2 = oc.SentenceTransformersV2()
    for n in names:
        vec[n]['v2-d'] = rp.cached(f'review-{n}-v2st-B-d', docs[n], v2.embed)
        for k in PROMPTS:
            vec[n][f'v2-{k}-q'] = rp.cached(f'review-grid-{n}-v2st-{k}-q', queries[n][k], v2.embed)
    del v2; gc.collect(); torch.mps.empty_cache()
    # Paper protocol P for v1: bf16 weights, 512 tokens.
    model, embed = st_v1('bf16', 512)
    for n in names:
        vec[n]['p-d'] = rp.cached(f'review-grid-{n}-v1st-bf16-512-d', docs[n], embed)
        for k in PROMPTS:
            vec[n][f'p-{k}-q'] = rp.cached(f'review-grid-{n}-v1st-bf16-512-{k}-q', queries[n][k], embed)
    del model, embed; gc.collect(); torch.mps.empty_cache()

    report = {'protocol': __doc__, 'prompts': PROMPTS, 'published_v1': oc.PUBLISHED_V1, 'datasets': {}}
    for n in names:
        rows, scores = {}, {}
        for tag in ('v1', 'v2', 'p'):
            for k in PROMPTS:
                r = rp.evaluate(sets[n], vec[n][f'{tag}-{k}-q'], vec[n][f'{tag}-d'])
                rows[(tag, k)] = r
                scores[f'{tag}-{k}'] = {m: float(np.mean([x[m] for x in r])) for m in rp.METRICS}
        best = {t: max(PROMPTS, key=lambda k: scores[f'{t}-{k}']['ndcg@10']) for t in ('v1', 'v2')}
        worst_v1 = min(PROMPTS, key=lambda k: scores[f'v1-{k}']['ndcg@10'])
        b = B_KEY[oc.QUERY_PROMPT_B[n]]

        def diff(k1: str, k2: str) -> dict:
            return {m: rp.paired([x[m] for x in rows[('v2', k2)]], [x[m] for x in rows[('v1', k1)]]) for m in rp.METRICS}
        entry = {'scores': scores, 'best_prompt': best, 'worst_prompt_v1': worst_v1, 'B_prompt': b,
                 'v1_minus_v2': {'B': diff(b, b), **{f'same-{k}': diff(k, k) for k in PROMPTS},
                                 'best': diff(best['v1'], best['v2']), 'worst': diff(worst_v1, best['v2'])}}
        if n in oc.PUBLISHED_V1:
            entry['published_v1'] = oc.PUBLISHED_V1[n]
        report['datasets'][n] = entry
        print(f"{n:28s} " + '  '.join(f"{t}-{k} {scores[f'{t}-{k}']['ndcg@10']:.4f}" for t in ('v1', 'v2', 'p') for k in PROMPTS), flush=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    fmt = lambda d: f"{d['mean_diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"  # noqa: E731
    lines = ['# Review 3a: every model-card query prompt for both models (nDCG@10, 768 dims, model-card documents)', '',
             'G: sentence-transformers fp32, 2048 tokens. P: v1 with bf16 weights and 512 tokens (the paper\'s protocol).', '',
             '| dataset | v1 search | v1 qa | v1 fact | v2 search | v2 qa | v2 fact | P v1 search | P v1 qa | P v1 fact | published v1 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n, e in report['datasets'].items():
        s = e['scores']
        pub = f"{e['published_v1']['ndcg@10']:.4f}" if 'published_v1' in e else '—'
        lines.append(f"| {n} | " + ' | '.join(f"{s[f'{t}-{k}']['ndcg@10']:.4f}" for t in ('v1', 'v2', 'p') for k in PROMPTS) + f' | {pub} |')
    lines += ['', '## v1 − v2 (nDCG@10, paired bootstrap 95% CI)', '',
              '| dataset | B (declared before) | same: search | same: qa | same: fact | best vs best | v1 worst vs v2 best |',
              '|---|---|---|---|---|---|---|']
    for n, e in report['datasets'].items():
        d = e['v1_minus_v2']
        lines.append(f"| {n} | {fmt(d['B']['ndcg@10'])} ({e['B_prompt']}) | {fmt(d['same-search']['ndcg@10'])} | {fmt(d['same-qa']['ndcg@10'])} | "
                     f"{fmt(d['same-fact']['ndcg@10'])} | {fmt(d['best']['ndcg@10'])} ({e['best_prompt']['v1']}/{e['best_prompt']['v2']}) | "
                     f"{fmt(d['worst']['ndcg@10'])} ({e['worst_prompt_v1']}/{e['best_prompt']['v2']}) |")
    OUT.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
