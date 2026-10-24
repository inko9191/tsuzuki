#!/usr/bin/env python3
"""Review item 1: v1 and v2 under the model card's evaluation conditions, next to the earlier generic setup.

Fixed and committed before any of these numbers were computed (2026-10-12, docs/REVIEW_PACKET.md section 8).

Condition A "generic" (the earlier results, results/eval/retrieval_public.json): query "task: search result |
query: ", document "title: none | text: " + "title text" joined; v1 also cut at 5000 characters. This is how MTEB 2.x
runs these models through its SentenceTransformer wrapper (the checkpoint prompts "Retrieval-query"/"Retrieval-document";
corpus title and text joined), and it reproduces the published v1 MTEB scores of the three JMTEB tasks.

Condition B "model card" (both model cards: "Documents with a real title should be formatted as `title: {title} |
text: {content}`", plus the use-case query prompts):
  documents   "title: {title} | text: {text}" when the corpus row has a non-empty title, else "title: none | text: {text}"
              (titles exist for SciFact, NFCorpus, ArguAna; JMTEB and LoCoMo have none)
  queries     per task, from the model card's use-case table, chosen by what the queries are:
              SciFact (claims to verify)              task: fact checking | query:
              JaGovFaqs, LoCoMo (questions)          task: question answering | query:
              NFCorpus, ArguAna, NLP Journal x2       task: search result | query:
  length      2048 tokens for both models (each model's own tokenizer, ST-style truncation); no 5000-character cut
  backends    v1 sentence-transformers fp32 MPS; v2 sentence-transformers fp32 MPS; v2 egemma-server fp16 (default)
Metrics as before (nDCG@10, Recall@10, ArguAna excludes the query's own document, LoCoMo per conversation).

ArguAna diagnostic (v1 only; explains the gap to the published 0.71535, mteb 1.34.7, model revision 64614b0, dataset
revision c22ab2a5 whose files are byte-identical to ours): the 8 combinations of query prompt {search result, fact
checking} x documents {A, B} x own-document exclusion {yes, no}. Exploratory: not used to choose any reported setting.

Check of the v2 input path: sentence-transformers encode(text, prompt=P) (the chat-template route) vs encode(P + text)
on 64 texts must agree (cos), since the egemma server renders prompt + text.

Output: results/eval/review/official_conditions.json (+ .md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
import retrieval_public as rp  # noqa: E402

SEARCH, FACT, QA = 'task: search result | query: ', 'task: fact checking | query: ', 'task: question answering | query: '
QUERY_PROMPT_B = {'mteb-scifact': FACT, 'jmteb-jagovfaqs_22k': QA, 'locomo': QA, 'mteb-nfcorpus': SEARCH,
                  'mteb-arguana': SEARCH, 'jmteb-nlp_journal_title_abs': SEARCH, 'jmteb-nlp_journal_abs_intro': SEARCH}
RAW_MTEB = {'mteb-scifact': 'mteb__scifact', 'mteb-nfcorpus': 'mteb__nfcorpus', 'mteb-arguana': 'mteb__arguana'}
PUBLISHED_V1 = {  # embeddings-benchmark/results @ dfc29a2, google__embeddinggemma-300m, ndcg_at_10 / recall_at_10
    'jmteb-jagovfaqs_22k': {'ndcg@10': 0.72094, 'recall@10': 0.86084, 'task': 'JaGovFaqsRetrieval', 'revision': '57c266a', 'mteb': '2.2.0'},
    'jmteb-nlp_journal_title_abs': {'ndcg@10': 0.96275, 'recall@10': 0.9902, 'task': 'NLPJournalTitleAbsRetrieval.V2', 'revision': '57c266a', 'mteb': '2.2.0'},
    'jmteb-nlp_journal_abs_intro': {'ndcg@10': 0.97571, 'recall@10': 0.99608, 'task': 'NLPJournalAbsIntroRetrieval.V2', 'revision': '57c266a', 'mteb': '2.2.0'},
    'mteb-arguana': {'ndcg@10': 0.71535, 'recall@10': 0.9431, 'task': 'ArguAna', 'revision': '64614b0', 'mteb': '1.34.7'},
}
OUT = ROOT / 'results/eval/review/official_conditions.json'


def titled_corpus(name: str, ds: dict) -> list[str]:
    """Condition B documents, in the order of ds['corpus']."""
    if name not in RAW_MTEB:
        return ['title: none | text: ' + d['text'] for d in ds['corpus']]
    rows = {}
    with open(ROOT / f'build/eval/raw/{RAW_MTEB[name]}/corpus.jsonl') as f:
        for line in f:
            r = json.loads(line)
            rows[str(r['_id'])] = r
    out = []
    for d in ds['corpus']:
        r = rows[d['id']]
        title = (r.get('title') or '').strip()
        out.append(f'title: {title if title else "none"} | text: {r["text"]}')
    return out


class SentenceTransformersV2:
    def __init__(self):
        import sentence_transformers
        import torch
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(ROOT / 'models/embeddinggemma-2'), device='mps', model_kwargs={'dtype': torch.float32},
                                         config_kwargs={'vision_config': None, 'audio_config': None})
        self.info = {'path': 'sentence-transformers', 'model': 'google/embeddinggemma-2@914f7f8', 'dtype': 'float32', 'device': 'mps',
                     'sentence_transformers': sentence_transformers.__version__}

    def embed(self, texts: list[str], prompt: str | None = None) -> np.ndarray:
        # MPS keeps per-shape allocations across many differently sized batches (the first run died at 86 GiB of
        # "other allocations" on JaGovFaqs); encode in chunks and release the cache in between. Results unchanged.
        import torch
        out = []
        for start in range(0, len(texts), 512):
            out.append(self.model.encode(texts[start:start + 512], batch_size=8, normalize_embeddings=True,
                                         convert_to_numpy=True, prompt=prompt))
            torch.mps.empty_cache()
        return np.concatenate(out)

    def stop(self):
        pass


def mean(rows: list[dict], m: str) -> float:
    return float(np.mean([r[m] for r in rows]))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=rp.EGEMMA_RETRIEVAL_PORT)
    args = ap.parse_args()
    from egemma_server.tokenizer import PromptTokenizer
    v1_tok, v2_tok = PromptTokenizer(rp.V1_ST_DIR), PromptTokenizer(ROOT / 'models/embeddinggemma-2')
    names = rp.DATASETS
    sets = {n: rp.read_json(rp.DATA / f'{n}.json') for n in names}
    inputs = {n: {'q': [QUERY_PROMPT_B[n] + q['text'] for q in sets[n]['queries']], 'd': titled_corpus(n, sets[n])} for n in names}
    vec: dict[str, dict[str, np.ndarray]] = {n: {} for n in names}
    report = {'protocol': __doc__, 'published_v1': PUBLISHED_V1, 'datasets': {}}

    st1 = rp.SentenceTransformersV1(rp.V1_ST_DIR)
    report['v1_info'] = st1.info
    for n in names:  # ST truncates at its max_seq_length (2048) itself
        vec[n]['v1-st-B-q'] = rp.cached(f'review-{n}-v1st-B-q', inputs[n]['q'], st1.embed)
        vec[n]['v1-st-B-d'] = rp.cached(f'review-{n}-v1st-B-d', inputs[n]['d'], st1.embed)
    # ArguAna diagnostic inputs
    ds = sets['mteb-arguana']
    diag_q = {'search': [SEARCH + q['text'] for q in ds['queries']], 'fact': [FACT + q['text'] for q in ds['queries']]}
    diag_d = {'A': rp.v1_inputs([d['text'] for d in ds['corpus']], 'title: none | text: '), 'B': inputs['mteb-arguana']['d']}
    for k, texts in diag_q.items():
        vec['mteb-arguana'][f'diag-q-{k}'] = rp.cached(f'review-arguana-v1st-diag-q-{k}', texts, st1.embed)
    for k, texts in diag_d.items():
        vec['mteb-arguana'][f'diag-d-{k}'] = rp.cached(f'review-arguana-v1st-diag-d-{k}', texts, st1.embed)
    del st1
    import gc
    import torch
    gc.collect()
    torch.mps.empty_cache()

    st2 = SentenceTransformersV2()
    report['v2_st_info'] = st2.info
    probe = [d['text'] for d in sets['locomo']['corpus'][:64]]
    a, b = st2.embed(probe, prompt=QA), st2.embed([QA + t for t in probe])
    report['v2_prompt_route_check'] = {'texts': len(probe), 'min_cos': float((a * b).sum(1).min())}
    for n in names:
        vec[n]['v2-st-B-q'] = rp.cached(f'review-{n}-v2st-B-q', inputs[n]['q'], st2.embed)
        vec[n]['v2-st-B-d'] = rp.cached(f'review-{n}-v2st-B-d', inputs[n]['d'], st2.embed)
    del st2

    egemma = rp.Egemma(argparse.Namespace(port=args.port, model_dir=ROOT / 'models/embeddinggemma-2', v1_dir=rp.V1_ST_DIR))
    try:
        report['v2_egemma_info'] = egemma.info
        for n in names:
            q = rp.fit_like_st(v2_tok, inputs[n]['q'])
            d = rp.fit_like_st(v2_tok, inputs[n]['d'])
            vec[n]['v2-eg-B-q'] = rp.cached(f'review-{n}-v2eg-B-q', q, lambda t: egemma.embed(t, 'none'))
            vec[n]['v2-eg-B-d'] = rp.cached(f'review-{n}-v2eg-B-d', d, lambda t: egemma.embed(t, 'none'))
    finally:
        egemma.stop()

    previous = rp.read_json(rp.OUT)
    for n in names:
        entry = {'query_prompt_B': QUERY_PROMPT_B[n]}
        for tag in ('v1-st-B', 'v2-st-B', 'v2-eg-B'):
            rows = rp.evaluate(sets[n], vec[n][f'{tag}-q'], vec[n][f'{tag}-d'])
            entry[tag] = {m: mean(rows, m) for m in rp.METRICS}
            entry[tag + '_per_query'] = {m: [r[m] for r in rows] for m in rp.METRICS}
        old = previous['datasets'][n]['variants']
        entry['A'] = {'v1-st': old['v1-prompt-768'], 'v2-eg': old['v2-prompt-768']}
        entry['v2_egemma_vs_st_B'] = {m: rp.paired(entry['v2-st-B_per_query'][m], entry['v2-eg-B_per_query'][m]) for m in rp.METRICS}
        entry['v1_vs_v2_B'] = {m: rp.paired(entry['v2-eg-B_per_query'][m], entry['v1-st-B_per_query'][m]) for m in rp.METRICS}
        cos = np.concatenate([(vec[n][f'v2-eg-B-{s}'] * vec[n][f'v2-st-B-{s}']).sum(1) for s in ('q', 'd')])
        entry['v2_egemma_vs_st_vector_cos'] = {'min': float(cos.min()), 'mean': float(cos.mean())}
        if n in PUBLISHED_V1:
            entry['published_v1'] = PUBLISHED_V1[n]
        report['datasets'][n] = entry
        print(f"{n:28s} A: v1 {entry['A']['v1-st']['ndcg@10']:.4f} v2 {entry['A']['v2-eg']['ndcg@10']:.4f} | "
              f"B: v1 {entry['v1-st-B']['ndcg@10']:.4f} v2st {entry['v2-st-B']['ndcg@10']:.4f} v2eg {entry['v2-eg-B']['ndcg@10']:.4f}", flush=True)

    diag = {}
    for qk in ('search', 'fact'):
        for dk in ('A', 'B'):
            for exclude in (True, False):
                d = dict(sets['mteb-arguana']) | {'name': 'mteb-arguana' if exclude else 'arguana-keep-own'}
                rows = rp.evaluate(d, vec['mteb-arguana'][f'diag-q-{qk}'], vec['mteb-arguana'][f'diag-d-{dk}'])
                diag[f'query={qk} docs={dk} exclude_own={exclude}'] = {m: mean(rows, m) for m in rp.METRICS}
    report['arguana_diagnostic_v1'] = diag
    for k, v in diag.items():
        print(f'  ArguAna v1 {k:45s} nDCG@10 {v["ndcg@10"]:.4f}  R@10 {v["recall@10"]:.4f}')

    for n in report['datasets']:
        for tag in ('v1-st-B', 'v2-st-B', 'v2-eg-B'):
            report['datasets'][n].pop(tag + '_per_query')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    lines = ['# Review 1: model-card conditions (B) next to the generic setup (A), nDCG@10, 768 dims', '',
             '| dataset | published v1 | A v1 (ST) | A v2 (egemma) | B v1 (ST) | B v2 (ST) | B v2 (egemma) | B v1 − v2 [95% CI] |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for n, e in report['datasets'].items():
        pub = f"{e['published_v1']['ndcg@10']:.4f}" if 'published_v1' in e else 'none published'
        d = e['v1_vs_v2_B']['ndcg@10']
        lines.append(f"| {n} | {pub} | {e['A']['v1-st']['ndcg@10']:.4f} | {e['A']['v2-eg']['ndcg@10']:.4f} | {e['v1-st-B']['ndcg@10']:.4f} | "
                     f"{e['v2-st-B']['ndcg@10']:.4f} | {e['v2-eg-B']['ndcg@10']:.4f} | {d['mean_diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    lines += ['', f"v2 prompt route check (ST prompt= vs prompt + text): min cos {report['v2_prompt_route_check']['min_cos']:.6f}", '',
              '## ArguAna diagnostic (v1, ST; published 0.71535 at model revision 64614b0)', '', '| setting | nDCG@10 | Recall@10 |', '|---|---:|---:|']
    lines += [f'| {k} | {v["ndcg@10"]:.4f} | {v["recall@10"]:.4f} |' for k, v in diag.items()]
    OUT.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
