#!/usr/bin/env python3
"""Retrieval quality on public benchmarks: egemma-server vs dsh-prime-memory's default local embedding.

Datasets: build/eval/data/*.json from scripts/eval/fetch_data.py (JMTEB x3, MTEB x3, LoCoMo).
Variants (dense retrieval, cosine = dot product of unit vectors, exact search), "<model>-<prompt>-<dims>":
  v1  EmbeddingGemma 300m, the model of dsh-prime-memory 0.22.1's default local embedding. The plugin runs the
      8-bit quantized ONNX export (onnx-community/embeddinggemma-300m-ONNX@5090578, onnx/model_quantized.onnx,
      dtype 'q8', transformers.js + onnxruntime-node on CPU, one text at a time), which is too slow for every
      corpus here. So v1 is measured with sentence-transformers (google/embeddinggemma-300m, fp32, MPS) on all
      datasets, and the plugin's own ONNX worker (dist/embedding-worker.cjs) is run on ONNX_CHECK_SETS only to
      confirm that the two agree (variants v1onnx-*). Both get the plugin's input: the text (prompt included)
      cut at 5000 characters. v1-none is the plugin default (raw text). v1-prompt prepends the EmbeddingGemma
      300m retrieval prompts of its model card: query "task: search result | query: ", document
      "title: none | text: ".
  v2  egemma-server (EmbeddingGemma 2) fp16 matmul=auto. v2-none: no prompt (what prime-memory sends to a
      remote model). v2-prompt: the checkpoint's Retrieval-query / Retrieval-document prompts
      (config_sentence_transformers.json; the same strings as v1's), selected with the ``task`` field.
  256 the same vectors truncated to 256 dims and renormalized (Matryoshka; both models support it); for v2
      checked once against the server's own ``dimensions`` truncation.
Documents longer than the server's 2048-token limit are cut at a token boundary to fit with the document
prompt; both v2 variants embed the same cut text. v1 (both paths) also stops at 2048 tokens (model limit).
Metrics per query: nDCG@10 (graded qrels, linear gain as in pytrec_eval / MTEB) and Recall@10
(relevant documents in the top 10 / all relevant documents). ArguAna excludes the query's own document
from the candidates (BEIR/MTEB convention: its query ids are corpus ids). LoCoMo ranks only the turns of
the question's own conversation. Every variant is compared with v1-none-768 per query (paired bootstrap 95% CI).
Engine check (backends egemma-v1-fp32 / egemma-v1-bf16): egemma-server serving EmbeddingGemma 300m embeds exactly
the ST inputs (prompt + text, 5000-character cut, then cut to 2048 tokens at a token boundary as ST truncates),
and its scores are compared with the ST v1 scores (variants v1e<dtype>-*).

Embeddings are cached in build/eval/cache/ (keyed by the sha256 of the texts), so an interrupted run resumes.
Output: results/eval/retrieval_public.json and .md.
Usage: .venv/bin/python scripts/eval/retrieval_public.py [--datasets a,b] [--backends egemma,st,onnx,egemma-v1-fp32,egemma-v1-bf16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/stage2'))
sys.path.insert(0, str(ROOT / 'server'))
from common import EGEMMA_RETRIEVAL_PORT, read_json, write_json  # noqa: E402

DATA = ROOT / 'build/eval/data'
CACHE = ROOT / 'build/eval/cache'
OUT = ROOT / 'results/eval/retrieval_public.json'
DATASETS = ['jmteb-jagovfaqs_22k', 'jmteb-nlp_journal_title_abs', 'jmteb-nlp_journal_abs_intro',
            'mteb-scifact', 'mteb-nfcorpus', 'mteb-arguana', 'locomo']
ONNX_CHECK_SETS = ['jmteb-nlp_journal_title_abs', 'locomo']
IGNORE_IDENTICAL_IDS = {'mteb-arguana'}
COLUMNS = ['v1-prompt', 'v1-none', 'v2-prompt', 'v2-none']
VARIANTS = [f'{c}-{d}' for d in (768, 256) for c in COLUMNS]
CHECK_VARIANTS = [f'v1onnx-{p}-{d}' for d in (768, 256) for p in ('prompt', 'none')]
ENGINE_V1_DTYPES = ('fp32', 'bf16')
ENGINE_V1_VARIANTS = [f'v1e{t}-{p}-{d}' for t in ENGINE_V1_DTYPES for d in (768, 256) for p in ('prompt', 'none')]
BASELINE = 'v1-none-768'
# v1's ONNX export has no prompt table; these are its model card's retrieval prompts.
V1_PROMPTS = {'query': 'task: search result | query: ', 'document': 'title: none | text: ', 'none': ''}
V1_MAX_CHARS = 5000  # dsh-prime-memory embedding.maxInputChars default (the worker cuts its input there)
V1_ST_DIR = ROOT / 'models/embeddinggemma-300m'
V1_ST_REVISION = '57c266a740f537b4dc058e1b0cda161fd15afa75'
MAX_TOKENS = 2048
K = 10


# -- metrics (pure) --------------------------------------------------------------
def query_metrics(ranking: list[str], qrels: dict[str, int], k: int = K) -> dict:
    relevant = {d: g for d, g in qrels.items() if g > 0}
    gains = [relevant.get(d, 0) for d in ranking[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return {'ndcg@10': float(dcg / idcg) if idcg else 0.0,
            'recall@10': sum(1 for g in gains if g > 0) / len(relevant) if relevant else 0.0}


def top_k(query_vectors: np.ndarray, doc_vectors: np.ndarray, k: int, exclude: list[int | None] | None = None) -> np.ndarray:
    """Indices of the k best documents per query (descending score, stable for ties)."""
    out = np.empty((len(query_vectors), min(k, len(doc_vectors))), dtype=np.int64)
    for start in range(0, len(query_vectors), 256):
        scores = query_vectors[start:start + 256] @ doc_vectors.T
        if exclude is not None:
            for row, j in enumerate(exclude[start:start + 256]):
                if j is not None:
                    scores[row, j] = -np.inf
        kk = out.shape[1]
        part = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind='stable')
        out[start:start + 256] = np.take_along_axis(part, order, axis=1)
    return out


def evaluate(dataset: dict, query_vectors: np.ndarray, doc_vectors: np.ndarray) -> list[dict]:
    corpus, queries = dataset['corpus'], dataset['queries']
    doc_ids = [d['id'] for d in corpus]
    groups: dict[str | None, list[int]] = {}
    for i, d in enumerate(corpus):
        groups.setdefault(d.get('group'), []).append(i)
    per_query: list[dict | None] = [None] * len(queries)
    by_group: dict[str | None, list[int]] = {}
    for qi, q in enumerate(queries):
        by_group.setdefault(q.get('group'), []).append(qi)
    for group, qis in by_group.items():
        cand = np.asarray(groups[group])
        position = {doc_ids[j]: n for n, j in enumerate(cand)}
        exclude = [position.get(queries[qi]['id']) for qi in qis] if dataset['name'] in IGNORE_IDENTICAL_IDS else None
        best = top_k(query_vectors[qis], doc_vectors[cand], K, exclude)
        for row, qi in enumerate(qis):
            ranking = [doc_ids[cand[j]] for j in best[row]]
            per_query[qi] = query_metrics(ranking, dataset['qrels'][queries[qi]['id']])
    return per_query  # type: ignore[return-value]


def paired(a: list[float], b: list[float], seed: int = 0, resamples: int = 10000) -> dict:
    """Mean of b - a over queries with a percentile bootstrap 95% CI."""
    diff = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    means = np.concatenate([diff[rng.integers(0, len(diff), (1000, len(diff)))].mean(axis=1) for _ in range(resamples // 1000)])
    return {'mean_diff': float(diff.mean()), 'ci95': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def truncate(vectors: np.ndarray, dims: int) -> np.ndarray:
    v = vectors[:, :dims]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# -- embedding --------------------------------------------------------------------
def cached(name: str, texts: list[str], compute) -> np.ndarray:
    digest = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()[:16]
    path = CACHE / f'{name}-{digest}.npy'
    if path.exists():
        return np.load(path)
    started = time.time()
    vectors = np.asarray(compute(texts), np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    print(f'    {name}: {len(texts)} texts in {time.time() - started:.1f}s', flush=True)
    return vectors


def fit_to_limit(tokenizer, texts: list[str], prompt: str) -> tuple[list[str], int]:
    """Cut texts so that prompt + text + BOS/EOS fits MAX_TOKENS (token boundary of the plain text)."""
    out, cut = [], 0
    for text in texts:
        if len(tokenizer.encode([text], prompt)[0]) <= MAX_TOKENS:
            out.append(text)
            continue
        offsets = tokenizer.tokenizer.encode(text, add_special_tokens=False).offsets
        keep = MAX_TOKENS - len(tokenizer.encode([''], prompt)[0]) - 8
        while keep > 0:
            candidate = text[:offsets[keep - 1][1]]
            if len(tokenizer.encode([candidate], prompt)[0]) <= MAX_TOKENS:
                break
            keep -= 16
        out.append(candidate)
        cut += 1
    return out, cut


class Egemma:
    def __init__(self, args):
        from bench import HttpBackend
        import retrieval
        spec = retrieval.default_specs(argparse.Namespace(port=args.port, model_dir=args.model_dir))['egemma']
        self.backend = HttpBackend(spec, ROOT / 'results/eval/logs/egemma_retrieval.log', 600)
        self.info = self.backend.wait_ready(300)

    def embed(self, texts: list[str], task: str, dims: int = 768) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), 32):
            reply = self.backend._request('POST', '/v1/embeddings', {'model': 'embeddinggemma-2', 'input': texts[start:start + 32],
                                                                    'task': task, 'dimensions': dims})
            vectors += [d['embedding'] for d in sorted(reply['data'], key=lambda d: d['index'])]
        return np.asarray(vectors, np.float32)

    def stop(self):
        self.backend.stop()


def fit_like_st(tokenizer, texts: list[str]) -> list[str]:
    """Cut texts (prompt included) the way ST truncates them: BOS + the first 2046 tokens + EOS."""
    out = []
    for text in texts:
        if len(tokenizer.encode([text], None)[0]) <= MAX_TOKENS:
            out.append(text)
            continue
        offsets = tokenizer.tokenizer.encode(text, add_special_tokens=False).offsets
        keep = MAX_TOKENS - 2
        while len(tokenizer.encode([text[:offsets[keep - 1][1]]], None)[0]) > MAX_TOKENS:
            keep -= 1
        out.append(text[:offsets[keep - 1][1]])
    return out


class EgemmaV1:
    """egemma-server serving EmbeddingGemma 300m; inputs arrive with the prompt already prepended (task none)."""

    def __init__(self, args, dtype: str):
        from bench import HttpBackend
        spec = {'kind': 'http', 'port': args.port, 'label': f'egemma-server v1 {dtype}', 'env': {'PYTHONPATH': str(ROOT / 'server')},
                'command': [sys.executable, '-m', 'egemma_server', '--port', str(args.port), '--dtype', dtype, '--matmul', 'auto',
                            '--model-dir', str(args.v1_dir), '--model-name', 'embeddinggemma-300m']}
        self.backend = HttpBackend(spec, ROOT / f'results/eval/logs/egemma_v1_{dtype}_retrieval.log', 600)
        self.info = self.backend.wait_ready(300)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), 32):
            reply = self.backend._request('POST', '/v1/embeddings', {'model': 'embeddinggemma-300m', 'input': texts[start:start + 32],
                                                                    'task': 'none'})
            vectors += [d['embedding'] for d in sorted(reply['data'], key=lambda d: d['index'])]
        return np.asarray(vectors, np.float32)

    def stop(self):
        self.backend.stop()


class SentenceTransformersV1:
    """v1 quality path: google/embeddinggemma-300m with sentence-transformers, fp32 on MPS."""

    def __init__(self, model_dir: Path):
        import sentence_transformers
        import torch
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(model_dir), device='mps').float()
        self.info = {'path': 'sentence-transformers', 'model': f'google/embeddinggemma-300m@{V1_ST_REVISION[:7]}',
                     'dtype': 'float32', 'device': 'mps', 'max_seq_length': self.model.max_seq_length,
                     'sentence_transformers': sentence_transformers.__version__, 'torch': torch.__version__}

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True)

    def stop(self):
        pass


class Onnx:
    def __init__(self, args):
        from bench import JsonlBackend, default_specs
        spec = default_specs(args.model_dir, sys.executable)['onnx-prime']
        self.backend = JsonlBackend(spec, ROOT / 'results/eval/logs/onnx_retrieval.log', 1800)
        self.info = self.backend.wait_ready(300)

    def embed(self, texts: list[str], prefix: str = '') -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), 16):
            vectors += self.backend.call({'op': 'embed', 'texts': texts[start:start + 16], 'prefix': prefix,
                                          'return_vectors': True})['vectors']
        return np.asarray(vectors, np.float32)

    def stop(self):
        self.backend.stop()


# -- report ------------------------------------------------------------------------
METRICS = ('ndcg@10', 'recall@10')
TITLES = {'v1-prompt': 'v1あり (ST)', 'v1-none': 'v1なし (ST)', 'v2-prompt': 'v2あり', 'v2-none': 'v2なし'}


def summarize(report: dict) -> None:
    groups = {'JMTEB (ja) mean': [d for d in report['datasets'] if d.startswith('jmteb-')],
              'MTEB (en) mean': [d for d in report['datasets'] if d.startswith('mteb-')]}
    report['means'] = {}
    for label, names in groups.items():
        names = [n for n in names if all(v in report['datasets'][n].get('variants', {}) for v in VARIANTS)]
        if names:
            report['means'][label] = {v: {m: float(np.mean([report['datasets'][n]['variants'][v][m] for n in names]))
                                          for m in METRICS} for v in VARIANTS} | {'datasets': names}


def write_markdown(report: dict, path: Path) -> None:
    lines = ['# Retrieval quality on public benchmarks (dsh plugin evaluation)', '',
             '- **v1** = EmbeddingGemma 300m (the model of dsh-prime-memory 0.22.1\'s default local embedding). '
             '**Measured with sentence-transformers** (google/embeddinggemma-300m, fp32, MPS) on every dataset; the plugin\'s own '
             'ONNX q8 path was run on ' + ', '.join(ONNX_CHECK_SETS) + ' only, to check that it matches (last table). '
             'Input as the plugin sends it (cut at 5000 characters). v1なし = the plugin default.',
             '- The plugin\'s ONNX path returns the mean-pooled last hidden state **without the model\'s 2_Dense/3_Dense '
             'projections** (transformers.js feature-extraction pipeline; see results/eval/v1_onnx_pooling_check.json), so its '
             'vectors differ from the model\'s sentence embedding. At 768 dims retrieval quality is close (check table), so v1 at '
             '768 uses the ST results. **v1 at 256 dims is the official model, not the plugin**: truncating the plugin\'s vectors '
             'loses much more (check table), but the plugin always stores 768 dims.',
             '- **v2** = EmbeddingGemma 2 through egemma-server (fp16, MPP) — this engine (vectors equal sentence-transformers: '
             'results/eval/v2_reference_check.json).',
             '- あり = retrieval prompts (query "task: search result | query: ", document "title: none | text: "; the same strings for v1 and v2), '
             'なし = raw text. Δ = variant minus v1なし (768) with a paired bootstrap 95% CI over queries.', '']
    for metric in METRICS:
        for dims in (768, 256):
            cols = [f'{c}-{dims}' for c in COLUMNS]
            lines += [f'## {metric}, {dims} dims', '',
                      '| dataset | queries / docs | ' + ' | '.join(TITLES[c] for c in COLUMNS) + ' |',
                      '|---|---:|' + '---:|' * len(cols)]
            for name, entry in report['datasets'].items():
                v = entry.get('variants', {})
                lines.append(f'| {name} | {entry["queries"]} / {entry["corpus"]} | ' +
                             ' | '.join(f'{v[x][metric]:.4f}' if x in v else '—' for x in cols) + ' |')
            for label, entry in report.get('means', {}).items():
                lines.append(f'| **{label}** | {len(entry["datasets"])} sets | ' +
                             ' | '.join(f'**{entry[x][metric]:.4f}**' for x in cols) + ' |')
            lines.append('')
        others = [v for v in VARIANTS if v != BASELINE]
        lines += [f'### Δ{metric} vs v1なし (ST, 768) [95% CI]', '', '| dataset | ' + ' | '.join(others) + ' |',
                  '|---|' + '---|' * len(others)]
        for name, entry in report['datasets'].items():
            d = entry.get('vs_baseline', {})
            lines.append(f'| {name} | ' + ' | '.join(
                f'{d[x][metric]["mean_diff"]:+.4f} [{d[x][metric]["ci95"][0]:+.4f}, {d[x][metric]["ci95"][1]:+.4f}]'
                if x in d else '—' for x in others) + ' |')
        lines.append('')
    lines += ['## v1 measurement check: plugin ONNX q8 (CPU) vs sentence-transformers fp32 (MPS)', '',
              'Same inputs (prompt + text cut at 5000 characters). The vector cosine is near 0 because the plugin skips the '
              'Dense projections (the vectors live in a different space); the comparison that matters is the retrieval quality.', '',
              '| dataset | variant | ONNX q8 nDCG@10 | ST nDCG@10 | Δ (ONNX − ST) [95% CI] | ONNX Recall@10 | ST Recall@10 | vector cos ONNX·ST mean / min |',
              '|---|---|---:|---:|---|---:|---:|---|']
    for name, entry in report['datasets'].items():
        for v, check in entry.get('onnx_check', {}).items():
            d = check['ndcg@10']
            lines.append(f'| {name} | {v} | {check["onnx"]["ndcg@10"]:.4f} | {check["st"]["ndcg@10"]:.4f} | '
                         f'{d["mean_diff"]:+.4f} [{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}] | {check["onnx"]["recall@10"]:.4f} | '
                         f'{check["st"]["recall@10"]:.4f} | {check.get("cos_mean", float("nan")):.4f} / {check.get("cos_min", float("nan")):.4f} |')
    if any(e.get('engine_v1_check') for e in report['datasets'].values()):
        lines += ['', '## Engine v1 check: egemma-server (EmbeddingGemma 300m) vs sentence-transformers fp32', '',
                  'Same input strings. Δ = engine − ST with a paired bootstrap 95% CI; vector cosine over all queries and documents.', '',
                  '| dataset | engine | variant | engine nDCG@10 | ST nDCG@10 | ΔnDCG@10 [95% CI] | ΔRecall@10 | vector cos min / mean |',
                  '|---|---|---|---:|---:|---|---|---|']
        for name, entry in report['datasets'].items():
            for v, c in entry.get('engine_v1_check', {}).items():
                d, r = c['ndcg@10'], c['recall@10']
                lines.append(f'| {name} | {v.split("-")[0][3:]} | {"-".join(v.split("-")[1:])} | {c["engine"]["ndcg@10"]:.4f} | '
                             f'{c["st"]["ndcg@10"]:.4f} | {d["mean_diff"]:+.4f} [{d["ci95"][0]:+.4f}, {d["ci95"][1]:+.4f}] | '
                             f'{r["mean_diff"]:+.4f} | {c.get("cos_min", float("nan")):.6f} / {c.get("cos_mean", float("nan")):.6f} |')
    notes = [f'- {n}: {e.get("v2_cut_docs", 0)} document(s) cut to the 2048-token limit for v2; '
             f'{e.get("v1_cut_docs", 0)} cut at 5000 characters for v1.'
             for n, e in report['datasets'].items() if e.get('v2_cut_docs') or e.get('v1_cut_docs')]
    if notes:
        lines += ['', '## Long documents', ''] + notes
    path.write_text('\n'.join(lines) + '\n')


def v1_inputs(texts: list[str], prompt: str) -> list[str]:
    """What the plugin worker embeds: prompt + text, cut at the plugin's 5000 characters."""
    return [(prompt + t)[:V1_MAX_CHARS] for t in texts]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--backends', default='egemma,st,onnx')
    ap.add_argument('--model-dir', type=Path, default=ROOT / 'models/embeddinggemma-2')
    ap.add_argument('--v1-dir', type=Path, default=V1_ST_DIR)
    ap.add_argument('--port', type=int, default=EGEMMA_RETRIEVAL_PORT)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    (ROOT / 'results/eval/logs').mkdir(parents=True, exist_ok=True)
    from egemma_server.tokenizer import PromptTokenizer
    tokenizer = PromptTokenizer(args.model_dir)
    doc_prompt, query_prompt = tokenizer.prompt_for('document'), tokenizer.prompt_for('query')
    report = read_json(args.out) or {}
    report.setdefault('datasets', {})
    report['settings'] = {'k': K, 'max_tokens_v2': MAX_TOKENS, 'max_chars_v1': V1_MAX_CHARS, 'variants': VARIANTS,
                          'baseline': BASELINE, 'onnx_check_sets': ONNX_CHECK_SETS,
                          'prompts': {'v2': {'query': query_prompt, 'document': doc_prompt}, 'v1': V1_PROMPTS},
                          'ignore_identical_ids': sorted(IGNORE_IDENTICAL_IDS),
                          'v1_method': 'sentence-transformers fp32 MPS on all datasets; plugin ONNX q8 CPU on onnx_check_sets'}
    names = args.datasets.split(',')
    sets = {n: read_json(DATA / f'{n}.json') for n in names}
    backends = args.backends.split(',')
    vectors: dict[str, dict[str, np.ndarray]] = {n: {} for n in names}
    timing = report.setdefault('embedding_seconds', {})
    sides = {'none': ('none', 'none'), 'prompt': ('query', 'document')}

    if 'egemma' in backends:
        egemma = Egemma(args)
        try:
            report['v2_info'] = egemma.info
            probe = [d['text'] for d in sets[names[0]]['corpus'][:8]]
            full, server_256 = egemma.embed(probe, 'document'), egemma.embed(probe, 'document', 256)
            report['matryoshka_check_max_abs_diff'] = float(np.abs(truncate(full, 256) - server_256).max())
            for n in names:
                ds = sets[n]
                docs, cut = fit_to_limit(tokenizer, [d['text'] for d in ds['corpus']], doc_prompt)
                queries, _ = fit_to_limit(tokenizer, [q['text'] for q in ds['queries']], query_prompt)
                report['datasets'].setdefault(n, {})['v2_cut_docs'] = cut
                print(f'{n}: v2 (egemma)', flush=True)
                for variant, (qt, dt) in sides.items():
                    started = time.time()
                    vectors[n][f'v2-{variant}-q'] = cached(f'{n}-egemma-{qt}-q', queries, lambda t, qt=qt: egemma.embed(t, qt))
                    vectors[n][f'v2-{variant}-d'] = cached(f'{n}-egemma-{dt}-d', docs, lambda t, dt=dt: egemma.embed(t, dt))
                    timing.setdefault(n, {})[f'v2-{variant}'] = time.time() - started
        finally:
            egemma.stop()
    # v1 through two paths: (tag, backend class, cache prefix, datasets).
    v1_paths = []
    if 'st' in backends:
        v1_paths.append(('v1', lambda: SentenceTransformersV1(args.v1_dir), 'st300m', names))
    if 'onnx' in backends:
        v1_paths.append(('v1onnx', lambda: Onnx(args), 'onnx', [n for n in names if n in ONNX_CHECK_SETS]))
    for tag, make, prefix, todo in v1_paths:
        if not todo:
            continue
        backend = make()
        try:
            report[f'{tag}_info'] = backend.info
            for n in todo:
                ds = sets[n]
                report['datasets'].setdefault(n, {})['v1_cut_docs'] = sum(len(V1_PROMPTS['document'] + d['text']) > V1_MAX_CHARS
                                                                         for d in ds['corpus'])
                print(f'{n}: {tag}', flush=True)
                for variant, (qt, dt) in sides.items():
                    started = time.time()
                    q_in = v1_inputs([q['text'] for q in ds['queries']], V1_PROMPTS[qt])
                    d_in = v1_inputs([d['text'] for d in ds['corpus']], V1_PROMPTS[dt])
                    # The ONNX worker gets the raw text and the prompt as its prefix, exactly as the plugin cuts it.
                    if prefix == 'onnx':
                        q_raw, d_raw = [q['text'] for q in ds['queries']], [d['text'] for d in ds['corpus']]
                        qtag, dtag = ('onnx-q', 'onnx-d') if variant == 'none' else ('onnx-query-q', 'onnx-document-d')
                        vectors[n][f'{tag}-{variant}-q'] = cached(f'{n}-{qtag}', q_raw, lambda t, p=V1_PROMPTS[qt]: backend.embed(t, p))
                        vectors[n][f'{tag}-{variant}-d'] = cached(f'{n}-{dtag}', d_raw, lambda t, p=V1_PROMPTS[dt]: backend.embed(t, p))
                    else:
                        vectors[n][f'{tag}-{variant}-q'] = cached(f'{n}-{prefix}-{qt}-q', q_in, backend.embed)
                        vectors[n][f'{tag}-{variant}-d'] = cached(f'{n}-{prefix}-{dt}-d', d_in, backend.embed)
                    timing.setdefault(n, {})[f'{tag}-{variant}'] = time.time() - started
        finally:
            backend.stop()

    engine_dtypes = [b.rsplit('-', 1)[1] for b in backends if b.startswith('egemma-v1-')]
    if engine_dtypes:
        from egemma_server.tokenizer import PromptTokenizer as V1Tokenizer
        v1_tokenizer = V1Tokenizer(args.v1_dir)
    for dtype in engine_dtypes:
        backend = EgemmaV1(args, dtype)
        try:
            report[f'v1e{dtype}_info'] = backend.info
            for n in names:
                ds = sets[n]
                print(f'{n}: v1 on egemma ({dtype})', flush=True)
                for variant, (qt, dt) in sides.items():
                    # Same strings as the ST path; ST truncates to 2048 tokens, the server refuses longer inputs.
                    q_in = fit_like_st(v1_tokenizer, v1_inputs([q['text'] for q in ds['queries']], V1_PROMPTS[qt]))
                    d_in = fit_like_st(v1_tokenizer, v1_inputs([d['text'] for d in ds['corpus']], V1_PROMPTS[dt]))
                    started = time.time()
                    vectors[n][f'v1e{dtype}-{variant}-q'] = cached(f'{n}-egemma300m-{dtype}-{qt}-q', q_in, backend.embed)
                    vectors[n][f'v1e{dtype}-{variant}-d'] = cached(f'{n}-egemma300m-{dtype}-{dt}-d', d_in, backend.embed)
                    timing.setdefault(n, {})[f'v1e{dtype}-{variant}'] = time.time() - started
        finally:
            backend.stop()

    for n in names:
        ds, vec = sets[n], vectors[n]
        entry = report['datasets'].setdefault(n, {})
        entry.update(lang=ds['lang'], source=ds['source'], queries=len(ds['queries']), corpus=len(ds['corpus']))
        variants = entry.setdefault('variants', {})
        per_query = entry.setdefault('per_query', {})
        for column in COLUMNS + ['v1onnx-prompt', 'v1onnx-none'] + [f'v1e{t}-{p}' for t in ENGINE_V1_DTYPES for p in ('prompt', 'none')]:
            if f'{column}-q' not in vec:
                continue
            for dims in (768, 256):
                rows = evaluate(ds, truncate(vec[f'{column}-q'], dims), truncate(vec[f'{column}-d'], dims))
                v = f'{column}-{dims}'
                variants[v] = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
                per_query[v] = {m: [round(r[m], 6) for r in rows] for m in METRICS}
        if BASELINE in per_query:
            entry['vs_baseline'] = {v: {m: paired(per_query[BASELINE][m], per_query[v][m]) for m in METRICS}
                                    for v in VARIANTS if v in per_query and v != BASELINE}
        check = {}
        for v in CHECK_VARIANTS:
            st = v.replace('v1onnx', 'v1')
            if v in per_query and st in per_query:
                c = {'onnx': variants[v], 'st': variants[st],
                     **{m: paired(per_query[st][m], per_query[v][m]) for m in METRICS}}
                kind, dims = v.split('-')[1], int(v.split('-')[2])
                if f'v1onnx-{kind}-d' in vec and f'v1-{kind}-d' in vec:
                    cos = np.concatenate([np.sum(truncate(vec[f'v1onnx-{kind}-{s}'], dims) * truncate(vec[f'v1-{kind}-{s}'], dims), axis=1)
                                          for s in ('q', 'd')])
                    c.update(cos_mean=float(cos.mean()), cos_min=float(cos.min()))
                check[v] = c
        if check:
            entry['onnx_check'] = check
        engine_check = {}
        for v in ENGINE_V1_VARIANTS:
            st = 'v1-' + v.split('-', 1)[1]
            if v in per_query and st in per_query:
                c = {'engine': variants[v], 'st': variants[st], **{m: paired(per_query[st][m], per_query[v][m]) for m in METRICS}}
                kind, dims, tag = v.split('-')[1], int(v.split('-')[2]), v.split('-')[0]
                if f'{tag}-{kind}-d' in vec and f'v1-{kind}-d' in vec:
                    cos = np.concatenate([np.sum(truncate(vec[f'{tag}-{kind}-{s}'], dims) * truncate(vec[f'v1-{kind}-{s}'], dims), axis=1)
                                          for s in ('q', 'd')])
                    c.update(cos_mean=float(cos.mean()), cos_min=float(cos.min()))
                engine_check[v] = c
        if engine_check:
            entry['engine_v1_check'] = engine_check
        print(f'{n:28s} ' + '  '.join(f'{v} {variants[v]["ndcg@10"]:.4f}' for v in VARIANTS + CHECK_VARIANTS + ENGINE_V1_VARIANTS
                                       if v in variants), flush=True)
    summarize(report)
    write_json(args.out, report)
    write_markdown(report, args.out.with_suffix('.md'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
