#!/usr/bin/env python3
"""Review 3b: reproduce the model-card MTEB(Code, v1) means (v1 68.76, v2 78.68; EmbeddingGemma 2 model card).

Fixed and committed before measuring (2026-10-12, second review).
  harness     mteb 2.24.9 (build/eval/venv-mteb, torch/transformers/sentence-transformers as in requirements-lock.txt),
              benchmark "MTEB(Code, v1)" (12 retrieval tasks, all subsets, main score nDCG@10), mteb.evaluate scoring
  embeddings  egemma-server: v1 EmbeddingGemma 300m fp32, v2 EmbeddingGemma 2 fp16 (both equal sentence-transformers,
              results/v1_validation_fp32.json and results/eval/review/official_conditions.json); 768 dims
  length      512 tokens per input including the prompt (the EmbeddingGemma paper's evaluation context for most tasks),
              cut at a token boundary of the text (BOS + 510 tokens + EOS)
  documents   mteb's corpus text ("title text" joined when a title exists) with the prompt below
  prompts     M "mteb default": what mteb's SentenceTransformer wrapper resolves from the checkpoint's own prompt
                table (task-name, then "Retrieval-query"/"Retrieval-document"): search result / title: none
              C "model card code": queries "task: code retrieval | query: ", documents "title: none | text: "
  score       Mean(Task): mean over the 12 tasks of each task's main score (mean over its subsets)
Reported: per task and the mean for {v1, v2} x {M, C}, next to the published means; and v1 - v2.
Numeric conditions differ from the published ones, and the table says so: the EmbeddingGemma paper evaluated v1 with
bfloat16 weights, the v2 model card reports full-precision results; here v2 runs fp16 and v1 fp32 on egemma-server.
Backend check (added before measuring, 2026-10-12): on the two smallest tasks (CodeTransOceanDL,
CodeTransOceanContest), both models also with
  st        sentence-transformers fp32 MPS on exactly the same strings (prompt + text cut at 512 tokens) -> isolates
            the embedding computation
  mteb-st   mteb's own SentenceTransformerEncoderWrapper on the local checkpoint, fp32 MPS, max_seq_length 512,
            condition M only (prompts resolved by mteb itself) -> the official harness path
and the score differences to egemma are reported.
Run with the mteb venv: build/eval/venv-mteb/bin/python scripts/eval/mteb_code.py [--tasks A,B] [--models v1,v2]
Output: results/eval/review/mteb_code.json (+ .md); mteb's own result files in build/eval/mteb_results/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'server'))
warnings.filterwarnings('ignore')
CACHE = ROOT / 'build/eval/cache/mteb_code'
BUDGET = 512
PUBLISHED = {'v1': 68.76, 'v2': 78.68}  # EmbeddingGemma 2 model card, "MTEB (code, v1) Mean(Task), NDCG@10"
MODELS = {
    'v1': {'dir': ROOT / 'models/embeddinggemma-300m', 'name': 'embeddinggemma-300m', 'dtype': 'fp32', 'revision': '57c266a'},
    'v2': {'dir': ROOT / 'models/embeddinggemma-2', 'name': 'embeddinggemma-2', 'dtype': 'fp16', 'revision': '914f7f8'},
}
CODE_PROMPTS = {'query': 'task: code retrieval | query: ', 'document': 'title: none | text: '}
OUT = ROOT / 'results/eval/review/mteb_code.json'


def cut(tokenizer, text: str) -> str:
    if len(tokenizer.encode([text], None)[0]) <= BUDGET:
        return text
    offsets = tokenizer.tokenizer.encode(text, add_special_tokens=False).offsets
    keep = BUDGET - 2
    while len(tokenizer.encode([text[:offsets[keep - 1][1]]], None)[0]) > BUDGET:
        keep -= 1
    return text[:offsets[keep - 1][1]]


class Daemon:
    def __init__(self, spec: dict, port: int):
        self.port = port
        self.proc = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-m', 'egemma_server', '--port', str(port), '--dtype', spec['dtype'],
                                      '--model-dir', str(spec['dir']), '--model-name', spec['name']],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=ROOT,
                                     env={'PYTHONPATH': str(ROOT / 'server'), 'HF_HUB_OFFLINE': '1', 'PATH': '/usr/bin:/bin'})
        event = json.loads(self.proc.stdout.readline() or '{}')
        if event.get('event') != 'ready':
            raise RuntimeError(f'egemma-server failed: {event}')
        self.name, self.ready = spec['name'], event

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), 64):
            body = json.dumps({'model': self.name, 'input': texts[start:start + 64], 'task': 'none'}).encode()
            req = urllib.request.Request(f'http://127.0.0.1:{self.port}/v1/embeddings', body, {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.load(r)['data']
            out += [d['embedding'] for d in sorted(data, key=lambda d: d['index'])]
        return np.asarray(out, np.float32)

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=60)


class STBackend:
    """sentence-transformers fp32 on MPS, same interface as Daemon.embed."""

    def __init__(self, spec: dict):
        import torch
        from sentence_transformers import SentenceTransformer
        kwargs = {'config_kwargs': {'vision_config': None, 'audio_config': None}} if spec['name'] == 'embeddinggemma-2' else {}
        self.model = SentenceTransformer(str(spec['dir']), device='mps', model_kwargs={'dtype': torch.float32}, **kwargs)
        self.model.max_seq_length = BUDGET

    def embed(self, texts: list[str]) -> np.ndarray:
        import torch
        out = []
        for start in range(0, len(texts), 256):
            out.append(self.model.encode(texts[start:start + 256], batch_size=16, normalize_embeddings=True))
            torch.mps.empty_cache()
        return np.concatenate(out).astype(np.float32)

    def close(self):
        pass


def make_encoder(key: str, condition: str, daemon, meta, stats: dict, backend: str = 'egemma'):
    from mteb.models.abs_encoder import AbsEncoder
    from mteb.models.sentence_transformer_wrapper import _resolve_prompt
    from mteb.types import PromptType
    from egemma_server.tokenizer import PromptTokenizer
    spec = MODELS[key]
    tokenizer = PromptTokenizer(spec['dir'])
    table = json.loads((spec['dir'] / 'config_sentence_transformers.json').read_text())['prompts']

    class EgemmaEncoder(AbsEncoder):
        mteb_model_meta = meta

        def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
            texts = [t for batch in inputs for t in batch['text']]
            if condition == 'M':
                prompt = _resolve_prompt(table, task_metadata, prompt_type) or ''
            else:
                prompt = CODE_PROMPTS['query' if prompt_type == PromptType.query else 'document']
            full = [cut(tokenizer, prompt + t) for t in texts]
            digest = hashlib.sha256(json.dumps(full, ensure_ascii=False).encode()).hexdigest()[:20]
            path = CACHE / f'{key}-{backend}-{digest}.npy' if backend != 'egemma' else CACHE / f'{key}-{digest}.npy'
            role = 'query' if prompt_type == PromptType.query else 'document'
            stats.setdefault(task_metadata.name, {})[f'{condition}-{role}-prompt'] = prompt
            if path.exists():
                return np.load(path)
            started = time.time()
            vectors = daemon.embed(full)
            CACHE.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors)
            print(f'    {key} {condition} {task_metadata.name}/{hf_subset} {role}: {len(full)} texts in {time.time() - started:.1f}s', flush=True)
            return vectors

    return EgemmaEncoder()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default='')
    ap.add_argument('--models', default='v1,v2')
    ap.add_argument('--port', type=int, default=8397)
    ap.add_argument('--backend', default='egemma', choices=['egemma', 'st', 'mteb-st'])
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    import mteb
    from mteb.cache import ResultCache
    tasks = list(mteb.get_benchmark('MTEB(Code, v1)').tasks)
    if args.tasks:
        tasks = [t for t in tasks if t.metadata.name in args.tasks.split(',')]
    base_meta = mteb.get_model_meta('google/embeddinggemma-300m')
    report = json.loads(OUT.read_text()) if OUT.exists() else {'protocol': __doc__, 'published_mean': PUBLISHED, 'runs': {}}
    stats: dict = report.setdefault('prompts_used', {})
    suffix = '' if args.backend == 'egemma' else f'-{args.backend}'
    for key in args.models.split(','):
        spec = MODELS[key]
        daemon = Daemon(spec, args.port) if args.backend == 'egemma' else STBackend(spec) if args.backend == 'st' else None
        try:
            for condition in (('M', 'C') if args.backend != 'mteb-st' else ('M',)):
                dtype = spec['dtype'] if args.backend == 'egemma' else 'fp32'
                meta = base_meta.model_copy(update={'name': f'egemma/{spec["name"]}-{dtype}-{condition}{suffix}', 'revision': spec['revision'],
                                                    'loader': None, 'max_tokens': BUDGET})
                if args.backend == 'mteb-st':
                    import torch
                    from sentence_transformers import SentenceTransformer
                    from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
                    kwargs = {'config_kwargs': {'vision_config': None, 'audio_config': None}} if key == 'v2' else {}
                    st = SentenceTransformer(str(spec['dir']), device='mps', model_kwargs={'dtype': torch.float32}, **kwargs)
                    st.max_seq_length = BUDGET
                    encoder = SentenceTransformerEncoderWrapper(st)
                    encoder.mteb_model_meta = meta
                else:
                    encoder = make_encoder(key, condition, daemon, meta, stats, args.backend)
                results = mteb.evaluate(encoder, tasks, cache=ResultCache(ROOT / 'build/eval/mteb_results'),
                                        overwrite_strategy='only-missing', encode_kwargs={'batch_size': 64 if args.backend != 'egemma' else 256,
                                                                                         'show_progress_bar': False})
                run = report['runs'].setdefault(f'{key}-{condition}{suffix}', {})
                for tr in results.task_results:
                    split = list(tr.scores)[0]
                    run[tr.task_name] = float(np.mean([s['main_score'] for s in tr.scores[split]])) * 100
                print(f'{key}-{condition}: ' + ', '.join(f'{k} {v:.2f}' for k, v in run.items()), flush=True)
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        finally:
            if daemon is not None:
                daemon.close()

    names = [t.metadata.name for t in mteb.get_benchmark('MTEB(Code, v1)').tasks]
    runs = report['runs']
    for r in runs.values():
        if all(n in r for n in names):
            r['Mean(Task)'] = float(np.mean([r[n] for n in names]))
    lines = ['# Review 3b: MTEB(Code, v1), nDCG@10 x 100 (512 tokens)', '',
             'Numerics: runs without a suffix use egemma-server with **v2 fp16** and **v1 fp32**; "-st" and "-mteb-st" use '
             'sentence-transformers fp32 on MPS. The published means were measured differently: the EmbeddingGemma paper '
             'evaluated v1 with **bfloat16** weights, and the v2 model card reports **full-precision** results.', '',
             '| task | ' + ' | '.join(sorted(runs)) + ' |', '|---|' + '---:|' * len(runs)]
    for n in names + ['Mean(Task)']:
        lines.append(f'| {n} | ' + ' | '.join(f'{runs[k][n]:.2f}' if n in runs[k] else '—' for k in sorted(runs)) + ' |')
    lines += ['', f'Published means (EmbeddingGemma 2 model card): v1 {PUBLISHED["v1"]}, v2 {PUBLISHED["v2"]}.']
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    OUT.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
