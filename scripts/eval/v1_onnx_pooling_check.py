#!/usr/bin/env python3
"""Which EmbeddingGemma 300m output does dsh-prime-memory's local embedding return?

The plugin's worker calls transformers.js pipeline('feature-extraction', ..., {pooling: 'mean', normalize: true})
on the q8 ONNX export, whose graph has two outputs: last_hidden_state and sentence_embedding (mean pooling + the
2_Dense/3_Dense projections + normalization, i.e. the model's real embedding). This compares the plugin's cached
vectors (scripts/eval/retrieval_public.py, raw texts, no prompt) with sentence-transformers fp32 on MPS twice:
the full model (sentence embedding) and the model without its Dense layers (mean-pooled last hidden state).
Output: results/eval/v1_onnx_pooling_check.json.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Normalize
    full = SentenceTransformer(str(ROOT / 'models/embeddinggemma-300m'), device='mps').float()
    no_dense = SentenceTransformer(modules=[full[0], full[1], Normalize()], device='mps')
    out = {'model': 'google/embeddinggemma-300m@57c266a (sentence-transformers fp32 MPS)', 'datasets': {}}
    for name in ('jmteb-nlp_journal_title_abs', 'locomo'):
        ds = json.loads((ROOT / f'build/eval/data/{name}.json').read_text())
        texts = [q['text'] for q in ds['queries']][:500]
        onnx = np.load(sorted(glob.glob(str(ROOT / f'build/eval/cache/{name}-onnx-q-*.npy')))[0])[:500]
        row = {'texts': len(texts)}
        for label, model in (('sentence_embedding (with Dense)', full), ('mean-pooled last_hidden_state (no Dense)', no_dense)):
            v = model.encode(texts, batch_size=32, normalize_embeddings=True)
            cos = (v * onnx).sum(1)
            row[label] = {'cos_mean': float(cos.mean()), 'cos_min': float(cos.min())}
        out['datasets'][name] = row
    (ROOT / 'results/eval/v1_onnx_pooling_check.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
