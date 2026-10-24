#!/usr/bin/env python3
"""Can EmbeddingGemma 300m (v1) run with fp16 activations? (input to the v1 engine-support estimate)

The model card says its activations do not support float16. This measures, on the 60 validation texts of
data/texts_ja_en.jsonl with and without the query prompt, (1) the largest absolute activation of every layer
output in fp32 (fp16 overflows above 65504) and (2) sentence-transformers in fp16 vs fp32 on MPS (non-finite
outputs, minimum cosine). Output: results/eval/v1_fp16_check.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    from sentence_transformers import SentenceTransformer
    texts = [json.loads(line)['text'] for line in open(ROOT / 'data/texts_ja_en.jsonl')]
    prompt = 'task: search result | query: '
    inputs = texts + [prompt + t for t in texts]
    st32 = SentenceTransformer(str(ROOT / 'models/embeddinggemma-300m'), device='mps').float()
    backbone = st32[0].model
    peaks = np.zeros(backbone.config.num_hidden_layers + 1)
    for start in range(0, len(inputs), 8):
        features = st32.tokenize(inputs[start:start + 8])
        features = {k: v.to('mps') for k, v in features.items() if hasattr(v, 'to')}
        with torch.no_grad():
            out = backbone(input_ids=features['input_ids'], attention_mask=features['attention_mask'], output_hidden_states=True)
        mask = features['attention_mask'].bool()
        for i, h in enumerate(out.hidden_states):
            peaks[i] = max(peaks[i], float(h[mask].abs().max()))
    ref = st32.encode(inputs, batch_size=8, normalize_embeddings=True)
    st16 = SentenceTransformer(str(ROOT / 'models/embeddinggemma-300m'), device='mps').half()
    half = st16.encode(inputs, batch_size=8, normalize_embeddings=True).astype(np.float32)
    finite = np.isfinite(half).all(axis=1)
    cos = (ref[finite] * half[finite]).sum(1)
    report = {'texts': len(inputs), 'fp16_max': 65504.0,
              'fp32_layer_output_abs_max': {('embeddings' if i == 0 else f'layer{i - 1}'): round(float(v), 1) for i, v in enumerate(peaks)},
              'fp32_overall_abs_max': round(float(peaks.max()), 1),
              'st_fp16': {'non_finite_vectors': int((~finite).sum()),
                          'min_cos_vs_fp32': float(cos.min()) if len(cos) else None,
                          'mean_cos_vs_fp32': float(cos.mean()) if len(cos) else None}}
    (ROOT / 'results/eval/v1_fp16_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
