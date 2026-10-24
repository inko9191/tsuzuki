#!/usr/bin/env python3
"""How close does EmbeddingGemma 300m itself get in bf16? sentence-transformers bf16 vs fp32 (MPS).

Context for the engine's bf16 result (scripts/validate_v1.py --dtype bf16): the same 6 prompts x 63 texts,
minimum cosine of the bf16 model's vectors against the fp32 model's. Output: results/eval/v1_bf16_reference.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'server'))
from validate_v1 import MODEL, PROMPT_CASES, load_texts  # noqa: E402


def main() -> int:
    from sentence_transformers import SentenceTransformer
    from egemma_server.tokenizer import PromptTokenizer
    texts, tokenizer = load_texts(), PromptTokenizer(MODEL)
    fp32 = SentenceTransformer(str(MODEL), device='mps').float()
    bf16 = SentenceTransformer(str(MODEL), device='mps').to(torch.bfloat16)
    out = {'reference': 'sentence-transformers fp32 MPS', 'compared': 'sentence-transformers bf16 MPS', 'per_prompt': {}}
    for case in PROMPT_CASES:
        inputs = [(tokenizer.prompt_for(case) or '' if case != '<none>' else '') + t for t in texts]
        a = fp32.encode(inputs, batch_size=8, normalize_embeddings=True)
        b = bf16.encode(inputs, batch_size=8, normalize_embeddings=True).astype(np.float32)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        cos = (a * b).sum(1)
        out['per_prompt'][case] = {'min_cos': float(cos.min()), 'below_0.999': int((cos < 0.999).sum()), 'finite': bool(np.isfinite(b).all())}
    out['min_cos'] = min(v['min_cos'] for v in out['per_prompt'].values())
    (ROOT / 'results/eval/v1_bf16_reference.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
