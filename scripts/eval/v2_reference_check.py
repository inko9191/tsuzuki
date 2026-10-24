#!/usr/bin/env python3
"""Cross-check of the v2 path on LoCoMo: egemma-server vectors vs sentence-transformers (EmbeddingGemma 2, fp32, MPS).

Re-embeds the LoCoMo queries and turns with the reference sentence-transformers model and the same prompts,
compares them with the cached egemma vectors of scripts/eval/retrieval_public.py (cosine per text) and
recomputes nDCG@10. Output: results/eval/v2_reference_check.json.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
sys.path.insert(0, str(ROOT / 'server'))
import retrieval_public as rp  # noqa: E402


def main() -> int:
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    from egemma_server.tokenizer import PromptTokenizer
    ds = json.loads((ROOT / 'build/eval/data/locomo.json').read_text())
    report = json.loads((ROOT / 'results/eval/retrieval_public.json').read_text())
    tokenizer = PromptTokenizer(ROOT / 'models/embeddinggemma-2')
    model = SentenceTransformer(str(ROOT / 'models/embeddinggemma-2'), device='mps').float()
    out = {'dataset': 'locomo', 'reference': f'sentence-transformers {sentence_transformers.__version__}, fp32, MPS', 'variants': {}}
    for variant, (qt, dt) in {'none': ('none', 'none'), 'prompt': ('query', 'document')}.items():
        qp, dp = tokenizer.prompt_for(qt) or '', tokenizer.prompt_for(dt) or ''
        q = model.encode([qp + x['text'] for x in ds['queries']], batch_size=32, normalize_embeddings=True)
        d = model.encode([dp + x['text'] for x in ds['corpus']], batch_size=32, normalize_embeddings=True)
        eq = np.load(sorted(glob.glob(str(ROOT / f'build/eval/cache/locomo-egemma-{qt}-q-*.npy')))[0])
        ed = np.load(sorted(glob.glob(str(ROOT / f'build/eval/cache/locomo-egemma-{dt}-d-*.npy')))[0])
        cos = np.concatenate([(q * eq).sum(1), (d * ed).sum(1)])
        out['variants'][f'v2-{variant}-768'] = {
            'ndcg@10_reference': float(np.mean([r['ndcg@10'] for r in rp.evaluate(ds, q, d)])),
            'ndcg@10_egemma': report['datasets']['locomo']['variants'][f'v2-{variant}-768']['ndcg@10'],
            'cos_min': float(cos.min()), 'cos_mean': float(cos.mean()), 'texts': int(len(cos))}
    (ROOT / 'results/eval/v2_reference_check.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
