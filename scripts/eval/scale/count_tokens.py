"""Token counts of the collected window texts as egemma-server will see them (client prefix + BOS/EOS). CPU only.

The server refuses an input over 2048 tokens (HTTP 400 for the whole request), so such a window can never be
indexed, and the HttpEmbedder's batch of 32 that contains it fails as a whole.

    .venv/bin/python -I scripts/eval/scale/count_tokens.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'server'))
from egemma_server.tokenizer import PromptTokenizer  # noqa: E402

OUT = ROOT/'build/scale'
PREFIX = {'document': 'title: none | text: ', 'query': 'task: search result | query: '}
LIMIT = 2048


def main() -> None:
    tok = PromptTokenizer(ROOT/'models/embeddinggemma-300m')
    rows = [json.loads(line) for line in (OUT/'windows.jsonl').open() if line.strip()]
    counts: dict[str, int] = {}
    for k in range(0, len(rows), 512):
        part = rows[k:k + 512]
        for r, ids in zip(part, tok.encode([PREFIX['document'] + r['text'] for r in part], None)):
            counts[r['h']] = len(ids)
    queries = [json.loads(line) for line in (OUT/'queries.jsonl').open() if line.strip()]
    qcounts = {q['h']: len(ids) for q, ids in zip(queries, tok.encode([PREFIX['query'] + q['text'] for q in queries], None))}
    out = {'limit': LIMIT, 'documents': counts, 'queries': qcounts}
    for name in ('fixed', 'shipped'):
        values = [counts[r['h']] for r in rows if name in r['sets']]
        over = [r['h'] for r in rows if name in r['sets'] and counts[r['h']] > LIMIT]
        out[name] = {'windows': len(values), 'tokens': sum(values), 'mean': round(statistics.mean(values), 1),
                     'p50': statistics.median(values), 'p95': sorted(values)[int(0.95 * (len(values) - 1))], 'max': max(values),
                     'over_limit': len(over)}
    (OUT/'window-tokens.json').write_text(json.dumps(out))
    print(json.dumps({k: out[k] for k in ('fixed', 'shipped')}))


if __name__ == '__main__':
    main()
