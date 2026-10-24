#!/usr/bin/env python3
"""Measure frozen C-final input construction separately, without model generation."""
from collections import defaultdict
from pathlib import Path
import time

import locomo_answer_audit_measure as measured

ae, lm, st, mm = measured.ae, measured.lm, measured.st, measured.mm
LOCAL = measured.LOCAL


def main():
    measured.checked()
    if lm.read(LOCAL / 'status.json').get('phase') != 'audit complete':
        raise RuntimeError('cache/memory audit must finish before separate CPU input measurement')
    measured.quiet()
    path = lm.RESULTS / 'answer-audit-payload.json'
    if path.exists():
        raise RuntimeError('already measured; no overwrite')
    search = lm.read(ae.LOCAL / 'production/search-inputs.json')
    rerank = lm.read(ae.LOCAL / 'production/rerank-responses.json')
    original = lm.read(ae.LOCAL / 'production/answer-inputs.json')
    count = ae.Tokenizer(original['server']['url'])
    # Fresh, supplementary token-count memo only. Never save over the original token-count cache.
    count.counts = {}
    count.path = LOCAL / 'payload-token-counts.json'
    by_group = defaultdict(list)
    for t in search['turns']:
        by_group[t['group']].append(t)
    rows = []
    for qid in search['schedule']:
        ae.gate()
        measured.quiet()
        q = search['queries'][qid]
        tick = time.perf_counter()
        content = rerank[qid]['response']['choices'][0]['message']['content']
        value = mm.parse('rerank', content, n_passages=len(q['orders']['C-final'][:mm.RERANK_CANDIDATES]))
        order = mm.distinct(measured.production.reranked_order(q['orders']['C-final'], value), search['units'])
        ranking = list(dict.fromkeys(t for i in order for t in search['units'][i]['turns']))
        payload = ae.am.payload('C-final', by_group[q['group']], {'C-final': ranking}, count.count)
        chat = ae.am.messages(payload['conversation'], payload['retrieved'], q['text'])
        before_tokencheck = time.perf_counter() - tick
        tokens = count.input_count(chat)
        with_tokencheck = time.perf_counter() - tick
        item = original['items'][f'C-final|{qid}']
        if chat != item['messages'] or tokens != item['input_tokens']:
            raise RuntimeError('supplementary input construction differs; do not change frozen inputs')
        rows.append({'id': qid, 'input_construction_seconds': before_tokencheck,
                     'including_full_input_preflight_seconds': with_tokencheck})
    count.save()
    lm.write(LOCAL / 'payload-rows.json', rows)
    result = {'created_at': lm.stamp(), 'questions': len(rows), 'code_sha256': lm.file_hash(Path(__file__)),
        'all302_messages_and_input_tokens_match': True, 'fresh_count_memo': True,
        'reuse': 'within-run token-count memo; same recent prefix reused between questions of a conversation',
        'input_construction_seconds': measured.audit.distribution([r['input_construction_seconds'] for r in rows]),
        'including_full_input_preflight_seconds': measured.audit.distribution([r['including_full_input_preflight_seconds'] for r in rows]),
        'per_conversation': {g: {k: measured.audit.distribution([r[k] for r in rows if r['id'].split('/')[0] == g])
            for k in ('input_construction_seconds', 'including_full_input_preflight_seconds')}
            for g in sorted(by_group)},
        'note': 'CPU and tokenizer API only; original messages/token counts are verified, no generation/scoring. '
                'Excludes per-question environment checks/file IO and cold backend/model startup.'}
    lm.write(path, result)


if __name__ == '__main__':
    main()
