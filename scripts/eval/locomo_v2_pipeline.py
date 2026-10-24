#!/usr/bin/env python3
"""Exploratory (dev only): does v2 close the gap to v1 once retrieval uses Qwen facts and p2 windows?

The stage 1 gap (raw single turns) was measured on chatty text, which v2 handles poorly. Facts are
self-contained sentences, closer to the documents v2 does well on. Same dev split, same frozen facts
(cached generation, no new Qwen calls), same prompts, dense only, no rerank. Not a registered stage;
the frozen C-final config and holdout results are untouched.
"""
from __future__ import annotations
import argparse

import numpy as np
import locomo_memory as lm
import locomo_memory_methods as mm
import locomo_memory_stages as st
import longmemeval_g1_methods as g1m
import retrieval_public as rp

VARIANTS = ('v1-fp32-prompt', 'v2-fp16-prompt')
CONFIGS = [mm.config(['t1']), mm.config(['p2']), mm.config(['p2', 'fact'])]
METRICS = ('recall@10', 'recall@T512', 'all_evidence@10')
OUT = lm.RESULTS/'v2-pipeline-dev'


def run(port):
    dataset, manifest = lm.load_dev()
    turns = mm.turn_table(dataset['corpus'], st.load_raw(manifest))
    from egemma_server.tokenizer import PromptTokenizer
    tokenizer = PromptTokenizer(rp.V1_ST_DIR)
    prompts = lm.PROTOCOL['prompts']

    def count(texts):
        return [len(e.ids) for e in tokenizer.tokenizer.encode_batch(texts, add_special_tokens=False)]
    turn_tokens = dict(zip([t['id'] for t in turns], count([t['text'] for t in turns])))
    first_date = {}
    for t in turns:
        first_date.setdefault(t['session'], t['date'])
    header_tokens = dict(zip(first_date, count([f'Date: {d}' for d in first_date.values()])))
    turn_session = {t['id']: t['session'] for t in turns}
    facts, generation = st.generated('fact', dataset, turns, st.Qwen.offline())
    units = st.build_units(['t1', 'p2', 'fact'], turns, {'fact': facts})
    args = argparse.Namespace(port=port, model_dir=lm.ROOT/'models/embeddinggemma-2', v1_dir=rp.V1_ST_DIR)
    rows, report = {}, {'started_at': lm.stamp(), 'split': 'dev', 'holdout_evaluated': False,
                        'exploratory': True, 'fact_generation': generation, 'variants': {}}
    for variant in VARIANTS:
        identity = lm.fingerprint(variant)
        vectors = np.zeros((len(units), 768), np.float32)
        with lm.start_backend(variant, args) as (info, embed):
            for set_name in ('t1', 'p2', 'fact'):
                index = [i for i, u in enumerate(units) if u['set'] == set_name]
                raw = [prompts['document']+units[i]['text'] for i in index]
                fitted = rp.fit_like_st(tokenizer, raw)
                side = 'd' if set_name == 't1' else f'u-{set_name}'
                vecs, _ = lm.cached_vectors(variant, side, fitted, identity, embed)
                vectors[index] = vecs
            raw = [prompts['query']+q['text'] for q in dataset['queries']]
            qvecs, _ = lm.cached_vectors(variant, 'q-dev', rp.fit_like_st(tokenizer, raw), identity, embed)
        qtexts = {'orig': [[(v, q['text'])] for v, q in zip(qvecs, dataset['queries'])]}
        index = mm.Index(units, vectors)
        report['variants'][variant] = {'server': info, 'configs': {}}
        for cfg in CONFIGS:
            r = mm.score_config(index, cfg, dataset, qtexts, turn_session, turn_tokens, header_tokens,
                                lambda q: lm.query_type(q, dataset['qrels'][q['id']]))
            rows[(variant, cfg['id'])] = r
            report['variants'][variant]['configs'][cfg['id']] = mm.aggregate(r, lm.TYPES)
    groups = {q['id']: q['group'] for q in dataset['queries']}
    report['v2_minus_v1'] = {}
    for cfg in CONFIGS:
        a, b = rows[('v2-fp16-prompt', cfg['id'])], rows[('v1-fp32-prompt', cfg['id'])]
        if [r['id'] for r in a] != [r['id'] for r in b]:
            raise RuntimeError('query order differs')
        report['v2_minus_v1'][cfg['id']] = {}
        for metric in METRICS:
            diffs = [x[metric]-y[metric] for x, y in zip(a, b)]
            report['v2_minus_v1'][cfg['id']][metric] = {
                'difference': float(np.mean(diffs)),
                'conversation_ci95': g1m.interval(diffs, [groups[r['id']] for r in a]),
                'question_ci95': g1m.interval(diffs, [r['id'] for r in a])}
    report['finished_at'] = lm.stamp()
    lm.write(OUT.with_suffix('.json'), report)
    OUT.with_suffix('.md').write_text(markdown(report))
    print(markdown(report))


def markdown(report):
    v = report['variants']
    lines = ['# v2 を記憶の仕組みに入れた場合（LoCoMo 開発用、探索的）', '',
             '開発用1233問。凍結の事実（Qwen 生成済み）と窓 p2 をそのまま使い、埋め込みだけ v1 fp32 と v2 fp16 を入れ替えた。'
             'プロンプトあり、768次元、密ベクトルのみ、再ランキングなし。登録ステージではない。', '',
             '|設定|指標|v1|v2|v2−v1|会話単位95%区間|', '|---|---|---:|---:|---:|---|']
    for cfg in CONFIGS:
        for metric in METRICS:
            a = v['v1-fp32-prompt']['configs'][cfg['id']]['overall'][metric]
            b = v['v2-fp16-prompt']['configs'][cfg['id']]['overall'][metric]
            d = report['v2_minus_v1'][cfg['id']][metric]
            lo, hi = d['conversation_ci95']
            lines.append(f'|`{cfg["id"]}`|{metric}|{a:.4f}|{b:.4f}|{d["difference"]:+.4f}|[{lo:+.4f}, {hi:+.4f}]|')
    lines += ['', '会話単位の区間は、10会話をまとめて再標本化（10000回、seed 0）。質問単位の区間は JSON。']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8397)
    run(parser.parse_args().port)
