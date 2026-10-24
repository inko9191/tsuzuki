#!/usr/bin/env python3
"""Supplementary read-only accounting of the completed frozen experiment."""
from __future__ import annotations

from collections import Counter
import numpy as np

import locomo_answer_eval as ae
import locomo_answer_report as ar

lm = ae.lm
LOCAL = ae.LOCAL / 'audit'
PUBLIC = lm.RESULTS / 'answer-audit-records.json'
STRATA = ('type:multiple_evidence', 'overall', 'type:single_fact', 'type:temporal')


def distribution(values):
    if not values:
        raise ValueError('empty measurement')
    return {**ar.distribution(values), 'min': float(min(values)), 'max': float(max(values)),
            'p25': float(np.percentile(values, 25)), 'p75': float(np.percentile(values, 75))}


def apportioned(total, counts):
    """Count allocation is explicitly an estimate, never a recovered measurement."""
    return {g: {'count': n, 'seconds_estimate': total * n / sum(counts.values()),
                'basis': 'count-proportional allocation; individual durations not recorded'}
            for g, n in sorted(counts.items())}


def collect():
    report = lm.read(lm.RESULTS / 'answer-report.json')
    rows = lm.read(ae.LOCAL / 'production/scored-rows.json')
    runtime = lm.read(lm.RESULTS / 'answer-runtime.json')
    rerank = lm.read(ae.LOCAL / 'production/rerank-responses.json')
    search = lm.read(ae.LOCAL / 'production/search-inputs.json')
    generation = lm.read(lm.RESULTS / 'generation-outputs.json')
    rw_counts = Counter(t['group'] for t in search['turns'])
    fact_counts = Counter()
    for t in search['turns']:
        fact_counts[(t['group'], t['session'])] = 1
    sessions = Counter(g for g, _ in fact_counts)
    original_embeddings = {}
    for name, stage in (('p2', 'stage2-dev.json'), ('rw', 'stage4-dev.json'), ('fact', 'stage4-dev.json')):
        original_embeddings[name] = lm.read(lm.RESULTS / stage)['units'][name]['embedding']
    result = {'created_at': lm.stamp(), 'source_result_sha256': lm.file_hash(lm.RESULTS / 'answer-report.json'),
        'source_runtime_sha256': lm.file_hash(lm.RESULTS / 'answer-runtime.json'),
        'frozen_results_changed': False,
        'scope': 'same ten LoCoMo conversations; heldout302 lexical scores; timing audit only',
        'tables': {s: {'f1': report['tables'][s]['f1'], 'abstention': report['tables'][s]['abstention']}
                   for s in STRATA},
        'per_condition_original': runtime['per_condition'],
        'A_input_tokens_per_conversation': {g: distribution([r['input_tokens'] for r in rows['A']
            if r['id'].split('/')[0] == g]) for g in sorted(rw_counts)},
        'rerank_seconds': distribution([v['wall_seconds'] for v in rerank.values()]),
        'rerank_per_conversation': {g: distribution([v['wall_seconds'] for q, v in rerank.items()
            if q.split('/')[0] == g]) for g in sorted(rw_counts)},
        'rerank_stage_seconds': sum(r['seconds'] for r in lm.read(ae.LOCAL / 'production/rerank-runs.json')),
        'query_rw_calls': 0,
        'time_inclusions': {'answer_wall': 'HTTP answer call only; excludes embedding/search/rerank/ingestion',
            'prefill': 'server prompt evaluation; already excludes reused cache prefix',
            'generation': 'server answer token generation',
            'query_rw': 'none; orig query; rw is a corpus-side transformation'},
        'original_query_embedding': report['search_preparation']['query_embedding'],
        'ingestion': {k: {'original_run_seconds': generation[k]['seconds_this_call'],
            'jobs': sum(counts.values()), 'individual_durations_recorded': False,
            'per_conversation_estimate': apportioned(generation[k]['seconds_this_call'], counts)}
            for k, counts in (('rw', rw_counts), ('fact', sessions))},
        'original_fresh_document_embeddings': original_embeddings,
        'limits': ['Original rw/fact individual durations unavailable. Per-conversation allocation is an estimate.',
            'Original server memory peak cannot be attributed to a condition after the fact.',
            'Original query embedding was batched; amortized time is not single-request latency.',
            'No external/Japanese generalization or semantic correctness assessment executed.']}
    lm.write(PUBLIC, result)
    write_markdown(result)
    return result


def write_markdown(d):
    out = ['# 回答評価：保存記録の追加監査', '',
        '本番結果の数字・設定は変更していない。語彙F1は意味的正答率ではない。', '',
        '## 種類別 F1', '', '差はC-final−A。区間は元の会話単位10000回seed0、質問macro比率。'
        '符号検定は会話平均差の厳密両側、ゼロ差除外、p値は未補正。', '',
        '| 種類 | 問数 | A | C-t1 | C-p2 | C-final | 差 | 会話単位95%区間 | 勝/負/同（n; p） |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for s in STRATA:
        t = d['tables'][s]['f1']; v = t['C-final']['A']; sign = v['sign_test']; ci = v['conversation_ci95']
        out.append(f'| {s.removeprefix("type:")} | {v["questions"]} | '
            + ' | '.join(f'{t[a]["A"]["mean"]:.6f}' for a in ('A', 'C-t1', 'C-p2', 'C-final'))
            + f' | {v["mean_diff"]:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | '
            + f'{sign["wins"]}/{sign["losses"]}/{sign["ties"]} ({sign["n"]}; {sign["two_sided_p"]:.6f}) |')
    out += ['', 'single_factの区間は±.03に収まらず、同等性は未確認。temporalは区間が正でも'
        '符号検定p=.109375。分類は元の固定代理分類のまま。', '', '### 会話別 C-final−A', '',
        '| 会話 | multiple_evidence | overall | single_fact | temporal |', '|---|---:|---:|---:|---:|']
    for g in d['A_input_tokens_per_conversation']:
        out.append(f'| {g} | ' + ' | '.join(f'{d["tables"][s]["f1"]["C-final"]["A"]["per_conversation"][g]["mean_diff"]:+.6f}' for s in STRATA) + ' |')
    out += ['', '## 定型棄権率', '', '| 条件 | multiple_evidence | overall | single_fact | temporal |',
            '|---|---:|---:|---:|---:|']
    for a in ae.am.CONDITIONS:
        out.append(f'| {a} | ' + ' | '.join(f'{100*d["tables"][s]["abstention"][a]["A"]["mean"]:.2f}%' for s in STRATA) + ' |')
    out += ['', '参照あり302問。定型Insufficient evidenceは0点。答えなし0問のため正しい拒否率は算出できない。', '',
        '## 時間表に含まれる範囲', '',
        'wallは回答HTTP呼び出し、prefill/generationはそのサーバー内処理。rw・検索・埋め込み・rerankは含まれない。'
        '質問rw生成は0回（orig）。rerankは302問に1回ずつ実行し、全C条件で共有。', '',
        f'本番rerankステージ {d["rerank_stage_seconds"]:.3f}秒、HTTP平均 {d["rerank_seconds"]["mean"]:.6f}秒/問。'
        '起動・hash確認・索引準備・tokenizer事前チェックは別費用。', '',
        '| 条件 | 回答wall秒/問 | prefill秒/問 | generation秒/問 | 入力token平均 | cache token平均 | 実処理token平均 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a, v in d['per_condition_original'].items():
        out.append(f'| {a} | ' + ' | '.join(f'{v[k]["mean"]:.6f}' for k in
            ('wall_seconds', 'prefill_seconds', 'generation_seconds', 'input_tokens', 'cache_tokens', 'processed_prompt_tokens')) + ' |')
    out += ['', 'Aは長い会話prefixを再利用済み。C8192-finalは質問ごとに検索断片が変わり、'
        'cache平均875 token、実処理6399 token。Aはcache27188 token、実処理1416 token。'
        '9.3秒対2.9秒というprefill差は総入力長だけでは比較できない。追加cache-off測定は別記録。', '',
        '## A入力長（Qwen tokenizer、system/chat/questionを含む）', '',
        '| 会話 | n | 最小 | p25 | 中央値 | 平均 | p75 | p90 | 最大 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g, v in d['A_input_tokens_per_conversation'].items():
        out.append(f'| {g} | ' + ' | '.join(f'{v[k]:.1f}' for k in ('n', 'min', 'p25', 'median', 'mean', 'p75', 'p90', 'max')) + ' |')
    out += ['', '## 取り込み時費用：実測合計と推定配分', '',
        'rw5882発言／fact272セッションの各生成時間は保存されていない。下記の会話別値は件数比例の推定であり、'
        '入力長・出力長・キャッシュ差を反映しない。全体秒数は元のrun実測。再生成は行っていない。', '',
        '| 会話 | rw件数 | rw秒（推定） | fact件数 | fact秒（推定） |', '|---|---:|---:|---:|---:|']
    for g in d['A_input_tokens_per_conversation']:
        r, f = (d['ingestion'][k]['per_conversation_estimate'][g] for k in ('rw', 'fact'))
        out.append(f'| {g} | {r["count"]} | {r["seconds_estimate"]:.3f} | {f["count"]} | {f["seconds_estimate"]:.3f} |')
    out.append('| 全体（実測） | ' + ' | '.join(f'{d["ingestion"][k]["jobs"]} | {d["ingestion"][k]["original_run_seconds"]:.3f}' for k in ('rw', 'fact')) + ' |')
    out += ['', '原embeddingのfresh計算（cache保存を含む）: ' + '; '.join(
        f'{k}={v["seconds"]:.3f}秒/{v["texts"]}単位' for k, v in d['original_fresh_document_embeddings'].items()) + '。'
        '会話別は未記録。追加の会話別fresh埋め込み実測を別に行う。', '',
        '## 確認範囲', '', '確認済み: 上記は凍結結果・ログの値。未確認: 既存条件別ピークメモリ、'
        'cacheなしAの本番302問全件時間、外部・日本語一般化、意味的正しさ。'
        '主張してよい範囲: 同じ10会話の調整に使わなかった質問の語彙F1比較。'
        '回答API時間だけでCの処理全体がAより速いとは主張できない。', '']
    PUBLIC.with_suffix('.md').write_text('\n'.join(out))


if __name__ == '__main__':
    collect()
