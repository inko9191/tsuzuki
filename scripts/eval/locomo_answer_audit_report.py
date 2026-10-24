#!/usr/bin/env python3
"""Supplementary report; preserves every original and prior audit result number."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import locomo_answer_audit_measure as measured

ae, lm, st = measured.ae, measured.lm, measured.st
LOCAL = measured.LOCAL
RESULT = lm.RESULTS / 'answer-audit-performance.json'


def initial_fact_accounting(records):
    path = lm.LOCAL / 'gen_sample_fact.log'
    text = path.read_text()
    log = json.loads(text[text.index('{'):])
    store = lm.read(lm.LOCAL / 'gen/fact.json')
    search = lm.read(ae.LOCAL / 'production/search-inputs.json')
    jobs = st.generation_jobs('fact', {}, search['turns'])[:log['items']]
    subset = {qid: store[qid]['reply'] for qid, _, _ in jobs}
    if lm.digest(dict(sorted(subset.items()))) != log['replies_sha256']:
        raise RuntimeError('initial fact outputs not identical to final cache; cannot attribute its time')
    counts = {g: v['count'] for g, v in records['ingestion']['fact']['per_conversation_estimate'].items()}
    first_groups = defaultdict(int)
    for qid, _, _ in jobs:
        first_groups[qid.split('#')[0]] += 1
    if len(first_groups) != 1:
        raise RuntimeError('sample spans groups; cannot assign sample duration to one conversation')
    first_group = next(iter(first_groups))
    remaining = {g: n-first_groups[g] for g, n in counts.items()}
    main = records['ingestion']['fact']['original_run_seconds']
    allocation = measured.audit.apportioned(main, remaining)
    return {'sample_seconds_actual': log['seconds_this_call'], 'sample_jobs': log['items'],
        'sample_group': first_group, 'sample_reply_digest_matches_final_cache': True,
        'main_run_seconds_actual': main, 'known_runs_total_seconds': main+log['seconds_this_call'],
        'sample_log_sha256': lm.file_hash(path),
        'sample_gpu_concurrency': 'not recorded in the initial sample log; wall time is not a controlled GPU benchmark',
        'per_conversation': {g: {'jobs': counts[g], 'sample_seconds_actual':
            log['seconds_this_call'] if g == first_group else 0.,
            'main_seconds_estimate': allocation[g]['seconds_estimate'],
            'total_seconds_estimate': allocation[g]['seconds_estimate']
                + (log['seconds_this_call'] if g == first_group else 0.)} for g in sorted(counts)},
        'note': 'Initial3 cache entries were reused. Main-run time apportioned among269 remaining jobs, '
                'not a recovered per-job duration; no original result file is corrected or overwritten.'}


def totals(components, payloads):
    by_id = {p['id']: p for p in payloads}
    if set(by_id) != {r['id'] for r in components} or len(by_id) != len(payloads):
        raise RuntimeError('accounting requires identical unique questions')
    return [{**r, 'input_construction_seconds': by_id[r['id']]['input_construction_seconds'],
             'including_full_input_preflight_seconds': by_id[r['id']]['including_full_input_preflight_seconds'],
             'total_with_input_preflight_seconds': r['total_component_seconds']
                 + by_id[r['id']]['including_full_input_preflight_seconds']} for r in components]


def main():
    measured.checked()
    audit = lm.read(lm.RESULTS / 'answer-audit-measurements.json')
    if audit.get('status') != 'complete':
        raise RuntimeError('measurement incomplete; never summarize a partial audit as complete')
    record = lm.read(measured.audit.PUBLIC)
    payload = lm.read(lm.RESULTS / 'answer-audit-payload.json')
    rows = totals(lm.read(LOCAL / 'component-rows.json'), lm.read(LOCAL / 'payload-rows.json'))
    result = {'created_at': lm.stamp(), 'scope': record['scope'],
        'source_result_sha256': record['source_result_sha256'],
        'measurement_sha256': lm.file_hash(lm.RESULTS / 'answer-audit-measurements.json'),
        'payload_sha256': lm.file_hash(lm.RESULTS / 'answer-audit-payload.json'),
        'fact_accounting': initial_fact_accounting(record),
        'original_files_unchanged': True,
        'C_final_seconds': {k: measured.audit.distribution([r[k] for r in rows]) for k in
            ('answer_seconds', 'rerank_seconds', 'query_embedding_seconds', 'ranking_seconds',
             'input_construction_seconds', 'including_full_input_preflight_seconds',
             'total_component_seconds', 'total_with_input_preflight_seconds')},
        'C_final_per_conversation': {g: {k: measured.audit.distribution([r[k] for r in rows if r['id'].split('/')[0] == g])
            for k in ('answer_seconds', 'rerank_seconds', 'query_embedding_seconds', 'ranking_seconds',
                      'including_full_input_preflight_seconds', 'total_with_input_preflight_seconds')}
            for g in record['A_input_tokens_per_conversation']},
        'cache_comparison': audit['cache_comparison'], 'memory': audit['memory'],
        'ingestion_embedding': audit['embedding_search']['ingestion'],
        'embedding_startup_seconds': audit['embedding_search']['startup_seconds'],
        'index_build_seconds': audit['embedding_search']['index_build_seconds'],
        'additional_response_failures': audit['failures'],
        'note': 'C total is a matched-question component sum measured on different occasions, '
                'not an online replay. Cache comparison uses ten paired development questions, '
                'not302 holdout no-cache responses. Memory uses fresh-process30-request workloads '
                'including cold/warm/cache-off requests per arm, not original302-condition peaks. '
                'Ingestion generation conversation times are estimates.'}
    lm.write(LOCAL / 'total-cost-rows.json', rows)
    lm.write(RESULT, result)
    markdown(result, record, payload)


def markdown(d, record, payload):
    out = ['# 回答評価：追加の時間・メモリ実測', '',
        '元の本番302問の数字・設定・本文は変更していない。追加生成は開発質問のタイミング監査用で、'
        '採点・選択に使わない。', '', '## C-finalの1問あたり時間', '',
        '下記は同じ本番302問に対応づけた成分合算。回答/rerankは本番記録、埋め込み/検索/入力準備は'
        '追加実測。異なる時点の処理を足した値であり、連続オンラインlatencyの実測ではない。', '',
        '| 処理 | n | 平均秒 | 中央値秒 | p90秒 |', '|---|---:|---:|---:|---:|']
    labels = {'answer_seconds': '回答HTTP（cacheあり）', 'rerank_seconds': 'rerank-qwen HTTP',
        'query_embedding_seconds': '質問のfresh埋め込み（batch1）', 'ranking_seconds': 'CPU検索',
        'input_construction_seconds': 'rerank parse/原文展開/予算内組み立て',
        'including_full_input_preflight_seconds': '上記組み立て＋全入力tokenizer検査',
        'total_component_seconds': '合計（回答＋rerank＋埋め込み＋検索）',
        'total_with_input_preflight_seconds': '合計（さらに組み立て・検査込み）'}
    for k, v in d['C_final_seconds'].items():
        out.append(f'| {labels[k]} | {v["n"]} | {v["mean"]:.6f} | {v["median"]:.6f} | {v["p90"]:.6f} |')
    out += ['', '質問のrw生成は0回（orig）。rwは取り込み時のみ。rerankは1問1回で全C条件に共有。'
        'p90合計は各問の成分合計のp90であり、成分ごとのp90を足したものではない。', '',
        f'一度だけの追加embedding起動 {d["embedding_startup_seconds"]:.6f}秒、CPU索引構築 {d["index_build_seconds"]:.6f}秒。'
        'これら、モデル読み込み・hash・GPU環境確認・ファイルIO・サーバー窓切替は上記1問合計に含まない。'
        '現在はrerank8192窓と回答65536窓をバッチ間で切替えた評価で、窓を切替えずに連続処理するオンライン構成は未測定。', '',
        '元の全7条件実行の一度だけ／全工程費用はanswer-runtime.json: '
        '検索準備10.158秒、入力準備80.219秒、2回の窓切替4.555秒、hash/採点/保存等の残差14.817秒。'
        '検索準備・入力準備には上記の処理が一部含まれるので二重加算しない。', '',
        '| 会話 | 回答 | rerank | 埋め込み | CPU検索 | 組み立て＋検査 | 合計（平均秒/問） |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g, v in d['C_final_per_conversation'].items():
        out.append(f'| {g} | ' + ' | '.join(f'{v[k]["mean"]:.6f}' for k in
            ('answer_seconds', 'rerank_seconds', 'query_embedding_seconds', 'ranking_seconds',
             'including_full_input_preflight_seconds', 'total_with_input_preflight_seconds')) + ' |')
    out += ['', '## Aなどのcacheあり／なし追加実測', '',
        '開発pilotからhash固定で各会話2問。第一質問の後、同じ第二質問をcacheあり／なし各1回。'
        '下記は10会話10組の第二質問。これは本番302問全件のcacheなし時間ではない。'
        '同じモデル・build・65536窓・256出力上限・1並列。cache=falseの全40件でcache_n=0を確認。', '',
        '| 条件 | cache | n | wall平均秒 | prefill平均秒 | generation平均秒 | wall中央値 | wall p90 | cache平均token | 実処理平均token |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in measured.ARMS:
        for role, label in (('warm', 'あり（warm）'), ('no_cache', 'なし')):
            v = d['cache_comparison'][arm][role]
            out.append(f'| {arm} | {label} | {v["wall_seconds"]["n"]} | '
                + ' | '.join(f'{v[k]["mean"]:.6f}' for k in ('wall_seconds','prefill_seconds','generation_seconds'))
                + f' | {v["wall_seconds"]["median"]:.6f} | {v["wall_seconds"]["p90"]:.6f} | '
                + ' | '.join(f'{v[k]["mean"]:.1f}' for k in ('cache_tokens', 'processed_prompt_tokens')) + ' |')
    out += ['', f'参考: 元の本番302問A（cacheあり、会話先頭のcoldも含む）の回答平均は'
        f'{record["per_condition_original"]["A"]["wall_seconds"]["mean"]:.6f}秒。追加10問warmだけの平均と混同しない。'
        'cacheなし時の出力長差・finish_reasonも記録し、出力差を新たな精度評価として扱わない。', '',
        '## サーバーのメモリピーク（追加測定）', '',
        '条件ごとに新規プロセス。各30要求（cold/warm/cache-offを各10）。同じ65536窓。'
        'KVとRAM prefix cacheが動作中のプロセス全体のOS指標。原本番302問全件のピークを復元した値ではない。'
        'kernel lifetime physical footprint peakは起動からの最大、RSSはready後50msサンプルの最大。', '',
        '| 条件 | 起動含むkernel最大footprint (GiB) | ready後sample最大footprint (GiB) | ready後sample最大RSS (GiB) | サンプル数 |',
        '|---|---:|---:|---:|---:|']
    for arm, v in d['memory'].items():
        out.append(f'| {arm} | {v["kernel_lifetime_peak_footprint_bytes"]/2**30:.3f} | '
                   f'{v["physical_footprint_bytes"]/2**30:.3f} | {v["rss_bytes"]/2**30:.3f} | {v["samples"]} |')
    out += ['', '両指標は重なっており足し合わせない。footprintはOSが負担として計上する値、RSSは'
        'cleanなfile mapping等を含む常駐値。GPU専用allocation／KV単体は分離していない。'
        '短いRSSピークをサンプルが逃す可能性はある。同じ大きな窓の予約・重みが共通なので、'
        '小さい窓でCを配置した場合の必要メモリや本番全体への削減率はこの値から主張できない。'
        '埋め込みプロセス・Python audit driver・システム全体のメモリはこのQwen指標に含まない。', '',
        '## 取り込み：会話別fresh埋め込みと生成時間推定', '',
        '追加embeddingはp2/rw/factの凍結本文・元native fp32、会話単位batch32。計算/転送/応答の実測、'
        'backend起動・本文準備・vector保存は含まない。fact272セッションから3452索引単位を抽出。', '',
        '| 会話 | rw件数 | rw生成秒（推定） | factセッション | fact生成秒（推定） | p2埋め込み秒 | rw埋め込み秒 | fact埋め込み秒 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    embeds = {(v['group'], v['set']): v['seconds'] for v in d['ingestion_embedding']}
    for g in record['A_input_tokens_per_conversation']:
        rw = record['ingestion']['rw']['per_conversation_estimate'][g]
        fact = d['fact_accounting']['per_conversation'][g]
        out.append(f'| {g} | {rw["count"]} | {rw["seconds_estimate"]:.3f} | {fact["jobs"]} | '
            f'{fact["total_seconds_estimate"]:.3f} | ' + ' | '.join(f'{embeds[g,k]:.6f}' for k in ('p2','rw','fact')) + ' |')
    out.append('| 全体（生成は既知run実測、埋め込みは追加実測） | 5882 | '
        + f'{record["ingestion"]["rw"]["original_run_seconds"]:.3f} | 272 | {d["fact_accounting"]["known_runs_total_seconds"]:.3f} | '
        + ' | '.join(f'{sum(v["seconds"] for v in d["ingestion_embedding"] if v["set"] == k):.6f}' for k in ('p2','rw','fact')) + ' |')
    out += ['', 'factの初期3件はconv-26、75.340760秒を別ログから復元し保存reply digest一致を確認。'
        '本実行7126.977374秒はその3件を再利用。既知run合計7202.318134秒。'
        '会話別は残269件に本実行時間を件数比例配分し、conv-26へ初期3件実測を加えた推定。'
        'rw会話別も件数比例推定。入力長・出力長・cache差を反映しない。元generation-outputsの数字は変更しない。', '',
        '初期fact3件のログにはGPU競合状態がなく、その75.34秒は専用GPU性能値としては扱わない。'
        '本生成はgeneration-run.jsonで他の既知GPU作業なしを記録している。', '',
        '取り込みを何問で償却するかによって合計は変わる。生成＋fresh埋め込みの既知成分の総和をI、'
        '質問数をNとすれば、平均成分費用は `C-finalの上記平均 + I/N`。'
        '読み込み・本文準備・索引保存など未計測成分はこのIに含まない。'
        '取り込みを無料とみなした速度／費用削減の主張はしない。', '',
        '## 検証・問題点・主張範囲', '',
        '元の全追跡済み評価ファイル／コードhashは開始前後で一致。追加CPU検索302順位、'
        '追加組み立て302messages/input tokensとも元のものと完全一致。追加HTTPは120件、各1試行、'
        f'非stop終了は{len(d["additional_response_failures"])}件（再生成せず保存）。生成は他GPU作業なしで1並列。', '',
        '確認できたこと: 時間表の対象、元のcache実利用、同じ開発質問でのcache-off差、'
        '成分合計、会話別embedding、追加ワークロードの条件別メモリ。', '',
        '確認できなかったこと: 過去の会話別生成秒数、元本番全件の条件別メモリ、'
        '元本番302問全件のcache-off時間、窓切替不要な連続オンラインlatency、'
        'GPU専用allocation量、独立会話／日本語一般化、意味的正しさ。', '',
        '主張してよい範囲: 元の語彙F1は同じ10会話の未調整質問への値。性能は記載した'
        '機械・窓・cache・batch・入力・測定時点に限る。C-finalは回答APIだけなら小さいが、'
        '元cacheありAより全処理が速いという主張は支持されない。メモリ削減や未知会話での性能を'
        '元精度から推論しない。一般化と意味評価の文書は未承認の提示案で、未実行。', '',
        '元の種類別F1、会話別区間・差・符号検定、全条件棄権率、A会話別入力長は'
        '[保存記録の監査表](answer-audit-records.md)。本書の追加fact内訳が初期生成分を補う。', '']
    RESULT.with_suffix('.md').write_text('\n'.join(out))


if __name__ == '__main__':
    main()
