"""Authorized post-outcome G1 CPU finalization. No HTTP, server start, or scoring calls."""
from copy import deepcopy
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import subprocess

import longmemeval_g1_report as reporting

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT/'results/eval/memory'
BASE = ROOT/'build/eval/memory/longmemeval-g1'
LOCAL = BASE/'continuation'
OUT = LOCAL/'cpu-finalization'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def resource_summary(value):
    result = deepcopy(value)
    samples = result.pop('OS_samples1s', [])
    result['OS_sample_count'] = len(samples)
    if samples:
        first, last = samples[0], samples[-1]
        result['final_OS'] = last
        result['OS_summary'] = {
            'minimum_available_bytes_estimate': min(s['available_bytes_estimate'] for s in samples),
            'maximum_swap_used_bytes': max(s['swap_used_bytes'] for s in samples),
            'maximum_pressure_level': max(s['pressure_level'] for s in samples),
            'swapin_increment_bytes': (last['swapins']-first['swapins'])*first['page_size'],
            'swapout_increment_bytes': (last['swapouts']-first['swapouts'])*first['page_size']}
    result['raw_series_scope'] = 'preserved in private pending-final-report.json; published aggregates only'
    return result


def unchanged_comparisons(before, after):
    keys = ('questions', 'A_eligible', 'A_overflow', 'A_overflow_ids',
            'C_minus_A_semantic_primary', 'C_minus_A_all30_operational', 'C_minus_B_all30',
            'F1_C_minus_A_semantic_cohort_reference', 'F1_C_minus_B_all30_reference')
    for key in keys:
        if before[key] != after[key]:
            raise RuntimeError('registered comparison changed: '+key)
    for metric in ('metrics_all30', 'metrics_A_eligible'):
        for arm in reporting.ARMS:
            if before[metric][arm]['overall'] != after[metric][arm]['overall']:
                raise RuntimeError('overall metric changed: '+metric+' '+arm)


def human_list(review, plan):
    blind = read(OUT/'codex-blind-labels.json')
    key = read(OUT/'codex-blind-key.json')
    labels = {r['id']: r for r in blind['labels']}
    lookup = {q['id']: deepcopy(q) for q in review}
    totals = {a: 0 for a in reporting.ARMS}
    disagreements = []
    for qid, q in lookup.items():
        label = labels[qid]
        q['registered_stratum'] = 'abstention' if qid.endswith('_abs') else (
            'single-session' if q['question_type'].startswith('single-session-') else q['question_type'])
        q['codex_rationale'] = label['rationale']
        for alias, value in label['labels'].items():
            arm = key[qid][alias]
            q['conditions'][arm]['codex_label'] = value
            totals[arm] += value
            if value != q['conditions'][arm]['judge']['label']:
                disagreements.append({'id': qid, 'arm': arm, 'codex_label': value,
                                      'Qwen_label': q['conditions'][arm]['judge']['label']})
    def differs(qid):
        c = lookup[qid]['conditions']
        return c['A']['judge']['label'] != c['C-final-norw']['judge']['label']
    ordered = sorted(plan['question_ids'], key=lambda q: (not differs(q),
                      not any(d['id'] == q for d in disagreements)))
    summary = {'protocol': blind['protocol'], 'reviewer': blind['reviewer'],
               'questions': 30, 'labels': 90, 'correct_counts': totals,
               'agreement_with_Qwen': 90-len(disagreements), 'disagreements': disagreements,
               'Qwen_C_A_disagreement_questions': sum(differs(q) for q in ordered),
               'scope': 'supplementary retrospective model review; not human or official score; no history inspection'}
    write(OUT/'human-review-with-labels.json', [lookup[q] for q in ordered])
    write(OUT/'codex-review-unblinded.json', summary)
    pages = ['<!doctype html><meta charset="utf-8"><title>G1 30問 人手確認一覧</title>',
             '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.6}',
             'section{border-top:2px solid #777;margin-top:32px}pre{white-space:pre-wrap;overflow-wrap:anywhere}',
             'article{border:1px solid #bbb;padding:12px;margin:12px 0}h2{overflow-wrap:anywhere}</style>',
             '<h1>G1 30問・90回答 人手確認一覧</h1>',
             '<p>C/AのQwen判定不一致を先頭、その後にCodex/Qwen不一致。条件・Qwen判定を表示した確認用一覧。',
             '元の匿名一覧と別keyも保持。Codex採点は後から行った補助モデルレビューで、人手採点ではない。</p>']
    for number, qid in enumerate(ordered, 1):
        q = lookup[qid]
        pages += [f'<section><h2>{number}. {html.escape(qid)}'+(' [C/A判定不一致]' if differs(qid) else '')+'</h2>',
                  '<p>登録層: '+html.escape(q['registered_stratum'])+' / 元種類: '+html.escape(q['question_type'])+'</p>',
                  '<h3>質問</h3><pre>'+html.escape(q['question'])+'</pre>',
                  '<h3>正解・rubric</h3><pre>'+html.escape(str(q['gold']))+'</pre>']
        for arm, label in [('A','A 全履歴'), ('B1536','B 短い履歴'), ('C-final-norw','C 記憶検索')]:
            v = q['conditions'][arm]
            pages += ['<article><h3>'+label+'</h3><pre>'+html.escape(v['answer'])+'</pre>',
                      f'<p>Qwen判定: {v["judge"]["label"]} / Codex判定: {v["codex_label"]}（1正答・0不正答）</p></article>']
        pages += ['<p>Codex根拠（別名X/Y/Z）: '+html.escape(q['codex_rationale'])+'</p>',
                  '<p>採点時の別名対応: '+html.escape(json.dumps(key[qid],ensure_ascii=False))+'</p>',
                  '<p>人手確認メモ: ____________________</p></section>']
    (OUT/'human-review-with-labels.html').write_text('\n'.join(pages))
    return summary


def finalize():
    OUT.mkdir(exist_ok=True)
    # Preserve the stopped worker and gate. This is a separate CPU completion record.
    status = read(LOCAL/'status.json')
    if status['status'] != 'stopped':
        raise RuntimeError('unexpected worker state')
    command = subprocess.run(['ps','-p',str(status['pid']),'-o','command='],capture_output=True,text=True).stdout
    if 'longmemeval_g1_continue.py' in command:
        raise RuntimeError('worker still running')
    plan = read(PUBLIC/'longmemeval-g1-window-deviation.json')
    deviation = read(PUBLIC/'longmemeval-g1-finalization-deviation.json')
    changed = deviation['approved_code_paths']
    protected = {**plan['original_source_sha256'], **plan['source_sha256']}
    code_hashes = {}
    for name, expected in protected.items():
        if name in changed:
            snapshot = OUT/(Path(name).name+'.before')
            if sha(snapshot) != expected or expected != changed[name]:
                raise RuntimeError('approved source snapshot mismatch: '+name)
            code_hashes[name] = {'registered_before':expected,'approved_after':sha(ROOT/name)}
        elif sha(ROOT/name) != expected:
            raise RuntimeError('protected hash changed: '+name)
    for filename, field in [('pending-final-report.json','before_pending_report_sha256'),
                            ('scored-rows.json','before_scored_rows_sha256'), ('review.json','before_review_sha256')]:
        if sha(LOCAL/filename) != deviation[field]:
            raise RuntimeError('saved output changed: '+filename)
    records = []
    counts = []
    for directory in (BASE/'run', LOCAL):
        if (directory/'inflight.json').exists():
            raise RuntimeError('unknown inflight; no completion')
        part = [json.loads(s) for s in (directory/'received.jsonl').read_text().splitlines()]
        counts.append(len(part)); records += part
    stopped = read(PUBLIC/'longmemeval-g1-continuation-stopped-report.json')
    if (sha(BASE/'run/received.jsonl') != stopped['source_journal_sha256'] or
            sha(LOCAL/'received.jsonl') != stopped['continuation_journal_sha256']):
        raise RuntimeError('journal changed since provisional report d0dfd13')
    if counts != [1441,210] or len({r['key'] for r in records}) != 1651:
        raise RuntimeError('incomplete/duplicate journal')
    if any(r['reply']['attempts'] != 1 for r in records):
        raise RuntimeError('multiple HTTP attempts')
    before = read(LOCAL/'pending-final-report.json'); after = deepcopy(before)
    rows = read(LOCAL/'scored-rows.json'); inputs = read(LOCAL/'answer-inputs.json')
    ids = plan['question_ids']; eligible = [q for q in ids if inputs['A'][q]['fits_context']]
    for arm in reporting.ARMS:
        if len(rows[arm]) != 30 or {r['id'] for r in rows[arm]} != set(ids):
            raise RuntimeError('row cohort mismatch')
        after['metrics_all30'][arm] = reporting.metrics(rows[arm])
        after['metrics_A_eligible'][arm] = reporting.metrics([r for r in rows[arm] if r['id'] in eligible])
        if any(after['metrics_all30'][arm][s]['questions'] != 6 for s in reporting.STRATA if s != 'overall'):
            raise RuntimeError('registered stratum count differs')
    # Recompute paired intervals from the saved numeric rows; no new correctness labels.
    for name, cohort, other, metric in [
        ('C_minus_A_semantic_primary',eligible,'A','correct'),
        ('C_minus_A_all30_operational',ids,'A','correct'),
        ('C_minus_B_all30',ids,'B1536','correct'),
        ('F1_C_minus_A_semantic_cohort_reference',eligible,'A','f1'),
        ('F1_C_minus_B_all30_reference',ids,'B1536','f1')]:
        after[name] = reporting.comparison(rows,cohort,other,metric)
    unchanged_comparisons(before,after)
    restore_log = (LOCAL/'qwen-original-restore.log').read_text()
    if not all(s in restore_log for s in ('model loaded','listening','n_ctx_slot = 131072')):
        raise RuntimeError('saved original restore readiness/context evidence incomplete')
    allocations = [json.loads(s)['reply'] for s in (LOCAL/'server-starts.jsonl').read_text().splitlines()]
    events = [json.loads(s)['reply'] for s in (BASE/'run/events.jsonl').read_text().splitlines()]
    ingest = next(e for e in events if e['kind']=='ingestion complete')
    fact = next(e for e in events if e['kind']=='ingestion milestone' and e['received_sessions']==1441)
    online = after['answer_timing']['C-final-norw']['registered_question_means']
    # Query embedding is already inside the observed ingestion wall, so do not count it twice.
    c_total = ingest['elapsed_seconds'] + 30*sum(online[k] for k in
               ('answer_HTTP_seconds','rerank_seconds','search_seconds','assembly_seconds'))
    after.update(time=datetime.now().astimezone().isoformat(), report_status='complete_cpu_finalization',
        worker_status_preserved='stopped; authorized CPU postprocessing completed separately',
        continuation_seconds_including_restore=status['elapsed_seconds'],
        machine_time_component_sum_seconds_excluding_human_wait=
            after['original_preflight_ingestion_embedding_and_stop_seconds']+status['elapsed_seconds'],
        allocation_records=allocations,
        restoration_verification={'scope':'saved log only, no live health or server operations',
            'ready_model_and_131072_logged':True,'allocation_values':'unrecorded, unknown; not zero',
            'restore_log_sha256':sha(LOCAL/'qwen-original-restore.log')},
        report_corrections={'approved_code_hashes':code_hashes,'overall_and_all_comparisons_exactly_unchanged':True,
            'registered_strata_each6':True,'type_table_scope':'descriptive tendencies only; no significance claim',
            'before_pending_report_sha256':sha(LOCAL/'pending-final-report.json')},
        C_total_cost={'one_history_one_question':True,'history_question_pairs':30,'sessions':1441,
            'fact_phase_seconds':fact['phase_elapsed_seconds'],
            'observed_preflight_fact_embedding_pipeline_seconds':ingest['elapsed_seconds'],
            'document_embedding_component_seconds':sum(v['document_seconds'] for v in after['embedding'].values()),
            'query_embedding_component_seconds':sum(v['query_seconds'] for v in after['embedding'].values()),
            'pipeline_plus_separate_online_components_seconds':c_total,'per_history_question_pair_seconds':c_total/30,
            'scope':'separately measured stage component sum, not end-to-end latency; query embedding already in ingestion wall; ingestion not amortized across questions'},
        judge_HTTP_seconds_sum=sum(r['reply']['wall_seconds'] for r in records if r['key'].startswith('judge|')),
        Codex_supplementary_review=human_list(read(LOCAL/'review.json'),plan))
    after['answer_resources'] = {a: resource_summary(v) for a,v in after['answer_resources'].items()}
    for field in ('rerank_resources', 'original_8k_rerank_idle_resources'):
        after[field] = resource_summary(after[field])
    write(PUBLIC/'longmemeval-g1-report.json',after)
    # Markdown includes the fixed type table and paired statistics plus explanatory cost/resource tables.
    primary = after['C_minus_A_semantic_primary']['overall']
    conclusion = ('主比較は判定不能。C−A −3.33pt、主履歴95%区間 −16.67〜+10.00pt。\n'
                  '登録Qwen正答はC 15/30、B 6/30、A 16/30。修正前後の全体値・比較区間は完全一致。\n'
                  '元8k並べ替え設計は24/30で窓超過し、登録逸脱で16kへ拡大した結果。英語30問・自己採点に限定。\n')
    if primary['classification'] != '判定不能':
        raise RuntimeError('unexpected registered final conclusion')
    md = '# G1 正式報告（CPU終了処理修正後）\n\n'+conclusion+'\n## 登録結果\n\n'+reporting.markdown(after).split('\n',2)[2]
    md += '\n## 費用・資源と主張範囲\n\n採点者は回答者と同じQwen3.8-27B。公式GPT4o scoreではない。英語30問、5層各6問。型別は傾向参考で有意性を主張しない。\n'
    md += '\n|工程|平均HTTP秒/問|KV GiB（ログ）|footprint peak GiB|RSS peak GiB|swap増分観測|\n|---|---:|---:|---:|---:|---|\n'
    resources = {'rerank':after['rerank_resources'],**after['answer_resources']}
    labels = {'rerank':'rerank','C-final-norw':'C','B1536':'B','A':'A'}
    for stage, resource in resources.items():
        record = next(r for r in allocations if r['label']==stage)
        mean = online['rerank_seconds'] if stage=='rerank' else after['answer_timing'][stage]['registered_question_means']['answer_HTTP_seconds']
        peak = resource['peak'];gib=1024**3
        md += f'|{labels[stage]}|{mean:.3f}|{record["allocations"]["bytes"]["KV"]/gib:.2f}|{peak["physical_footprint_bytes"]/gib:.3f}|{peak["rss_bytes"]/gib:.3f}|{resource["swap_affected"]}|\n'
    cost=after['C_total_cost']
    md += f'\n**Cの取り込み費用は別枠で総費用に含める。** 今回は1記憶に1問、30履歴に30問、1441セッション。fact段階 {cost["fact_phase_seconds"]:.3f}秒、文書embedding成分 {cost["document_embedding_component_seconds"]:.3f}秒。前処理・fact・embeddingの観測pipeline {cost["observed_preflight_fact_embedding_pipeline_seconds"]:.3f}秒。オンライン成分を加えたC総費用は {c_total:.3f}秒、1履歴1問あたり {c_total/30:.3f}秒。複数問への償却はしていない。query embeddingは取り込みwallに既に含まれ、二重加算しない。\n'
    md += '\n|回答条件|prefill秒/問|生成秒/問|組立preflight秒/問|\n|---|---:|---:|---:|\n'
    for arm in reporting.ARMS:
        means=after['answer_timing'][arm]['registered_question_means']
        md += f'|{labels[arm]}|{means["prefill_seconds"]:.3f}|{means["generation_seconds"]:.3f}|{means["assembly_preflight_seconds"]:.6f}|\n'
    md += f'\nC rerank {online["rerank_seconds"]:.3f}秒、CPU検索 {online["search_seconds"]:.6f}秒、組立 {online["assembly_seconds"]:.6f}秒、query embedding {online["query_embedding_seconds"]:.6f}秒（各問平均）。段階を別々に測った成分和でend-to-end待ち時間ではない。prefill/生成はHTTP内訳でHTTP秒に再加算しない。\n'
    md += f'\n旧機械時間 {after["original_preflight_ingestion_embedding_and_stop_seconds"]:.3f}秒、新continuation {status["elapsed_seconds"]:.3f}秒、合計 {after["machine_time_component_sum_seconds_excluding_human_wait"]:.3f}秒。人間待機・今回のCPU報告作業を含めない。開始swap約8.99GiB（1GiB以上の大印）、Aでswap増分観測あり、時間は参考値。footprint/RSSはQwen全体でGPU専用/純KVではない。ログ確保量とOS値は加算しない。model/compute/output/state/OSbaselineはJSON allocation_recordsで別保存。\n'
    md += '\n**元8k rerankは24/30で窓超過し、16kが必要だった。** KVは8k 0.50GiB→16k 1.00GiB。旧8kは0HTTPでidleのみ、16kの実測peakとの差を因果メモリ差とせず、速度差も未測定。精度と別の運用上の一般化限界。C/B入力超過0、A適格30/30・超過0。fact失敗3件は凍結fallbackのまま再生成なし。rerank/回答/judgeも再試行なし。\n'
    md += '\n終了復帰の通常verbosityログ必須チェックと、棄権を元種類へ重複計上する表示を人間許可の事後逸脱として修正。全体値と全paired比較/CI/符号検定の修正前後完全一致を確認。元worker stopped証跡・stopgateを保持し、CPU報告だけ完了。復帰は保存ログのmodel loaded/listening/131072まで確認、ライブhealth/確保量は未検査・不明。\n'
    sup=after['Codex_supplementary_review']
    md += f'\n匿名Qwen judge90件のHTTP合計 {after["judge_HTTP_seconds_sum"]:.3f}秒は評価費用として別計上し、Cの回答時総費用へ加算しない。\n'
    md += f'\nCodex補助採点: A {sup["correct_counts"]["A"]}/30、B {sup["correct_counts"]["B1536"]}/30、C {sup["correct_counts"]["C-final-norw"]}/30、Qwenと{sup["agreement_with_Qwen"]}/90判定一致。条件とQwen個別判定を隠して単回採点したが、全体集計は既知の事後レビュー。人手採点/登録採点を置換しない。差は同じ1問のA/C: 固有名詞が地理補足の省略を含めて正解を特定するかの判断。詳細はprivate一覧。人間確認対象。履歴照合はしていない。\n'
    md += '\n確認できた: 1651 unique/attempt1、unknown inflightなし、保護hashは承認した2コードの前後差を除き一致、5層各6、主比較不変。確認できない: 人手の最終正誤・学習混入・履歴独立性・一般的な意味精度・旧8k速度差。新しい実験は開始しない。private人手一覧と匿名一覧を保存、公開は集計/hash/修正コードのみ、pushなし。\n'
    (PUBLIC/'longmemeval-g1-report.md').write_text(md)
    verification={'time':after['time'],'status':'complete_cpu_finalization','source_ingestion_preserved':1441,
        'new_unique_records':210,'total_unique_records':1651,'all_HTTP_attempts_1':True,'no_unknown_inflight':True,
        'protected_hashes':{'total':len(protected),'unchanged':len(protected)-len(code_hashes),
            'approved_changed_code_before_snapshots_match':True,'approved_code_hashes':code_hashes},
        'exact_before_after_overall_and_all_pairwise_comparisons_match':True,
        'before_after_primary':{'before':before['C_minus_A_semantic_primary']['overall'],
            'after':after['C_minus_A_semantic_primary']['overall']},
        'registered_strata_each6':True,'tests':'synthetic CPU regression suite; see final commit and reported run; no HTTP/server',
        'old_received_journal_sha256':sha(BASE/'run/received.jsonl'),
        'new_received_journal_sha256':sha(LOCAL/'received.jsonl'),
        'report_sha256':sha(PUBLIC/'longmemeval-g1-report.json'),
        'markdown_sha256':sha(PUBLIC/'longmemeval-g1-report.md'),
        'private_review_sha256':sha(OUT/'human-review-with-labels.json'),
        'private_Codex_blind_labels_sha256':sha(OUT/'codex-blind-labels.json'),
        'finalization_source_sha256':{p:sha(ROOT/p) for p in
            ('scripts/eval/longmemeval_g1_finalize.py','scripts/eval/tests/test_longmemeval_g1_finalize.py')},
        'deviation_sha256':sha(PUBLIC/'longmemeval-g1-finalization-deviation.json'),
        'no_live_server_operation':True,'journal_hashes_equal_d0dfd13':True}
    write(PUBLIC/'longmemeval-g1-verification.json',verification)
    write(OUT/'completion.json',{'status':'complete_cpu_finalization','time':after['time'],
        'verification_sha256':sha(PUBLIC/'longmemeval-g1-verification.json'),
        'original_gate_and_worker_status_preserved':True,'automation':'PAUSED; human confirmation; no next experiment'})
    print(json.dumps({'status':after['report_status'],'primary':after['C_minus_A_semantic_primary']['overall']['classification'],
                      'unchanged_comparisons':True,'review':sup['correct_counts'],'records':1651},ensure_ascii=False))


if __name__ == '__main__':
    finalize()
