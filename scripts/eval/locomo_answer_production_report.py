"""Final report, reachable only after every registered heldout response is received."""
from __future__ import annotations

from collections import Counter

import locomo_answer_production as production
import locomo_answer_report as ar
import locomo_answer_methods as am
import locomo_memory_reporting as inference

lm, ae = production.lm, production.ae


def ready(prepared: dict, replies: dict, reg: dict, state: dict) -> None:
    if state.get('phase') != 'answer complete' or state.get('status') != 'complete':
        raise RuntimeError('answer runner must finalize before any heldout scoring')
    if len(prepared['items']) != reg['plan']['responses']:
        raise RuntimeError('prepared response count differs from registration')
    if len({item['id'] for item in prepared['items'].values()}) != reg['plan']['questions']:
        raise RuntimeError('prepared question count differs from registration')
    ar.complete(prepared, replies, {'plan': reg['plan']['answer'], 'plan_sha256': reg['plan_sha256']})


def markdown(record: dict) -> str:
    lines = ['# LoCoMo回答評価（本番用302問）', '',
        '評価範囲はLoCoMoの同じ10会話の、調整に使わなかった質問への性能。'
        '質問単位splitであり会話単位では独立でない。以前の公開検索評価で同じ質問への接触がある。'
        '未知の会話や日本語会話への一般化は評価していない。token F1は語彙一致であり意味的正答率ではない。',
        '会話単位95%区間：10000回seed0、質問macro比。会話平均差の符号検定は厳密両側、'
        'ゼロ差を除外。個別区間と未補正p値。Bは末尾の原文だけという弱い比較対象。',
        '主比較は凍結済みC-final対AとB1536。C8192-finalは追加の探索的比較。'
        '±0.03許容差と+0.05実用改善幅を変更していない。0をまたぐだけでは同等性を主張しない。', '']
    for stratum, metrics in record['tables'].items():
        lines += [f'## {stratum}', '']
        shown = ('f1',) if stratum.startswith('category:') else ar.METRICS
        for metric in shown:
            lines += [f'### {metric}：全文Aとの差（t1/p2併記）', '']
            lines += inference.table(metrics[metric], 'A', record['groups']) + ['']
        for reference in ('B1536', 'B8192', 'C-final'):
            lines += [f'### F1：{reference}との差（追加比較を含む）', '']
            lines += inference.table(metrics['f1'], reference, record['groups']) + ['']
    for metric in ('wall_seconds', 'prefill_seconds', 'generation_seconds'):
        lines += [f'## 時間：{metric}', '']
        lines += inference.table(record['tables']['overall'][metric], 'A', record['groups']) + ['']
    lines += ['## 解釈（登録済み基準）', '']
    for key, value in record['interpretation_checks'].items():
        lines.append(f'- {key}: {value}')
    lines += ['', '注釈上の答えなし質問は0件。定型Insufficient evidenceは0点とし棄権率を上に併記。'
        '出力上限・パース失敗は0点、受信済み応答の再生成はしない。'
        '本文と生成本文はgitignoreされたbuild/eval/にのみ保存。', '']
    return '\n'.join(lines)


def report() -> None:
    reg = production.checked()
    path = lm.RESULTS / 'answer-report.json'
    if path.exists():
        if lm.read(path)['plan_sha256'] != reg['plan_sha256']:
            raise RuntimeError('existing final report refers to different registration')
        return
    prepared = lm.read(production.LOCAL / 'answer-inputs.json')
    replies = ae.recover(production.LOCAL / 'answer-responses.json', production.LOCAL / 'answer-received.jsonl')
    state = lm.read(production.LOCAL / 'status.json')
    ready(prepared, replies, reg, state)
    rows = ar.rows_for(prepared, replies, reg['plan']['answer']['aliases'])
    tables = ar.paired_tables(rows)
    checks = {}
    for kind in ('type:multiple_evidence', 'overall', 'type:single_fact', 'type:temporal'):
        a = tables[kind]['f1']['C-final']['A']['conversation_ci95']
        b = tables[kind]['f1']['C-final']['B1536']['conversation_ci95']
        checks[kind] = {'C-final_minus_A_inside_registered_plus_minus_0_03': a[0] >= -.03 and a[1] <= .03,
            'C-final_minus_B1536_lower_above_zero': b[0] > 0,
            'C-final_minus_B1536_lower_above_practical_0_05': b[0] > .05}
    rerank = ae.recover(production.LOCAL / 'rerank-responses.json', production.LOCAL / 'rerank-received.jsonl')
    record = {'created_at': lm.stamp(), 'split': 'heldout302', 'scope': reg['plan']['scope'],
        'plan_sha256': reg['plan_sha256'], 'groups': sorted(reg['plan']['conversation_counts']),
        'questions': 302, 'responses': len(replies), 'conditions': am.CONDITIONS, 'tables': tables,
        'interpretation_checks': checks,
        'failure_counts': {a: dict(Counter(r['failure_kind'] or 'none' for r in part)) for a, part in rows.items()},
        'response_attempts': dict(Counter(str(r['attempts']) for r in replies.values())),
        'rerank_response_attempts': dict(Counter(str(r['attempts']) for r in rerank.values())),
        'generation': reg['plan']['answer']['generation'],
        'search_preparation': lm.read(lm.RESULTS / 'answer-search-preparation.json'),
        'answer_preparation': lm.read(lm.RESULTS / 'answer-preparation.json'),
        'context_switches': lm.read(production.LOCAL / 'context-switches.json'),
        'received_sha256': lm.file_hash(production.LOCAL / 'answer-received.jsonl')}
    lm.write(production.LOCAL / 'scored-rows.json', rows)
    lm.write(path, record)
    (lm.RESULTS / 'answer-report.md').write_text(markdown(record))
    production.status('answer evaluation complete', status='complete', completed=len(replies), total=len(replies))
    print('all heldout responses complete; final report written', flush=True)
