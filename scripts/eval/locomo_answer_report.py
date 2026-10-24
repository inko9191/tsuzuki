#!/usr/bin/env python3
"""CPU-only pilot reporting; refuses incomplete responses and never opens the holdout file."""
from __future__ import annotations

from collections import Counter, defaultdict
import json

import numpy as np

import locomo_answer_eval as ae
import locomo_answer_methods as am
import locomo_memory_reporting as inference

lm = ae.lm
TYPES = ('multiple_evidence', 'single_fact', 'temporal')
METRICS = ('f1', 'em', 'abstention', 'failure', 'empty_answer', 'inserted_all', 'inserted_recall')
CONTRASTS = (('C-final', 'A'), ('C-final', 'B1536'),
             ('C8192-final', 'A'), ('C8192-final', 'B8192'))


def complete(prepared: dict, replies: dict, registered: dict) -> None:
    if set(prepared['items']) != set(replies):
        raise RuntimeError('all pilot conditions must finish before scoring or forecasting')
    if prepared['plan_sha256'] != registered['plan_sha256']:
        raise RuntimeError('pilot input registration differs')
    for key, item in prepared['items'].items():
        expected = lm.digest({'request': ae.request(registered['plan'], item),
                              'plan': registered['plan_sha256'], 'server': prepared['server']})
        if replies[key]['digest'] != expected:
            raise RuntimeError('successful response is not from the registered request')
        ae.checked_timings(replies[key])


def rows_for(prepared: dict, replies: dict, sharing: dict) -> dict:
    rows = {}
    ids = sorted({item['id'] for item in prepared['items'].values()})
    for arm in am.CONDITIONS:
        part = []
        for qid in ids:
            key = f'{sharing[arm]}|{qid}'
            item, received = prepared['items'][key], replies[key]
            t = ae.checked_timings(received)
            prefill, generated = t['prompt_ms'] / 1000, t['predicted_ms'] / 1000
            part.append({'id': qid, 'type': item['type'], 'category': item['category'],
                         **am.score(received['response'], item['gold']),
                         'inserted_all': item['inserted_all'], 'inserted_recall': item['inserted_recall'],
                         'wall_seconds': received['wall_seconds'], 'prefill_seconds': prefill,
                         'generation_seconds': generated,
                         'overhead_seconds': received['wall_seconds'] - prefill - generated,
                         'cache_tokens': t['cache_n'], 'processed_prompt_tokens': t['prompt_n'],
                         'generated_tokens': t['predicted_n'], 'input_tokens': item['input_tokens'],
                         'conversation_tokens': item['conversation_tokens'], 'memory_tokens': item['memory_tokens']})
        rows[arm] = part
    return rows


def paired_tables(rows: dict) -> dict:
    strata = {'type:multiple_evidence': lambda r: r['type'] == 'multiple_evidence',
              'overall': lambda r: True,
              'type:single_fact': lambda r: r['type'] == 'single_fact',
              'type:temporal': lambda r: r['type'] == 'temporal'}
    for category in sorted({r['category'] for r in rows['A']}):
        strata[f'category:{category}'] = lambda r, c=category: r['category'] == c
    references = ['A', 'B1536', 'B8192', 'C-p2', 'C-final']
    result = {}
    for name, keep in strata.items():
        parts = {arm: [r for r in part if keep(r)] for arm, part in rows.items()}
        result[name] = {metric: {arm: {ref: inference.paired(part, parts[ref], metric)
                                      for ref in references} for arm, part in parts.items()}
                        for metric in (*METRICS, 'wall_seconds', 'prefill_seconds', 'generation_seconds')}
    return result


def distribution(values) -> dict:
    v = np.asarray(values, float)
    return {'n': len(v), 'mean': float(v.mean()), 'median': float(np.median(v)),
            'p90': float(np.percentile(v, 90))}


def timing_profile(prepared: dict, rows: dict, sharing: dict, target_counts: dict) -> dict:
    """Cold = observed first request of the block, not a forced cache eviction."""
    by_id = {arm: {r['id']: r for r in part} for arm, part in rows.items()}
    blocks = defaultdict(list)
    for key in prepared['schedule']:
        item = prepared['items'][key]
        blocks[(item['id'].split('/')[0], item['condition'])].append(by_id[item['condition']][item['id']])
    metrics = ('prefill_seconds', 'generation_seconds', 'overhead_seconds', 'wall_seconds',
               'input_tokens', 'cache_tokens', 'processed_prompt_tokens', 'generated_tokens')
    result, totals = {}, {arm: {'mean_seconds': 0., 'p90_components_seconds': 0.}
                          for arm in dict.fromkeys(sharing.values())}
    for (group, arm), part in blocks.items():
        cold, warm = part[0], part[1:]
        # Record the observation; use cold prefill if no additional prefix was reused on warm requests.
        cache_effective = np.mean([r['cache_tokens'] for r in warm]) > cold['cache_tokens']
        entry = {'first': {m: cold[m] for m in metrics},
                 'warm': {m: distribution([r[m] for r in warm]) for m in metrics},
                 'all': {m: distribution([r[m] for r in part]) for m in metrics},
                 'warm_cache_exceeds_first': bool(cache_effective)}
        n = target_counts[group]
        entry['target_questions'] = n
        entry['projected'] = {}
        for label, estimator in [('mean_seconds', 'mean'), ('p90_components_seconds', 'p90')]:
            pcold = cold['prefill_seconds']
            pwarm = entry['warm']['prefill_seconds'][estimator] if cache_effective else pcold
            g = entry['all']['generation_seconds'][estimator]
            o = entry['all']['overhead_seconds'][estimator]
            entry['projected'][label] = pcold + (n - 1) * pwarm + n * (g + o)
            totals[arm][label] += entry['projected'][label]
        result.setdefault(group, {})[arm] = entry
    return {'blocks': result, 'per_condition_projection': totals,
            'all_pilot': {arm: {m: distribution([r[m] for r in rows[arm]]) for m in metrics}
                          for arm in am.CONDITIONS},
            'total_answer_seconds': {label: sum(t[label] for t in totals.values())
                                     for label in ('mean_seconds', 'p90_components_seconds')},
            'note': 'p90 of components is a conservative planning scenario, not a calibrated total-time percentile; '
                    'first/warm depend on the registered order and are not forced cold/warm cache states'}


def precision_forecast(rows: dict, dev_metadata: list[dict], target_counts: dict,
                       simulations: int = 500, resamples: int = 10000) -> dict:
    """Paired within-cell sampling, then question-weighted whole-conversation bootstrap."""
    ids = [r['id'] for r in rows['A']]
    if any([r['id'] for r in part] != ids for part in rows.values()):
        raise ValueError('paired pilot question order differs')
    groups = sorted(target_counts)
    cells = {(g, t): [i for i, r in enumerate(rows['A']) if r['id'].split('/')[0] == g and r['type'] == t]
             for g in groups for t in TYPES}
    if any(not cell for cell in cells.values()):
        raise ValueError('missing conversation/type pilot cell')
    counts = Counter((r['group'], r['type']) for r in dev_metadata)
    weights = {g: np.array([counts[(g, t)] for t in TYPES], float) for g in groups}
    weights = {g: v / v.sum() for g, v in weights.items()}
    deltas = {f'{a}-{b}': np.array([ra['f1'] - rb['f1'] for ra, rb in zip(rows[a], rows[b])])
              for a, b in CONTRASTS}
    expected = {}
    for label, delta in deltas.items():
        expected[label] = {}
        for kind in ('overall', *TYPES):
            numerator = denominator = 0.
            for g in groups:
                for ti, t in enumerate(TYPES):
                    if kind != 'overall' and t != kind:
                        continue
                    n = target_counts[g] * weights[g][ti]
                    numerator += n * delta[cells[(g, t)]].mean()
                    denominator += n
            expected[label][kind] = numerator / denominator
    rng = np.random.default_rng(0)
    samples = {label: {kind: [] for kind in ('overall', *TYPES)} for label in deltas}
    for _ in range(simulations):
        chosen, sample_groups, sample_types = [], [], []
        for gi, g in enumerate(groups):
            kinds = rng.choice(len(TYPES), target_counts[g], p=weights[g])
            for ti in kinds:
                chosen.append(rng.choice(cells[(g, TYPES[ti])]))
                sample_groups.append(gi)
                sample_types.append(ti)
        chosen, sample_groups, sample_types = map(np.asarray, (chosen, sample_groups, sample_types))
        for kind in ('overall', *TYPES):
            keep = np.ones(len(chosen), bool) if kind == 'overall' else sample_types == TYPES.index(kind)
            active = np.unique(sample_groups[keep])
            if not len(active):
                raise ValueError('simulated stratum has no questions; cannot estimate its interval width')
            n = np.array([(sample_groups[keep] == gi).sum() for gi in active])
            picks = np.random.default_rng(0).integers(0, len(active), (resamples, len(active)))
            for label, delta in deltas.items():
                sums = np.array([delta[chosen[keep][sample_groups[keep] == gi]].sum() for gi in active])
                means = sums[picks].sum(axis=1) / n[picks].sum(axis=1)
                lo, hi = np.percentile(means, [2.5, 97.5])
                samples[label][kind].append((float(hi - lo), float(lo - expected[label][kind])))
    output = {}
    for label, strata in samples.items():
        output[label] = {}
        for kind, values in strata.items():
            v = np.asarray(values)
            grid = np.arange(61) * .005
            probabilities = [float(np.mean(v[:, 1] + shift > 0)) for shift in grid]
            passing = [float(shift) for shift, probability in zip(grid, probabilities) if probability >= .8]
            width = np.percentile(v[:, 0], [10, 50, 90]).tolist()
            output[label][kind] = {'width_p10_median_p90': width,
                'conditional_detectable_difference_80_percent': passing[0] if passing else None,
                'dev_type_weighted_difference': float(expected[label][kind]),
                'zero_center_width_exceeds_equivalence_span': bool(width[1] > .06),
                'shift_sensitivity': [{'difference': float(s), 'detection_fraction': p}
                                      for s, p in zip(grid, probabilities)]}
    return {'comparisons': output, 'simulations': simulations, 'conversation_bootstrap_resamples': resamples,
            'seed': 0, 'target_counts': target_counts,
            'development_type_weights': {g: dict(zip(TYPES, w.tolist())) for g, w in weights.items()},
            'note': 'forecast from four questions per conversation/type; not holdout performance or a power '
                    'guarantee. Constant shifts do not respect F1 bounds or changing conversation effects. '
                    'Bootstrap errors are centered on the development-weighted expected simulation mean.'}


def markdown(record: dict) -> str:
    groups = record['groups']
    lines = ['# 回答評価：開発用120問の実測', '',
             '同じ10会話の開発用pilotのみ。本番用は未計測。種類ごと40問の均等pilotであり、'
             '素の平均を本番性能の予測値とは呼ばない。token F1は意味的正答率ではない。',
             '会話単位95%区間は10000回seed0の質問macro比。符号検定は会話平均差の厳密両側検定、'
             '同点を除外。個別区間・未補正p値による探索的比較。', '']
    for name, metrics in record['tables'].items():
        lines += [f'## {name}', '']
        shown_metrics = ('f1',) if name.startswith('category:') else METRICS
        for metric in shown_metrics:
            lines += [f'### {metric}', '']
            lines += inference.table(metrics[metric], 'A', groups) + ['']
        lines += ['### F1：弱い末尾B1536との比較', '']
        lines += inference.table(metrics['f1'], 'B1536', groups) + ['']
    lines += ['## 時間：すべての条件', '',
              '平均wall秒の対応差はAを基準。読み込み・生成・overhead・cache token数の'
              '各会話cold/warm、平均・中央値・p90はJSONに記録。']
    for metric in ('wall_seconds', 'prefill_seconds', 'generation_seconds'):
        lines += ['', f'### {metric}', '']
        lines += inference.table(record['tables']['overall'][metric], 'A', groups) + ['']
    lines += ['## 区間幅と検出差の見込み', '',
              '500模擬集合を開発の会話内種類比で生成し、各集合で10000回の会話bootstrap。'
              '各枠4問からの仮定付き見積もりであり、検出力保証ではない。']
    for label, strata in record['precision_forecast']['comparisons'].items():
        for kind in ('multiple_evidence', 'overall', 'single_fact', 'temporal'):
            s = strata[kind]
            lo, med, hi = s['width_p10_median_p90']
            detect = s['conditional_detectable_difference_80_percent']
            lines.append(f'- {label} / {kind}: 区間幅中央値 {med:.4f}（10〜90百分位 {lo:.4f}〜{hi:.4f}）、'
                         f'仮定付き80%検出差 {detect if detect is not None else "0.30まででは未達"}。')
    lines += ['', '幅が0.06を超える比較では、中心が0でも±0.03の基準で「ほぼ同じ」とは主張できない。'
              '本番の件数・回数・条件・閾値はこの結果から変更しない。',
              '条件ごとの時間投影と補助費用はJSONを参照。本番計画は費用欄をすべて確認してから固定する。', '']
    return '\n'.join(lines)


def main() -> None:
    registered = ae.checked_plan()
    analysis = ae.st.frozen(lm.RESULTS / 'answer-pilot-analysis-plan.json')
    if analysis['pilot_plan_sha256'] != lm.file_hash(ae.PLAN_PATH):
        raise RuntimeError('analysis refers to a different pilot plan')
    for name, expected in analysis['code_sha256'].items():
        if lm.file_hash(lm.ROOT / 'scripts/eval' / name) != expected:
            raise RuntimeError('registered pilot analysis implementation changed')
    prepared = lm.read(ae.LOCAL / 'pilot-inputs.json')
    replies = ae.recover(ae.LOCAL / 'pilot-responses.json', ae.LOCAL / 'pilot-received.jsonl')
    complete(prepared, replies, registered)
    state = lm.read(ae.LOCAL / 'status.json')
    if state.get('phase') != 'pilot complete' or state.get('status') != 'complete':
        raise RuntimeError('pilot process has not finalized')
    plan = registered['plan']
    rows = rows_for(prepared, replies, plan['aliases'])
    dev, manifest = lm.load_dev()
    dev_metadata = [{'group': q['group'], 'type': lm.query_type(q, dev['qrels'][q['id']])} for q in dev['queries']]
    counts = manifest['splits']['holdout']['conversations']  # Counts only; holdout file never opened.
    timing = timing_profile(prepared, rows, plan['aliases'], counts)
    preparation = lm.read(lm.RESULTS / 'answer-pilot-preparation.json')['seconds']
    switch = lm.read(lm.RESULTS / 'answer-pilot-server.json')['restart_seconds']
    dev_actual = state['seconds_this_call'] + preparation + switch
    rerank = lm.read(lm.RESULTS / 'stage5-dev.json')['configs'][plan['configs']['C-final']['id']]['seconds']
    rerank_projection = rerank / manifest['splits']['dev']['queries'] * sum(counts.values())
    inputs_projection = preparation / len(prepared['items']) * sum(counts.values()) * len(set(plan['aliases'].values()))
    auxiliary = {'heldout_rerank_seconds_mean_projection': rerank_projection,
                 'heldout_embedding_search_and_input_preparation_seconds_projection': inputs_projection,
                 'two_planned_context_switches_seconds_projection': 2 * switch,
                 'scoring_save_seconds': None,
                 'note': 'rerank from stage5 dev wall average; input preparation includes cached CPU search, '
                         'tokenization and preflight. Fresh holdout query embedding and scoring/save costs '
                         'must be measured or bounded before the complete production time plan is frozen.'}
    record = {'created_at': lm.stamp(), 'split': 'development pilot', 'holdout_evaluated': False,
              'plan_sha256': registered['plan_sha256'], 'groups': sorted(counts),
              'questions': 120, 'responses': len(replies), 'conditions': am.CONDITIONS,
              'tables': paired_tables(rows), 'timing': timing, 'auxiliary_costs': auxiliary,
              'development_preparation_actual_seconds': dev_actual,
              'precision_forecast': precision_forecast(rows, dev_metadata, counts),
              'received_sha256': lm.file_hash(ae.LOCAL / 'pilot-received.jsonl'),
              'failure_counts': {a: dict(Counter(r['failure_kind'] or 'none' for r in rs)) for a, rs in rows.items()},
              'response_attempts': dict(Counter(str(r['attempts']) for r in replies.values()))}
    lm.write(ae.LOCAL / 'pilot-scored-rows.json', rows)
    lm.write(lm.RESULTS / 'answer-pilot-report.json', record)
    (lm.RESULTS / 'answer-pilot-report.md').write_text(markdown(record))
    print('complete development pilot report written; no holdout questions opened', flush=True)


if __name__ == '__main__':
    main()
