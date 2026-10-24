"""Report-only paired conversation inference; never participates in configuration selection."""
from __future__ import annotations

from math import comb

import numpy as np

RESAMPLES, SEED = 10000, 0
T1, P2 = 't1/dense/orig/none', 'p2/dense/orig/none'
METRICS = ('recall@T512', 'recall@5', 'recall@10', 'recall@20', 'all_evidence@10', 'all_evidence@T512')


def paired(rows: list[dict], reference: list[dict], metric: str) -> dict:
    if [r['id'] for r in rows] != [r['id'] for r in reference]:
        raise ValueError('reporting requires identical query order')
    if not rows:
        return {'questions': 0, 'conversations': 0, 'per_conversation': {}, 'mean': None,
                'mean_diff': None, 'conversation_ci95': None, 'sign_test': None}
    delta = np.array([r[metric] - b[metric] for r, b in zip(rows, reference)])
    groups = [r['id'].split('/')[0] for r in rows]
    names = sorted(set(groups))
    sums = np.array([delta[np.array(groups) == g].sum() for g in names])
    counts = np.array([groups.count(g) for g in names])
    means = sums / counts
    pick = np.random.default_rng(SEED).integers(0, len(names), (RESAMPLES, len(names)))
    samples = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    wins, losses, ties = int((means > 0).sum()), int((means < 0).sum()), int((means == 0).sum())
    n = wins + losses
    p = min(1., 2 * sum(comb(n, j) for j in range(min(wins, losses) + 1)) / 2 ** n)
    return {'questions': len(rows), 'conversations': len(names),
            'mean': float(np.mean([r[metric] for r in rows])), 'mean_diff': float(delta.mean()),
            'conversation_ci95': np.percentile(samples, [2.5, 97.5]).tolist(),
            'per_conversation': {g: {'questions': int(c), 'mean_diff': float(m)}
                                 for g, c, m in zip(names, counts, means)},
            'sign_test': {'wins': wins, 'losses': losses, 'ties': ties, 'n': n, 'two_sided_p': p}}


def comparisons(candidates: dict, references: dict, base: str) -> dict:
    """Add frozen t1/p2 rows for display, with strict alignment; don't mutate candidates."""
    for cid in (T1, P2):
        if cid in candidates:
            if [{k: r[k] for k in ('id', *METRICS)} for r in candidates[cid]] != [
                    {k: r[k] for k in ('id', *METRICS)} for r in references[cid]]:
                raise ValueError('t1/p2 baseline reproduction differs')
    display = {T1: references[T1], P2: references[P2], **candidates}
    ids = [r['id'] for r in candidates[base]]
    for rows in display.values():
        if [r['id'] for r in rows] != ids:
            raise ValueError('baseline and stage development query order differ')
        if any(r['type'] != b['type'] or r['category'] != b['category']
               for r, b in zip(rows, candidates[base])):
            raise ValueError('baseline and stage query strata differ')
    strata = {'overall': lambda r: True}
    for kind in ('multiple_evidence', 'single_fact', 'temporal'):
        strata[f'type:{kind}'] = lambda r, kind=kind: r['type'] == kind
    for category in sorted({r['category'] for r in candidates[base]}):
        strata[f'category:{category}'] = lambda r, category=category: r['category'] == category
    output = {}
    for label, keep in strata.items():
        subsets = {cid: [r for r in rows if keep(r)] for cid, rows in display.items()}
        output[label] = {metric: {cid: {ref: paired(rows, subsets[ref], metric)
                                      for ref in dict.fromkeys((T1, P2, base))}
                                 for cid, rows in subsets.items()} for metric in METRICS}
    return {'method': 'inference-policy.json; 10000 whole-conversation resamples, seed 0; question-macro ratio; '
                      'exact two-sided sign test, exact zero ties excluded; pointwise CI and unadjusted p',
            'references': [T1, P2], 'stage_base': base,
            'groups': sorted({r['id'].split('/')[0] for r in candidates[base]}), 'strata': output}


def table(entries: dict, reference: str, groups: list[str]) -> list[str]:
    lines = [f'比較対象 `{reference}`。会話別Δの順序：' + ', '.join(groups) + '。', '',
             '| 設定 | 問数 | 値 | Δ | 会話単位95%区間 | 符号検定 勝/負/同 (n; p) | 会話別Δ |',
             '|---|---:|---:|---:|---|---|---|']
    for cid, refs in entries.items():
        s = refs[reference]
        if not s['questions']:
            lines.append(f'| `{cid}` | 0 | — | — | — | — | — |')
            continue
        lo, hi = s['conversation_ci95']
        sign = s['sign_test']
        per = ', '.join(f'{s["per_conversation"][g]["mean_diff"]:+.4f}'
                        if g in s['per_conversation'] else '—' for g in groups)
        lines.append(f'| `{cid}` | {s["questions"]} | {s["mean"]:.4f} | {s["mean_diff"]:+.4f} '
                     f'| [{lo:+.4f}, {hi:+.4f}] | {sign["wins"]}/{sign["losses"]}/{sign["ties"]} '
                     f'({sign["n"]}; {sign["two_sided_p"]:.5f}) | {per} |')
    return lines


def markdown(report: dict) -> str:
    stats, selected = report['reporting'], report['selection']['selected']
    base = report['base']
    lines = [f'# 外部記憶・検索評価：ステージ{report["stage"]}（開発用のみ）', '',
             '本番用は未計測。以下は同じ10会話の開発用質問への探索的な比較であり、'
             '「LoCoMoの同じ10会話の、調整に使わなかった質問への性能」はまだ測っていない。',
             '未知の会話・日本語会話への一般化は評価していない。区間は会話単位の対応あり'
             'ブートストラップ（10000回、seed 0）。点推定は質問macro平均。'
             '符号検定は会話ごとの平均差、両側・厳密検定。ゼロ差の会話は除外して併記する。',
             '全候補を掲載。区間は個別の95%区間、p値は多重比較補正なし。'
             '選択は変更していない事前登録の質問単位の区間による探索的な規則。', '',
             f'**選択**：`{selected}`。ステージ基準：`{base}`。',
             'R@kは上位k検索単位の返した原文、R@T512は既定の埋め込みtokenizerで'
             '原文を512トークンまで詰めた根拠Recall。Allは必要な根拠が全て入った質問の割合。',
             '種類は凍結した代理分類（カテゴリ2→時系列、それ以外でカテゴリ1または複数根拠→'
             'multiple_evidence、残り→single_fact）。厳密な意味分類ではない。', '']
    labels = {'type:multiple_evidence': '**複数根拠を要する質問（最重要）**', 'overall': '全体',
              'type:single_fact': '単一の事実', 'type:temporal': '時系列'}
    order = ['type:multiple_evidence', 'overall', 'type:single_fact', 'type:temporal']
    order += [s for s in stats['strata'] if s.startswith('category:')]
    for stratum in order:
        lines += [f'## {labels.get(stratum, stratum)}', '']
        metrics = METRICS if not stratum.startswith('category:') else METRICS[:1]
        for metric in metrics:
            lines += [f'### {metric}（t1・p2・全候補）', '']
            lines += table(stats['strata'][stratum][metric], T1, stats['groups']) + ['']
        lines += ['### ステージ基準との比較（選択指標 recall@T512）', '']
        lines += table(stats['strata'][stratum]['recall@T512'], base, stats['groups']) + ['']
    if report['generation']:
        lines += ['生成物（失敗時は登録済みの代替）：' + ', '.join(
            f'{k} {v["items"]}件、代替{v["fallback"]}件' for k, v in report['generation'].items()), '']
    lines += ['生の質問・会話・生成本文はgitignoreされたbuild/eval/に保存。'
              '検索・生成条件はstages.jsonとgeneration.json、解釈はinference-policy.jsonに従う。', '']
    return '\n'.join(lines)
