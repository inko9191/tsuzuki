"""CPU-only reporting for the amended G1 protocol."""
from collections import Counter
import numpy as np
import locomo_answer_methods as am
import longmemeval_g1_methods as methods

ARMS = ('C-final-norw', 'B1536', 'A')
STRATA = ('multi-session', 'overall', 'temporal-reasoning', 'knowledge-update',
          'single-session', 'abstention')


def classification(ci):
    if ci is None:
        return '判定不能'
    if ci[0] >= -.03:
        return '成立'
    if ci[1] < 0:
        return '不成立'
    return '判定不能'


def stratum(row):
    if row['question_id'].endswith('_abs'):
        return 'abstention'
    kind = row['question_type']
    return 'single-session' if kind.startswith('single-session-') else kind


def answer_value(reply, gold):
    if reply.get('failure_kind'):
        return dict(answer='', f1=0., abstention=0., failure=1., failure_kind=reply['failure_kind'])
    parsed, failure = am.parse(reply['response'])
    score = am.score(reply['response'], gold)
    return {'answer': parsed['answer'] if parsed else '', **score, 'failure_kind': failure}


def metrics(rows):
    result = {}
    kinds = (*STRATA, 'single-session-user', 'single-session-assistant', 'single-session-preference')
    for kind in kinds:
        if kind == 'overall':
            part = rows
        elif kind in STRATA:
            part = [r for r in rows if r['stratum'] == kind]
        else:
            part = [r for r in rows if r['stratum'] == 'single-session' and r['type'] == kind]
        result[kind] = {'questions': len(part)}
        if part:
            result[kind].update({k: float(np.mean([r[k] for r in part]))
                                for k in ('correct', 'f1', 'abstention', 'failure', 'judge_failure')})
    return result


def comparison(rows, ids, other, metric='correct'):
    index = {a: {r['id']: r for r in rows[a]} for a in ARMS}
    result = {}
    for kind in STRATA:
        chosen = [q for q in ids if kind == 'overall' or index['C-final-norw'][q]['stratum'] == kind]
        if not chosen:
            result[kind] = {'questions': 0, 'classification': '判定不能', 'ci95': None}
            continue
        differences = [index['C-final-norw'][q][metric]-index[other][q][metric] for q in chosen]
        groups = [index[other][q]['history_group'] for q in chosen]
        strata = [index[other][q]['stratum'] for q in chosen]
        ci = methods.interval(differences, groups)
        supplemental = methods.interval(differences, chosen, strata)
        result[kind] = {'questions': len(chosen), 'difference': float(np.mean(differences)),
                        'ci95': ci, 'stratified_question_ci95': supplemental,
                        'classification': classification(ci) if metric == 'correct' else '参考語彙F1',
                        'supplemental_classification': classification(supplemental) if metric == 'correct' else '参考語彙F1',
                        'equivalence_both_intervals_within_0_03': all(-.03 <= low and high <= .03
                            for low, high in (ci, supplemental)),
                        'sign_test': methods.sign_test(differences),
                        'by_question': dict(zip(chosen, differences)),
                        'A_input_eligible_strata': dict(Counter(strata))}
    return result


def analyze(questions, inputs, answers, judges):
    rows, review = {a: [] for a in ARMS}, []
    ids = list(questions)
    if any(set(answers[a]) != set(ids) or set(judges[a]) != set(ids) for a in ARMS):
        raise RuntimeError('incomplete registered paired cohort')
    eligible = [q for q in ids if inputs['A'][q]['fits_context']]
    for qid, question in questions.items():
        item = {'id': qid, 'question': question['question'], 'gold': question['answer'],
                'question_type': question['question_type'], 'conditions': {},
                'history_reference': 'prepared.json: questions/'+qid}
        for arm in ARMS:
            answer = answer_value(answers[arm][qid], question['answer'])
            judge = judges[arm][qid]
            row = {k: v for k, v in answer.items() if k != 'answer'}
            row.update(id=qid, stratum=stratum(question), type=question['question_type'],
                       history_group=question['history_group'], correct=judge['label'],
                       judge_failure=float(bool(judge.get('failure'))),
                       input_tokens=inputs[arm][qid]['input_tokens'])
            rows[arm].append(row)
            item['conditions'][arm] = {'answer': answer['answer'], 'judge': judge,
                                       'failure': answer['failure_kind'],
                                       'evidence_ids': (am.parse(answers[arm][qid].get('response', {}))[0] or {}).get('evidence_ids', [])}
        review.append(item)
    public = {'questions': len(ids), 'A_eligible': len(eligible), 'A_overflow': len(ids)-len(eligible),
              'metrics_all30': {a: metrics(rows[a]) for a in ARMS},
              'metrics_A_eligible': {a: metrics([r for r in rows[a] if r['id'] in eligible]) for a in ARMS},
              'C_minus_A_semantic_primary': comparison(rows, eligible, 'A'),
              'C_minus_A_all30_operational': comparison(rows, ids, 'A'),
              'C_minus_B_all30': comparison(rows, ids, 'B1536'),
              'F1_C_minus_A_semantic_cohort_reference': comparison(rows, eligible, 'A', 'f1'),
              'F1_C_minus_B_all30_reference': comparison(rows, ids, 'B1536', 'f1'),
              'A_overflow_ids': [q for q in ids if q not in eligible],
              'scope': 'fixed30 stratified English benchmark; self judge; training contamination unknown'}
    ordered = methods.review_order(ids, {a: {r['id']: r['conditions'][a]['answer'] for r in review} for a in ARMS})
    lookup = {r['id']: r for r in review}
    return public, rows, [lookup[q] for q in ordered]


def markdown(report):
    lines = ['# LongMemEval G1（登録英語標本）', '',
             '固定30問、Qwen自己judge、背景履歴の依存/学習混入は未確認。時間はswap状態を併記する。', '',
             '|条件|種類|問数|judge正誤率|語彙F1（参考）|棄権率|生成失敗率|judge失敗率|',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for arm, metrics_ in report['metrics_all30'].items():
        for kind in STRATA:
            m = metrics_[kind]
            if m['questions']:
                values = '|'.join(f'{m[k]:.4f}' for k in ('correct', 'f1', 'abstention', 'failure', 'judge_failure'))
                lines.append(f'|{arm}|{kind}|{m["questions"]}|{values}|')
    lines += ['', f'A-eligible={report["A_eligible"]}, A-window-overflow={report["A_overflow"]}。意味的主比較と全30の運用比較を混同しない。', '',
              '|比較|種類|問数|点差|履歴95%区間|層別95%区間|判定|勝/負/同/n|符号検定p|',
              '|---|---|---:|---:|---|---|---|---|---:|']
    for comparison_ in ('C_minus_A_semantic_primary', 'C_minus_A_all30_operational', 'C_minus_B_all30'):
        for kind, v in report[comparison_].items():
            if not v['questions']:
                lines.append(f'|{comparison_}|{kind}|0|—|—|—|判定不能|—|—|')
                continue
            s = v['sign_test']
            verdict = v['classification'] if kind == 'overall' else '傾向参考・有意性主張なし'
            lines.append(f'|{comparison_}|{kind}|{v["questions"]}|{v["difference"]:+.4f}|{v["ci95"]}|'
                         f'{v["stratified_question_ci95"]}|{verdict}|'
                         f'{s["wins"]}/{s["losses"]}/{s["ties"]}/{s["n"]}|{s["p_two_sided"]:.6f}|')
    lines += ['', '判定不能では再現した/再現しなかったとは述べない。型別の点差と区間を確認する。',
              '種類別の表・区間は傾向の参考であり、有意性を主張しない。',
              '生成失敗/窓超過/judge失敗は保持。原文と人間レビュー一覧はignored privateファイルのみ。',
              'タイミング/footprint/RSS/swap/元hashはreport.jsonに保存。日本語/実長期/独立ユーザー/汎用意味精度へ一般化しない。']
    return '\n'.join(lines)+'\n'
