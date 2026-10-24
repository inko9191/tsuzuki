"""Pure G1 adapter and judge/statistical rules. No model/network/file operations."""
import hashlib
import math
import numpy as np


def turns(row):
    """Only original role/content/date data; never gold or has_answer annotations."""
    sessions = row['haystack_sessions']
    dates = row['haystack_dates']
    if len(sessions) != len(dates) or len(sessions) != len(row['haystack_session_ids']):
        raise ValueError('unaligned history')
    qid = row['question_id']
    result = []
    for i, (session, date) in enumerate(zip(sessions, dates), 1):
        if not session:
            raise ValueError('empty session')
        for j, message in enumerate(session, 1):
            role, content = message['role'], message['content']
            if role not in ('user', 'assistant') or not isinstance(content, str):
                raise ValueError('unsupported original message')
            result.append({'id': f'{qid}/D{i}:{j}', 'group': qid,
                           'session': f'{qid}#{i}', 'pos': j-1,
                           'date': date, 'text': f'{role}: {content}'})
    return result


def judge_label(response):
    try:
        choice = response['choices'][0]
        if choice['finish_reason'] != 'stop':
            return 0, 'unfinished'
        text = choice['message']['content'].strip().casefold()
        if text not in ('yes', 'no'):
            return 0, 'parse'
        return int(text == 'yes'), None
    except (KeyError, TypeError, IndexError, AttributeError):
        return 0, 'parse'


def paired_rows(arms, left, right):
    """Refuse successful-only or partial-pair analyses."""
    a, b = arms[left], arms[right]
    if not a or set(a) != set(b):
        raise ValueError('incomplete paired cohort')
    return [(qid, a[qid]-b[qid]) for qid in sorted(a)]


def interval(differences, groups, strata=None, samples=10000, seed=0):
    """Paired percentile bootstrap, clusters kept intact; optional stratum resampling."""
    if not differences or len(differences) != len(groups):
        raise ValueError('invalid bootstrap cohort')
    if strata is not None and len(strata) != len(groups):
        raise ValueError('invalid strata')
    pools = {}
    for i, (value, group) in enumerate(zip(differences, groups)):
        if not math.isfinite(value):
            raise ValueError('nonfinite difference')
        key = strata[i] if strata is not None else 'all'
        pools.setdefault(key, {}).setdefault(group, []).append(value)
    if strata is not None:
        memberships = {}
        for s, clusters in pools.items():
            for group in clusters:
                if group in memberships and memberships[group] != s:
                    raise ValueError('cluster crosses strata')
                memberships[group] = s
    rng = np.random.default_rng(seed)
    values = np.empty(samples)
    ordered = [list(pools[s].values()) for s in sorted(pools)]
    for k in range(samples):
        draw = []
        for clusters in ordered:
            for index in rng.integers(0, len(clusters), size=len(clusters)):
                draw.extend(clusters[index])
        values[k] = np.mean(draw)
    return np.quantile(values, [.025, .975]).tolist()


def noninferiority(history_ci, stratified_ci, margin=.03):
    if min(history_ci[0], stratified_ci[0]) >= -margin:
        return 'confirmed_in_registered_sample'
    if max(history_ci[1], stratified_ci[1]) < -margin:
        return 'not_reproduced'
    return 'inconclusive'


def sign_test(differences):
    wins = sum(d > 0 for d in differences)
    losses = sum(d < 0 for d in differences)
    ties = len(differences)-wins-losses
    n = wins+losses
    p = min(1., 2*sum(math.comb(n, i) for i in range(min(wins, losses)+1))/2**n) if n else 1.
    return dict(wins=wins, losses=losses, ties=ties, n=n, p_two_sided=p)


def review_order(ids, answers):
    """All 30 questions; disagreement first, no exclusion or selective relabeling."""
    def key(qid):
        disagree = answers['C-final-norw'][qid] != answers['A'][qid]
        return not disagree, hashlib.sha256(('longmemeval-g3-v1|' + qid).encode()).hexdigest(), qid
    if set(ids) != set(answers['A']) or set(ids) != set(answers['C-final-norw']):
        raise ValueError('incomplete review cohort')
    return sorted(ids, key=key)
