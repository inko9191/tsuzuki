"""CPU-only G1 sampling. Never inspect/export question, gold or history content."""
import argparse
import hashlib
import json
from pathlib import Path

SEED = 'longmemeval-g1-v1'
QUOTAS = {'multi-session': 6, 'temporal-reasoning': 6, 'knowledge-update': 6,
          'single-session-user': 2, 'single-session-assistant': 2,
          'single-session-preference': 2, 'abstention': 6}
SHA256 = 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'


def rank(qid):
    return hashlib.sha256((SEED + '|' + qid).encode()).hexdigest(), qid


def metadata(rows):
    """Allowlist only. Session IDs/dates support cost estimates, never selection."""
    result = []
    seen = set()
    for row in rows:
        qid, kind = row['question_id'], row['question_type']
        if not isinstance(qid, str) or qid in seen:
            raise ValueError('invalid or duplicate question ID')
        seen.add(qid)
        if kind not in QUOTAS or kind == 'abstention':
            raise ValueError('unsupported question type')
        ids, dates = row['haystack_session_ids'], row['haystack_dates']
        if len(ids) != len(dates) or not ids:
            raise ValueError('invalid session metadata')
        result.append({'id': qid, 'type': kind,
                       'stratum': 'abstention' if qid.endswith('_abs') else kind,
                       'sessions': list(zip(ids, dates))})
    return result


def select(rows):
    selected = []
    for kind, quota in QUOTAS.items():
        pool = sorted((r for r in rows if r['stratum'] == kind), key=lambda r: rank(r['id']))
        if len(pool) < quota:
            raise ValueError('insufficient stratum: ' + kind)
        selected.extend(pool[:quota])
    return sorted(selected, key=lambda r: rank(r['id']))


def counts(rows):
    strata = {k: sum(r['stratum'] == k for r in rows) for k in QUOTAS}
    sessions = [tuple(s) for r in rows for s in r['sessions']]
    histories = {tuple(tuple(s) for s in r['sessions']) for r in rows}
    return {'questions': len(rows), 'strata': strata,
            'session_occurrences': len(sessions),
            'histories_with_repeated_session_ids': sum(
                len({s[0] for s in r['sessions']}) < len(r['sessions']) for r in rows),
            'unique_session_id_date_pairs': len(set(sessions)),
            'unique_history_metadata_sequences': len(histories),
            'sessions_per_question': sorted(len(r['sessions']) for r in rows)}


def audit(path):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise ValueError('dataset hash mismatch')
    rows = metadata(json.loads(raw))
    chosen = select(rows)
    return {'dataset_sha256': digest, 'seed': SEED, 'quotas': QUOTAS,
            'all': counts(rows), 'selected': counts(chosen),
            'selected_question_ids': [r['id'] for r in chosen],
            'selected_metadata': chosen,
            'content_access': 'No question/answer/history fields accessed; metadata allowlist only.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    value = audit(args.dataset)
    if args.output.exists():
        raise SystemExit('refusing to overwrite existing metadata audit')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k: v for k, v in value.items() if k not in ('selected_metadata', 'selected_question_ids')}))
