"""Synthetic only; no benchmark question/history/gold is read."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import longmemeval_g1_metadata as m


def fixture():
    rows = []
    for kind in m.QUOTAS:
        if kind == 'abstention':
            continue
        for i in range(12):
            rows.append(dict(question_id=f'{kind}-{i}', question_type=kind,
                             haystack_session_ids=['s'], haystack_dates=['date']))
    for i in range(12):
        rows.append(dict(question_id=f'{i}_abs', question_type='multi-session',
                         haystack_session_ids=['s'], haystack_dates=['date']))
    return rows


class MetadataTests(unittest.TestCase):
    def test_no_content_or_gold_access(self):
        class OnlyMetadata(dict):
            def __getitem__(self, key):
                if key not in ('question_id', 'question_type', 'haystack_session_ids', 'haystack_dates'):
                    raise AssertionError('access to forbidden field')
                return super().__getitem__(key)
        self.assertEqual(len(m.select(m.metadata([OnlyMetadata(r) for r in fixture()]))), 30)

    def test_order_invariant_and_quotas(self):
        rows = m.metadata(fixture())
        chosen = m.select(rows)
        self.assertEqual(chosen, m.select(list(reversed(rows))))
        self.assertEqual(m.counts(chosen)['strata'], m.QUOTAS)

    def test_abstention_suffix_overrides_type(self):
        rows = m.metadata(fixture())
        self.assertEqual(sum(r['stratum'] == 'abstention' for r in rows), 12)

    def test_session_count_not_used_for_selection(self):
        rows = m.metadata(fixture())
        original = [r['id'] for r in m.select(rows)]
        for r in rows:
            r['sessions'] = [('different', 'later')] * 10
        self.assertEqual(original, [r['id'] for r in m.select(rows)])

    def test_missing_stratum_stops_no_replacement(self):
        rows = [r for r in m.metadata(fixture()) if r['stratum'] != 'abstention']
        with self.assertRaises(ValueError):
            m.select(rows)

    def test_duplicate_id_stops(self):
        rows = fixture()
        with self.assertRaises(ValueError):
            m.metadata(rows + [rows[0]])

    def test_mismatched_dates_stop(self):
        rows = fixture()
        rows[0]['haystack_dates'] = []
        with self.assertRaises(ValueError):
            m.metadata(rows)

    def test_unique_session_date_not_id_alone(self):
        rows = m.metadata(fixture()[:2])
        rows[1]['sessions'] = [('s', 'other date')]
        self.assertEqual(m.counts(rows)['unique_session_id_date_pairs'], 2)

    def test_repeated_session_id_preserves_occurrence_order(self):
        rows = fixture()[:1]
        rows[0]['haystack_session_ids'] = ['s', 's']
        rows[0]['haystack_dates'] = ['first', 'second']
        result = m.metadata(rows)
        self.assertEqual(result[0]['sessions'], [('s', 'first'), ('s', 'second')])
        self.assertEqual(m.counts(result)['session_occurrences'], 2)
        self.assertEqual(m.counts(result)['histories_with_repeated_session_ids'], 1)


if __name__ == '__main__':
    unittest.main()
