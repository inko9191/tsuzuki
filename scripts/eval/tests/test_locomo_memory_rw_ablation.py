"""Synthetic-only safeguards for the registered rw removal experiment."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locomo_memory_rw_ablation as ab


class RWRemoval(unittest.TestCase):
    def test_equivalence_requires_entire_interval_not_just_zero_crossing(self):
        self.assertTrue(ab.equivalent([-.03, .03]))
        self.assertTrue(ab.equivalent([.01, .02]))
        self.assertFalse(ab.equivalent([-.04, .02]))
        self.assertFalse(ab.equivalent([-.02, .031]))
        self.assertFalse(ab.equivalent([.02, .01]))
        self.assertFalse(ab.equivalent(None))

    def test_timing_sample_fixed_by_id_and_contains302_unique_development_ids(self):
        ids = [f'conv-{i%10}/q{i}' for i in range(1233)]
        picked = ab.timing_sample(ids)
        self.assertEqual(len(set(picked)), 302)
        self.assertEqual(picked, ab.timing_sample(list(reversed(ids))))
        self.assertTrue(set(picked) <= set(ids))

    def test_all_unreranked_rows_must_reproduce_not_only_aggregate(self):
        rows = [{'id': 'a/q1', 'ranking': ['p2|a/t1'], 'recall@T512': .5}]
        ab.require_reproduction(rows, rows)
        changed = [{'id': 'a/q1', 'ranking': ['fact|a/t2'], 'recall@T512': .5}]
        with self.assertRaisesRegex(RuntimeError, 'differ from stage4'):
            ab.require_reproduction(changed, rows)

    def test_unknown_inflight_never_resends_and_received_digest_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(ab, 'LOCAL', Path(tmp)):
            job = {'query': {'id': 'a/q1'}, 'digest': 'registered'}
            ab.lm.write(Path(tmp)/'inflight.json', {'id': 'a/q1'})
            with self.assertRaisesRegex(RuntimeError, 'unknown inflight'):
                ab.replies_for({'queries': [job]})
            ab.lm.write(Path(tmp)/'responses.json', {'a/q1': {'digest': 'different'}})
            with self.assertRaisesRegex(RuntimeError, 'registered request'):
                ab.replies_for({'queries': [job]})
            ab.lm.write(Path(tmp)/'responses.json', {'a/q1': {'digest': 'registered'}})
            self.assertIn('a/q1', ab.replies_for({'queries': [job]}))
            self.assertFalse((Path(tmp)/'inflight.json').exists())

    def test_pause_only_targets_detached_python_runner_and_leaves_server(self):
        lines = ('210 210 python scripts/eval/locomo_memory_rw_ablation.py run\n'
                 '220 220 /opt/homebrew/bin/llama-server -c 8192\n'
                 '230 231 python scripts/eval/locomo_memory_rw_ablation.py run\n')
        with tempfile.TemporaryDirectory() as tmp, patch.object(ab.lm, 'LOCAL', Path(tmp)), \
                patch.object(ab.subprocess, 'check_output', return_value=lines), \
                patch.object(ab.os, 'getpid', return_value=999), patch.object(ab.os, 'killpg') as kill:
            ab.pause()
            kill.assert_called_once_with(210, ab.signal.SIGINT)
            self.assertTrue((Path(tmp)/'user-pause.json').exists())

    def test_distinct_original_rerank_merge_preserves_tail_and_filters_duplicates(self):
        order = list(range(40))
        output = ab.production.reranked_order(order, [3, 0, 3])
        self.assertEqual(output[:2], [3, 0])
        self.assertEqual(len(output), len(order))
        self.assertEqual(set(output), set(order))


if __name__ == '__main__':
    unittest.main()
