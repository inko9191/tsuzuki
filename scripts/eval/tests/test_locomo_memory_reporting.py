"""Reporting inference and checkpoint tests on synthetic data only."""
import itertools
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locomo_memory_reporting as report
import locomo_memory_stages as stages
import locomo_memory_resume as resume


def rows(groups, values):
    return [{'id': f'{g}/q{i}', 'type': 'multiple_evidence', 'category': 1,
             **{m: v for m in report.METRICS}} for i, (g, v) in enumerate(zip(groups, values))]


class Inference(unittest.TestCase):
    def test_cluster_ratio_preserves_question_macro_weighting(self):
        baseline = rows(['a'] * 9 + ['b'], [0] * 10)
        candidate = rows(['a'] * 9 + ['b'], [1] * 9 + [0])
        s = report.paired(candidate, baseline, report.METRICS[0])
        self.assertEqual(s['mean_diff'], .9)
        self.assertEqual(s['per_conversation']['a'], {'questions': 9, 'mean_diff': 1.})
        pick = np.random.default_rng(0).integers(0, 2, (10000, 2))
        expected = np.array([9, 0])[pick].sum(axis=1) / np.array([9, 1])[pick].sum(axis=1)
        self.assertEqual(s['conversation_ci95'], np.percentile(expected, [2.5, 97.5]).tolist())

    def test_sign_test_matches_exhaustive_ten_conversation_patterns(self):
        groups = [str(i) for i in range(10)]
        s = report.paired(rows(groups, [1] * 7 + [0] * 3), rows(groups, [.5] * 10), report.METRICS[0])
        exact = sum(min(sum(x), 10 - sum(x)) <= 3 for x in itertools.product((0, 1), repeat=10)) / 1024
        self.assertEqual(s['sign_test']['two_sided_p'], exact)
        self.assertEqual(exact, .34375)

    def test_ties_excluded_and_empty_stratum_not_zero_performance(self):
        s = report.paired(rows('abc', [1, 0, .5]), rows('abc', [.5] * 3), report.METRICS[0])
        self.assertEqual(s['sign_test'], {'wins': 1, 'losses': 1, 'ties': 1, 'n': 2, 'two_sided_p': 1.})
        self.assertIsNone(report.paired([], [], report.METRICS[0])['mean'])

    def test_reference_rows_do_not_enter_or_mutate_candidates(self):
        x = rows('ab', [.5, .5])
        candidates = {'candidate': x}
        stats = report.comparisons(candidates, {report.T1: rows('ab', [0, 0]), report.P2: x}, 'candidate')
        self.assertEqual(list(candidates), ['candidate'])
        self.assertEqual(list(stats['strata']['overall'][report.METRICS[0]]), [report.T1, report.P2, 'candidate'])
        with self.assertRaisesRegex(ValueError, 'order differ'):
            report.comparisons(candidates, {report.T1: list(reversed(x)), report.P2: x}, 'candidate')

    def test_mismatched_strata_rejected(self):
        x = rows('ab', [0, 0]); y = rows('ab', [0, 0]); y[0]['category'] = 2
        with self.assertRaisesRegex(ValueError, 'strata differ'):
            report.comparisons({'base': x}, {report.T1: y, report.P2: x}, 'base')

    def test_baseline_drift_is_not_silently_overwritten(self):
        x = rows('ab', [0, 0]); y = rows('ab', [0, 1])
        with self.assertRaisesRegex(ValueError, 'reproduction differs'):
            report.comparisons({report.P2: y}, {report.T1: x, report.P2: x}, report.P2)


class Checkpoint(unittest.TestCase):
    def test_every_25_items_survive_restart_with_digest_validation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(stages, 'GEN', Path(tmp)):
            store = stages.Store('para')
            for i in range(25):
                store.put(str(i), {'digest': 'same', 'reply': 'synthetic', 'value': ['synthetic']})
            restarted = stages.Store('para')
            self.assertEqual(len(restarted.items), 25)
            self.assertIsNotNone(restarted.get('24', 'same'))
            self.assertIsNone(restarted.get('24', 'changed'))
            store.put('25', {'digest': 'same', 'reply': 'synthetic', 'value': None})
            store.save()
            self.assertEqual(len(stages.Store('para').items), 26)

    def test_rerank_overflow_stops_before_any_generation(self):
        qwen = stages.Qwen.__new__(stages.Qwen)
        qwen.url = 'http://synthetic'
        qwen.settings = {'server': {'n_ctx': 8192}, 'request': {'chat_template_kwargs': {}}}
        with patch.object(stages, 'http_json', side_effect=[{'prompt': 'synthetic'}, {'tokens': [0] * 8100}]) as http:
            with self.assertRaisesRegex(RuntimeError, 'stop without truncation'):
                qwen.chat('rerank', [])
            self.assertEqual(http.call_count, 2)

    def test_completed_stage_cannot_bypass_split_or_holdout_guard(self):
        base = stages.mm.config(['p2'])
        data = {'plan_sha256': stages.lm.digest(stages.PLAN), 'split_manifest_sha256': 'split',
                'holdout_evaluated': True, 'base': base['id'], 'configs': {base['id']: {}}, 'reporting': {}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(stages, 'RESULTS', Path(tmp)), \
                patch.object(stages, 'committed', return_value=True), \
                patch.object(stages, 'previous_selection', return_value=base), \
                patch.object(stages, 'stage_configs', return_value=(base, [base])), \
                patch.object(stages.lm, 'file_hash', return_value='split'):
            stages.lm.write(Path(tmp) / 'stage3-dev.json', data)
            with self.assertRaisesRegex(RuntimeError, 'stop for review'):
                resume.complete('3')

    def test_single_worker_stops_on_fatal_api_error_and_saves_completed_items(self):
        class Qwen:
            settings = {}
            calls = 0

            def __init__(self, model_path):
                pass

            def item_digest(self, kind, chat):
                return 'digest'

            def chat(self, kind, chat):
                Qwen.calls += 1
                if Qwen.calls == 2:
                    raise RuntimeError('server unavailable')
                return '{"paraphrases": ["synthetic"]}'

        jobs = [(str(i), [], {}) for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(stages, 'GEN', Path(tmp)), \
                patch.object(stages, 'check_plan'), patch.object(stages.lm, 'load_dev', return_value=({'corpus': []}, {})), \
                patch.object(stages, 'load_raw'), patch.object(stages.mm, 'turn_table'), \
                patch.object(stages, 'generation_jobs', return_value=jobs), patch.object(stages, 'Qwen', Qwen):
            with self.assertRaisesRegex(RuntimeError, 'server unavailable'):
                stages.generate('para', 1, None)
            self.assertEqual(Qwen.calls, 2)
            self.assertEqual(list(stages.Store('para').items), ['0'])


if __name__ == '__main__':
    unittest.main()
