"""Regression tests for authorized CPU reporting corrections; no server calls."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import longmemeval_g1_continue as continuation
import longmemeval_g1_report as reporting
import longmemeval_g1_finalize as finalize
from copy import deepcopy


class FinalizationTests(unittest.TestCase):
    def test_resource_aggregate_preserves_swap_units_and_private_series(self):
        samples = [dict(available_bytes_estimate=100, swap_used_bytes=10, pressure_level=1,
                        swapins=3, swapouts=5, page_size=16384),
                   dict(available_bytes_estimate=80, swap_used_bytes=20, pressure_level=2,
                        swapins=7, swapouts=5, page_size=16384)]
        value = {'OS_samples1s': samples, 'peak': {'rss_bytes': 50}}
        summary = finalize.resource_summary(value)
        self.assertNotIn('OS_samples1s', summary)
        self.assertEqual(summary['OS_summary']['swapin_increment_bytes'], 65536)
        self.assertEqual(summary['OS_summary']['minimum_available_bytes_estimate'], 80)
        self.assertEqual(summary['peak'], value['peak'])
        self.assertEqual(len(value['OS_samples1s']), 2)

    def test_finalization_rejects_changed_main_interval_or_overall_score(self):
        before = {k: {} for k in ('questions', 'A_eligible', 'A_overflow', 'A_overflow_ids',
                  'C_minus_A_semantic_primary', 'C_minus_A_all30_operational', 'C_minus_B_all30',
                  'F1_C_minus_A_semantic_cohort_reference', 'F1_C_minus_B_all30_reference')}
        for kind in ('metrics_all30', 'metrics_A_eligible'):
            before[kind] = {a: {'overall': {'correct': .5}} for a in reporting.ARMS}
        after = deepcopy(before)
        finalize.unchanged_comparisons(before, after)
        after['C_minus_A_semantic_primary'] = {'ci95': [-.01, .01]}
        with self.assertRaisesRegex(RuntimeError, 'registered comparison changed'):
            finalize.unchanged_comparisons(before, after)
        after = deepcopy(before)
        after['metrics_all30']['A']['overall']['correct'] = .6
        with self.assertRaisesRegex(RuntimeError, 'overall metric changed'):
            finalize.unchanged_comparisons(before, after)

    def test_normal_restore_allows_missing_allocations_without_mutation(self):
        missing = {'required_recorded': False, 'bytes': {'KV': None}}
        continuation.validate_allocations(missing, restore=True)
        self.assertIsNone(missing['bytes']['KV'])

    def test_measurement_start_still_requires_allocations(self):
        with self.assertRaisesRegex(RuntimeError, 'required model/KV/compute'):
            continuation.validate_allocations({'required_recorded': False})
        continuation.validate_allocations({'required_recorded': True})

    def test_abstention_is_not_counted_in_original_type_or_single_subtype(self):
        rows = []
        for stratum in ('multi-session', 'temporal-reasoning', 'knowledge-update',
                        'single-session', 'abstention'):
            for n in range(6):
                kind = ('single-session-user' if stratum == 'single-session'
                        else 'multi-session' if stratum == 'abstention' else stratum)
                rows.append(dict(stratum=stratum, type=kind, correct=int(stratum == 'abstention'),
                                 f1=0., abstention=1., failure=0., judge_failure=0.))
        rows[-1]['type'] = 'single-session-user'
        result = reporting.metrics(rows)
        self.assertEqual(result['overall']['questions'], 30)
        self.assertEqual(result['overall']['correct'], .2)
        for stratum in reporting.STRATA:
            if stratum != 'overall':
                self.assertEqual(result[stratum]['questions'], 6)
        self.assertEqual(result['multi-session']['correct'], 0.)
        self.assertEqual(result['single-session-user']['questions'], 6)


if __name__ == '__main__':
    unittest.main()
