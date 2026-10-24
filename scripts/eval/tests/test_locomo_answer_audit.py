import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout
from io import StringIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locomo_answer_audit as audit
import locomo_answer_audit_measure as measure
import locomo_answer_audit_report as report


class AuditTests(unittest.TestCase):
    def test_ingestion_allocation_is_labelled_estimate_and_conserves_total(self):
        result = audit.apportioned(90., {'short': 1, 'long': 2})
        self.assertEqual(sum(x['seconds_estimate'] for x in result.values()), 90.)
        self.assertEqual(result['long']['seconds_estimate'], 60.)
        self.assertTrue(all('not recorded' in x['basis'] for x in result.values()))
        self.assertTrue(all('seconds' not in x for x in result.values()))

    def test_selection_ignores_scores_answers_conditions_and_input_order(self):
        items = {str(i): {'id': f'conv-{g}/q{i}', 'gold': 'private', 'f1': i}
                 for g in (1, 2) for i in range(g*10, g*10+5)}
        changed = {k: {'id': v['id'], 'gold': 'changed', 'f1': -1} for k, v in reversed(list(items.items()))}
        self.assertEqual(measure.pick_questions(items), measure.pick_questions(changed))
        self.assertEqual([len(v) for v in measure.pick_questions(items).values()], [2, 2])

    def test_distribution_preserves_tail_and_count(self):
        v = audit.distribution([1., 2., 3., 100.])
        self.assertEqual(v['n'], 4)
        self.assertEqual(v['mean'], 26.5)
        self.assertEqual(v['median'], 2.5)
        self.assertGreater(v['p90'], 3)

    def test_empty_measurement_rejected(self):
        with self.assertRaises(ValueError):
            audit.distribution([])

    def test_main_results_and_private_outputs_have_distinct_destinations(self):
        self.assertNotEqual(audit.PUBLIC, audit.lm.RESULTS / 'answer-report.json')
        self.assertEqual(measure.LOCAL, audit.ae.LOCAL / 'audit')
        self.assertNotEqual(measure.PLAN, measure.production.PLAN)

    def test_total_is_matched_by_id_not_accidental_row_order(self):
        components = [{'id': 'a', 'total_component_seconds': 10.},
                      {'id': 'b', 'total_component_seconds': 20.}]
        payloads = [{'id': 'b', 'input_construction_seconds': 2., 'including_full_input_preflight_seconds': 3.},
                    {'id': 'a', 'input_construction_seconds': 4., 'including_full_input_preflight_seconds': 5.}]
        joined = report.totals(components, payloads)
        self.assertEqual([r['total_with_input_preflight_seconds'] for r in joined], [15., 23.])
        # Input construction is already inside full preflight; never double-count it.
        self.assertNotEqual(joined[0]['total_with_input_preflight_seconds'], 19.)

    def test_incomplete_or_duplicate_cost_rows_are_rejected(self):
        components = [{'id': 'a', 'total_component_seconds': 10.}]
        with self.assertRaises(RuntimeError):
            report.totals(components, [])
        p = {'id': 'a', 'input_construction_seconds': 1., 'including_full_input_preflight_seconds': 2.}
        with self.assertRaises(RuntimeError):
            report.totals(components, [p, p])

    def test_pause_persists_gate_and_signals_only_detached_audit_group(self):
        processes = ('101 101 /usr/bin/python locomo_answer_audit_measure.py run\n'
                     '202 202 /opt/homebrew/bin/llama-server -c 65536\n'
                     '303 303 /usr/bin/python another_evaluation.py\n'
                     '404 400 /usr/bin/python locomo_answer_audit_measure.py run\n')
        with TemporaryDirectory() as directory, patch.object(measure.lm, 'LOCAL', Path(directory)), \
                patch.object(measure.subprocess, 'check_output', return_value=processes), \
                patch.object(measure.os, 'killpg') as kill, redirect_stdout(StringIO()):
            measure.pause()
            self.assertTrue((Path(directory) / 'user-pause.json').exists())
            kill.assert_called_once_with(101, measure.signal.SIGINT)


if __name__ == '__main__':
    unittest.main()
