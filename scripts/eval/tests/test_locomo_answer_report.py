from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locomo_answer_report as report
import locomo_answer_methods as am


def synthetic():
    rows = {arm: [] for arm in am.CONDITIONS}
    metadata = []
    for gi in range(10):
        group = f'g{gi:02}'
        for kind in report.TYPES:
            for qi in range(4):
                row = {'id': f'{group}/{kind}-{qi}', 'type': kind, 'category': 1, 'f1': 0.,
                       'em': 0., 'abstention': 0., 'failure': 0., 'empty_answer': 0.,
                       'inserted_all': 0., 'inserted_recall': 0., 'wall_seconds': 1.,
                       'prefill_seconds': .2, 'generation_seconds': .8}
                for arm in am.CONDITIONS:
                    rows[arm].append(dict(row))
            metadata += [{'group': group, 'type': kind}] * (80 if kind == 'single_fact' else 10)
    return rows, metadata, {f'g{i:02}': 50 for i in range(10)}


class AnswerForecast(unittest.TestCase):
    def test_incomplete_input_refused_before_scoring(self):
        prepared = {'items': {'a': {}}, 'plan_sha256': 'x'}
        with patch.object(report.am, 'score') as score:
            with self.assertRaises(RuntimeError):
                report.complete(prepared, {}, {'plan_sha256': 'x'})
            score.assert_not_called()

    def test_development_type_weights_replace_balanced_pilot_mean(self):
        rows, metadata, target = synthetic()
        for r in rows['C-final']:
            r['f1'] = float(r['type'] != 'single_fact')
        x = report.precision_forecast(rows, metadata, target, simulations=5, resamples=100)
        self.assertAlmostEqual(x['comparisons']['C-final-A']['overall']['dev_type_weighted_difference'], .2)
        self.assertEqual(x['development_type_weights']['g00'],
                         {'multiple_evidence': .1, 'single_fact': .8, 'temporal': .1})

    def test_zero_variance_simulation_and_strict_positive_detection(self):
        rows, metadata, target = synthetic()
        x = report.precision_forecast(rows, metadata, target, simulations=3, resamples=100)
        for strata in x['comparisons'].values():
            for s in strata.values():
                self.assertEqual(s['width_p10_median_p90'], [0., 0., 0.])
                self.assertEqual(s['conditional_detectable_difference_80_percent'], .005)
                self.assertEqual(s['shift_sensitivity'][0]['detection_fraction'], 0.)

    def test_forecast_reproducible_and_missing_cell_rejected(self):
        rows, metadata, target = synthetic()
        a = report.precision_forecast(rows, metadata, target, simulations=2, resamples=100)
        b = report.precision_forecast(rows, metadata, target, simulations=2, resamples=100)
        self.assertEqual(a, b)
        for part in rows.values():
            part[:] = [r for r in part if not (r['id'].startswith('g00/') and r['type'] == 'temporal')]
        with self.assertRaises(ValueError):
            report.precision_forecast(rows, metadata, target, simulations=2, resamples=100)

    def test_pairing_mismatch_rejected(self):
        rows, metadata, target = synthetic()
        rows['C-final'].reverse()
        with self.assertRaises(ValueError):
            report.precision_forecast(rows, metadata, target, simulations=2, resamples=100)

    def test_timing_projection_respects_first_and_warm_cost(self):
        rows, _, _ = synthetic()
        prepared = {'items': {}, 'schedule': []}
        for arm in am.CONDITIONS:
            for i, row in enumerate(rows[arm]):
                cold = i % 12 == 0
                p = 10. if cold else 1.
                row.update({'prefill_seconds': p, 'generation_seconds': 2., 'overhead_seconds': .5,
                    'wall_seconds': p + 2.5, 'input_tokens': 100, 'cache_tokens': 0 if cold else 90,
                    'processed_prompt_tokens': 100 if cold else 10, 'generated_tokens': 20})
                key = f'{arm}|{row["id"]}'
                prepared['items'][key] = {'id': row['id'], 'condition': arm}
                prepared['schedule'].append(key)
        result = report.timing_profile(prepared, rows, {a: a for a in am.CONDITIONS},
                                       {f'g{i:02}': 10 for i in range(10)})
        self.assertAlmostEqual(result['per_condition_projection']['A']['mean_seconds'], 440.)
        self.assertAlmostEqual(result['total_answer_seconds']['mean_seconds'], 440. * 7)
        self.assertEqual(result['blocks']['g00']['A']['warm']['prefill_seconds']['n'], 11)

    def test_report_contains_t1_p2_and_exact_sign_with_multiple_first(self):
        rows, _, _ = synthetic()
        tables = report.paired_tables(rows)
        self.assertEqual(next(iter(tables)), 'type:multiple_evidence')
        entry = tables['overall']['f1']['C-final']['A']
        self.assertEqual(entry['sign_test']['ties'], 10)
        self.assertEqual(entry['sign_test']['two_sided_p'], 1.)
        self.assertIn('C-t1', tables['overall']['f1'])
        self.assertIn('C-p2', tables['overall']['f1'])


if __name__ == '__main__':
    unittest.main()
