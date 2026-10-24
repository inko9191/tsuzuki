import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locomo_answer_costs as costs


class TimePlanTests(unittest.TestCase):
    def test_all_questions_include_auxiliary_costs_and_margin(self):
        report = {'timing': {'total_answer_seconds': {'mean_seconds': 100., 'p90_components_seconds': 150.}},
                  'auxiliary_costs': {'heldout_rerank_seconds_mean_projection': 10.,
                    'heldout_embedding_search_and_input_preparation_seconds_projection': 20.,
                    'two_planned_context_switches_seconds_projection': 2.},
                  'development_preparation_actual_seconds': 200.}
        measured = {'target_questions': 300, 'questions': 120, 'query_seconds': 12.,
                    'slowest_batch_seconds_per_query': .2, 'scoring_reporting_save_seconds': 4.,
                    'embedding_startup_seconds': 3., 'measurement_wall_seconds': 40.}
        result = costs.totals(report, measured)
        self.assertEqual(result['heldout_total_seconds_mean_projection'], 175.)
        self.assertEqual(result['heldout_total_seconds_component_scenario'], 255.)
        self.assertEqual(result['all_422_questions_seconds_mean_projection'], 415.)
        self.assertEqual(result['all_422_questions_seconds_with_20_percent_margin'], [498., 594.])


if __name__ == '__main__':
    unittest.main()
