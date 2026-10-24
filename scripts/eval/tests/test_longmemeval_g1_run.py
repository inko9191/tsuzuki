"""Synthetic CPU protocol tests; no server, tokenizer endpoint or dataset accessed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import longmemeval_g1_report as r
import longmemeval_g1_run as run


def reply(answer='yes'):
    return {'response': {'choices': [{'finish_reason': 'stop', 'message': {
        'content': json.dumps({'answer': answer, 'evidence_ids': []})}}]},
        'attempts': 1, 'wall_seconds': 1.}


class ExecutionTests(unittest.TestCase):
    def test_classification_overlap_prefers_noninferiority(self):
        self.assertEqual(r.classification([-.02, -.01]), '成立')

    def test_not_established_and_negative_upper_is_failure(self):
        self.assertEqual(r.classification([-.2, -.001]), '不成立')

    def test_uncertain_not_failure(self):
        self.assertEqual(r.classification([-.2, .1]), '判定不能')
        self.assertEqual(r.classification([-.2, 0.]), '判定不能')
        self.assertEqual(r.classification(None), '判定不能')

    def test_exact_boundary_pass(self):
        self.assertEqual(r.classification([-.03, .1]), '成立')

    def test_window_reservations_include_output_and_margin(self):
        self.assertTrue(run.fits(131072-256-128, 256, 131072))
        self.assertFalse(run.fits(131072-256-127, 256, 131072))

    def test_budget_reserves_transport_timeout_before_new_request(self):
        with patch.object(run.time, 'monotonic', return_value=43200-900):
            with self.assertRaises(RuntimeError):
                run.reserve_budget(0)

    def test_A_overflow_kept_operational_excluded_from_semantic_pair(self):
        questions = {'q': dict(question_id='q', question_type='multi-session',
                                question='synthetic?', answer='yes', history_group='h')}
        inputs = {a: {'q': dict(fits_context=a != 'A', input_tokens=1)} for a in r.ARMS}
        answers = {a: {'q': reply()} for a in r.ARMS}
        answers['A']['q'] = dict(failure_kind='input_overflow', response={}, attempts=0)
        judges = {a: {'q': dict(label=int(a != 'A'), failure=None)} for a in r.ARMS}
        result, rows, review = r.analyze(questions, inputs, answers, judges)
        self.assertEqual(result['A_eligible'], 0)
        self.assertEqual(result['A_overflow'], 1)
        self.assertEqual(result['C_minus_A_semantic_primary']['overall']['classification'], '判定不能')
        self.assertEqual(result['C_minus_A_all30_operational']['overall']['difference'], 1)
        self.assertEqual(rows['A'][0]['failure'], 1)
        self.assertEqual(len(review), 1)
        self.assertNotIn('synthetic?', json.dumps(result))

    def test_incomplete_cohort_cannot_be_scored(self):
        with self.assertRaises(RuntimeError):
            r.analyze({'q': {}}, {}, {a: {} for a in r.ARMS}, {a: {} for a in r.ARMS})

    def test_generation_parse_failure_retained_as_zero(self):
        value = r.answer_value({'response': {'choices': [{'finish_reason': 'stop',
                                                        'message': {'content': 'not JSON'}}]}}, 'yes')
        self.assertEqual(value['f1'], 0)
        self.assertEqual(value['failure'], 1)

    def test_duplicate_key_stops_before_transport(self):
        with tempfile.TemporaryDirectory() as t, patch.object(run, 'LOCAL', Path(t)), \
             patch.object(run.ae, 'gate'), patch.object(run.st, 'http_json') as http:
            watch = type('Watch', (), {'error': None})()
            with self.assertRaises(RuntimeError):
                run.send('q', {}, {}, {'q': reply()}, watch, 1)
            http.assert_not_called()

    def test_unknown_inflight_stops_before_transport(self):
        with tempfile.TemporaryDirectory() as t, patch.object(run, 'LOCAL', Path(t)), \
             patch.object(run.ae, 'gate'), patch.object(run.st, 'http_json') as http:
            (Path(t)/'inflight.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                run.send('q', {}, {}, {}, type('Watch', (), {'error': None})(), 1)
            http.assert_not_called()

    def test_received_transport_is_journaled_once_without_retry(self):
        with tempfile.TemporaryDirectory() as t, patch.object(run, 'LOCAL', Path(t)), \
             patch.object(run.ae, 'gate'), patch.object(run.ae, 'checked_timings'), \
             patch.object(run.st, 'http_json', return_value=reply()['response']) as http:
            replies = {}; watch = type('Watch', (), {'error': None})()
            run.send('q', {}, {'model_sha256': 'm', 'url': 'test'}, replies, watch, 1)
            self.assertEqual(http.call_count, 1)
            self.assertEqual(len((Path(t)/'received.jsonl').read_text().splitlines()), 1)
            self.assertFalse((Path(t)/'inflight.json').exists())
            self.assertEqual(replies['q']['attempts'], 1)

    def test_transport_failure_leaves_unknown_inflight_and_no_retry(self):
        with tempfile.TemporaryDirectory() as t, patch.object(run, 'LOCAL', Path(t)), \
             patch.object(run.ae, 'gate'), patch.object(run.st, 'http_json', side_effect=TimeoutError) as http:
            with self.assertRaises(TimeoutError):
                run.send('q', {}, {'model_sha256': 'm', 'url': 'test'}, {}, type('W', (), {'error': None})(), 1)
            self.assertEqual(http.call_count, 1)
            self.assertTrue((Path(t)/'inflight.json').exists())

    def test_execution_order_is_C_B_A(self):
        self.assertEqual(r.ARMS, ('C-final-norw', 'B1536', 'A'))

    def test_human_pause_is_durable_without_signal(self):
        with tempfile.TemporaryDirectory() as t, patch.object(run.ae, 'PAUSE', Path(t)/'pause.json'), \
             patch.object(run.os, 'kill') as kill:
            run.pause()
            self.assertEqual(json.loads((Path(t)/'pause.json').read_text())['reason'], 'human requested pause')
            kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
