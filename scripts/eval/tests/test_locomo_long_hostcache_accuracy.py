"""CPU tests: exact per-question comparison, failures, memory references and isolation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locomo_long_hostcache_accuracy as c


def reply(answer='one', evidence=None, finish='stop', compact=False):
    content = json.dumps({'answer': answer, 'evidence_ids': evidence or []},
                         separators=(',', ':') if compact else None)
    return {'response': {'choices': [{'finish_reason': finish, 'message': {'content': content}}]}}


def row(qid, value):
    return {'id': qid, 'type': 'single_fact', 'f1': value, 'em': 0., 'abstention': 0.,
            'failure': 0., 'inserted_all': 1., 'inserted_recall': 1.}


class HostCacheAccuracy(unittest.TestCase):
    def test_exact_answer_is_not_whitespace_normalized(self):
        self.assertFalse(c.answer_match(reply('one '), reply('one'))['answer_exact'])

    def test_serialization_difference_does_not_change_parsed_answer(self):
        value = c.answer_match(reply(compact=True), reply())
        self.assertTrue(value['answer_exact'])
        self.assertTrue(value['parsed_payload_exact'])
        self.assertFalse(value['raw_content_exact'])

    def test_evidence_difference_is_reported_separately(self):
        value = c.answer_match(reply(evidence=['x']), reply(evidence=['y']))
        self.assertTrue(value['answer_exact'])
        self.assertFalse(value['parsed_payload_exact'])

    def test_matching_unfinished_output_does_not_pass_answer_match(self):
        self.assertFalse(c.answer_match(reply(finish='length'), reply(finish='length'))['answer_exact'])

    def test_same_F1_different_answer_not_claimed_answer_match(self):
        rows = [row('g/q1', .5)]
        refs = {'ref': {'rows': {'g/q1': row('g/q1', .5)}, 'replies': {'g/q1': reply('first')}}}
        result, differences = c.compare(rows, {'g/q1': reply('second')}, refs)
        self.assertTrue(result['ref']['all_stratum_F1_differences_zero'])
        self.assertFalse(result['ref']['all_answers_exact'])
        self.assertEqual(differences[0]['id'], 'g/q1')

    def test_cancelling_F1_changes_are_not_per_question_equality(self):
        rows = [row('g/q1', .8), row('g/q2', .2)]
        refs = {'ref': {'rows': {q: row(q, .5) for q in ('g/q1', 'g/q2')},
                        'replies': {q: reply() for q in ('g/q1', 'g/q2')}}}
        result, _ = c.compare(rows, refs['ref']['replies'], refs)
        self.assertTrue(result['ref']['all_stratum_F1_differences_zero'])
        self.assertFalse(result['ref']['all_question_F1_differences_zero'])

    def test_partial_reference_metrics_use_same_questions(self):
        rows = [row('g/q1', 1.)]
        ref = {'rows': {'g/q1': row('g/q1', 1.), 'g/q2': row('g/q2', 0.)},
               'replies': {'g/q1': reply(), 'g/q2': reply()}}
        result, _ = c.compare(rows, {'g/q1': reply()}, {'ref': ref})
        self.assertEqual(result['ref']['reference_metrics']['overall']['questions'], 1)
        self.assertEqual(result['ref']['reference_metrics']['overall']['f1'], 1.)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(RuntimeError):
            c.compare([row('q', 1.), row('q', 1.)], {}, {})

    def test_missing_reference_rejected(self):
        with self.assertRaises(RuntimeError):
            c.compare([row('q', 1.)], {}, {'ref': {'rows': {}, 'replies': {}}})

    def test_range_comparison_has_no_posthoc_tolerance(self):
        p = {'memory_reference_files': ['r1', 'r2']}
        old = [dict(curve={'30': dict(peak=dict(physical_footprint_bytes=v, rss_bytes=v+50))}) for v in (100, 110)]
        current = {'30': {'peak': {'physical_footprint_bytes': 111, 'rss_bytes': 160}}}
        with patch.object(c.lm, 'read', side_effect=old):
            value = c.memory_comparison(current, p)['30']
        self.assertFalse(value['physical_footprint_bytes']['inside_prior_repeat_range'])
        self.assertTrue(value['rss_bytes']['inside_prior_repeat_range'])

    def test_runtime_changes_only_host_cache_and_logging_then_restores(self):
        previous = c.base.LOCAL
        with patch.dict(c.os.environ, {'LLAMA_ARG_CACHE_RAM': '8192', 'LLAMA_ARG_LOG_VERBOSITY': '1'}):
            with c.runtime():
                self.assertEqual(c.os.environ['LLAMA_ARG_CACHE_RAM'], '0')
                self.assertEqual(c.os.environ['LLAMA_ARG_LOG_VERBOSITY'], '4')
                self.assertEqual(c.audit.PLAN, c.PLAN)
            self.assertEqual(c.os.environ['LLAMA_ARG_CACHE_RAM'], '8192')
        self.assertEqual(c.base.LOCAL, previous)

    def test_restore_runtime_clears_probe_environment(self):
        with patch.dict(c.os.environ, {'LLAMA_ARG_CACHE_RAM': '0'}):
            with c.runtime(off=False):
                self.assertNotIn('LLAMA_ARG_CACHE_RAM', c.os.environ)
            self.assertEqual(c.os.environ['LLAMA_ARG_CACHE_RAM'], '0')

    def test_pause_uses_no_process_signal(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c.ae, 'PAUSE', Path(tmp)/'pause.json'), patch.object(c.os, 'kill') as kill:
            c.pause()
            self.assertEqual(json.loads(c.ae.PAUSE.read_text())['reason'], 'human requested pause')
        kill.assert_not_called()

    def test_received_key_is_rejected_before_http(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c.audit, 'LOCAL', Path(tmp)), patch.object(c.ae, 'gate'), patch.object(c.base.st, 'http_json') as http:
            with self.assertRaises(RuntimeError):
                c.audit.send('q', {}, {}, {'q': reply()}, type('Watch', (), {'error': None})())
        http.assert_not_called()


if __name__ == '__main__':
    unittest.main()
