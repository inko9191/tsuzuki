import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locomo_answer_norw as n


class NorwTests(unittest.TestCase):
    def test_frozen_config_only_removes_rw(self):
        cf = n.mm.config(['p2', 'rw', 'fact'], rerank='qwen')
        expected = {**cf, 'id': n.CFG['id'], 'units': ['p2', 'fact']}
        self.assertEqual(n.CFG, expected)

    def test_equivalence_requires_whole_interval(self):
        self.assertTrue(n.equivalent([-.03, .03]))
        self.assertFalse(n.equivalent([-.031, .001]))
        self.assertFalse(n.equivalent([-.001, .031]))
        self.assertFalse(n.equivalent(None))

    def test_order_is_original_c_final_order_once(self):
        queries = [{'id': f'conv-{g}/q{j}', 'group': f'conv-{g}'} for g in range(10) for j in range(31)]
        expected = [qid for qid, arm in n.am.schedule(queries, {a:a for a in n.am.CONDITIONS}) if arm=='C-final']
        self.assertEqual(n.schedule(queries), expected)
        self.assertEqual(len(set(expected)), len(queries))
        self.assertEqual(n.schedule(queries[::-1]), expected)

    def test_switch_destination_restored_on_exception(self):
        old = n.ab.LOCAL
        with self.assertRaises(RuntimeError):
            with n.redirected_switch():
                self.assertEqual(n.ab.LOCAL, n.LOCAL)
                raise RuntimeError('planned failure')
        self.assertEqual(n.ab.LOCAL, old)

    def test_tokenizer_save_cannot_write_original(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(n, 'LOCAL', Path(tmp)), \
                patch.object(n.ae.lm, 'read', return_value={}):
            tok = n.Tokenizer('no-network')
            tok.counts['fake'] = 2
            tok.save()
            self.assertEqual(json.loads((Path(tmp)/'token-counts.json').read_text()), {'fake':2})
            self.assertEqual(tok.path.parent, Path(tmp))

    def test_all_answers_and_reranks_required(self):
        items = {str(i):{} for i in range(302)}
        state = {'phase':'answer complete', 'status':'complete'}
        n.ready({'items':items}, items, items, state)
        for responses in (dict(list(items.items())[:301]), {**items, 'extra':{}}):
            with self.assertRaises(RuntimeError):
                n.ready({'items':items}, responses, items, state)
        with self.assertRaises(RuntimeError):
            n.ready({'items':items}, items, items, {'phase':'answer generation', 'status':'running'})

    def test_generation_transport_failure_is_not_retried(self):
        jobs = {str(i): {'digest':str(i), 'body':{}, 'input_tokens':1} for i in range(302)}
        with tempfile.TemporaryDirectory() as tmp, patch.object(n, 'LOCAL', Path(tmp)), \
                patch.object(n.ae, 'gate'), patch.object(n.st, 'http_json', side_effect=TimeoutError) as http:
            with self.assertRaises(TimeoutError):
                n.generate('answer', jobs, list(jobs), {'url':'no-network'})
            self.assertEqual(http.call_count, 1)
            self.assertTrue((Path(tmp)/'inflight.json').exists())
            self.assertEqual(json.loads((Path(tmp)/'answer-responses.json').read_text()), {})

    def test_success_saved_before_timing_failure(self):
        jobs = {str(i): {'digest':str(i), 'body':{}, 'input_tokens':1} for i in range(302)}
        with tempfile.TemporaryDirectory() as tmp, patch.object(n, 'LOCAL', Path(tmp)), \
                patch.object(n.ae, 'gate'), patch.object(n.st, 'http_json', return_value={}) as http:
            with self.assertRaises(RuntimeError):
                n.generate('answer', jobs, list(jobs), {'url':'no-network'})
            self.assertEqual(http.call_count, 1)
            self.assertFalse((Path(tmp)/'inflight.json').exists())
            self.assertEqual(len((Path(tmp)/'answer-received.jsonl').read_text().splitlines()), 1)
            self.assertEqual(len(json.loads((Path(tmp)/'answer-responses.json').read_text())), 1)

    def test_preexisting_inflight_prevents_any_send(self):
        jobs = {str(i):{} for i in range(302)}
        with tempfile.TemporaryDirectory() as tmp, patch.object(n, 'LOCAL', Path(tmp)), \
                patch.object(n.st, 'http_json') as http:
            (Path(tmp)/'inflight.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                n.generate('answer', jobs, list(jobs), {'url':'no-network'})
            http.assert_not_called()

    def test_missing_cache_never_generates(self):
        with patch.object(n.ab.lm, 'cached_vectors', side_effect=lambda v,s,t,i,f: f(t)):
            with self.assertRaises(RuntimeError):
                n.ab.cached(['q'], 'missing', {})

    def test_modified_original_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(n.lm, 'ROOT', Path(tmp)):
            (Path(tmp)/'old').write_text('new')
            with self.assertRaises(RuntimeError):
                n.verify_originals({'original_files_sha256': {'old': 'wrong'}})

    def test_rerank_digest_change_rejected(self):
        s = {'jobs':{'q':{'digest':'a'}}, 'queries':{'q':{}}}
        with self.assertRaises(RuntimeError):
            n.orders_for(s, {'q':{'digest':'b'}})

    def test_original_answer_failures_score_zero_without_regeneration(self):
        response = {'choices':[{'finish_reason':'length', 'message':{'content':'{"answer":"yes","evidence_ids":[]}'}}]}
        self.assertEqual(n.am.score(response, 'yes')['f1'], 0)
        self.assertEqual(n.am.score(response, 'yes')['failure'], 1)

    def test_pause_signals_only_registered_runner_group(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(n, 'LOCAL', Path(tmp)), \
                patch.object(n.ae, 'PAUSE', Path(tmp)/'pause.json'), \
                patch.object(n.subprocess, 'run') as ps, patch.object(n.os, 'getpgid', side_effect=[123,999]), \
                patch.object(n.os, 'killpg') as kill:
            (Path(tmp)/'status.json').write_text('{"pid":123,"pgid":123}')
            ps.return_value.stdout='python scripts/eval/locomo_answer_norw.py run'
            n.pause()
            kill.assert_called_once_with(123, n.signal.SIGINT)
            self.assertTrue((Path(tmp)/'pause.json').exists())


if __name__ == '__main__':
    unittest.main()
