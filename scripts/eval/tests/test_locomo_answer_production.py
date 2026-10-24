from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locomo_answer_production as production
import locomo_answer_production_report as report
import locomo_memory_pause as pause


class ProductionTests(unittest.TestCase):
    def test_full_overflow_refused_without_truncation(self):
        production.ensure_length(65152, 256, 65536, 128)
        with self.assertRaises(RuntimeError):
            production.ensure_length(65153, 256, 65536, 128)
        with self.assertRaises(RuntimeError):
            production.ensure_length(8073, 120, 8192)

    def test_rerank_prompt_and_reordering_equal_frozen_implementation(self):
        turns = [{'id': 'g/t0', 'text': 'first', 'date': '2020', 'group': 'g'},
                 {'id': 'g/t1', 'text': 'second', 'date': '2020', 'group': 'g'}]
        units = [{'id': 'u0', 'turns': ['g/t1', 'g/t0']}, {'id': 'u1', 'turns': ['g/t0']}]
        q, cfg = {'id': 'g/q', 'text': 'question'}, production.mm.config(['p2'], rerank='qwen')
        seen = []
        class Qwen:
            def item_digest(self, kind, chat):
                return 'digest'
            def chat(self, kind, chat):
                seen.append(chat)
                return '{"ranking": [2, 1]}'
        class Store:
            def get(self, key, digest):
                return None
            def put(self, key, value):
                self.value = value['value']
        store = Store()
        rerank, _ = production.st.reranker(Qwen(), cfg, units, turns, store)
        actual = rerank(q, [0, 1])
        self.assertEqual(seen[0], production.rerank_chat(q, [0, 1], units, turns, cfg))
        self.assertEqual(actual, production.reranked_order([0, 1], store.value))
        self.assertEqual(actual, [1, 0])

    def test_unknown_inflight_refused_known_received_is_recovered(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'inflight.json'
            production.lm.write(path, {'key': 'k', 'digest': 'd'})
            with self.assertRaises(RuntimeError):
                production.recover_pending(path, {})
            self.assertTrue(path.exists())
            production.recover_pending(path, {'k': {'digest': 'd'}})
            self.assertFalse(path.exists())

    def test_success_saved_once_even_when_unfinished(self):
        response = {'choices': [{'finish_reason': 'length', 'message': {'content': 'unfinished'}}],
                    'timings': {'cache_n': 0, 'prompt_n': 1, 'prompt_ms': 1, 'predicted_n': 1, 'predicted_ms': 1}}
        jobs = {'C-final|g/q': {'digest': 'd', 'body': {'messages': []}, 'input_tokens': 1}}
        with tempfile.TemporaryDirectory() as d, patch.object(production, 'LOCAL', Path(d)), \
                patch.object(production.ae, 'gate'), patch.object(production.st, 'http_json', return_value=response) as http:
            first = production.generate_jobs('answer', jobs, list(jobs), 'url', True)
            second = production.generate_jobs('answer', jobs, list(jobs), 'url', True)
            self.assertEqual(first, second)
            self.assertEqual(http.call_count, 1)
            self.assertEqual(production.am.score(response, 'gold')['f1'], 0.)
            self.assertEqual(len((Path(d) / 'answer-received.jsonl').read_text().splitlines()), 1)

    def test_changed_saved_digest_refused_before_generation(self):
        with tempfile.TemporaryDirectory() as d, patch.object(production, 'LOCAL', Path(d)), \
                patch.object(production.ae, 'gate'), patch.object(production.st, 'http_json') as http:
            production.lm.write(Path(d) / 'answer-responses.json', {'k': {'digest': 'old'}})
            with self.assertRaises(RuntimeError):
                production.generate_jobs('answer', {'k': {'digest': 'new'}}, ['k'], 'url', False)
            http.assert_not_called()

    def test_persistent_pause_prevents_unsent_request(self):
        with tempfile.TemporaryDirectory() as d, patch.object(production, 'LOCAL', Path(d)), \
                patch.object(production.ae, 'gate', side_effect=RuntimeError('paused')), \
                patch.object(production.st, 'http_json') as http:
            with self.assertRaises(RuntimeError):
                production.generate_jobs('rerank', {'q': {'digest': 'd'}}, ['q'], 'url', False)
            http.assert_not_called()
            self.assertEqual(production.lm.read(Path(d) / 'rerank-responses.json'), {})

    def test_stopped_server_never_started_by_context_switch(self):
        with patch.object(production, 'checked', return_value={'plan': {'answer': {'model': {'url': 'url'}}}}), \
                patch.object(production.st, 'model_identity', side_effect=ConnectionError('stopped')), \
                patch.object(production.subprocess, 'Popen') as start:
            with self.assertRaises(ConnectionError):
                production.switch_context(8192)
            start.assert_not_called()

    def test_final_report_refuses_incomplete_conditions_before_scoring(self):
        with patch.object(report.ar, 'complete') as complete:
            with self.assertRaises(RuntimeError):
                report.ready({'items': {'one': {'id': 'g/q'}}}, {},
                    {'plan': {'responses': 7, 'questions': 1}}, {'phase': 'answer complete', 'status': 'complete'})
            complete.assert_not_called()
            with self.assertRaises(RuntimeError):
                report.ready({}, {}, {}, {'phase': 'answer generation', 'status': 'running'})
            complete.assert_not_called()

    def test_pause_targets_production_python_not_qwen_or_shell(self):
        root = production.lm.ROOT
        self.assertEqual(pause.script('.venv/bin/python -u scripts/eval/locomo_answer_production.py run', root),
                         root / 'scripts/eval/locomo_answer_production.py')
        self.assertIsNone(pause.script('/bin/zsh -c "python scripts/eval/locomo_answer_production.py run"', root))
        self.assertIsNone(pause.script('/opt/homebrew/bin/llama-server -c 65536', root))

    def test_environment_allows_one_qwen_and_rejects_other_gpu_work(self):
        with patch.object(production.subprocess, 'check_output', return_value='12 /opt/homebrew/bin/llama-server -m /a b/model -c 65536\n'):
            self.assertEqual(production.environment()['qwen_servers'][0]['pid'], 12)
        with patch.object(production.subprocess, 'check_output', return_value='12 /opt/homebrew/bin/llama-server\n13 python /ComfyUI/main.py\n'):
            with self.assertRaises(RuntimeError):
                production.environment()


if __name__ == '__main__':
    unittest.main()
