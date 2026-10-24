from pathlib import Path
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locomo_answer_methods as am
import locomo_answer_eval as ae
import locomo_memory_pause as pause


def response(answer, finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {
        'content': json.dumps({'answer': answer, 'evidence_ids': []})}}]}


def turns():
    return [{'id': f'g/D{i}', 'group': 'g', 'session': 'g#1' if i < 3 else 'g#2',
             'date': 'today', 'text': f'P: message {i}'} for i in range(4)]


class AnswerRules(unittest.TestCase):
    def test_f1_counts_repetitions_and_articles(self):
        self.assertAlmostEqual(am.score(response('a blue blue car'), 'the blue car')['f1'], .8)

    def test_normalization_and_integer_reference(self):
        self.assertEqual(am.score(response('２０２４!'), 2024)['em'], 1)

    def test_empty_reference_rejected(self):
        for bad in ('', 'the', [], None, True):
            with self.assertRaises(ValueError):
                am.score(response('x'), bad)

    def test_abstention_exact_and_zero(self):
        self.assertEqual(am.score(response(' Insufficient  evidence. '), 'evidence')['f1'], 0)
        self.assertEqual(am.score(response(' Insufficient  evidence. '), 'evidence')['abstention'], 1)
        self.assertFalse(am.abstains('Insufficient evidence because context is absent'))

    def test_unfinished_is_zero_even_valid_json(self):
        s = am.score(response('blue', 'length'), 'blue')
        self.assertEqual((s['f1'], s['failure']), (0, 1))

    def test_parse_and_empty_are_not_abstention(self):
        self.assertEqual(am.score({}, 'blue')['abstention'], 0)
        s = am.score(response(''), 'blue')
        self.assertEqual((s['empty_answer'], s['failure'], s['abstention']), (1, 0, 0))

    def test_schema_is_strict(self):
        x = response('blue'); x['choices'][0]['message']['content'] = '{"answer":"blue"}'
        self.assertEqual(am.score(x, 'blue')['failure'], 1)

    def test_recent_contiguous_suffix_no_partial_turn(self):
        ts = turns()
        text, ids = am.recent(ts, len(am.render(ts[-2:])), len)
        self.assertEqual(ids, [t['id'] for t in ts[-2:]])
        self.assertEqual(text, am.render(ts[-2:]))
        self.assertEqual(am.recent(ts, 1, len), ('', []))

    def test_memory_skips_recent_and_duplicate_then_stops(self):
        ts = turns(); lookup = {t['id']: t for t in ts}
        order = ['g/D3', 'g/D0', 'g/D0', 'g/D1', 'g/D2']
        text, ids = am.memory(order, lookup, ['g/D3'], len(am.render(ts[:2])), len)
        self.assertEqual(ids, ['g/D0', 'g/D1'])
        self.assertEqual(text.count('Date:'), 1)

    def test_headers_paid_once_across_sessions(self):
        self.assertEqual(am.render(turns()).count('Date:'), 2)

    def test_full_A_has_every_turn(self):
        p = am.payload('A', turns(), {}, len)
        self.assertEqual(p['conversation'], am.render(turns()))
        self.assertEqual(p['memory_ids'], [])

    def test_prompts_are_identical_across_conditions(self):
        chat = am.messages('c', 'm', 'q')
        self.assertEqual(chat[0]['content'], am.SYSTEM)
        self.assertEqual(chat[1]['content'], 'Conversation:\nc\n\nRetrieved conversation excerpts:\nm\n\nQuestion:\nq')
        self.assertNotIn('gold', json.dumps(chat))

    def test_pilot_balanced_and_stable(self):
        qs = [{'id': f'g{g}/q{k}{i}', 'group': f'g{g}'}
              for g in range(10) for k in am.CONDITIONS[:3] for i in range(5)]
        kinds = ('multiple_evidence', 'single_fact', 'temporal')
        mapping = {q['id']: kinds[i // 5 % 3] for i, q in enumerate(qs)}
        selected = am.pilot(qs, mapping)
        self.assertEqual(len(selected), 120)
        self.assertEqual(selected, am.pilot(list(reversed(qs)), mapping))

    def test_pilot_small_cell_rejected(self):
        with self.assertRaises(ValueError):
            am.pilot([{'id': 'g/q', 'group': 'g'}], {'g/q': 'temporal'})

    def test_alias_means_one_request_per_identical_arm(self):
        alias = am.aliases({'C-final': {'x': 1}, 'C-t1': {'x': 1}, 'C-p2': {'x': 2}})
        qs = [{'id': 'g/q1', 'group': 'g'}, {'id': 'g/q2', 'group': 'g'}]
        order = am.schedule(qs, alias)
        self.assertEqual(len(order), 12)
        self.assertNotIn('C-final', [arm for _, arm in order])
        self.assertEqual(len(set(order)), len(order))


class AnswerIO(unittest.TestCase):
    def test_missing_timings_saves_success_and_stops_before_next_question(self):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            record = {'plan_sha256': 'p', 'plan': {'model': {'model_id': 'm'}}}
            now = {'url': 'local'}
            item = {'id': 'g/q', 'condition': 'A', 'messages': [], 'input_tokens': 20}
            prepared = {'plan_sha256': 'p', 'items': {'first': item, 'next': item},
                        'schedule': ['first', 'next']}
            (root / 'pilot-inputs.json').write_text(json.dumps(prepared))
            stack.enter_context(patch.object(ae, 'LOCAL', root))
            stack.enter_context(patch.object(ae, 'gate'))
            stack.enter_context(patch.object(ae, 'checked_plan', return_value=record))
            stack.enter_context(patch.object(ae, 'server', return_value=now))
            api = stack.enter_context(patch.object(ae.st, 'http_json', return_value=response('blue')))
            with self.assertRaises(RuntimeError): ae.generate()
            self.assertEqual(api.call_count, 1)
            self.assertIn('first', json.loads((root / 'pilot-responses.json').read_text()))
            self.assertFalse((root / 'inflight.json').exists())
            with self.assertRaises(RuntimeError): ae.generate()
            self.assertEqual(api.call_count, 1)

    def test_parse_failure_is_saved_and_never_regenerated(self):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            record = {'plan_sha256': 'p', 'plan': {'model': {'model_id': 'm'}}}
            now = {'url': 'local'}
            item = {'id': 'g/q', 'condition': 'A', 'messages': [], 'input_tokens': 20}
            prepared = {'plan_sha256': 'p', 'items': {'key': item}, 'schedule': ['key']}
            (root / 'pilot-inputs.json').write_text(json.dumps(prepared))
            stack.enter_context(patch.object(ae, 'LOCAL', root))
            stack.enter_context(patch.object(ae, 'gate'))
            stack.enter_context(patch.object(ae, 'checked_plan', return_value=record))
            stack.enter_context(patch.object(ae, 'server', return_value=now))
            reply = {'choices': [], 'timings': {k: 0 for k in
                      ('cache_n', 'prompt_n', 'prompt_ms', 'predicted_n', 'predicted_ms')}}
            api = stack.enter_context(patch.object(ae.st, 'http_json', return_value=reply))
            ae.generate(); ae.generate()
            self.assertEqual(api.call_count, 1)

    def test_persistent_pause_gate(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'pause.json'; p.write_text('{}')
            with patch.object(ae, 'PAUSE', p), self.assertRaises(RuntimeError):
                ae.gate()

    def test_pause_targets_answer_runner_not_server(self):
        self.assertEqual(pause.script('python3 -u scripts/eval/locomo_answer_eval.py generate-pilot', pause.ROOT),
                         pause.ROOT / 'scripts/eval/locomo_answer_eval.py')
        self.assertIsNone(pause.script('llama-server -c 65536', pause.ROOT))

    def test_journal_recovers_response_before_batch_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p, j = Path(d) / 'replies.json', Path(d) / 'received.jsonl'
            ae.received(j, 'A|g/q', {'digest': 'd', 'response': response('blue')})
            self.assertEqual(ae.recover(p, j)['A|g/q']['digest'], 'd')

    def test_conflicting_journal_stops(self):
        with tempfile.TemporaryDirectory() as d:
            p, j = Path(d) / 'replies.json', Path(d) / 'received.jsonl'
            p.write_text(json.dumps({'key': {'digest': 'old'}}))
            ae.received(j, 'key', {'digest': 'new'})
            with self.assertRaises(RuntimeError): ae.recover(p, j)

    def test_model_and_context_checked_before_generation(self):
        plan = {'model': {'url': 'local', 'model_id': 'm', 'model_sha256': 'hash', 'build': 'b'}}
        with patch.object(ae.st, 'model_identity', return_value={**plan['model'], 'n_ctx': 8192}):
            with self.assertRaises(RuntimeError): ae.server(plan)


if __name__ == '__main__':
    unittest.main()
