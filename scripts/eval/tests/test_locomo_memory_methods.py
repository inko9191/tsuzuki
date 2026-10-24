"""Stage 2-5 retrieval rules on synthetic conversations; no model, server, GPU or licensed data."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locomo_memory as lm
import locomo_memory_methods as mm
import locomo_memory_stages as st


def raw_fixture():
    """Conversation a: session 1 (4 turns), session 2 (3 turns). Conversation b: one session (3 turns)."""
    def turns(prefix, n, session):
        return [{'speaker': 'Ann' if i % 2 else 'Bob', 'dia_id': f'D{session}:{i + 1}', 'text': f'{prefix} {session}.{i + 1}'}
                for i in range(n)]
    return [{'sample_id': 'a', 'conversation': {'session_1': turns('alpha', 4, 1), 'session_1_date_time': '1 May, 2023',
                                                'session_2': turns('beta', 3, 2), 'session_2_date_time': '9 May, 2023'}},
            {'sample_id': 'b', 'conversation': {'session_1': turns('gamma', 3, 1), 'session_1_date_time': '2 June, 2023'}}]


def corpus_fixture(raw):
    corpus = []
    for sample in raw:
        conv = sample['conversation']
        for key in sorted((k for k in conv if k.startswith('session_') and not k.endswith('time')), key=lambda k: int(k[8:])):
            corpus += [{'id': f'{sample["sample_id"]}/{t["dia_id"]}', 'group': sample['sample_id'],
                        'text': f'{t["speaker"]}: {t["text"]}'} for t in conv[key]]
    return corpus


def unit_vectors(n, dims=8, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, dims))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class Units(unittest.TestCase):
    def setUp(self):
        self.turns = mm.turn_table(corpus_fixture(raw_fixture()), raw_fixture())

    def test_turn_table_sessions_positions_dates(self):
        t = {x['id']: x for x in self.turns}
        self.assertEqual((t['a/D1:4']['session'], t['a/D1:4']['pos'], t['a/D1:4']['date']), ('a#1', 3, '1 May, 2023'))
        self.assertEqual((t['a/D2:1']['session'], t['a/D2:1']['pos']), ('a#2', 0))
        shuffled = corpus_fixture(raw_fixture())
        shuffled[0], shuffled[5] = shuffled[5], shuffled[0]
        with self.assertRaisesRegex(ValueError, 'session order'):
            mm.turn_table(shuffled, raw_fixture())

    def test_windows_stay_in_session_and_return_centre_first(self):
        units = {u['id']: u for u in mm.window_units(self.turns, 'c5', False)}
        last = units['c5|a/D1:4']
        self.assertEqual(last['turns'], ['a/D1:4', 'a/D1:3', 'a/D1:2'])  # no a/D2:* across the boundary
        self.assertEqual(units['c5|a/D1:2']['turns'], ['a/D1:2', 'a/D1:1', 'a/D1:3', 'a/D1:4'])
        self.assertEqual(last['text'].splitlines(), ['Ann: alpha 1.2', 'Bob: alpha 1.3', 'Ann: alpha 1.4'])

    def test_context_only_window_returns_centre_and_dates_prefix(self):
        unit = {u['id']: u for u in mm.window_units(self.turns, 'x3', True)}['x3+d|a/D2:3']
        self.assertEqual(unit['turns'], ['a/D2:3'])
        self.assertEqual(unit['text'].splitlines(), ['Date: 9 May, 2023', 'Bob: beta 2.1', 'Ann: beta 2.2', 'Bob: beta 2.3'])

    def test_t1_is_the_stage1_document(self):
        corpus = corpus_fixture(raw_fixture())
        self.assertEqual([u['text'] for u in mm.window_units(self.turns, 't1', False)], [d['text'] for d in corpus])

    def test_facts_return_only_cited_turns(self):
        units = mm.fact_units(self.turns, {'a#2': [{'fact': 'Ann likes tea.', 'turns': ['a/D2:2']}]}, True)
        self.assertEqual(units[0]['turns'], ['a/D2:2'])
        self.assertEqual(units[0]['text'], 'Date: 9 May, 2023\nAnn likes tea.')


class Ranking(unittest.TestCase):
    def test_t1_pipeline_reproduces_stage1_evaluate(self):
        raw = raw_fixture()
        corpus = corpus_fixture(raw)
        turns = mm.turn_table(corpus, raw)
        docs = unit_vectors(len(corpus), seed=1)
        queries = unit_vectors(4, seed=2)
        dataset = {'corpus': corpus, 'queries': [{'id': f'{g}/q{i}', 'group': g, 'category': 1 + i % 4, 'text': 'q'}
                                                  for i, g in enumerate('abab')],
                   'qrels': {'a/q0': {'a/D1:2': 1, 'a/D2:3': 1}, 'b/q1': {'b/D1:1': 1}, 'a/q2': {'a/D2:1': 1}, 'b/q3': {'b/D1:3': 1}}}
        before = lm.evaluate(dataset, queries, docs)
        index = mm.Index(mm.window_units(turns, 't1', False), docs)
        rows = mm.score_config(index, mm.config(['t1']), dataset, {'orig': [[(q, 'q')] for q in queries]},
                               {t['id']: t['session'] for t in turns}, {t['id']: 3 for t in turns},
                               {}, lambda q: lm.query_type(q, dataset['qrels'][q['id']]))
        for a, b in zip(before, rows):
            self.assertEqual(a['ranking'], [u.split('|', 1)[1] for u in b['ranking']])
            for k in mm.KS:
                self.assertEqual(a[f'recall@{k}'], b[f'recall@{k}'])

    def test_distinct_skips_units_without_new_turns(self):
        units = [{'turns': ['a', 'b']}, {'turns': ['b', 'a']}, {'turns': ['b', 'c']}]
        self.assertEqual(mm.distinct([0, 1, 2], units), [0, 2])

    def test_budgets_pay_each_date_line_once_and_stop_at_first_misfit(self):
        units = [{'turns': ['x1', 'x2']}, {'turns': ['y1']}, {'turns': ['x3']}]
        session = {'x1': 's1', 'x2': 's1', 'x3': 's1', 'y1': 's2'}
        tokens = {'x1': 100, 'x2': 100, 'y1': 100, 'x3': 10}
        row = mm.score([0, 1, 2], units, {'x3'}, session, tokens, {'s1': 20, 's2': 20}, dates=True)
        self.assertEqual(row['recall@T256'], 0)  # x1 120 + x2 100 = 220; y1 needs 120 -> stop (x3 never reached)
        self.assertEqual(row['recall@T512'], 1)  # 220 + 120 + 10
        self.assertEqual((row['payload_turns@10'], row['payload_tokens@10']), (4, 350))
        self.assertEqual(row['recall@5'], 1)
        undated = mm.score([0, 1, 2], units, {'x3'}, session, tokens, {'s1': 20, 's2': 20}, dates=False)
        self.assertEqual(undated['payload_tokens@10'], 310)

    def test_k_counts_units_not_turns(self):
        units = [{'turns': [f't{i}', f'u{i}']} for i in range(30)]
        session = {t: 's' for u in units for t in u['turns']}
        row = mm.score(list(range(30)), units, {'u4', 'u5'}, session, {t: 1 for t in session}, {}, False)
        self.assertEqual(row['recall@5'], 0.5)
        self.assertEqual(row['recall@N10'], 0.5)  # first 10 turns are units 0-4
        self.assertEqual(row['payload_turns@10'], 20)

    def test_bm25_prefers_rare_terms_and_drops_zero_scores(self):
        bm = mm.BM25(['the cat sat', 'the dog sat', 'the the the', 'pottery class today'])
        scores = bm.scores('Who went to a POTTERY class?')
        self.assertEqual(int(np.argmax(scores)), 3)
        units = [{'set': 't1', 'group': 'g', 'session': 's', 'text': t} for t in
                 ['the cat sat', 'the dog sat', 'pottery class today']]
        index = mm.Index(units, unit_vectors(3))
        self.assertEqual([i for i, _ in index.ranked('t1', 'g', 'bm25', None, 'pottery')], [2])

    def test_rrf_sums_reciprocal_ranks(self):
        fused = mm.rrf([[(1, .9), (2, .8)], [(2, 5.0), (3, 4.0)]])
        self.assertEqual([i for i, _ in fused], [2, 1, 3])
        self.assertAlmostEqual(dict(fused)[2], 1 / 62 + 1 / 61)
        self.assertEqual(mm.rrf([[(1, .9)]]), [(1, .9)])

    def test_session_cap_defers_extra_units(self):
        ranked = [(i, 1.0) for i in range(6)]
        self.assertEqual([i for i, _ in mm.session_cap(ranked, ['s', 's', 's', 's', 't', 's'], cap=3)], [0, 1, 2, 4, 3, 5])

    def test_mmr_demotes_duplicates_and_is_identity_at_lambda_one(self):
        v = np.array([[1, 0], [1, 0], [0, 1], [0, -1]], float)
        ranked = [(0, .9), (1, .89), (2, .85), (3, .0)]
        self.assertEqual([i for i, _ in mm.mmr(ranked, v, lam=1.0)], [0, 1, 2, 3])
        self.assertEqual([i for i, _ in mm.mmr(ranked, v, lam=0.7)], [0, 2, 1, 3])  # duplicate of 0 moves down

    def test_rank_fuses_every_set_ranker_and_query(self):
        units = [{'set': s, 'group': 'g', 'session': 's', 'text': t, 'turns': [t]} for s in ('t1', 'rw')
                 for t in ('apple pie', 'banana bread')]
        index = mm.Index(units, np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float))
        cfg = mm.config(['t1', 'rw'], ['dense', 'bm25'], ['orig', 'para'])
        with patch.object(index, 'ranked', wraps=index.ranked) as spy:
            index.rank(cfg, 'g', [(np.array([1., 0]), 'apple'), (np.array([0., 1]), 'banana')])
        self.assertEqual(spy.call_count, 2 * 2 * 2)

    def test_rerank_merge_keeps_valid_picks_then_original_order(self):
        self.assertEqual(mm.merge_rerank([5, 6, 7, 8], [7, 99, 7, 5]), [7, 5, 6, 8])


class Selection(unittest.TestCase):
    def rows(self, values, tokens=10):
        return [{'id': str(i), mm.PRIMARY: v, 'recall@10': v, 'payload_tokens@10': tokens} for i, v in enumerate(values)]

    def test_base_stays_without_a_significant_gain(self):
        base = self.rows([0, 1] * 50)
        noisy = self.rows([1, 0] * 25 + [0, 1] * 25)
        self.assertEqual(mm.select('base', {'base': base, 'noisy': noisy})['selected'], 'base')

    def test_best_significant_configuration_wins_and_order_must_match(self):
        base = self.rows([0] * 100)
        good, better = self.rows([1] * 60 + [0] * 40), self.rows([1] * 80 + [0] * 20)
        result = mm.select('base', {'base': base, 'good': good, 'better': better})
        self.assertEqual(result['selected'], 'better')
        self.assertGreater(result['vs_base']['good']['ci95'][0], 0)
        with self.assertRaises(ValueError):
            mm.select('base', {'base': base, 'bad': list(reversed(good))})

    def test_tie_prefers_smaller_payload(self):
        base = self.rows([0] * 100)
        result = mm.select('base', {'base': base, 'wide': self.rows([1] * 80 + [0] * 20, 300),
                                    'narrow': self.rows([1] * 80 + [0] * 20, 100)})
        self.assertEqual(result['selected'], 'narrow')


class Parsing(unittest.TestCase):
    def test_json_object_tolerates_thinking_and_fences(self):
        self.assertEqual(mm.json_object('<think>{"x": 0}</think>```json\n{"rewrite": "A"}\n```'), {'rewrite': 'A'})
        self.assertIsNone(mm.json_object('no json'))
        self.assertIsNone(mm.parse('rw', '{"rewrite": ""}'))

    def test_query_generators(self):
        self.assertEqual(mm.parse('para', '{"paraphrases": ["A  b", "A b", "c", "d"]}'), ['A b', 'c'])
        self.assertEqual(mm.parse('decomp', '{"subquestions": ["a", "b", "c", "d"]}'), ['a', 'b', 'c'])
        self.assertIsNone(mm.parse('decomp', '{"subquestions": []}'))
        self.assertEqual(mm.parse('hyde', '{"message": "I went hiking"}'), 'I went hiking')

    def test_facts_keep_only_real_citations(self):
        reply = json.dumps({'facts': [{'fact': 'Ann paints.', 'ids': ['D1:1', 'D9:9', 'D1:1']},
                                      {'fact': 'Invented.', 'ids': ['D7:1']}, {'fact': '', 'ids': ['D1:2']}]})
        self.assertEqual(mm.parse('fact', reply, valid_ids=['D1:1', 'D1:2']), [{'fact': 'Ann paints.', 'turns': ['D1:1']}])

    def test_rerank_picks_are_in_range_and_unique(self):
        self.assertEqual(mm.parse('rerank', '{"ranking": [3, 3, 0, 31, 1, 2.5]}', n_passages=30), [2, 0])

    def test_prompts_format_without_leftover_braces(self):
        fields = {'question': 'Q', 'date': 'D', 'context': 'C', 'message': 'M', 'messages': 'L', 'passages': 'P'}
        for kind in mm.PROMPTS:
            text = mm.messages(kind, **fields)[1]['content']
            self.assertNotIn('{question}', text)
            self.assertIn('JSON', text)


class Stages(unittest.TestCase):
    def test_plan_is_json_stable(self):
        self.assertEqual(lm.digest(json.loads(json.dumps(st.PLAN))), lm.digest(st.PLAN))

    def test_stage_configs_build_on_the_previous_selection(self):
        base, configs = st.stage_configs('2', None)
        self.assertEqual(base['id'], 't1/dense/orig/none')
        self.assertEqual(len(configs), 16)
        self.assertIn(base, configs)
        prev = mm.config(['c3+d'])
        base, configs = st.stage_configs('2b', prev)
        self.assertEqual(len(configs), 18)
        self.assertIn(base, configs)
        prev = mm.config(['c3+d'], ['dense', 'bm25'], post='cap3')
        base, configs = st.stage_configs('3', prev)
        self.assertEqual([c['queries'] for c in configs][-1], ['orig', 'para', 'hyde', 'decomp'])
        self.assertTrue(all(c['post'] == 'cap3' for c in configs))
        base, configs = st.stage_configs('4', prev)
        self.assertIn(['c3+d', 'rw+d', 'fact+d'], [c['units'] for c in configs])
        self.assertIn(base, configs)
        base, configs = st.stage_configs('5', prev)
        self.assertEqual([c['rerank'] for c in configs], [None, 'qwen'])

    def test_generation_inputs_hold_no_answers_and_only_earlier_turns(self):
        raw = raw_fixture()
        turns = mm.turn_table(corpus_fixture(raw), raw)
        dataset = {'queries': [{'id': 'a/q0', 'group': 'a', 'category': 2, 'text': 'When did Ann paint?',
                                'answer': 'SECRET'}], 'qrels': {'a/q0': {'a/D1:1': 1}}}
        prompt = st.generation_jobs('hyde', dataset, turns)[0][1][1]['content']
        self.assertIn('When did Ann paint?', prompt)
        self.assertNotIn('SECRET', prompt)
        self.assertNotIn('D1:1', prompt)
        jobs = {item: chat[1]['content'] for item, chat, _ in st.generation_jobs('rw', dataset, turns)}
        self.assertIn('alpha 1.2', jobs['a/D1:3'])
        self.assertNotIn('alpha 1.4', jobs['a/D1:3'])
        self.assertNotIn('alpha', jobs['a/D2:1'])  # previous session is not context
        facts = st.generation_jobs('fact', dataset, turns)
        self.assertEqual([(item, extra['valid_ids']) for item, _, extra in facts][2], ('b#1', ['D1:1', 'D1:2', 'D1:3']))

    def test_generated_outputs_reparse_apply_fallbacks_and_require_frozen_digests(self):
        raw = raw_fixture()
        turns = mm.turn_table(corpus_fixture(raw), raw)
        dataset = {'queries': [{'id': 'a/q0', 'text': 'Q'}, {'id': 'b/q1', 'text': 'Q'}]}
        qwen = FakeQwen()

        def stored(kind, replies):
            jobs = st.generation_jobs(kind, dataset, turns)
            lm.write(st.GEN / f'{kind}.json', {item: {'digest': qwen.item_digest(kind, chat), 'reply': replies(item),
                                                       'value': 'ignored: parsed again'} for item, chat, _ in jobs})
        with tempfile.TemporaryDirectory() as tmp, patch.object(st, 'GEN', Path(tmp)):
            stored('rw', lambda item: 'broken' if item == 'a/D1:1' else '{"rewrite": "R"}')
            rewrites, record = st.generated('rw', dataset, turns, qwen)
            self.assertEqual((rewrites['a/D1:1'], rewrites['a/D1:2'], record['fallback']), ('Bob: alpha 1.1', 'R', 1))
            stored('fact', lambda item: '{"facts": [{"fact": "F", "ids": ["D1:2"]}]}')
            facts, _ = st.generated('fact', dataset, turns, qwen)
            self.assertEqual(facts['a#1'], [{'fact': 'F', 'turns': ['a/D1:2']}])
            self.assertEqual(facts['a#2'], [])  # D1:2 is not a message of session 2
            stored('hyde', lambda item: '{"message": "H"}' if item == 'a/q0' else '')
            self.assertEqual(st.generated('hyde', dataset, turns, qwen)[0], {'a/q0': ['H'], 'b/q1': []})
            items = lm.read(Path(tmp) / 'hyde.json')
            items['b/q1']['digest'] = 'other settings'
            lm.write(Path(tmp) / 'hyde.json', items)
            with self.assertRaisesRegex(RuntimeError, 'frozen settings'):
                st.generated('hyde', dataset, turns, qwen)
            del items['b/q1']
            lm.write(Path(tmp) / 'hyde.json', items)
            with self.assertRaisesRegex(RuntimeError, '1 of 2 items'):
                st.generated('hyde', dataset, turns, qwen)

    def test_qwen_request_disables_thinking_and_constrains_json(self):
        qwen = st.Qwen.__new__(st.Qwen)
        qwen.model = 'qwen'
        qwen.settings = {'request': {'temperature': 0, 'seed': 0, 'json_schema': True,
                                     'chat_template_kwargs': {'enable_thinking': False}}, 'server': {'model_sha256': 'h'}}
        body = qwen.request('rw', mm.messages('rw', date='D', context='C', message='M'))
        self.assertEqual(body['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual(body['response_format']['json_schema']['schema'], mm.PROMPTS['rw']['schema'])
        self.assertEqual((body['temperature'], body['max_tokens']), (0, 200))



class FakeTokenizer:
    """Whitespace tokens; enough for budgets, cuts and the prompt check."""

    class _Encoding:
        def __init__(self, text):
            self.ids = list(range(len(text.split())))

    def __init__(self, model_dir):
        self.tokenizer = self

    def prompt_for(self, side):
        return lm.PROTOCOL['prompts'][side]

    def encode(self, texts, prompt):
        return [self._Encoding(t).ids for t in texts]

    def encode_batch(self, texts, add_special_tokens=False):
        return [self._Encoding(t) for t in texts]


def fake_embed(texts):
    out = np.zeros((len(texts), 768))
    for row, text in enumerate(texts):
        for term in mm.terms(text):
            out[row, hash(term) % 768] += 1
        out[row, 767] += 1e-3
    return out / np.linalg.norm(out, axis=1, keepdims=True)


class FakeQwen:
    replies = {'rw': '{"rewrite": "Ann talked about painting"}', 'para': '{"paraphrases": ["what did ann", "ann what"]}',
               'hyde': '<think>hmm</think>{"message": "I painted a lake"}', 'decomp': '{"subquestions": ["ann paint"]}',
               'fact': '{"facts": [{"fact": "Ann paints lakes.", "ids": ["D1:1", "D1:2"]}]}', 'rerank': '{"ranking": [2, 1]}'}
    calls = []

    def __init__(self, model_path=None):
        self.settings = {'server': {'model_id': 'qwen', 'model_sha256': 'h'}}

    @classmethod
    def offline(cls):
        return cls()

    def chat(self, kind, chat):
        FakeQwen.calls.append(kind)
        return self.replies[kind]

    def item_digest(self, kind, chat):
        return lm.digest([kind, chat])


class EndToEnd(unittest.TestCase):
    """Every stage command against fakes: registration, embedding, scoring, selection chaining, reports."""

    def test_all_stages_chain_and_never_touch_holdout(self):
        raw = raw_fixture()
        corpus = corpus_fixture(raw)
        queries = [{'id': 'a/q0', 'group': 'a', 'category': 1, 'text': 'What did Ann say about alpha 1.2?'},
                   {'id': 'a/q1', 'group': 'a', 'category': 2, 'text': 'When was beta 2.3?'},
                   {'id': 'b/q2', 'group': 'b', 'category': 4, 'text': 'gamma 1.1 who'}]
        dataset = {'corpus': corpus, 'queries': queries,
                   'qrels': {'a/q0': {'a/D1:2': 1, 'a/D2:2': 1}, 'a/q1': {'a/D2:3': 1}, 'b/q2': {'b/D1:1': 1}}}
        manifest = {'source': {'files': [{'sha256': 'x'}]}}

        @__import__('contextlib').contextmanager
        def backend(variant, args):
            yield {'event': 'ready'}, fake_embed

        with tempfile.TemporaryDirectory() as tmp:
            local, results = Path(tmp) / 'local', Path(tmp) / 'results'
            results.mkdir()
            (results / 'split.json').write_text('{}')
            patches = [patch.object(lm, 'LOCAL', local), patch.object(st, 'LOCAL', local), patch.object(st, 'GEN', local / 'gen'),
                       patch.object(lm, 'RESULTS', results), patch.object(st, 'RESULTS', results),
                       patch.object(lm, 'load_dev', lambda: (json.loads(json.dumps(dataset)), manifest)),
                       patch.object(st, 'load_raw', lambda m: raw), patch.object(st, 'committed', lambda p: True),
                       patch.object(lm, 'gpu_state', lambda: {}), patch.object(lm, 'fingerprint', lambda v: {'m': 'x'}),
                       patch.object(lm, 'start_backend', backend), patch.object(st, 'Qwen', FakeQwen),
                       patch('egemma_server.tokenizer.PromptTokenizer', FakeTokenizer)]
            patches.append(patch('sys.stdout', new=__import__('io').StringIO()))
            for p in patches:
                p.start()
            try:
                with self.assertRaisesRegex(RuntimeError, 'register'):
                    st.run('2', 0, None)
                st.register()
                for stage in ('2', '2b'):
                    st.run(stage, 0, None)
                with self.assertRaisesRegex(RuntimeError, 'not generated'):
                    st.run('3', 0, None)
                for kind in st.CORPUS_KINDS + st.QUERY_GEN_KINDS:
                    st.generate(kind, 2, None)
                calls = len(FakeQwen.calls)
                st.generate('rw', 1, None)  # resumable: nothing left to generate
                self.assertEqual(len(FakeQwen.calls), calls)
                for stage in ('3', '4', '5'):
                    st.run(stage, 0, None)
                reports = {s: lm.read(results / f'stage{s}-dev.json') for s in st.STAGES}
                self.assertEqual(len(reports['2']['configs']), 16)
                self.assertFalse(reports['2']['stage1_reproduction']['checked'])
                for s in st.STAGES[1:]:
                    self.assertEqual(reports[s]['previous_selection'], reports[st.PREVIOUS[s]]['selection']['selected_config'])
                    self.assertFalse(reports[s]['holdout_evaluated'])
                self.assertIn('rerank', FakeQwen.calls)
                self.assertEqual(reports['4']['generation']['fact']['fallback'], 0)
                self.assertTrue((results / 'stage5-dev.md').read_text().startswith('# 外部記憶・検索評価：ステージ5'))
                summary = lm.read(results / 'generation-outputs.json')
                self.assertEqual(summary['rw']['items'], len(corpus))
                for entry in reports['4']['configs'].values():
                    self.assertGreaterEqual(entry['overall']['recall@20'], entry['overall']['recall@5'])
                self.assertFalse(list(Path(tmp).rglob('holdout*')))
            finally:
                for p in reversed(patches):
                    p.stop()

if __name__ == '__main__':
    unittest.main()
