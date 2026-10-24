"""Meaningful evaluation guards; no model/GPU or licensed data required."""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locomo_memory as lm


class Split(unittest.TestCase):
    def test_deterministic_complete_disjoint_and_stratified(self):
        queries = [{'id': f'{g}/{c}/{i}', 'group': g, 'category': c, 'text': f'{g} category {c} question {i}'}
                   for g in ('a', 'b') for c in range(1, 5) for i in range(10)]
        assignment = lm.split_questions(queries)
        self.assertEqual(assignment, lm.split_questions(list(reversed(queries))))
        self.assertEqual(set(assignment), {q['id'] for q in queries})
        for g in ('a', 'b'):
            for c in range(1, 5):
                part = [assignment[q['id']] for q in queries if q['group'] == g and q['category'] == c]
                self.assertEqual(part.count('holdout'), 2)
                self.assertEqual(part.count('dev'), 8)

    def test_duplicate_questions_cannot_cross_split_even_with_different_categories(self):
        queries = [{'id': f'q{i}', 'group': 'a', 'category': 1, 'text': f'Question {i}'} for i in range(10)]
        queries.append({'id': 'duplicate', 'group': 'a', 'category': 2, 'text': '  QUESTION 1?  '})
        assigned = lm.split_questions(queries)
        self.assertEqual(assigned['q1'], assigned['duplicate'])

    def test_dev_loader_never_opens_holdout_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            local, results = Path(tmp) / 'local', Path(tmp) / 'results'
            with patch.object(lm, 'LOCAL', local), patch.object(lm, 'RESULTS', results):
                lm.write(local / 'corpus.json', [{'id': 'a/1', 'group': 'a', 'text': 'fixture'}])
                lm.write(local / 'dev.json', {'queries': [{'id': 'a/q'}], 'qrels': {'a/q': {'a/1': 1}}})
                lm.write(local / 'split-map.json', {'a/q': 'dev', 'a/hidden': 'holdout'})
                manifest = {'protocol': copy.deepcopy(lm.PROTOCOL), 'protocol_sha256': lm.digest(lm.PROTOCOL),
                            'corpus_sha256': lm.file_hash(local / 'corpus.json'),
                            'split_map_sha256': lm.file_hash(local / 'split-map.json'),
                            'split_files_sha256': {'dev': lm.file_hash(local / 'dev.json'), 'holdout': 'not-readable'}}
                lm.write(results / 'split.json', manifest)
                # There is deliberately no holdout file: loading dev still succeeds.
                dataset, _ = lm.load_dev()
                self.assertEqual([q['id'] for q in dataset['queries']], ['a/q'])
                changed = copy.deepcopy(manifest)
                changed['protocol']['ks'] = [10]
                lm.write(results / 'split.json', changed)
                with self.assertRaisesRegex(ValueError, 'protocol changed'):
                    lm.load_dev()
                lm.write(results / 'split.json', manifest)
                (local / 'dev.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'dev changed'):
                    lm.load_dev()


class Metrics(unittest.TestCase):
    def test_multi_evidence_denominator_and_deduplication(self):
        row = lm.evidence_metrics(['a', 'a', 'x', 'b'] + ['x'] * 6 + ['c'], {'a', 'b', 'c'})
        self.assertEqual(row['recall@5'], 2 / 3)
        self.assertEqual(row['all_evidence@10'], 0)
        self.assertEqual(row['recall@20'], 1)
        with self.assertRaises(ValueError):
            lm.evidence_metrics(['a'], set())

    def test_conversation_isolation_and_stable_ties(self):
        dataset = {'corpus': [{'id': 'b/best', 'group': 'b'}, {'id': 'a/first', 'group': 'a'},
                              {'id': 'a/second', 'group': 'a'}],
                   'queries': [{'id': 'a/q', 'group': 'a', 'category': 4}], 'qrels': {'a/q': {'a/second': 1}}}
        row = lm.evaluate(dataset, np.array([[1, 0]]), np.array([[1, 0], [0, 1], [0, 1]]))[0]
        self.assertEqual(row['ranking'], ['a/first', 'a/second'])
        self.assertEqual(row['recall@5'], 1)

    def test_question_types_prioritize_temporal_and_keep_multiple_evidence(self):
        self.assertEqual(lm.query_type({'category': 2}, {'a': 1, 'b': 1}), 'temporal')
        self.assertEqual(lm.query_type({'category': 4}, {'a': 1, 'b': 1}), 'multiple_evidence')
        self.assertEqual(lm.query_type({'category': 1}, {'a': 1}), 'multiple_evidence')
        self.assertEqual(lm.query_type({'category': 3}, {'a': 1}), 'single_fact')

    def test_recall_is_macro_average_not_hit_rate_or_micro_average(self):
        a = {'id': 'a', 'category': 1, 'type': 'multiple_evidence', **lm.evidence_metrics(['a'], {'a', 'b', 'c'})}
        b = {'id': 'b', 'category': 4, 'type': 'single_fact', **lm.evidence_metrics(['d'], {'d'})}
        result = lm.aggregate([a, b])['overall']
        self.assertAlmostEqual(result['recall@10'], 2 / 3)
        self.assertEqual(result['all_evidence@10'], 0.5)
        self.assertEqual(result['hit@10'], 1)

    def test_invalid_vectors_fail_before_scoring(self):
        for vectors in (np.ones((1, 256)), np.zeros((1, 768)), np.full((1, 768), np.nan)):
            with self.assertRaises(ValueError):
                lm.checked_vectors(vectors, 1)


if __name__ == '__main__':
    unittest.main()
