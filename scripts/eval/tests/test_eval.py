#!/usr/bin/env python3
"""Pure tests of the evaluation metrics and candidate handling (no model, no server).

Run: .venv/bin/python -m unittest discover -s scripts/eval/tests -v
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retrieval_public as rp  # noqa: E402


def unit(rows):
    v = np.asarray(rows, np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class Metrics(unittest.TestCase):
    def test_graded_ndcg_and_recall(self):
        # relevant a (gain 2), b (gain 1); ranking puts b first, a third.
        m = rp.query_metrics(['b', 'x', 'a'], {'a': 2, 'b': 1})
        dcg = 1 / math.log2(2) + 2 / math.log2(4)
        idcg = 2 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(m['ndcg@10'], dcg / idcg)
        self.assertEqual(m['recall@10'], 1.0)
        self.assertEqual(rp.query_metrics(['x'] * 10 + ['a'], {'a': 1}), {'ndcg@10': 0.0, 'recall@10': 0.0})
        self.assertEqual(rp.query_metrics(['a'], {'a': 1, 'b': 1, 'c': 0})['recall@10'], 0.5)

    def test_top_k_order_and_exclusion(self):
        docs = unit([[1, 0], [0.9, 0.1], [0, 1]])
        queries = unit([[1, 0], [1, 0]])
        best = rp.top_k(queries, docs, 2, exclude=[None, 0])
        self.assertEqual(best[0].tolist(), [0, 1])
        self.assertEqual(best[1].tolist(), [1, 2])

    def test_groups_and_identical_ids(self):
        corpus = [{'id': 'q1', 'text': '', 'group': None}, {'id': 'd1', 'text': '', 'group': None}]
        ds = {'name': 'mteb-arguana', 'corpus': corpus, 'queries': [{'id': 'q1', 'text': ''}], 'qrels': {'q1': {'d1': 1}}}
        docs = unit([[1, 0], [0.5, 0.5]])
        self.assertEqual(rp.evaluate(ds, unit([[1, 0]]), docs)[0]['ndcg@10'], 1.0)  # own document excluded
        ds['name'] = 'other'
        self.assertLess(rp.evaluate(ds, unit([[1, 0]]), docs)[0]['ndcg@10'], 1.0)
        # Per-conversation candidates: the better match in another group is never returned.
        corpus = [{'id': 'a/1', 'group': 'a'}, {'id': 'b/1', 'group': 'b'}, {'id': 'a/2', 'group': 'a'}]
        ds = {'name': 'locomo', 'corpus': corpus, 'queries': [{'id': 'a/q', 'group': 'a'}], 'qrels': {'a/q': {'a/2': 1}}}
        docs = unit([[0, 1], [1, 0], [0.6, 0.4]])
        self.assertEqual(rp.evaluate(ds, unit([[1, 0]]), docs)[0]['ndcg@10'], 1.0)

    def test_truncate_renormalizes(self):
        v = rp.truncate(unit([[3, 4, 12]]), 2)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=6)
        np.testing.assert_allclose(v, [[0.6, 0.8]], rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
