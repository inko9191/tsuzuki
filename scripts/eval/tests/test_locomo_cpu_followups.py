"""Small artificial examples; real tokenizer/server/corpus construction is prohibited."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import locomo_long_memory as long
import locomo_both_answered as both


def conversation(group):
    return {'group': group, 'turns': [{'id': f'{group}/t{s}-{i}', 'group': group,
        'session': f'{group}#{s}', 'date': 'synthetic', 'text': ' '.join(['word']*10)}
        for s in range(3) for i in range(2)]}


class LongMemory(unittest.TestCase):
    def test_full_target_complete_sessions_deterministic_nested_lengths_without_network(self):
        data = [conversation(g) for g in ('a','b','c')]
        counter = lambda text: len(text.split())
        with patch.object(long.urllib.request, 'urlopen', side_effect=AssertionError('no Qwen allowed')):
            small = long.construct(data, 'a', counter, 144, tolerance=0)
            large = long.construct(data, 'a', counter, 216, tolerance=0,
                                   required_turn_ids=small['selected_turn_ids'])
            self.assertEqual(small, long.construct(list(reversed(data)), 'a', counter, 144, tolerance=0))
        self.assertTrue(small['target_complete'])
        self.assertEqual(small['memory_tokens'], 144)
        self.assertTrue(set(small['selected_turn_ids']) <= set(large['selected_turn_ids']))
        for c in data:
            original = long.sessions(c['turns'])
            for block in original:
                included = [t['id'] in small['selected_turn_ids'] for t in block]
                self.assertTrue(all(included) or not any(included))

    def test_shortage_and_duplicate_distractors_stop_instead_of_padding_or_truncating(self):
        data = [conversation(g) for g in ('a','b')]
        counter = lambda text: len(text.split())
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            long.construct(data, 'a', counter, 400)
        with self.assertRaisesRegex(ValueError, 'never truncate'):
            long.construct(data, 'a', counter, 20)
        data[1]['turns'][0]['id'] = data[0]['turns'][0]['id']
        with self.assertRaisesRegex(ValueError, 'unique'):
            long.construct(data, 'a', counter, 100)

    def test_reused_units_include_distractors_in_shared_group_and_keep_vector_ids(self):
        units = [{'id': 'p2|a/t1', 'group':'a', 'turns':['a/t1']},
                 {'id': 'rw|b/t1', 'group':'b', 'turns':['b/t1']},
                 {'id': 'fact|b/s1', 'group':'b', 'turns':['b/t1','b/t2']}]
        used = long.eligible_units(units, ['a/t1','b/t1'], 'joined-a')
        self.assertEqual([u['original_vector_row'] for u in used], [0,1])
        self.assertEqual({u['group'] for u in used}, {'joined-a'})
        self.assertEqual(units[1]['group'], 'b')

    def test_question_sample_is_fixed_per_conversation_and_type_ignoring_scores(self):
        queries = [{'id':f'{g}/{k}{i}', 'group':g, 'score':100-i}
                   for g in ('a','b') for k in both.STRATA if k != 'overall' for i in (0,1)]
        types={q['id']:q['id'].split('/')[1][:-1] for q in queries}
        chosen=long.sample_questions(queries,types)
        self.assertEqual(len(chosen),6)
        self.assertEqual(chosen,long.sample_questions(list(reversed(queries)),types))

    def test_short_control_pool_is_development_only_and_requires_every_cell(self):
        queries = [{'id':f'{g}/{k}{i}', 'group':g}
                   for g in ('a','b') for k in both.STRATA if k != 'overall' for i in (0,1)]
        types={q['id']:q['id'].split('/')[1][:-1] for q in queries}
        pool=[q['id'] for q in queries if q['id'].endswith('1')]
        self.assertEqual(set(long.sample_questions(queries,types,pool)),set(pool))
        with self.assertRaisesRegex(ValueError,'development-only'):
            long.sample_questions(queries,types,pool+['heldout/q'])
        with self.assertRaisesRegex(ValueError,'cell is empty'):
            long.sample_questions(queries,types,pool[:-1])


def scores():
    ids=['a/q1','a/q2','b/q1','b/q2']
    def row(qid):
        return {'id':qid,'type':'multiple_evidence','category':1,'f1':.5,
                'abstention':0,'failure':0,'empty_answer':0}
    return {arm:[row(q) for q in ids] for arm in both.ARMS}


class BothAnswered(unittest.TestCase):
    def test_main_and_sensitivity_masks_keep_failure_zero_and_no_f1_selection(self):
        rows=scores()
        rows['A'][0]['abstention']=1
        rows['C-final'][1]['failure']=1;rows['C-final'][1]['f1']=0
        rows['C-final'][2]['f1']=1
        summaries,ids=both.collect(rows)
        self.assertEqual(ids['both_valid_nonempty_nonabstaining'],['b/q1','b/q2'])
        self.assertEqual(ids['both_nonabstaining_sensitivity'],['a/q2','b/q1','b/q2'])
        self.assertEqual(summaries['both_valid_nonempty_nonabstaining']['tables']['overall']['C-final']['A']['mean'],.75)
        self.assertIsNone(summaries['both_valid_nonempty_nonabstaining']['tables']['temporal']['A']['A']['mean'])

    def test_id_alignment_duplicates_missing_or_strata_drift_rejected(self):
        rows=scores();rows['C-final'].reverse()
        self.assertEqual([r['id'] for r in both.align(rows)['C-final']],[r['id'] for r in rows['A']])
        rows['C-final'][0]['category']=2
        with self.assertRaisesRegex(ValueError,'strata differ'):
            both.align(rows)
        rows=scores();rows['C-final'].pop()
        with self.assertRaisesRegex(ValueError,'identical unique'):
            both.align(rows)


if __name__ == '__main__':
    unittest.main()
