import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import longmemeval_g1_methods as m


class MethodsTests(unittest.TestCase):
    def test_duplicate_session_ids_use_position_and_gold_is_not_in_adapter(self):
        row = dict(question_id='q', haystack_session_ids=['s', 's'], haystack_dates=['d', 'd'],
                   haystack_sessions=[[dict(role='user', content='original', has_answer=True)],
                                      [dict(role='assistant', content='other', has_answer=False)]],
                   answer='hidden', answer_session_ids=['s'])
        value = m.turns(row)
        self.assertEqual([t['id'] for t in value], ['q/D1:1', 'q/D2:1'])
        self.assertEqual(value[0]['text'], 'user: original')
        self.assertNotIn('hidden', str(value))
        self.assertNotIn('has_answer', str(value))

    def test_invalid_role_refuses_no_silent_drop(self):
        row = dict(question_id='q', haystack_session_ids=['s'], haystack_dates=['d'],
                   haystack_sessions=[[dict(role='system', content='x')]])
        with self.assertRaises(ValueError):
            m.turns(row)

    def test_judge_strict_yes_no(self):
        def reply(text, finish='stop'):
            return dict(choices=[dict(finish_reason=finish, message=dict(content=text))])
        self.assertEqual(m.judge_label(reply(' YES ')), (1, None))
        self.assertEqual(m.judge_label(reply('no')), (0, None))
        self.assertEqual(m.judge_label(reply('not yes')), (0, 'parse'))
        self.assertEqual(m.judge_label(reply('yes', 'length')), (0, 'unfinished'))

    def test_partial_pairs_rejected(self):
        with self.assertRaises(ValueError):
            m.paired_rows({'A': {'1': 1}, 'C': {'1': 1, '2': 0}}, 'C', 'A')

    def test_zero_difference_is_exact_zero_interval(self):
        self.assertEqual(m.interval([0, 0], ['1', '2'], samples=100), [0., 0.])

    def test_cluster_members_not_independently_resampled(self):
        self.assertEqual(m.interval([-1, 1], ['same', 'same'], samples=100), [0., 0.])

    def test_stratified_and_order_deterministic(self):
        ci = m.interval([-1, 1, 0, 0], ['a', 'b', 'c', 'd'], ['s', 's', 't', 't'], samples=100)
        self.assertEqual(ci, m.interval([-1, 1, 0, 0], ['a', 'b', 'c', 'd'], ['s', 's', 't', 't'], samples=100))

    def test_ci_crossing_zero_not_automatically_noninferior(self):
        self.assertEqual(m.noninferiority([-.1, .1], [-.1, .1]), 'inconclusive')
        self.assertEqual(m.noninferiority([-.1, -.04], [-.1, -.04]), 'not_reproduced')
        self.assertEqual(m.noninferiority([-.03, .1], [0, .1]), 'confirmed_in_registered_sample')

    def test_exact_sign_excludes_ties(self):
        self.assertEqual(m.sign_test([1, 1, 1, 0]), dict(wins=3, losses=0, ties=1, n=3, p_two_sided=.25))

    def test_review_disagreement_first_all_questions_retained(self):
        answers = {'A': {'1': 'a', '2': 'a'}, 'C-final-norw': {'1': 'a', '2': 'b'}}
        self.assertEqual(m.review_order(['1', '2'], answers), ['2', '1'])


if __name__ == '__main__':
    unittest.main()
