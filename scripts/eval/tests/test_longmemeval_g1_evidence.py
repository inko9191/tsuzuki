"""Synthetic CPU tests for final-prompt evidence ID attribution."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import longmemeval_g1_evidence as evidence


class EvidenceTests(unittest.TestCase):
    def question(self):
        return {'question_id':'q','answer_session_ids':['old','new'],
                'haystack_session_ids':['old','distractor','new'],
                'haystack_sessions':[[{'has_answer':True},{'has_answer':False}],
                                     [{'has_answer':False}],
                                     [{'has_answer':True},{'has_answer':False}]]}

    def test_partial_evidence_is_not_complete_even_when_answer_is_correct(self):
        row=evidence.coverage(self.question(),['q/D3:1'])
        self.assertFalse(row['all_evidence_sessions'])
        self.assertTrue(row['any_evidence_session'])
        row['C_correct']=True
        self.assertEqual(evidence.summarize([row])['missing_correct'],1)

    def test_recent_plus_retrieved_can_cover_both_evidence_sessions(self):
        recent=['q/D3:1'];retrieved=['q/D1:1']
        self.assertFalse(evidence.coverage(self.question(),retrieved)['all_evidence_sessions'])
        both=evidence.coverage(self.question(),recent+retrieved)
        self.assertTrue(both['all_evidence_sessions'])
        self.assertTrue(both['all_marked_turns_selected'])

    def test_session_presence_does_not_imply_relevant_turn_presence(self):
        row=evidence.coverage(self.question(),['q/D1:2','q/D3:2'])
        self.assertTrue(row['all_evidence_sessions'])
        self.assertFalse(row['all_marked_turns_selected'])
        self.assertEqual(row['selected_marked_turns'],[])

    def test_duplicate_session_ids_are_flagged_without_false_turn_sufficiency(self):
        q=self.question();q['haystack_session_ids']=['old','old','new']
        q['haystack_sessions'][1][0]['has_answer']=True
        row=evidence.coverage(q,['q/D1:1','q/D3:1'])
        self.assertTrue(row['all_evidence_sessions'])
        self.assertEqual(row['duplicate_haystack_session_ids'],['old'])
        self.assertFalse(row['all_marked_turns_selected'])

    def test_invalid_ids_or_evidence_metadata_fail_closed(self):
        for tid in ('another/D1:1','q/D0:1','q/D4:1','q/D1:3'):
            with self.assertRaises(ValueError):evidence.coverage(self.question(),[tid])
        q=self.question();q['answer_session_ids']=['absent']
        with self.assertRaises(ValueError):evidence.coverage(q,['q/D1:1'])

    def test_no_annotated_turns_is_unknown_not_vacuously_sufficient(self):
        q=self.question()
        for session in q['haystack_sessions']:
            for turn in session:turn.pop('has_answer')
        self.assertIsNone(evidence.coverage(q,['q/D1:1','q/D3:1'])['all_marked_turns_selected'])


if __name__=='__main__':unittest.main()
