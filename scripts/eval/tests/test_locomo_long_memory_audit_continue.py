"""CPU-only deviation/swap/continuation tests; no server/model calls."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import locomo_long_memory_audit_continue as c


class Continuation(unittest.TestCase):
    def test_exclude_only_runtime_logs_not_response_journals(self):
        p={'private_source_sha256':{'qwen-A.log':'a','runner.log':'b','answer-received.jsonl':'c','input.json':'d','responses.json':'e'}}
        self.assertEqual(c.protected_private(p),{'answer-received.jsonl':'c','input.json':'d','responses.json':'e'})

    def test_same_seven_remaining_cells_without_first_A(self):
        self.assertEqual(c.remaining_schedule(),c.audit.schedule()[1:])
        self.assertEqual(len(c.remaining_schedule()),7)
        self.assertEqual(sum(4 if a=='A' else 3 for r,a,x in c.remaining_schedule())+sum(3 for r,a,x in c.remaining_schedule() if x==8192),28)

    def test_runtime_bindings_restore_after_failure(self):
        saved=(c.audit.LOCAL,c.audit.checked,c.audit.base.restart,c.audit.summary)
        with self.assertRaises(RuntimeError):
            with c.amended_runtime():
                self.assertEqual(c.audit.LOCAL,c.LOCAL)
                self.assertIs(c.audit.checked,c.checked)
                raise RuntimeError('synthetic')
        self.assertEqual((c.audit.LOCAL,c.audit.checked,c.audit.base.restart,c.audit.summary),saved)

    def test_large_existing_swap_can_be_stable(self):
        values=[{'swap_used_bytes':2*c.audit.GIB,'swapins':4,'swapouts':5}]*3
        with patch.object(c.ae,'gate'),patch.object(c.audit.base.resources,'observe_os',side_effect=values),patch.object(c.audit.base.resources,'violation',return_value=None),patch.object(c.time,'sleep') as sleep:
            r=c.swap_check('synthetic')
        self.assertTrue(r['started_with_large_swap']);self.assertTrue(r['stable_in_observation_window']);self.assertFalse(r['time_is_reference']);self.assertEqual(sleep.call_count,2)

    def test_any_sample_increment_marks_even_if_usage_later_decreases(self):
        values=[{'swap_used_bytes':2*c.audit.GIB,'swapins':4,'swapouts':5},{'swap_used_bytes':2*c.audit.GIB,'swapins':5,'swapouts':5},{'swap_used_bytes':c.audit.GIB,'swapins':5,'swapouts':5}]
        with patch.object(c.ae,'gate'),patch.object(c.audit.base.resources,'observe_os',side_effect=values),patch.object(c.audit.base.resources,'violation',return_value=None),patch.object(c.time,'sleep'):
            r=c.swap_check('synthetic')
        self.assertTrue(r['increment_observed']);self.assertTrue(r['time_is_reference'])

    def test_resource_violation_stops_before_wait(self):
        with patch.object(c.ae,'gate'),patch.object(c.audit.base.resources,'observe_os',return_value={}),patch.object(c.audit.base.resources,'violation',return_value='critical'),patch.object(c.time,'sleep') as sleep:
            with self.assertRaises(RuntimeError):c.swap_check('synthetic')
        sleep.assert_not_called()

    def test_original_unknown_inflight_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'ORIGINAL',Path(tmp)),patch.object(c.lm,'read',return_value={}):
            (Path(tmp)/'inflight.json').write_text('{}')
            with self.assertRaises(RuntimeError):c.load_original()

    def test_original_four_keys_not_missing_or_extra(self):
        import tempfile
        p={'question_ids':['g/0','g/1','g/2'],'cache_off_id':'g/0'}
        with tempfile.TemporaryDirectory() as tmp,patch.object(c,'ORIGINAL',Path(tmp)),patch.object(c.lm,'read',side_effect=[{},p]),patch.object(c.ae,'recover',return_value={'unexpected':{}}):
            with self.assertRaises(RuntimeError):c.load_original()


if __name__=='__main__':unittest.main()
