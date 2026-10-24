"""Memory-only audit synthetic tests; no corpus/network/GPU/server signals."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import locomo_long_memory_audit as audit


class MemoryAudit(unittest.TestCase):
    def test_selection_uses_length_and_ids_never_score(self):
        p={'questions':{f'g{g}/q{i}':{'group':f'g{g}'} for g in range(2) for i in range(3)},
           'schedule':[f'g{g}/q{i}' for g in range(2) for i in range(3)]}
        items={q:{'id':q,'input_tokens':100 if q.startswith('g0') else 200,'f1':1,'gold':'private'} for q in p['schedule']}
        selected=audit.selected(p,items)
        changed={k:{**v,'f1':-99,'gold':'different'} for k,v in reversed(list(items.items()))}
        self.assertEqual(audit.selected(p,changed),selected)
        self.assertEqual(selected[0],['g1/q0','g1/q1','g1/q2'])

    def test_selected_group_must_be_three_original_questions(self):
        with self.assertRaises(RuntimeError):audit.selected({'questions':{'q':{'group':'g'}},'schedule':['q']},{'q':{'id':'q','input_tokens':4}})

    def test_twenty_six_answers_in_reversed_orders(self):
        p={'question_ids':['g/q0','g/q1','g/q2'],'cache_off_id':'g/q1'}
        jobs=[item for r,a,c in audit.schedule() for item in audit.jobs(p,a,f'r{r}-{a}-c{c}')] 
        self.assertEqual(len(jobs),26);self.assertEqual(len({key for key,q,off in jobs}),26)
        self.assertEqual(sum(off for key,q,off in jobs),2)
        self.assertTrue(all(key.startswith('r') for key,q,off in jobs))

    def test_received_before_timing_failure_no_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(audit,'LOCAL',Path(tmp)),patch.object(audit.ae,'gate'),patch.object(audit.lm,'file_hash',return_value='plan'),patch.object(audit.base.st,'http_json',return_value={'choices':[]}) as http:
            replies={};watch=type('Watch',(),{'error':None})()
            with self.assertRaises(RuntimeError):audit.send('k',{}, {'url':'unused'},replies,watch)
            self.assertIn('k',replies);self.assertTrue((Path(tmp)/'received.jsonl').exists())
            self.assertFalse((Path(tmp)/'inflight.json').exists())
            with self.assertRaises(RuntimeError):audit.send('k',{}, {'url':'unused'},replies,watch)
            self.assertEqual(http.call_count,1)

    def test_unknown_inflight_retained_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(audit,'LOCAL',Path(tmp)),patch.object(audit.ae,'gate'),patch.object(audit.lm,'file_hash',return_value='plan'),patch.object(audit.base.st,'http_json',side_effect=TimeoutError) as http:
            replies={};watch=type('Watch',(),{'error':None})()
            with self.assertRaises(TimeoutError):audit.send('k',{}, {'url':'unused'},replies,watch)
            self.assertTrue((Path(tmp)/'inflight.json').exists())
            with self.assertRaises(RuntimeError):audit.send('k',{}, {'url':'unused'},replies,watch)
            self.assertEqual(http.call_count,1)

    def test_restart_records_do_not_touch_accuracy_directory(self):
        original=audit.base.LOCAL
        with self.assertRaises(RuntimeError):
            with audit.isolated_restart_records():
                self.assertEqual(audit.base.LOCAL,audit.LOCAL)
                raise RuntimeError('synthetic')
        self.assertEqual(audit.base.LOCAL,original)

    def test_no_quality_scores_and_swap_reference_preserved(self):
        rows=[{'cache_off':False,'wall_seconds':1.,'prefill_seconds':.5,'generation_seconds':.2,'cache_tokens':0,'input_tokens':4,'finish_reason':'stop'}]*3
        resources={'peak':{'rss_bytes':3,'physical_footprint_bytes':2},'swap_affected':True,'initial_OS':{'swap_used_bytes':2*audit.GIB},'OS_samples1s':[{}],'samples50ms':1,'time_interpretation':'reference only'}
        r=audit.summary('A',rows,resources,{}, {},{},3.)
        self.assertNotIn('f1',r);self.assertTrue(r['swap_affected']);self.assertEqual(r['time_interpretation'],'reference only')
        self.assertEqual(r['main_requests'],3);self.assertEqual(r['cache_off_requests'],0)
        self.assertEqual(r['peak'],resources['peak'])

    def test_complete_registration_includes_two_rerank_samples(self):
        self.assertEqual(sum(4 if arm=='A' else 3 for r,arm,c in audit.schedule())+sum(3 for r,arm,c in audit.schedule() if c==8192),32)

    def test_parse_distinct_allocations_no_repeated_double_count(self):
        value=audit.allocation_table('Metal model buffer size = 100.0 MiB\nCPU_Mapped model buffer size = 5 MiB\nMetal KV buffer size = 20 MiB\nMetal compute buffer size = 3 MiB\nMetal compute buffer size = 3 MiB\nCPU output buffer size = 1 MiB\n')
        self.assertTrue(value['required_recorded'])
        self.assertEqual(value['bytes']['model'],105*1024**2)
        self.assertEqual(value['complete_logged_buffer_sum_bytes'],129*1024**2)

    def test_missing_allocation_not_zero_or_complete(self):
        value=audit.allocation_table('Metal model buffer size = 100.0 MiB')
        self.assertFalse(value['required_recorded']);self.assertIsNone(value['bytes']['KV'])
        self.assertIsNone(value['complete_logged_buffer_sum_bytes'])

    def test_logging_environment_restored_on_failure(self):
        import os
        with patch.dict(os.environ,{'LLAMA_ARG_LOG_VERBOSITY':'3'}):
            with self.assertRaises(RuntimeError):
                with audit.isolated_restart_records():
                    self.assertEqual(os.environ['LLAMA_ARG_LOG_VERBOSITY'],'4')
                    raise RuntimeError('synthetic')
            self.assertEqual(os.environ['LLAMA_ARG_LOG_VERBOSITY'],'3')

    def test_preflight_minimum_context_keeps_same_questions(self):
        self.assertEqual(audit.schedule(),[(1,'A',131072),(1,'B1536',131072),(1,'C-final-norw',131072),(1,'C-final-norw',8192),(2,'C-final-norw',8192),(2,'C-final-norw',131072),(2,'B1536',131072),(2,'A',131072)])

    def test_vmmap_failure_not_zero_or_fake_success(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(audit,'LOCAL',Path(tmp)),patch.object(audit.subprocess,'run',side_effect=audit.subprocess.TimeoutExpired('vmmap',20)):
            value=audit.mapping_snapshot(1,'synthetic')
            self.assertFalse(value['recorded']);self.assertIn('timeout',value['reason'])


if __name__=='__main__':unittest.main()
