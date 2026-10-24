"""Synthetic CPU tests for registered continuation; no dataset or server calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import longmemeval_g1_continue as c
import longmemeval_g1_window as w


class ContinuationTests(unittest.TestCase):
    def test_measured_max_selects16k_including_reservations(self):
        self.assertEqual(w.select_window(13422),(15038,16384))

    def test10percent_rule_can_require32k(self):
        self.assertEqual(w.select_window(15000),(16773,32768))

    def test_no_allowed_context_stops_instead_of_truncating(self):
        self.assertIsNone(w.select_window(60000)[1])

    def test64k_is_last_allowed_choice(self):
        self.assertEqual(w.select_window(30000)[1],65536)

    def test_CPU_transport_rejects_completion_before_HTTP(self):
        count=w.CPUCount.__new__(w.CPUCount)
        with patch.object(w.st,'http_json') as http:
            with self.assertRaises(RuntimeError):count.request('/v1/chat/completions',{})
            http.assert_not_called()

    def source(self,path,count=1441,replace=None):
        rows=[{'key':'fact|s'+str(i),'reply':{'attempts':1}} for i in range(count)]
        if replace:rows[-1]=replace
        (path/'received.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_saved_ingestion_reused_without_resending(self):
        with tempfile.TemporaryDirectory() as t,patch.object(c,'SOURCE',Path(t)),patch.object(c.st,'http_json') as http:
            self.source(Path(t)); result=c.saved_ingestion()
            self.assertEqual(len(result),1441);http.assert_not_called()

    def test_incomplete_source_stops(self):
        with tempfile.TemporaryDirectory() as t,patch.object(c,'SOURCE',Path(t)):
            self.source(Path(t),1440)
            with self.assertRaises(RuntimeError):c.saved_ingestion()

    def test_existing_evaluation_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as t,patch.object(c,'SOURCE',Path(t)):
            self.source(Path(t),replace={'key':'answer|A|q','reply':{'attempts':1}})
            with self.assertRaises(RuntimeError):c.saved_ingestion()

    def test_duplicate_fact_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as t,patch.object(c,'SOURCE',Path(t)):
            self.source(Path(t),replace={'key':'fact|s0','reply':{'attempts':1}})
            with self.assertRaises(RuntimeError):c.saved_ingestion()

    def test_unknown_source_inflight_is_not_recovered(self):
        with tempfile.TemporaryDirectory() as t,patch.object(c,'SOURCE',Path(t)):
            self.source(Path(t));(Path(t)/'inflight.json').write_text('{}')
            with self.assertRaises(RuntimeError):c.saved_ingestion()

    def test_multiple_attempt_source_rejected(self):
        with tempfile.TemporaryDirectory() as t,patch.object(c,'SOURCE',Path(t)):
            self.source(Path(t),replace={'key':'fact|s1440','reply':{'attempts':2}})
            with self.assertRaises(RuntimeError):c.saved_ingestion()

    def test_runtime_paths_restore_on_error(self):
        old=c.original.LOCAL,c.original.PLAN,c.ae.LOCAL,c.lm.LOCAL
        with self.assertRaises(ValueError):
            with c.runtime():
                self.assertEqual(c.original.LOCAL,c.LOCAL)
                self.assertEqual(c.original.PLAN,c.PLAN)
                self.assertEqual(c.ae.LOCAL,c.LOCAL)
                raise ValueError('synthetic')
        self.assertEqual((c.original.LOCAL,c.original.PLAN,c.ae.LOCAL,c.lm.LOCAL),old)

    def test_unregistered_window_cannot_start_process(self):
        manager=c.Servers({'selected_rerank_context':16384});manager.up=False
        with patch.object(c.ae,'gate'),patch.object(c.original,'other_workers'), \
             patch.object(c.subprocess,'check_output',return_value=''),patch.object(c.subprocess,'Popen') as popen:
            with self.assertRaises(RuntimeError):manager.start(32768,'invalid')
            popen.assert_not_called()

    def test_actual_C_overflow_stops_before_answer_generation(self):
        class Count:
            count=staticmethod(lambda text:1)
            input_count=staticmethod(lambda chat:9000)
            save=staticmethod(lambda:None)
        class Watch:
            error=None
            def __init__(self,*args):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def begin(self,*args):pass
            def record(self):return {}
        prepared={'questions':{'q':{'question':'synthetic'}},'turns':{'q':[]},'inputs':{}}
        job={'body':{},'body_sha256':c.lm.digest({}),'input_tokens':1,'order':[],'units':[],'search_seconds':0}
        reply={'response':{'choices':[{'finish_reason':'stop','message':{'content':'{"ranking":[]}'}}]},'wall_seconds':1}
        with tempfile.TemporaryDirectory() as t,patch.object(c,'LOCAL',Path(t)), \
             patch.object(c.ae,'gate'),patch.object(c.ae,'Tokenizer',return_value=Count()), \
             patch.object(c.fp,'SeriesWatch',Watch),patch.object(c.original,'event'), \
             patch.object(c.original,'send',return_value=reply) as send:
            with self.assertRaisesRegex(RuntimeError,'actual final C input overflow'):
                c.rerank_and_preflight(prepared,{'q':job},1,{'url':'synthetic'}, {},{'question_ids':['q'],'selected_rerank_context':16384})
            self.assertEqual(send.call_args.args[0],'rerank|q')
            self.assertEqual(send.call_count,1)
            self.assertNotIn('C-final-norw',prepared['inputs'])


if __name__=='__main__':unittest.main()
