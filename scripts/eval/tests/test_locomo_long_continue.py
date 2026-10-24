"""Synthetic continuation guards: no GPU, network, signals or corpus."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import locomo_long_continue as cont


class Continuation(unittest.TestCase):
    def prepared(self):
        ids=[f'g{g}/q{i}' for g in range(10) for i in range(3)]
        return {'schedule':ids,'questions':{q:{'group':q.split('/')[0]} for q in ids}}

    def test_original_question_keys_and_cache_off_sample(self):
        p=self.prepared();jobs=list(cont.jobs(p,'A'))
        self.assertEqual(len(jobs),40)
        self.assertEqual(len({k for k,q,off in jobs}),40)
        self.assertEqual({q for k,q,off in jobs if not off},set(p['schedule']))
        for group in {q['group'] for q in p['questions'].values()}:
            qids=[q for q in p['schedule'] if p['questions'][q]['group']==group]
            expected=min(qids,key=lambda q:cont.base.am.hashed(cont.base.builder.QUESTION_SEED,q))
            self.assertEqual([q for k,q,off in jobs if off and q.startswith(group+'/')],[expected])

    def test_B_C_have_thirty_keys_without_auxiliary(self):
        for arm in ('B1536','C-final-norw'):
            jobs=list(cont.jobs(self.prepared(),arm))
            self.assertEqual(len(jobs),30);self.assertFalse(any(off for k,q,off in jobs))

    def test_received_reply_never_regenerated(self):
        with self.assertRaises(RuntimeError):
            cont.guarded_send('k','answer',{},'d',4,{}, {'k':{}},None,{},lambda *a:self.fail('must not send'))

    def test_unreceived_request_never_retried(self):
        with self.assertRaises(RuntimeError):
            cont.guarded_send('k','answer',{},'d',4,{}, {},None,{'k':{}},lambda *a:self.fail('must not send'))

    def test_server_identity_saved_before_send(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(cont,'LOCAL',Path(tmp)):
            servers={}
            def send(*args):
                self.assertEqual(cont.lm.read(Path(tmp)/'continuation-request-servers.json'),{'k':{'n_ctx':131072}})
                return 'reply'
            result=cont.guarded_send('k','answer',{},'d',4,{'n_ctx':131072},{},None,servers,send)
            self.assertEqual(result,'reply')

    def test_unknown_inflight_stops_without_deleting(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(cont,'LOCAL',Path(tmp)):
            path=Path(tmp)/'inflight.json';path.write_text('{}')
            with self.assertRaises(RuntimeError):cont.recover({})
            self.assertTrue(path.exists())

    def test_reuse_rejects_window_change_and_restores_functions(self):
        original=cont.base.restart
        with cont.reuse_server({}):
            with self.assertRaises(RuntimeError):cont.base.restart(65536,'bad')
        self.assertIs(cont.base.restart,original)

    def test_saved_original_request_and_changed_body(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(cont,'LOCAL',Path(tmp)),patch.object(cont.lm,'file_hash',return_value='plan'):
            path=Path(tmp)/'128000-A-inputs.json';cont.lm.write(path,{'128000|A|g/q0':{'messages':[]}})
            plan={'answer_plan':{'model':{'model_id':'synthetic'}}};server={'n_ctx':131072}
            body=cont.ae.request(plan['answer_plan'],{'messages':[]})
            reply={'attempts':1,'digest':cont.lm.digest({'request':body,'plan':'plan','server':server}),
                   'timings':{'cache_n':0,'prompt_n':4,'prompt_ms':1,'predicted_n':1,'predicted_ms':1}}
            cont.verify_reply('128000|A|g/q0',reply,plan,{'128000|A|g/q0':server})
            cont.lm.write(path,{'128000|A|g/q0':{'messages':[{'role':'user','content':'changed'}]}})
            with self.assertRaises(RuntimeError):cont.verify_reply('128000|A|g/q0',reply,plan,{'128000|A|g/q0':server})

    def test_memory_summary_marks_shared_process(self):
        with patch.object(cont.report,'condition_summary',return_value={}),patch.object(cont.lm,'file_hash',return_value='amendment'):
            with cont.reuse_server({}):
                result=cont.report.condition_summary()
                self.assertIn('deferred',result['memory_interpretation'])
                self.assertEqual(result['order_amendment_sha256'],'amendment')


if __name__=='__main__':unittest.main()
