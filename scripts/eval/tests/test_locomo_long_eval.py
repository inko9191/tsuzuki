"""Synthetic tests: no corpus, network, GPU, signals or real server control."""
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import locomo_long_eval as run
import locomo_long_report as report
import locomo_long_resources as resource


def row(qid,**changes):
    return {'id':qid,'type':('multiple_evidence','single_fact','temporal')[int(qid.rsplit('q',1)[1])%3],
        'category':1,'f1':.5,'em':0.,'abstention':0.,'failure':0.,'failure_kind':None,
        'inserted_all':1.,'inserted_recall':1.,'wall_seconds':2.,'prefill_seconds':1.,
        'generation_seconds':.5,'cache_tokens':0,'input_tokens':100,'assembly_seconds':.1,'attempts':1,**changes}


class Resources(unittest.TestCase):
    def test_parse_available_and_swap(self):
        vm='Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 2.\nPages inactive: 3.\nPages speculative: 4.\nPages purgeable: 100.\nSwapins: 5.\nSwapouts: 6.\n'
        s=resource.parse_os(vm,'total = 20.00G used = 13.80G free = 6.20G','1\n')
        self.assertEqual(s['available_bytes_estimate'],9*16384)
        self.assertAlmostEqual(s['swap_used_bytes'],13.8*1024**3)
        self.assertEqual(s['swapouts'],6)

    def test_unknown_OS_stops(self):
        with self.assertRaises(RuntimeError):resource.parse_os('unavailable','','1')

    def test_existing_swap_is_not_increment(self):
        s={'swapins':20,'swapouts':30,'swap_used_bytes':13*resource.GIB}
        self.assertFalse(resource.swap_increased(s,s))
        self.assertTrue(resource.swap_increased(s,{**s,'swapins':21}))

    def test_resource_boundaries(self):
        m={'physical_footprint_bytes':48*resource.GIB}
        s={'available_bytes_estimate':8*resource.GIB,'pressure_level':1}
        self.assertIsNone(resource.violation(m,s))
        self.assertIsNotNone(resource.violation({**m,'physical_footprint_bytes':48*resource.GIB+1},s))
        self.assertIsNotNone(resource.violation(m,{**s,'available_bytes_estimate':8*resource.GIB-1}))
        self.assertIsNotNone(resource.violation(m,{**s,'pressure_level':4}))

    def test_initial_resource_problem_no_start(self):
        with patch.object(resource,'observe_os',return_value={'available_bytes_estimate':7*resource.GIB,'pressure_level':1}),patch.object(resource.audit,'memory',return_value={'physical_footprint_bytes':1}):
            with self.assertRaises(RuntimeError):resource.Watch(123,lambda *x:None)


class Durability(unittest.TestCase):
    def test_received_saved_before_timing_error_and_no_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'LOCAL',Path(tmp)),patch.object(run.ae,'gate'),patch.object(run.st,'http_json',return_value={'choices':[]}) as http:
            replies={};watch=type('Watch',(),{'error':None})()
            with self.assertRaises(RuntimeError):run.send('k','answer',{},'digest',4,{'url':'unused'},replies,watch)
            self.assertIn('k',replies)
            self.assertTrue((Path(tmp)/'answer-received.jsonl').exists())
            self.assertFalse((Path(tmp)/'inflight.json').exists())
            with self.assertRaises(RuntimeError):run.send('k','answer',{},'digest',4,{'url':'unused'},replies,watch)
            self.assertEqual(http.call_count,1)

    def test_unknown_inflight_retained_no_retry(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'LOCAL',Path(tmp)),patch.object(run.ae,'gate'),patch.object(run.st,'http_json',side_effect=TimeoutError) as http:
            replies={};watch=type('Watch',(),{'error':None})()
            with self.assertRaises(TimeoutError):run.send('k','answer',{},'digest',4,{'url':'unused'},replies,watch)
            self.assertTrue((Path(tmp)/'inflight.json').exists())
            self.assertEqual(replies,{})
            self.assertEqual(http.call_count,1)

    def test_context_failure_is_zero_and_unattempted(self):
        r=run.main_row({'id':'g/q1','type':'single_fact','category':1,'gold':'correct'},run.failed_reply('answer_context_overflow'))
        self.assertEqual((r['f1'],r['failure'],r['attempts']),(0.,1.,0))

    def test_context_dependency_skips_C_payload(self):
        p={'questions':{'g/q1':{'group':'g'}},'memories':{'64000|g':{}},
            'rerank_jobs':{'64000|C-final|g/q1':{}}}
        with patch.object(run.am,'payload') as payload:
            self.assertIsNone(run.make_C(p,'64000','C-final','g/q1',{'64000|C-final|g/q1':run.failed_reply('rerank_context_overflow')},None))
            payload.assert_not_called()

    def test_unregistered_window_rejected_before_mutation(self):
        with patch.object(run.ae,'gate'),patch.object(run.norw.ab,'server',return_value={'n_ctx':65536}),patch.object(run,'quiet'),patch.object(run.os,'kill') as kill:
            with self.assertRaises(RuntimeError):run.restart(262144,'forbidden')
            kill.assert_not_called()


class Builder(unittest.TestCase):
    def test_distractors_kept_in_shared_retrieval_group(self):
        units=[{'group':'a','turns':['a1'],'text':'x'},{'group':'b','turns':['b1'],'text':'y'},
            {'group':'b','turns':['b1','b2'],'text':'z'}]
        result=run.builder.eligible_units(units,['a1','b1'],'synthetic')
        self.assertEqual([u['group'] for u in result],['synthetic','synthetic'])
        self.assertEqual([u['original_vector_row'] for u in result],[0,1])
        self.assertEqual([u['original_group'] for u in result],['a','b'])

    def test_sampling_is_dev_only_and_balanced(self):
        qs=[{'id':f'g/q{i}','group':'g'} for i in range(6)]
        types={q['id']:('multiple_evidence','single_fact','temporal')[i%3] for i,q in enumerate(qs)}
        self.assertEqual(run.builder.sample_questions(qs,types,[q['id'] for q in qs[:3]]),[q['id'] for q in qs[:3]])
        with self.assertRaises(ValueError):run.builder.sample_questions(qs,types,['unknown/q1'])


class Reporting(unittest.TestCase):
    def setUp(self):
        self.rows=[row(f'g{g}/q{i}') for g in range(10) for i in range(3)]
        self.prepared={'search_seconds':{f'64000|C-final-norw|{r["id"]}':.02 for r in self.rows}}
        self.reranks={f'64000|C-final-norw|{r["id"]}':{'wall_seconds':4.} for r in self.rows}
        self.resources={'peak':{'physical_footprint_bytes':10},'swap_affected':True,'initial_OS':{},
            'time_interpretation':'reference only','samples50ms':20}

    def summary(self,arm='C-final-norw',aux=()):
        return report.condition_summary({},self.prepared,'64000',arm,self.rows,list(aux),self.reranks,self.resources,99)

    def test_component_sum_and_swap_are_explicit(self):
        s=self.summary()
        self.assertAlmostEqual(s['components_seconds']['complete_sum']['mean'],6.12)
        self.assertTrue(s['swap_affected']);self.assertTrue(s['provisional'])
        self.assertEqual(s['metrics']['multiple_evidence']['questions'],10)

    def test_reused_unknown_HTTP_is_not_zero_time(self):
        self.reranks[next(iter(self.reranks))]['wall_seconds']=None
        s=self.summary()
        self.assertEqual(s['components_seconds']['complete_sum']['n'],29)
        self.assertEqual(s['components_seconds']['known_sum']['n'],30)

    def test_auxiliary_not_added_to_main_F1(self):
        s=self.summary('A',[row(f'g{g}/q1',f1=0.) for g in range(10)])
        self.assertEqual(s['metrics']['overall']['f1'],.5)
        self.assertEqual(s['main_questions'],30)
        self.assertEqual(s['auxiliary_questions'],10)
        self.assertTrue(s['A_cache_off_auxiliary']['excluded_from_main_F1'])

    def test_incomplete_condition_refused(self):
        self.rows.pop()
        with self.assertRaises(RuntimeError):self.summary()

    def test_interval_crossing_zero_does_not_imply_equivalent(self):
        left=[{**r,'f1':.5+(.2 if int(r['id'].split('/')[0][1:])<5 else -.2)} for r in self.rows]
        result=report.paired(left,self.rows)['overall']['f1']
        self.assertLess(result['conversation_ci95'][0],0.)
        self.assertGreater(result['conversation_ci95'][1],0.)
        self.assertFalse(result['equivalent_within_0.03'])

    def test_zero_ties_excluded_exact_sign(self):
        s=report.paired(self.rows,self.rows)['overall']['f1']
        self.assertEqual(s['sign_test'],{'wins':0,'losses':0,'ties':10,'n':0,'two_sided_p':1.})
        self.assertTrue(s['equivalent_within_0.03'])

    def test_missing_final_jobs_refused(self):
        p={'design':{'conditions':{'64000':['A']}}}
        prepared={'schedule':[r['id'] for r in self.rows],
            'questions':{r['id']:{'group':r['id'].split('/')[0]} for r in self.rows},'rerank_jobs':{}}
        with self.assertRaises(RuntimeError):report.verify_complete(p,prepared,{}, {},{})

    def test_A_block_generates_exactly_thirty_main_ten_cache_off(self):
        class Counter:
            def __init__(self,*args):pass
            def save(self):pass
        class Watch:
            error=None
            def __init__(self,*args):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def record(self):return {'peak':{'physical_footprint_bytes':10},'swap_affected':False,
                'initial_OS':{},'time_interpretation':'synthetic','samples50ms':0}
        prepared={'questions':{r['id']:{'group':r['id'].split('/')[0]} for r in self.rows},
            'schedule':[r['id'] for r in self.rows],'search_seconds':{},
            'answer_inputs':{f'64000|A|{r["id"]}':{'id':r['id'],'type':r['type'],'category':1,
                'gold':'synthetic','messages':[],'input_tokens':4,'fits_context':True} for r in self.rows}}
        response={'choices':[{'finish_reason':'stop','message':{'content':'{"answer":"synthetic","evidence_ids":[]}'}}],
            'timings':{'cache_n':0,'prompt_n':4,'prompt_ms':1,'predicted_n':1,'predicted_ms':1}}
        with tempfile.TemporaryDirectory() as tmp:
            local=Path(tmp)/'private';local.mkdir();public=Path(tmp)/'public';public.mkdir()
            plan=Path(tmp)/'plan.json';plan.write_text('{}')
            with patch.object(run,'LOCAL',local),patch.object(run,'PLAN',plan),patch.object(run.lm,'RESULTS',public),\
                patch.object(run,'restart',return_value=(1,{'url':'unused'})),patch.object(run,'Tokenizer',Counter),\
                patch.object(run.resources,'Watch',Watch),patch.object(run.ae,'gate'),\
                patch.object(run.st,'http_json',return_value=response) as http,patch.object(run,'event'),redirect_stdout(io.StringIO()):
                replies={};run.run_condition({'answer_plan':{'model':{'model_id':'synthetic'}}},prepared,'64000','A',{},replies)
                self.assertEqual(http.call_count,40)
                self.assertEqual(sum(c.args[1]['cache_prompt'] is False for c in http.call_args_list),10)
                result=json.loads((local/'64000-A-completed.json').read_text())
                self.assertEqual(len(result['rows']),30);self.assertEqual(len(result['auxiliary']),10)
                self.assertEqual(result['summary']['metrics']['overall']['f1'],1.)


if __name__=='__main__':unittest.main()
