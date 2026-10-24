#!/usr/bin/env python3
"""Registered synthetic long-context evaluation; single sequential Qwen, durable events."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.error

import numpy as np

import locomo_answer_norw as norw
import locomo_long_memory as builder
import locomo_long_resources as resources
import locomo_answer_report as ar

lm, ae, am, st, mm, rp = norw.lm, norw.ae, norw.am, norw.st, norw.mm, norw.rp
LOCAL = lm.LOCAL / 'long-context'
DESIGN = lm.RESULTS / 'long-context-design-plan.json'
PLAN = lm.RESULTS / 'long-context-execution-plan.json'
CODE = ('locomo_long_eval.py', 'locomo_long_resources.py', 'locomo_long_report.py')
EVENT_LOCK = threading.Lock()


def status(phase, **extra):
    lm.write(LOCAL/'status.json', {'time':lm.stamp(),'pid':os.getpid(),'pgid':os.getpgid(0),'phase':phase,**extra})


def event(kind, payload):
    with EVENT_LOCK:
        ae.received(LOCAL/'events.jsonl', kind, {'time':lm.stamp(),'kind':kind,**payload})
    print(f'event: {kind}', flush=True)


def design():
    ae.gate()
    d = st.frozen(DESIGN)
    for name,digest in d['source_sha256'].items():
        if lm.file_hash(lm.ROOT/name) != digest:
            raise RuntimeError(f'design source changed: {name}')
    return d


def quiet():
    env = norw.ab.quiet()
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        parts=line.strip().split(None,1)
        if len(parts)==2 and int(parts[0])!=os.getpid() and Path(parts[1].split()[0]).name.startswith('python'):
            if any(tag in parts[1] for tag in ('locomo_long_eval.py run','locomo_answer_norw.py run')):
                raise RuntimeError('another evaluation is active')
    return env


class Tokenizer(ae.Tokenizer):
    def __init__(self,url):
        super().__init__(url)
        self.path=LOCAL/'token-counts.json'
        if self.path.exists():
            self.counts=lm.read(self.path)


def cfg(arm,d):
    return d['configs'][arm]


def order_ids(queries):
    return norw.schedule(queries)


def prepare():
    d=design()
    quiet()
    now=norw.ab.server(65536)
    if (LOCAL/'prepared.json').exists():
        raise RuntimeError('preparation already exists; no overwrite')
    status('preparation',status='preparing')
    tick=time.perf_counter()
    memory=lm.read(LOCAL/'memories.json')
    if memory['code_sha256']!=lm.file_hash(Path(builder.__file__)):
        raise RuntimeError('memory builder code changed')
    data,manifest=lm.load_dev()
    if memory['dev_sha256']!=lm.file_hash(lm.LOCAL/'dev.json') or memory['corpus_sha256']!=lm.file_hash(lm.LOCAL/'corpus.json'):
        raise RuntimeError('memory source changed')
    raw=st.load_raw(manifest)
    turns=mm.turn_table(data['corpus'],raw)
    kinds={q['id']:lm.query_type(q,data['qrels'][q['id']]) for q in data['queries']}
    pilot=ae.checked_plan()
    ids=builder.sample_questions(data['queries'],kinds,pilot['plan']['question_ids'])
    if ids!=memory['development_question_ids'] or len(ids)!=30:
        raise RuntimeError('registered development sample differs')
    qlookup={q['id']:q for q in data['queries']}
    questions={qid:{**qlookup[qid],'type':kinds[qid], 'evidence':list(data['qrels'][qid])} for qid in ids}
    gold={f'{s["sample_id"]}/q{i}':am.reference(qa.get('answer')) for s in raw
        for i,qa in enumerate(s['qa']) if f'{s["sample_id"]}/q{i}' in ids}
    for qid,q in questions.items():q['gold']=gold[qid]
    qwen=st.Qwen.offline()
    generations={kind:st.generated(kind,data,turns,qwen)[0] for kind in ('rw','fact')}
    units=st.build_units(['p2','rw','fact'],turns,generations)
    from egemma_server.tokenizer import PromptTokenizer
    tk=PromptTokenizer(rp.V1_ST_DIR)
    identity=lm.fingerprint(st.VARIANT)
    if identity!=lm.read(lm.RESULTS/'final-config.json')['embedding_artifact_sha256']:
        raise RuntimeError('embedding model artifacts differ')
    vectors=np.zeros((len(units),768),np.float32)
    for name in ('p2','rw','fact'):
        ix=[i for i,u in enumerate(units) if u['set']==name]
        texts=rp.fit_like_st(tk,[lm.PROTOCOL['prompts']['document']+units[i]['text'] for i in ix])
        vectors[ix],_=norw.ab.cached(texts,f'u-{name}',identity)
    texts=rp.fit_like_st(tk,[lm.PROTOCOL['prompts']['query']+q['text'] for q in data['queries']])
    qv,_=norw.ab.cached(texts,'q-dev',identity)
    qvectors=dict(zip([q['id'] for q in data['queries']],qv))
    count=Tokenizer(now['url'])
    prepared={'memories':{},'questions':questions,'schedule':order_ids(list(questions.values())),
        'rerank_jobs':{},'orders':{},'search_seconds':{},'answer_inputs':{},'design_sha256':lm.file_hash(DESIGN)}
    try:
        for mem in memory['memories']:
            n=str(mem['requested_tokens']);group=mem['target_group'];mkey=f'{n}|{group}'
            if mem['rendered_memory_sha256']!=lm.digest(builder.render(mem['turns'])):
                # builder hashes plain UTF8, whereas lm.digest hashes JSON; compare explicitly below.
                import hashlib
                if mem['rendered_memory_sha256']!=hashlib.sha256(builder.render(mem['turns']).encode()).hexdigest():
                    raise RuntimeError('constructed memory text changed')
            selected=set(mem['selected_turn_ids'])
            if len(selected)!=len(mem['turns']) or not mem['target_complete']:
                raise RuntimeError('memory identity/target incomplete')
            eligible=builder.eligible_units(units,selected,mkey)
            ev=vectors[[u['original_vector_row'] for u in eligible]]
            index=mm.Index(eligible,ev)
            prepared['memories'][mkey]={**mem,'units':eligible}
            groupqueries=[qid for qid in prepared['schedule'] if questions[qid]['group']==group]
            for qid in groupqueries:
                ae.gate();q=questions[qid]
                for arm in d['conditions'][n]:
                    key=f'{n}|{arm}|{qid}'
                    if arm.startswith('C'):
                        start=time.perf_counter()
                        order=mm.distinct([i for i,_ in index.rank(cfg(arm,d),mkey,[(qvectors[qid],q['text'])])],eligible)
                        prepared['search_seconds'][key]=time.perf_counter()-start
                        prepared['orders'][key]=order
                        chat=norw.prod.rerank_chat(q,order,eligible,mem['turns'],cfg(arm,d))
                        inp=count.input_count(chat)
                        # Oversize is a registered failure, not an input truncation or condition substitution.
                        fits=inp+mm.PROMPTS['rerank']['max_tokens']<=8192
                        prepared['rerank_jobs'][key]={'body':qwen.request('rerank',chat),'digest':qwen.item_digest('rerank',chat),
                            'input_tokens':inp,'fits_context':fits}
                    else:
                        start=time.perf_counter()
                        payload=am.payload(arm,mem['turns'],{},count.count)
                        messages=am.messages(payload.pop('conversation'),payload.pop('retrieved'),q['text'])
                        inp=count.input_count(messages)
                        ctx=65536 if n=='64000' else 131072
                        inserted=set(payload['recent_ids']+payload['memory_ids']);evidence=set(q['evidence'])
                        prepared['answer_inputs'][key]={'id':qid,'condition':arm,'category':q['category'],'type':q['type'],
                            'gold':q['gold'],'messages':messages,'input_tokens':inp,'fits_context':inp+256+128<=ctx,
                            'inserted_all':float(evidence<=inserted),'inserted_recall':len(evidence&inserted)/len(evidence),
                            'assembly_seconds':time.perf_counter()-start,**payload}
    finally:count.save()
    # Short controls are verified without opening production questions or generating short answers.
    pilot_inputs=lm.read(ae.LOCAL/'pilot-inputs.json')
    pilot_replies=ae.recover(ae.LOCAL/'pilot-responses.json',ae.LOCAL/'pilot-received.jsonl')
    ar.complete(pilot_inputs,pilot_replies,pilot)
    short=ar.rows_for(pilot_inputs,pilot_replies,pilot['plan']['aliases'])
    short={a:[r for r in rows if r['id'] in ids] for a,rows in short.items() if a in ('A','B1536','C-final')}
    if any(len(rows)!=30 for rows in short.values()):raise RuntimeError('short controls incomplete')
    lm.write(LOCAL/'short-controls.json',short)
    lm.write(LOCAL/'prepared.json',prepared)
    summary={'time':lm.stamp(),'prepare_seconds':time.perf_counter()-tick,'questions':30,
        'type_counts':dict(Counter(q['type'] for q in questions.values())),'memories':len(memory['memories']),
        'memory_tokens':[{'length':m['requested_tokens'],'group':m['target_group'],'tokens':m['memory_tokens']} for m in memory['memories']],
        'rerank_jobs':len(prepared['rerank_jobs']),'oversize_rerank':sum(not j['fits_context'] for j in prepared['rerank_jobs'].values()),
        'A_B_preflight':{f'{n}|{a}':{'questions':len(part),'input_max':max(j['input_tokens'] for j in part),
            'oversize':sum(not j['fits_context'] for j in part)} for n in ('64000','128000') for a in ('A','B1536')
            if (part:=[j for k,j in prepared['answer_inputs'].items() if k.startswith(f'{n}|{a}|')])},
        'C_preflight':'actual prompts frozen and exact-token checked immediately after registered rerank, before each answer condition',
        'prepared_sha256':lm.file_hash(LOCAL/'prepared.json'),'short_controls_verified':True}
    lm.write(lm.RESULTS/'long-context-preparation.json',summary)
    status('prepared',status='prepared',**summary)


def register():
    d=design();ae.gate()
    if PLAN.exists():raise RuntimeError('execution already registered; never overwrite')
    source=lm.read(LOCAL/'prepared.json')
    if source['design_sha256']!=lm.file_hash(DESIGN):raise RuntimeError('prepared design differs')
    source['shared_reranks']=shared_reranks(source['rerank_jobs'])
    lm.write(LOCAL/'prepared.json',source)
    summary=lm.read(lm.RESULTS/'long-context-preparation.json')
    summary['prepared_sha256']=lm.file_hash(LOCAL/'prepared.json')
    summary['same_digest_reuse']=len(source['shared_reranks'])
    lm.write(lm.RESULTS/'long-context-preparation.json',summary)
    p={'version':1,'design':d,'design_sha256':lm.file_hash(DESIGN),'authorization':'human: 測定を開始して,2026-10-13',
        'created_at':lm.stamp(),'question_ids_sha256':lm.digest(sorted(source['questions'])),
        'question_count':30,'code_sha256':{name:lm.file_hash(lm.ROOT/'scripts/eval'/name) for name in CODE},
        'test_sha256':lm.file_hash(lm.ROOT/'scripts/eval/tests/test_locomo_long_eval.py'),
        'private_inputs_sha256':{name:lm.file_hash(LOCAL/name) for name in ('memories.json','prepared.json','short-controls.json')},
        'original_files_sha256':norw.original_hashes(),'answer_plan':ae.checked_plan()['plan'],
        'resources_available_definition':'vm_stat free+inactive+speculative pages, pressure sysctl>=4 critical',
        'conditional_C_inputs':'rerank-derived original turns under fixed1024/512 budgets; exact preflight before condition; no A/B or rule changes',
        'reporting':'durable completed-condition snapshots and anomaly events; no generation waits for reporting',
        'no_auto_resume':True,'no_256k':True,'no_holdout_questions_opened':True,'outputs':'ignored build/eval/memory/long-context; aggregates/hash public only'}
    lm.write(PLAN,p);print('registered; commit execution code/tests/preparation/plan before run',flush=True)


def shared_reranks(jobs):
    """Only identical saved development requests; no holdout responses opened."""
    by_digest={}
    for entry in lm.read(lm.LOCAL/'gen/rerank.json').values():
        content=entry['reply']
        if entry['digest'] in by_digest and by_digest[entry['digest']]['response']['choices'][0]['message']['content']!=content:
            raise RuntimeError('conflicting saved rerank digest')
        by_digest[entry['digest']]={'digest':entry['digest'],'response':{'choices':[{'message':{'content':content}}]},
            'wall_seconds':None,'attempts':0,'source':'saved identical development digest; HTTP/finish timing unrecorded'}
    path=norw.ab.LOCAL/'responses.json'
    if path.exists():
        for entry in lm.read(path).values():
            by_digest[entry['digest']]={**entry,'attempts':0,'source':'saved identical development digest; recovered HTTP time'}
    return {key:by_digest[job['digest']] for key,job in jobs.items() if job['digest'] in by_digest}


def checked():
    ae.gate();p=st.frozen(PLAN);design()
    for name,digest in p['code_sha256'].items():
        if lm.file_hash(lm.ROOT/'scripts/eval'/name)!=digest:raise RuntimeError(f'code changed:{name}')
    if lm.file_hash(lm.ROOT/'scripts/eval/tests/test_locomo_long_eval.py')!=p['test_sha256']:raise RuntimeError('tests changed')
    for name,digest in p['private_inputs_sha256'].items():
        if lm.file_hash(LOCAL/name)!=digest:raise RuntimeError(f'private input changed:{name}')
    norw.verify_originals(p)
    return p


def restart(context,label):
    ae.gate();now=norw.ab.server();env=quiet()
    if context not in (8192,65536,131072) or now['n_ctx'] not in (8192,65536,131072):
        raise RuntimeError('unregistered context')
    system=resources.observe_os();issue=resources.violation({'physical_footprint_bytes':0},system)
    if issue:raise RuntimeError(issue)
    argv=lm.read(lm.RESULTS/'answer-pilot-server.json')['new_args'].copy()
    expected=argv.copy();expected[expected.index('-c')+1]=str(now['n_ctx'])
    instance=env['qwen_servers'][0]
    if instance['command']!=' '.join(expected):raise RuntimeError('server argv differs')
    if lm.file_hash(Path(argv[argv.index('-m')+1]))!=now['model_sha256']:raise RuntimeError('fresh model hash differs')
    tick=time.perf_counter();os.kill(instance['pid'],signal.SIGTERM)
    for _ in range(100):
        try:os.waitpid(instance['pid'],os.WNOHANG)
        except ChildProcessError:pass
        try:os.kill(instance['pid'],0)
        except ProcessLookupError:break
        time.sleep(.1)
    else:raise RuntimeError('planned stop failed; no duplicate start')
    argv[argv.index('-c')+1]=str(context)
    with (LOCAL/f'qwen-{label}.log').open('a') as stream:
        proc=subprocess.Popen(argv,cwd=lm.ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(60):
        if proc.poll() is not None:raise RuntimeError('planned restart failed; no retry')
        try:new=norw.ab.server(context);break
        except (urllib.error.URLError,ConnectionError):time.sleep(1)
    else:raise RuntimeError('planned restart timeout; no retry')
    records=lm.read(LOCAL/'context-switches.json') if (LOCAL/'context-switches.json').exists() else []
    records.append({'time':lm.stamp(),'label':label,'pid':proc.pid,'old':now,'new':new,'args':argv,
        'seconds':time.perf_counter()-tick,'initial_OS':system,'fresh_hash_verified':True})
    lm.write(LOCAL/'context-switches.json',records)
    ae.gate();return proc.pid,new


def failed_reply(kind):
    return {'digest':None,'response':{},'wall_seconds':0.,'timings':{},'attempts':0,'failure_kind':kind}


def send(key,kind,body,digest,inp,now,replies,watch):
    ae.gate()
    if key in replies:raise RuntimeError('successful or failed registered job already exists; no overwrite')
    if watch.error:raise RuntimeError(watch.error)
    lm.write(LOCAL/'inflight.json',{'key':key,'kind':kind,'digest':digest,'time':lm.stamp()})
    tick=time.perf_counter()
    response=st.http_json(now['url']+'/v1/chat/completions',body,timeout=900)
    reply={'digest':digest,'response':response,'wall_seconds':time.perf_counter()-tick,'timings':response.get('timings') or {},
        'input_tokens':inp,'attempts':1,'finished_at':lm.stamp()}
    replies[key]=reply;ae.received(LOCAL/f'{kind}-received.jsonl',key,reply)
    (LOCAL/'inflight.json').unlink()
    ae.checked_timings(reply)
    return reply


def make_C(prepared,n,arm,qid,reranks,count):
    key=f'{n}|{arm}|{qid}';q=prepared['questions'][qid];mem=prepared['memories'][f'{n}|{q["group"]}']
    reply=reranks[key];job=prepared['rerank_jobs'][key]
    if reply.get('failure_kind'):return None
    if reply['digest']!=job['digest']:raise RuntimeError('rerank digest differs')
    order=prepared['orders'][key]
    content=reply['response']['choices'][0]['message'].get('content') or ''
    value=mm.parse('rerank',content,n_passages=len(order[:mm.RERANK_CANDIDATES]))
    order=mm.distinct(norw.prod.reranked_order(order,value),mem['units'])
    ids=list(dict.fromkeys(t for i in order for t in mem['units'][i]['turns']))
    tick=time.perf_counter()
    payload=am.payload(arm,mem['turns'],{arm:ids},count.count)
    messages=am.messages(payload.pop('conversation'),payload.pop('retrieved'),q['text'])
    inp=count.input_count(messages);ctx=65536 if n=='64000' else 131072
    inserted=set(payload['recent_ids']+payload['memory_ids']);evidence=set(q['evidence'])
    return {'id':qid,'condition':arm,'category':q['category'],'type':q['type'],'gold':q['gold'],
        'messages':messages,'input_tokens':inp,'fits_context':inp+256+128<=ctx,
        'inserted_all':float(evidence<=inserted),'inserted_recall':len(evidence&inserted)/len(evidence),
        'assembly_seconds':time.perf_counter()-tick,'rerank_parse_fallback':value is None,**payload}


def main_row(item,reply):
    score=am.score(reply['response'],item['gold']) if not reply.get('failure_kind') else {
        'f1':0.,'em':0.,'abstention':0.,'failure':1.,'empty_answer':0.,'failure_kind':reply['failure_kind']}
    t=reply.get('timings') or {}
    return {'id':item['id'],'type':item['type'],'category':item['category'],**score,
        'wall_seconds':reply['wall_seconds'],'prefill_seconds':t.get('prompt_ms',0)/1000,
        'generation_seconds':t.get('predicted_ms',0)/1000,'cache_tokens':t.get('cache_n',0),
        'input_tokens':item.get('input_tokens'), 'assembly_seconds':item.get('assembly_seconds',0),
        'inserted_all':item.get('inserted_all',0),'inserted_recall':item.get('inserted_recall',0),
        'attempts':reply['attempts']}


def run_rerank(p,prepared,n,replies):
    arms=[a for a in p['design']['conditions'][n] if a.startswith('C')]
    pid,now=restart(8192,f'{n}-rerank')
    watch=resources.Watch(pid,lambda kind,data:event(kind,{'length':int(n),'condition':'rerank',**data}))
    started=time.perf_counter()
    try:
        with watch:
            for arm in arms:
                for qid in prepared['schedule']:
                    key=f'{n}|{arm}|{qid}';job=prepared['rerank_jobs'][key]
                    status('rerank',status='running',length=int(n),condition=arm,completed=len(replies))
                    if key in prepared['shared_reranks']:
                        reply=prepared['shared_reranks'][key];replies[key]=reply
                        ae.received(LOCAL/'rerank-received.jsonl',key,reply)
                    elif not job['fits_context']:
                        reply=failed_reply('rerank_context_overflow');replies[key]=reply
                        ae.received(LOCAL/'rerank-received.jsonl',key,reply)
                        event('input overflow',{'length':int(n),'condition':arm,'stage':'rerank'})
                    else:
                        reply=send(key,'rerank',job['body'],job['digest'],job['input_tokens'],now,replies,watch)
                        finish=reply['response']['choices'][0].get('finish_reason')
                        if finish!='stop':event('nonstop finish',{'length':int(n),'condition':arm,'stage':'rerank','finish_reason':finish})
                    if len(replies)%25==0:lm.write(LOCAL/'rerank-responses.json',replies)
    finally:
        lm.write(LOCAL/'rerank-responses.json',replies)
        lm.write(LOCAL/f'{n}-rerank-resource.json',watch.record())
        lm.write(LOCAL/f'{n}-rerank-runtime.json',{'seconds':time.perf_counter()-started,'completed':len(replies)})
    if watch.error:raise RuntimeError(watch.error)


def run_condition(p,prepared,n,arm,reranks,replies):
    ctx=65536 if n=='64000' else 131072
    pid,now=restart(ctx,f'{n}-{arm}')
    watch=resources.Watch(pid,lambda kind,data:event(kind,{'length':int(n),'condition':arm,**data}))
    started=time.perf_counter();rows=[];aux=[]
    # Include input preflight in OS observations, while timing HTTP separately.
    watch.__enter__()
    count=Tokenizer(now['url']);items={}
    try:
        for qid in prepared['schedule']:
            ae.gate();key=f'{n}|{arm}|{qid}'
            if arm.startswith('C'):
                item=make_C(prepared,n,arm,qid,reranks,count)
                if item is None:
                    q=prepared['questions'][qid];item={'id':qid,'condition':arm,'category':q['category'],'type':q['type'],
                        'gold':q['gold'],'fits_context':False,'dependency_failure':'rerank_context_overflow'}
            else:item=prepared['answer_inputs'][key]
            items[key]=item
    except BaseException:
        watch.__exit__();lm.write(LOCAL/f'{n}-{arm}-resources.json',watch.record())
        raise
    finally:count.save()
    lm.write(LOCAL/f'{n}-{arm}-inputs.json',items)
    groups=sorted({q['group'] for q in prepared['questions'].values()},key=lambda g:am.hashed('locomo-answer-v2',g))
    try:
        for group in groups:
                qids=[q for q in prepared['schedule'] if prepared['questions'][q]['group']==group]
                requests=[(q,False) for q in qids]
                if arm=='A':requests.append((min(qids,key=lambda q:am.hashed(builder.QUESTION_SEED,q)),True))
                for qid,off in requests:
                    ae.gate();key=f'{n}|{arm}|{qid}'+('|cache-off' if off else '')
                    item=items[f'{n}|{arm}|{qid}']
                    status('answer',status='running',length=int(n),condition=arm,completed=len(rows),auxiliary=len(aux))
                    if not item['fits_context']:
                        reply=failed_reply(item.get('dependency_failure','answer_context_overflow'));replies[key]=reply
                        ae.received(LOCAL/'answer-received.jsonl',key,reply)
                        event('input overflow',{'length':int(n),'condition':arm,'stage':'answer','auxiliary':off})
                    else:
                        body=ae.request(p['answer_plan'],item)
                        if off:body['cache_prompt']=False
                        digest=lm.digest({'request':body,'plan':lm.file_hash(PLAN),'server':now})
                        reply=send(key,'answer',body,digest,item['input_tokens'],now,replies,watch)
                        finish=reply['response']['choices'][0].get('finish_reason')
                        if finish!='stop':event('nonstop finish',{'length':int(n),'condition':arm,'stage':'answer','finish_reason':finish,'auxiliary':off})
                        if off and reply['timings']['cache_n']!=0:raise RuntimeError('cache-off response reports reused tokens')
                    row=main_row(item,reply);(aux if off else rows).append(row)
                    if row['failure_kind']=='parse':event('parse failure',{'length':int(n),'condition':arm,'stage':'answer','auxiliary':off})
                    if len(replies)%25==0:lm.write(LOCAL/'answer-responses.json',replies)
    finally:
        watch.__exit__()
        lm.write(LOCAL/'answer-responses.json',replies)
        lm.write(LOCAL/f'{n}-{arm}-partial-rows.json',{'main':rows,'auxiliary':aux})
        lm.write(LOCAL/f'{n}-{arm}-resources.json',watch.record())
    if watch.error:raise RuntimeError(watch.error)
    if len(rows)!=30 or (arm=='A' and len(aux)!=10):raise RuntimeError('condition incomplete')
    import locomo_long_report as reporting
    result=reporting.condition_summary(p,prepared,n,arm,rows,aux,reranks,watch.record(),time.perf_counter()-started)
    lm.write(LOCAL/f'{n}-{arm}-completed.json',{'rows':rows,'auxiliary':aux,'summary':result})
    lm.write(lm.RESULTS/f'long-context-{n}-{arm}-interim.json',result)
    event('condition complete',{'length':int(n),'condition':arm,'summary':result})
    print(f'completed {n} {arm}:30 main, {len(aux)} auxiliary',flush=True)


def run():
    p=checked();quiet()
    if (LOCAL/'run-start.json').exists() or (LOCAL/'inflight.json').exists():
        raise RuntimeError('run already started; no automatic duplicate/resume')
    lm.write(LOCAL/'run-start.json',{'time':lm.stamp(),'pid':os.getpid(),'plan_sha256':lm.file_hash(PLAN)})
    prepared=lm.read(LOCAL/'prepared.json');reranks={};answers={};start=time.perf_counter()
    for n in ('64000','128000'):
        ae.gate();norw.verify_originals(p)
        run_rerank(p,prepared,n,reranks)
        for arm in p['design']['conditions'][n]:
            run_condition(p,prepared,n,arm,reranks,answers)
    restart(65536,'restore')
    norw.verify_originals(p)
    lm.write(LOCAL/'runtime.json',{'seconds':time.perf_counter()-start,'time':lm.stamp()})
    from locomo_long_report import final
    final(p,prepared)
    status('evaluation complete',status='complete')
    event('evaluation complete',{'seconds':time.perf_counter()-start})


def pause():
    lm.write(ae.PAUSE,{'time':lm.stamp(),'reason':'human requested pause','runner':'long-context'})
    s=lm.read(LOCAL/'status.json') if (LOCAL/'status.json').exists() else {}
    if s.get('pid'):
        cmd=subprocess.run(['ps','-p',str(s['pid']),'-o','command='],capture_output=True,text=True).stdout.strip()
        if 'locomo_long_eval.py run' in cmd and os.getpgid(s['pid'])==s['pgid'] and s['pgid']!=os.getpgid(0):
            os.killpg(s['pgid'],signal.SIGINT)
    print('persistent human pause; Qwen untouched',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','register','run','pause'))
    command=parser.parse_args().command
    try:{'prepare':prepare,'register':register,'run':run,'pause':pause}[command]()
    except BaseException as error:
        if command in ('prepare','run'):
            owner=lm.read(LOCAL/'status.json') if (LOCAL/'status.json').exists() else {}
            if owner.get('pid') in (None,os.getpid()) or command=='prepare':
                status('stopped',status='stopped',error=f'{type(error).__name__}: {error}')
                event('stopped',{'error':f'{type(error).__name__}: {error}'})
        raise


if __name__=='__main__':main()
