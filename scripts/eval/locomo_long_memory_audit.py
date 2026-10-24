#!/usr/bin/env python3
"""Preregistered repeated memory audit; accuracy responses remain immutable."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import os
from pathlib import Path
import re
import subprocess
import threading
import time

import locomo_long_eval as base
import locomo_long_report as reporting
import locomo_long_continue as continuation

lm, ae = base.lm, base.ae
SOURCE = base.LOCAL
LOCAL = lm.LOCAL/'long-context-memory-audit'
PLAN = lm.RESULTS/'long-context-memory-audit-plan.json'
ARMS = ('A','B1536','C-final-norw')
GIB = 1024**3
EVENT_LOCK = threading.Lock()


def selected(prepared, A_items):
    item=max(A_items.values(),key=lambda x:(x['input_tokens'],x['id']))
    group=prepared['questions'][item['id']]['group']
    ids=[q for q in prepared['schedule'] if prepared['questions'][q]['group']==group]
    if len(ids)!=3:raise RuntimeError('registered group must have exactly three questions')
    off=min(ids,key=lambda q:base.am.hashed(base.builder.QUESTION_SEED,q))
    return ids,off


def schedule():
    arms=[('A',131072),('B1536',131072),('C-final-norw',131072),('C-final-norw',8192)]
    return [(r,a,c) for r,order in ((1,arms),(2,list(reversed(arms)))) for a,c in order]


def jobs(selection,arm,label):
    for q in selection['question_ids']:yield f'{label}|answer|{q}',q,False
    if arm=='A':
        q=selection['cache_off_id'];yield f'{label}|answer|{q}|cache-off',q,True


def status(phase,**extra):
    lm.write(LOCAL/'status.json',{'time':lm.stamp(),'pid':os.getpid(),'pgid':os.getpgid(0),'phase':phase,**extra})


def event(kind,**extra):
    with EVENT_LOCK:
        ae.received(LOCAL/'events.jsonl',kind,{'time':lm.stamp(),'kind':kind,**extra})
    print(kind,flush=True)


def checked():
    ae.gate();p=base.st.frozen(PLAN)
    for name,digest in p['source_sha256'].items():
        if lm.file_hash(lm.ROOT/name)!=digest:raise RuntimeError(f'protected file changed:{name}')
    for name,digest in p['private_source_sha256'].items():
        if lm.file_hash(SOURCE/name)!=digest:raise RuntimeError(f'protected private source changed:{name}')
    if lm.file_hash(LOCAL/'selection.json')!=p['selection_sha256']:raise RuntimeError('selection changed')
    base.checked();continuation.other_workers()
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        parts=line.strip().split(None,1)
        if len(parts)==2 and int(parts[0])!=os.getpid() and 'python' in Path(parts[1].split()[0]).name:
            if 'locomo_long_memory_audit.py run' in parts[1]:raise RuntimeError('another memory audit worker exists')
    return p


@contextmanager
def isolated_restart_records(logging=True):
    saved=base.LOCAL;previous=os.environ.get('LLAMA_ARG_LOG_VERBOSITY')
    base.LOCAL=LOCAL
    if logging:os.environ['LLAMA_ARG_LOG_VERBOSITY']='4'
    else:os.environ.pop('LLAMA_ARG_LOG_VERBOSITY',None)
    try:yield
    finally:
        base.LOCAL=saved
        if previous is None:os.environ.pop('LLAMA_ARG_LOG_VERBOSITY',None)
        else:os.environ['LLAMA_ARG_LOG_VERBOSITY']=previous


def allocation_table(text):
    # Keep individual backend allocations. Do not use repeated totals or GGUF file size.
    components={k:[] for k in ('model','KV','compute','output','recurrent_state')}
    seen=set()
    pattern=r'(\S+)\s+(model|KV|compute|output|RS) buffer size\s*=\s*([0-9.]+)\s*MiB'
    for backend,kind,value in re.findall(pattern,text):
        key=(backend,kind,value)
        if key in seen:continue
        seen.add(key);kind='recurrent_state' if kind=='RS' else kind
        components[kind].append({'backend':backend,'MiB':float(value),'bytes':float(value)*1024**2})
    sums={k:sum(x['bytes'] for x in values) if values else None for k,values in components.items()}
    required=all(sums[k] is not None for k in ('model','KV','compute'))
    return {'components':components,'bytes':sums,'required_recorded':required,
        'known_buffer_sum_bytes':sum(x for x in sums.values() if x is not None),
        'complete_logged_buffer_sum_bytes':sum(x for x in sums.values() if x is not None) if required else None,
        'scope':'logged buffer allocations including other recorded output/state; not OS RSS/footprint, not all overhead; mapped backends may share pages'}


def heavy_processes(server_pid):
    text=subprocess.check_output(['ps','-axo','pid=,%cpu=,rss=,comm='],text=True)
    result=[]
    for line in text.splitlines():
        parts=line.strip().split(None,3)
        if len(parts)!=4:continue
        pid,cpu,rss=int(parts[0]),float(parts[1]),int(parts[2])*1024
        if pid not in (server_pid,os.getpid()) and (cpu>=10 or rss>=GIB):
            result.append({'pid':pid,'cpu_percent':cpu,'rss_bytes':rss,'executable':Path(parts[3]).name})
    return {'threshold':'CPU>=10% or RSS>=1GiB at snapshot; CPU includes non-GPU work; observation not proof of GPU inactivity',
        'other_heavy_processes':result,'time':lm.stamp()}


def mapping_snapshot(pid,label):
    try:
        value=subprocess.run(['/usr/bin/vmmap','-summary',str(pid)],capture_output=True,text=True,timeout=20)
        path=LOCAL/f'vmmap-{label}.txt';path.write_text(value.stdout+value.stderr)
        return {'recorded':value.returncode==0,'returncode':value.returncode,'sha256':lm.file_hash(path),
                'scope':'OS vmmap summary outside HTTP stopwatch; not isolated GPU bytes'}
    except subprocess.TimeoutExpired:
        return {'recorded':False,'reason':'vmmap timeout; no retry; primary resource observations retained'}


def send(key,body,now,replies,watch):
    ae.gate()
    if key in replies or (LOCAL/'inflight.json').exists():raise RuntimeError('received/unknown inflight; no resend')
    if watch.error:raise RuntimeError(watch.error)
    digest=lm.digest({'body':body,'plan':lm.file_hash(PLAN),'server':now})
    lm.write(LOCAL/'inflight.json',{'key':key,'digest':digest,'time':lm.stamp()})
    tick=time.perf_counter()
    response=base.st.http_json(now['url']+'/v1/chat/completions',body,timeout=900)
    reply={'digest':digest,'response':response,'timings':response.get('timings') or {},'wall_seconds':time.perf_counter()-tick,
           'attempts':1,'time':lm.stamp()}
    replies[key]=reply;ae.received(LOCAL/'received.jsonl',key,reply)
    (LOCAL/'inflight.json').unlink()
    lm.write(LOCAL/'responses.json',replies)
    ae.checked_timings(reply)
    return reply


def timing_row(key,item,reply,off=False):
    t=reply['timings'];finish=reply['response']['choices'][0].get('finish_reason')
    return {'key_sha256':lm.digest(key),'cache_off':off,'wall_seconds':reply['wall_seconds'],
        'prefill_seconds':t['prompt_ms']/1000,'generation_seconds':t['predicted_ms']/1000,
        'cache_tokens':t['cache_n'],'input_tokens':item['input_tokens'],'finish_reason':finish}


def summary(arm,rows,resource,before,after,maps,elapsed):
    main=[r for r in rows if not r['cache_off']];off=[r for r in rows if r['cache_off']]
    initial=resource['initial_OS'];large=initial['swap_used_bytes']>=GIB
    return {'provisional':False,'condition':arm,'main_requests':len(main),'cache_off_requests':len(off),
        'sample_scope':'one longest-input synthetic conversation; three original development questions; extra memory-only responses',
        'request_order':rows,'timing_main':reporting.timing(main),'timing_cache_off':reporting.timing(off),
        'cache':{'cold':reporting.timing(main[:1]),'warm':reporting.timing(main[1:])},
        'finish_counts':dict(Counter(r['finish_reason'] for r in rows)),
        'baseline':before,'after_requests':after,'peak':resource['peak'],'swap_affected':resource['swap_affected'],
        'initial_OS':initial,'last_OS':resource['OS_samples1s'][-1],'started_with_large_swap':large,
        'observations50ms':resource['samples50ms'],'time_interpretation':resource['time_interpretation'],
        'vmmap':maps,'elapsed_seconds':elapsed,'quality':'not scored; never enters original accuracy evaluation',
        'memory_scope':'fresh process; Qwen including model/KV/compute; RSS and footprint overlap, never add them',
        'limit':'one conversation/few requests, not guaranteed global maximum; A has extra cache-off exposure; initial swap>=1GiB marked'}


def measure(p,selection,prepared,round_number,arm,context,replies):
    label=f'r{round_number}-{arm}-c{context}'
    checked();status('planned fresh server',condition=arm,context=context,round=round_number,status='running')
    with isolated_restart_records():pid,now=base.restart(context,label)
    before=base.resources.audit.memory(pid)
    start_OS=base.resources.observe_os();heavy=heavy_processes(pid)
    log=LOCAL/f'qwen-{label}.log'
    allocations=allocation_table(log.read_text())
    allocation_record={'round':round_number,'condition':arm,'context':context,'server_pid':pid,
        'allocations':allocations,'baseline':before,'initial_OS':start_OS,'heavy_processes':heavy,
        'log_sha256':lm.file_hash(log)}
    lm.write(lm.RESULTS/f'long-context-memory-audit-{label}-allocations.json',allocation_record)
    event('allocation ready',**allocation_record)
    if not allocations['required_recorded']:raise RuntimeError('required model/KV/compute allocation log missing; no HTTP or retry')
    maps={'baseline':mapping_snapshot(pid,f'{label}-baseline')}
    watch=base.resources.Watch(pid,lambda kind,data:event(kind,condition=arm,context=context,round=round_number,stage='answer',**data))
    rows=[];tick=time.perf_counter();after={}
    items=lm.read(SOURCE/f'128000-{arm}-inputs.json')
    try:
        with watch:
            for key,qid,off in jobs(selection,arm,label):
                status('answer',condition=arm,context=context,round=round_number,completed=len(rows),status='running')
                item=items[f'128000|{arm}|{qid}']
                if not item['fits_context'] or item['input_tokens']+384>context:
                    event('input overflow',condition=arm,context=context,round=round_number)
                    raise RuntimeError('input overflow; no larger-window retry without human confirmation')
                body=ae.request(p['answer_plan'],item)
                if off:body['cache_prompt']=False
                reply=send(key,body,now,replies,watch)
                if off and reply['timings']['cache_n']!=0:raise RuntimeError('cache-off reused tokens')
                row=timing_row(key,item,reply,off);rows.append(row)
                if row['finish_reason']!='stop':event('nonstop finish',condition=arm,context=context,round=round_number,stage='answer',cache_off=off,finish_reason=row['finish_reason'])
            after=base.resources.audit.memory(pid)
    finally:
        lm.write(LOCAL/'responses.json',replies);lm.write(LOCAL/f'{label}-rows.json',rows)
        lm.write(LOCAL/f'{label}-resources.json',watch.record())
    if watch.error:raise RuntimeError(watch.error)
    maps['after']=mapping_snapshot(pid,f'{label}-after')
    r=summary(arm,rows,watch.record(),before,after,maps,time.perf_counter()-tick)
    r.update({'round':round_number,'context':context,'server_pid':pid,'allocations':allocations,'start_OS_before_mapping':start_OS,'heavy_processes':heavy})
    lm.write(lm.RESULTS/f'long-context-memory-audit-{label}.json',r)
    event('memory condition complete',condition=arm,context=context,round=round_number,summary=r)
    if context==8192:
        # Same process, after the answer sample: isolated stage observations plus whole-process lifetime peak.
        rerank_start_heavy=heavy_processes(pid)
        rerank_baseline=base.resources.audit.memory(pid)
        watch=base.resources.Watch(pid,lambda kind,data:event(kind,condition=arm,context=context,round=round_number,stage='rerank',**data))
        rr=[]
        try:
            with watch:
                for qid in selection['question_ids']:
                    key=f'{label}|rerank|{qid}';job=prepared['rerank_jobs'][f'128000|C-final-norw|{qid}']
                    if not job['fits_context'] or job['input_tokens']+384>8192:raise RuntimeError('rerank input overflow; no rescue')
                    status('rerank memory',context=context,round=round_number,completed=len(rr),status='running')
                    reply=send(key,job['body'],now,replies,watch)
                    rr.append(timing_row(key,job,reply))
                    if rr[-1]['finish_reason']!='stop':event('nonstop finish',condition=arm,context=context,round=round_number,stage='rerank')
        finally:
            lm.write(LOCAL/f'{label}-rerank-rows.json',rr);lm.write(LOCAL/f'{label}-rerank-resources.json',watch.record())
        if watch.error:raise RuntimeError(watch.error)
        r['rerank']={'same_process_as_minimum_window_answer':True,'server_pid':pid,'rows':rr,'timing':reporting.timing(rr),
            'resource':watch.record(),'order':'after three answers; replies unused for input selection or accuracy',
            'initial_OS':watch.initial,'heavy_processes':rerank_start_heavy,'baseline':rerank_baseline}
        r['server_combined_peak']={k:max(v,r['peak'][k]) for k,v in watch.record()['peak'].items()}
        lm.write(lm.RESULTS/f'long-context-memory-audit-{label}.json',r)
        event('rerank memory complete',context=context,round=round_number,summary=r['rerank'])
    return label,r


def run():
    p=checked();LOCAL.mkdir(parents=True,exist_ok=True)
    if any((LOCAL/name).exists() for name in ('run-start.json','received.jsonl','inflight.json')):
        raise RuntimeError('memory audit already started; no auto repeat/resume')
    prepared=lm.read(SOURCE/'prepared.json');selection=lm.read(LOCAL/'selection.json')
    if selected(prepared,lm.read(SOURCE/'128000-A-inputs.json'))!=(selection['question_ids'],selection['cache_off_id']):
        raise RuntimeError('registered selection differs')
    lm.write(LOCAL/'run-start.json',{'time':lm.stamp(),'pid':os.getpid(),'pgid':os.getpgid(0),'plan_sha256':lm.file_hash(PLAN)})
    replies={};results={};start=time.perf_counter()
    try:
        for round_number,arm,context in schedule():
            label,r=measure(p,selection,prepared,round_number,arm,context,replies);results[label]=r
        checked()
        if len(replies)!=32:raise RuntimeError('registered memory request count differs')
        # Restore original logging and131072 argv for the separately authorized A continuation.
        status('planned restore for A',status='running')
        with isolated_restart_records(logging=False):base.restart(131072,'restore-original-128k')
        result={'time':lm.stamp(),'plan_sha256':lm.file_hash(PLAN),'all32_requests_complete':True,
            'original_files_hash_match':True,'source_files':len(p['source_sha256']),'private_sources':len(p['private_source_sha256']),
            'conditions':results,'seconds':time.perf_counter()-start,'server_restarts':9,'window_value_changes':2,
            'A_accuracy_stays_paused':True,'A_resume_authorized_after_report':True,
            'no_F1_or_selection':'audit responses not scored or added to frozen evaluation',
            'responses_sha256':lm.file_hash(LOCAL/'received.jsonl')}
        lm.write(lm.RESULTS/'long-context-memory-audit-report.json',result)
        lm.write(ae.PAUSE,{'time':lm.stamp(),'reason':'memory audit complete; CPU report before separately authorized unchanged A resume','runner':'long-context-memory-audit'})
        status('memory audit complete',status='complete');event('memory audit complete',seconds=result['seconds'])
    except BaseException as error:
        lm.write(LOCAL/'responses.json',replies)
        status('stopped',status='stopped',error=f'{type(error).__name__}: {error}')
        event('stopped',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('run',))
    parser.parse_args();run()
