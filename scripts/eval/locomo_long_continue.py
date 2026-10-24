#!/usr/bin/env python3
"""Human-registered continuation; frozen jobs and received answers stay immutable."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import time

import locomo_long_eval as base
import locomo_long_report as report

lm, ae = base.lm, base.ae
AMEND = lm.RESULTS/'long-context-order-amendment.json'
LOCAL = base.LOCAL


def jobs(prepared, arm):
    groups=sorted({q['group'] for q in prepared['questions'].values()},
                  key=lambda g:base.am.hashed('locomo-answer-v2',g))
    for group in groups:
        qids=[q for q in prepared['schedule'] if prepared['questions'][q]['group']==group]
        for qid in qids:yield f'128000|{arm}|{qid}',qid,False
        if arm=='A':
            qid=min(qids,key=lambda q:base.am.hashed(base.builder.QUESTION_SEED,q))
            yield f'128000|A|{qid}|cache-off',qid,True


def verify_reply(key,reply,p,servers):
    if reply.get('failure_kind'):
        if reply['attempts']!=0:raise RuntimeError('failure attempts differ')
        return
    ae.checked_timings(reply)
    n,arm,qid,*extra=key.split('|')
    item=lm.read(LOCAL/f'{n}-{arm}-inputs.json')[f'{n}|{arm}|{qid}']
    body=ae.request(p['answer_plan'],item)
    if extra:body['cache_prompt']=False
    identity=servers.get(key)
    if identity is None:
        identity=next(s['new'] for s in lm.read(LOCAL/'context-switches.json') if s['label']==f'{n}-{arm}')
    digest=lm.digest({'request':body,'plan':lm.file_hash(base.PLAN),'server':identity})
    if reply['digest']!=digest:raise RuntimeError(f'received request identity differs:{key}')


def recover(p):
    if (LOCAL/'inflight.json').exists():raise RuntimeError('unknown inflight; no retry or deletion')
    answers=ae.recover(LOCAL/'answer-responses.json',LOCAL/'answer-received.jsonl')
    reranks=ae.recover(LOCAL/'rerank-responses.json',LOCAL/'rerank-received.jsonl')
    servers=lm.read(LOCAL/'continuation-request-servers.json') if (LOCAL/'continuation-request-servers.json').exists() else {}
    for key,reply in answers.items():verify_reply(key,reply,p,servers)
    original=LOCAL/'A-original-received.json'
    if original.exists():
        for key,reply in lm.read(original).items():
            if answers.get(key)!=reply:raise RuntimeError('original received A reply changed or missing')
    return answers,reranks,servers


def other_workers():
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        parts=line.strip().split(None,1)
        if len(parts)==2 and int(parts[0])!=os.getpid() and Path(parts[1].split()[0]).name.startswith('python'):
            if any(tag in parts[1] for tag in ('locomo_long_eval.py run','locomo_long_continue.py bc','locomo_long_continue.py resume-a')):
                raise RuntimeError('another long evaluation worker exists')


def checked():
    ae.gate();m=lm.read(AMEND)
    for name,digest in m['code_sha256'].items():
        if lm.file_hash(lm.ROOT/name)!=digest:raise RuntimeError('continuation code/test changed')
    if lm.file_hash(base.PLAN)!=m['original_plan_sha256']:raise RuntimeError('original plan changed')
    if lm.file_hash(lm.ROOT/'docs/MEMORY_LONG_CONTEXT_ORDER_AMENDMENT.md')!=m['document_sha256']:
        raise RuntimeError('amendment document changed')
    for name,digest in m['private_pause_snapshot_sha256'].items():
        if lm.file_hash(LOCAL/name)!=digest:raise RuntimeError('paused A snapshot changed')
    p=base.checked();other_workers()
    return p,m


def current_server(label):
    env=base.quiet();now=base.norw.ab.server(131072)
    instance=env['qwen_servers'][0]
    argv=lm.read(lm.RESULTS/'answer-pilot-server.json')['new_args'].copy()
    argv[argv.index('-c')+1]='131072'
    if instance['command']!=' '.join(argv):raise RuntimeError('server argv differs')
    if lm.file_hash(Path(argv[argv.index('-m')+1]))!=now['model_sha256']:raise RuntimeError('model hash differs')
    records=lm.read(LOCAL/'context-switches.json')
    if not any(s['label']==label for s in records):
        records.append({'time':lm.stamp(),'label':label,'pid':instance['pid'],'old':now,'new':now,
            'args':argv,'seconds':0.,'restart':False,'shared_process_memory_not_condition_comparison':True})
        lm.write(LOCAL/'context-switches.json',records)
    return instance['pid'],now


def guarded_send(key,kind,body,digest,inp,now,replies,watch,servers,send):
    if key in replies:raise RuntimeError('received job cannot be regenerated')
    if key in servers:raise RuntimeError('unreceived recorded request; stop without retry')
    servers[key]=now;lm.write(LOCAL/'continuation-request-servers.json',servers)
    return send(key,kind,body,digest,inp,now,replies,watch)


@contextmanager
def reuse_server(servers):
    restart,send,summary=base.restart,base.send,report.condition_summary
    def same_window(context,label):
        if context!=131072:raise RuntimeError('B/C continuation changes no window')
        return current_server(label)
    def tracked_send(*args):return guarded_send(*args,servers,send)
    def marked_summary(*args):
        value=summary(*args)
        value['order_amendment_sha256']=lm.file_hash(AMEND)
        value['memory_interpretation']='shared existing131072 process; condition memory comparison deferred; lifetime peak includes earlier arms'
        value['cache_interpretation']='same frozen cache_prompt; shared process/order changed by human request'
        return value
    base.restart,base.send,report.condition_summary=same_window,tracked_send,marked_summary
    try:yield
    finally:base.restart,base.send,report.condition_summary=restart,send,summary


def pause_after_bc():
    lm.write(ae.PAUSE,{'time':lm.stamp(),'reason':'128k B/C completed; human deferred A; explicit A resume required',
                     'runner':'long-context-continuation','A_resume':'not authorized automatically'})
    base.status('waiting for A resume',status='paused',length=128000,condition='A')
    base.event('B/C complete; A deferred',{'length':128000,'human_confirmation_required':True})


def bc():
    p,m=checked();answers,reranks,servers=recover(p);prepared=lm.read(LOCAL/'prepared.json')
    if set(reranks)!=set(prepared['rerank_jobs']):raise RuntimeError('registered reranks incomplete')
    for key,reply in reranks.items():
        if not reply.get('failure_kind') and reply['digest']!=prepared['rerank_jobs'][key]['digest']:
            raise RuntimeError('saved rerank identity differs')
    if not m['other_apps_closed_confirmed']:raise RuntimeError('other apps closure not confirmed')
    if (LOCAL/'bc-start.json').exists():raise RuntimeError('B/C already started; no automatic retry')
    if any(k.startswith(('128000|B1536|','128000|C-final-norw|')) for k in answers):
        raise RuntimeError('B/C has existing answers; no duplicate run')
    lm.write(LOCAL/'bc-start.json',{'time':lm.stamp(),'pid':os.getpid(),'pgid':os.getpgid(0),'amendment_sha256':lm.file_hash(AMEND)})
    tick=time.perf_counter()
    with reuse_server(servers):
        for arm in ('B1536','C-final-norw'):base.run_condition(p,prepared,'128000',arm,reranks,answers)
    base.norw.verify_originals(p)
    lm.write(LOCAL/'bc-runtime.json',{'seconds':time.perf_counter()-tick,'time':lm.stamp()})
    pause_after_bc()


def resume_a():
    """Only run after a later explicit human instruction, never automatically after B/C."""
    p,m=checked();answers,reranks,servers=recover(p);prepared=lm.read(LOCAL/'prepared.json')
    if not m['other_apps_closed_confirmed']:raise RuntimeError('other apps closure not confirmed')
    if (LOCAL/'a-resume-start.json').exists():raise RuntimeError('A resume already started; no automatic retry')
    if not all((LOCAL/f'128000-{a}-completed.json').exists() for a in ('B1536','C-final-norw')):
        raise RuntimeError('B/C must complete before A resume')
    pid,now=current_server('128000-A-resume')
    items=lm.read(LOCAL/'128000-A-inputs.json')
    rows=[];aux=[];remaining=0
    for key,qid,off in jobs(prepared,'A'):
        if key in answers:(aux if off else rows).append(base.main_row(items[f'128000|A|{qid}'],answers[key]))
        else:remaining+=1
    lm.write(LOCAL/'a-resume-start.json',{'time':lm.stamp(),'remaining':remaining,'pid':os.getpid(),'pgid':os.getpgid(0)})
    watch=base.resources.Watch(pid,lambda kind,data:base.event(kind,{'length':128000,'condition':'A','stage':'resume',**data}))
    tick=time.perf_counter()
    try:
        with watch:
            for key,qid,off in jobs(prepared,'A'):
                ae.gate()
                if key in answers:continue
                item=items[f'128000|A|{qid}']
                base.status('answer',status='running',length=128000,condition='A',completed=len(rows),auxiliary=len(aux),stage='resume')
                if not item['fits_context']:raise RuntimeError('unexpected A input overflow; unchanged preflight required')
                body=ae.request(p['answer_plan'],item)
                if off:body['cache_prompt']=False
                digest=lm.digest({'request':body,'plan':lm.file_hash(base.PLAN),'server':now})
                reply=guarded_send(key,'answer',body,digest,item['input_tokens'],now,answers,watch,servers,base.send)
                if off and reply['timings']['cache_n']!=0:raise RuntimeError('cache-off reused tokens')
                row=base.main_row(item,reply);(aux if off else rows).append(row)
                finish=reply['response']['choices'][0].get('finish_reason')
                if finish!='stop':base.event('nonstop finish',{'length':128000,'condition':'A','stage':'resume','auxiliary':off,'finish_reason':finish})
                if row['failure_kind']=='parse':base.event('parse failure',{'length':128000,'condition':'A','stage':'resume','auxiliary':off})
                if len(answers)%25==0:lm.write(LOCAL/'answer-responses.json',answers)
    finally:
        lm.write(LOCAL/'answer-responses.json',answers)
        lm.write(LOCAL/'128000-A-resume-partial-rows.json',{'main':rows,'auxiliary':aux})
        lm.write(LOCAL/'128000-A-resume-resources.json',watch.record())
    if watch.error:raise RuntimeError(watch.error)
    if len(rows)!=30 or len(aux)!=10:raise RuntimeError('A incomplete; no final inference')
    # Rebuild in the original question order, regardless of interrupted execution epoch.
    rows=[base.main_row(items[f'128000|A|q'],answers[k]) for k,q,off in jobs(prepared,'A') if not off]
    aux=[base.main_row(items[f'128000|A|q'],answers[k]) for k,q,off in jobs(prepared,'A') if off]
    result=report.condition_summary(p,prepared,'128000','A',rows,aux,reranks,watch.record(),time.perf_counter()-tick)
    result['condition_elapsed_seconds']=None
    result['resume_epoch_seconds']=time.perf_counter()-tick
    result['memory_interpretation']='separate sessions/shared process; memory comparison deferred; original and resume observations stored separately'
    result['cache_interpretation']='interrupted A; original received cache timings preserved; resumed cache state differs; no continuous-session speed claim'
    result['order_amendment_sha256']=lm.file_hash(AMEND)
    lm.write(LOCAL/'128000-A-completed.json',{'rows':rows,'auxiliary':aux,'summary':result})
    lm.write(lm.RESULTS/'long-context-128000-A-interim.json',result)
    base.event('condition complete',{'length':128000,'condition':'A','summary':result})
    base.status('all answers complete; CPU final verification pending',status='generated')
    lm.write(ae.PAUSE,{'time':lm.stamp(),'reason':'all answers complete; final CPU verification only, no further generation'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('bc','resume-a'))
    command=parser.parse_args().command
    try:{'bc':bc,'resume-a':resume_a}[command]()
    except BaseException as error:
        base.status('stopped',status='stopped',error=f'{type(error).__name__}: {error}',continuation=command)
        base.event('stopped',{'error':f'{type(error).__name__}: {error}','continuation':command})
        raise


if __name__=='__main__':main()
