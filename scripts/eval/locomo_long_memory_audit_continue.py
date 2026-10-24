#!/usr/bin/env python3
"""Human-authorized log-hash deviation; continue seven memory cells only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import time

import locomo_long_memory_audit as audit

lm,ae=audit.lm,audit.ae
ORIGINAL=audit.LOCAL
LOCAL=lm.LOCAL/'long-context-memory-audit-continuation'
AMEND=lm.RESULTS/'long-context-memory-audit-deviation.json'


def protected_private(p):
    # Runtime logs are appendable. Response journals remain protected, including JSONL.
    return {n:h for n,h in p['private_source_sha256'].items() if not n.endswith('.log')}


def remaining_schedule():return audit.schedule()[1:]


def load_original():
    p=lm.read(audit.PLAN)
    selection=lm.read(ORIGINAL/'selection.json')
    if (ORIGINAL/'inflight.json').exists():raise RuntimeError('original unknown inflight; no retry')
    answers=ae.recover(ORIGINAL/'responses.json',ORIGINAL/'received.jsonl')
    expected={k for k,q,off in audit.jobs(selection,'A','r1-A-c131072')}
    if set(answers)!=expected:raise RuntimeError('original four received keys differ; no regeneration')
    for reply in answers.values():
        ae.checked_timings(reply)
        if reply['attempts']!=1:raise RuntimeError('original attempt count differs')
    return p,selection,answers


def checked():
    ae.gate();d=audit.base.st.frozen(AMEND);p=lm.read(audit.PLAN)
    if lm.file_hash(audit.PLAN)!=d['original_plan_sha256']:raise RuntimeError('original plan changed')
    if d['excluded_private_runtime_logs']!=sorted(set(p['private_source_sha256'])-set(protected_private(p))):
        raise RuntimeError('registered log exclusions differ')
    for n,h in {**p['source_sha256'],**d['extra_public_sha256']}.items():
        if lm.file_hash(lm.ROOT/n)!=h:raise RuntimeError('protected public file changed:'+n)
    for n,h in protected_private(p).items():
        if lm.file_hash(audit.SOURCE/n)!=h:raise RuntimeError('protected private source changed:'+n)
    for n,h in d['original_audit_snapshot_sha256'].items():
        if lm.file_hash(ORIGINAL/n)!=h:raise RuntimeError('original audit snapshot changed:'+n)
    audit.base.checked();audit.continuation.other_workers()
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        fields=line.strip().split(None,1)
        if len(fields)==2 and int(fields[0])!=os.getpid() and 'python' in Path(fields[1].split()[0]).name:
            if any(s in fields[1] for s in ('locomo_long_memory_audit.py run','locomo_long_memory_audit_continue.py run')):
                raise RuntimeError('another memory worker exists')
    return p


def swap_check(stage,pid=None):
    observations=[]
    for i in range(3):
        ae.gate();v=audit.base.resources.observe_os()
        memory=audit.base.resources.audit.memory(pid) if pid else {'physical_footprint_bytes':0}
        error=audit.base.resources.violation(memory,v)
        if error:raise RuntimeError(error)
        observations.append(v)
        if i<2:time.sleep(1)
    first,last=observations[0],observations[-1]
    increment=any(audit.base.resources.swap_increased(first,v) for v in observations[1:])
    return {'stage':stage,'time':lm.stamp(),'samples':observations,'started_with_large_swap':first['swap_used_bytes']>=audit.GIB,
        'increment_observed':increment,'stable_in_observation_window':not increment,
        'definition':'three OS snapshots1second apart; no increased swapins/out/usage relative to first; this is not a guarantee of future stability',
        'time_is_reference':increment}


@contextmanager
def amended_runtime():
    local,check,restart,summary=audit.LOCAL,audit.checked,audit.base.restart,audit.summary
    audit.LOCAL,audit.checked=LOCAL,checked
    current={}
    def tracked_restart(context,label):
        before=swap_check('before planned restart')
        pid,now=restart(context,label)
        after=swap_check('fresh ready before requests',pid)
        current.clear();current.update({'before_restart':before,'before_requests':after,'label':label})
        lm.write(LOCAL/f'{label}-swap-prechecks.json',current)
        audit.event('swap precheck',label=label,context=context,observations=current)
        return pid,now
    def marked_summary(*args):
        value=summary(*args)
        value['deviation_sha256']=lm.file_hash(AMEND)
        value['swap_prechecks']=dict(current)
        value['started_with_large_swap']=value['started_with_large_swap'] or current['before_requests']['started_with_large_swap']
        if current['before_restart']['increment_observed'] or current['before_requests']['increment_observed']:
            value['time_interpretation']='reference only; swap increments observed during precheck or workload'
            value['swap_precheck_affected']=True
        else:value['swap_precheck_affected']=False
        return value
    audit.base.restart,audit.summary=tracked_restart,marked_summary
    try:yield
    finally:audit.LOCAL,audit.checked,audit.base.restart,audit.summary=local,check,restart,summary


def run():
    checked();p,selection,replies=load_original();prepared=lm.read(audit.SOURCE/'prepared.json')
    if any((LOCAL/n).exists() for n in ('run-start.json','received.jsonl','inflight.json')):
        raise RuntimeError('continuation already started; no automatic repeat')
    LOCAL.mkdir(parents=True,exist_ok=True)
    results={'r1-A-c131072':lm.read(lm.RESULTS/'long-context-memory-audit-r1-A-c131072.json')}
    with amended_runtime():
        lm.write(LOCAL/'run-start.json',{'time':lm.stamp(),'pid':os.getpid(),'pgid':os.getpgid(0),
            'deviation_sha256':lm.file_hash(AMEND),'original_received_preserved':4})
        tick=time.perf_counter()
        try:
            for round_number,arm,context in remaining_schedule():
                label,r=audit.measure(p,selection,prepared,round_number,arm,context,replies);results[label]=r
            checked()
            if len(replies)!=32:raise RuntimeError('registered32responses not complete')
            # Original four never enter the new journal; they are only copied to the merged checkpoint.
            original=load_original()[2]
            if any(replies[k]!=v for k,v in original.items()):raise RuntimeError('original received response changed')
            audit.status('planned restore for A',status='running')
            with audit.isolated_restart_records(logging=False):audit.base.restart(131072,'restore-original-128k')
            d=lm.read(AMEND)
            result={'time':lm.stamp(),'plan_sha256':lm.file_hash(audit.PLAN),'deviation_sha256':lm.file_hash(AMEND),
                'all32_requests_complete':True,'original_received4_preserved':True,'new_received28':28,
                'original_files_hash_match_except_registered_runtime_logs':True,
                'excluded_runtime_logs':d['excluded_private_runtime_logs'],'source_files':len(p['source_sha256']),
                'protected_private_sources':len(protected_private(p)),'conditions':results,
                'continuation_seconds':time.perf_counter()-tick,'original_first_cell_seconds':results['r1-A-c131072']['elapsed_seconds'],
                'seconds':'split epochs; continuation_seconds and original first cell separately; human waiting excluded',
                'server_starts_total':9,'window_value_changes':2,'A_accuracy_stays_paused':True,
                'no_F1_or_selection':'memory audit responses never scored or mixed into accuracy evaluation',
                'original_journal_sha256':lm.file_hash(ORIGINAL/'received.jsonl'),
                'continuation_journal_sha256':lm.file_hash(LOCAL/'received.jsonl')}
            lm.write(lm.RESULTS/'long-context-memory-audit-report.json',result)
            lm.write(ae.PAUSE,{'time':lm.stamp(),'reason':'memory audit complete; CPU report before separately authorized unchanged A resume','runner':'long-context-memory-audit-continuation'})
            audit.status('memory audit complete',status='complete');audit.event('memory audit complete',new_received=28,total_received=32)
        except BaseException as error:
            lm.write(LOCAL/'responses.json',replies)
            audit.status('stopped',status='stopped',error=f'{type(error).__name__}: {error}')
            audit.event('stopped',error=f'{type(error).__name__}: {error}')
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('run',));parser.parse_args();run()
