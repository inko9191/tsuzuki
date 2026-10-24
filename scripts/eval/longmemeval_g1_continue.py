"""Registered G1 continuation: reuse ingestion/vectors, widen rerank only, C/B/A then judge."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import numpy as np
import longmemeval_g1_run as original
import longmemeval_g1_window as diagnosis

lm, ae, mm, st, am, fp = original.lm, original.ae, original.mm, original.st, original.am, original.fp
SOURCE = original.LOCAL
LOCAL = SOURCE.parent/'continuation'
PLAN = lm.RESULTS/'longmemeval-g1-window-deviation.json'


def validate_allocations(allocations, restore=False):
    # Normal restoration uses the original verbosity, which may omit buffer logs.
    # Measurement starts still require all registered allocation records.
    if not restore and not allocations['required_recorded']:
        raise RuntimeError('required model/KV/compute logs missing; no inference')


def saved_ingestion():
    replies = {}
    for line in (SOURCE/'received.jsonl').read_text().splitlines():
        record = json.loads(line)
        if not record['key'].startswith('fact|') or record['key'] in replies or record['reply']['attempts'] != 1:
            raise RuntimeError('saved source has duplicate/non-fact/non-first-attempt record')
        replies[record['key']] = record['reply']
    if len(replies) != 1441 or (SOURCE/'inflight.json').exists():
        raise RuntimeError('ingestion not complete or unknown source inflight')
    return replies


@contextmanager
def runtime():
    saved = original.LOCAL, original.PLAN
    original.LOCAL, original.PLAN = LOCAL, PLAN
    try:
        with original.isolated():
            yield
    finally:
        original.LOCAL, original.PLAN = saved


class Servers(original.Servers):
    def start(self, context, label, restore=False):
        # Same fresh-process lifecycle as the original; the registered context allowlist is extended.
        ae.gate()
        if self.up:
            self.stop()
        original.other_workers()
        if any(Path(s.strip()).name == 'llama-server' for s in
               subprocess.check_output(['ps', '-axo', 'comm='], text=True).splitlines()):
            raise RuntimeError('server exists before fresh start')
        if context not in (8192, self.p['selected_rerank_context'], 131072):
            raise RuntimeError('unregistered window')
        system = original.base.resources.observe_os()
        issue = original.base.resources.violation({'physical_footprint_bytes': 0}, system)
        if issue:
            raise RuntimeError(issue)
        argv = self.argv.copy(); argv[argv.index('-c')+1] = str(context)
        if lm.file_hash(Path(argv[argv.index('-m')+1])) != self.p['model_sha256']:
            raise RuntimeError('model hash mismatch')
        env = os.environ.copy()
        if restore:
            env.pop('LLAMA_ARG_CACHE_RAM', None); env.pop('LLAMA_ARG_LOG_VERBOSITY', None)
        tick = time.monotonic(); path = LOCAL/f'qwen-{label}.log'
        with path.open('a') as log:
            proc = subprocess.Popen(argv, cwd=lm.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        self.up = True
        for _ in range(60):
            if proc.poll() is not None:
                raise RuntimeError('fresh server exited; no retry')
            try:
                now = original.base.norw.ab.server(context)
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(1)
        else:
            raise RuntimeError('fresh server timeout; no retry')
        if now['build'] != self.p['build']:
            raise RuntimeError('build differs')
        allocations = fp.audit.allocation_table(path.read_text())
        validate_allocations(allocations, restore)
        baseline = original.base.resources.audit.memory(proc.pid)
        record = {'pid': proc.pid, 'context': context, 'label': label, 'server': now, 'argv': argv,
                  'initial_OS': system, 'ready_OS': original.base.resources.observe_os(),
                  'heavy_processes': fp.audit.heavy_processes(proc.pid), 'allocations': allocations,
                  'OS_baseline': baseline, 'seconds': time.monotonic()-tick, 'restored_original': restore}
        ae.received(LOCAL/'server-starts.jsonl', label, record)
        original.event('allocation ready', label=label, context=context, pid=proc.pid,
                       allocations=allocations, OS_baseline=baseline, initial_swap_bytes=system['swap_used_bytes'])
        return proc.pid, now


def rerank_and_preflight(prepared, jobs, pid, now, replies, p):
    count = ae.Tokenizer(now['url']); C, timings = {}, {}
    watch = fp.SeriesWatch(pid, lambda kind, data: original.event(kind, stage='rerank', **data))
    try:
        with watch:
            for number, qid in enumerate(p['question_ids'], 1):
                ae.gate(); watch.begin(number); job = jobs[qid]
                if lm.digest(job['body']) != job['body_sha256']:
                    raise RuntimeError('unchanged rerank body hash mismatch')
                if not original.fits(job['input_tokens'], 120, p['selected_rerank_context']):
                    raise RuntimeError('rerank overflow even after registered change')
                reply = original.send('rerank|'+qid, job['body'], now, replies, watch, job['input_tokens'])
                text = reply['response']['choices'][0]['message'].get('content') or ''
                value = mm.parse('rerank', text, n_passages=min(len(job['order']), mm.RERANK_CANDIDATES))
                if value is None or reply['response']['choices'][0]['finish_reason'] != 'stop':
                    original.event('generation failure', stage='rerank', fallback='unchanged frozen order rule; no resend')
                order = mm.distinct(original.base.norw.prod.reranked_order(job['order'], value), job['units'])
                raw_ids = list(dict.fromkeys(t for i in order for t in job['units'][i]['turns']))
                tick = time.monotonic()
                payload = am.payload('C-final-norw', prepared['turns'][qid], {'C-final-norw': raw_ids}, count.count)
                chat = am.messages(payload.pop('conversation'), payload.pop('retrieved'), prepared['questions'][qid]['question'])
                tokens = count.input_count(chat)
                if not original.fits(tokens, 256, 8192):
                    original.event('input overflow', stage='answer preflight', condition='C-final-norw',
                                   question_id=qid, input_tokens=tokens, context=8192)
                    raise RuntimeError('actual final C input overflow; human review, no answer/window change')
                C[qid] = {'messages': chat, 'input_tokens': tokens, 'fits_context': True,
                          'assembly_seconds': time.monotonic()-tick, **payload}
                timings[qid] = {'search_seconds': job['search_seconds'], 'search_scope': 'separate CPU reconstruction',
                                'assembly_seconds': C[qid]['assembly_seconds'], 'rerank_seconds': reply['wall_seconds']}
                watch.boundary()
                original.status('rerank and actual C preflight', status='running', completed=number, total=30,
                                ingestion_sessions_received=1441, ingestion_sessions_total=1441)
                if number % 25 == 0:
                    lm.write(LOCAL/'responses.json', replies)
    finally:
        count.save(); lm.write(LOCAL/'responses.json', replies)
        lm.write(LOCAL/'rerank-resources.json', watch.record())
        lm.write(LOCAL/'actual-C-inputs-partial.json', C)
    if watch.error:
        raise RuntimeError(watch.error)
    prepared['inputs']['C-final-norw'] = C
    lm.write(LOCAL/'answer-inputs.json', prepared['inputs']); lm.write(LOCAL/'search-timings.json', timings)
    public = {'time': lm.stamp(), 'questions': 30, 'all_actual_C_fit8192': True,
              'actual_C_token_distribution': diagnosis.summary([i['input_tokens'] for i in C.values()]),
              'actual_C_input_tokens': {q:i['input_tokens'] for q,i in C.items()},
              'rerank_context': p['selected_rerank_context'], 'resources': watch.record(),
              'rerank_HTTP_seconds_mean': float(np.mean([r['rerank_seconds'] for r in timings.values()])),
              'ingestion_sessions_received':1441, 'ingestion_sessions_total':1441,
              'continuation_elapsed_seconds':time.time()-lm.read(LOCAL/'run-start.json')['unix_time'], 'unscored':True}
    lm.write(lm.RESULTS/'longmemeval-g1-rerank-interim.json', public)
    original.event('rerank complete', questions=30, C_all_inputs_fit=True,
                   public_file='longmemeval-g1-rerank-interim.json')
    return timings


def finish(p, prepared, answers, judges, timings, replies, start):
    original.checked()
    report, rows, review = original.reporting.analyze(prepared['questions'], prepared['inputs'], answers, judges)
    original.human_review(review); lm.write(LOCAL/'scored-rows.json', rows)
    embedding = lm.read(SOURCE/'embedding.json'); old_status = lm.read(SOURCE/'status.json')
    report.update(time=lm.stamp(), execution_plan_sha256=lm.file_hash(PLAN), original_hash_match=True,
        unchanged_source_ingestion=1441, unchanged_fact_failure_events=3, new_registered_records=len(replies)-1441,
        unique_registered_records=len(replies), HTTP_responses=sum(v['attempts'] for v in replies.values()),
        embedding=embedding['timings'], search_timings=timings, answer_resources=lm.read(LOCAL/'answer-resources.json'),
        rerank_resources=lm.read(LOCAL/'rerank-resources.json'),
        original_preflight_ingestion_embedding_and_stop_seconds=old_status['elapsed_seconds'],
        continuation_seconds_before_restore=time.monotonic()-start,
        generalization_operational_limit='original8k rerank overflow24/30; widened16k under pre-fixed rule; accuracy separate',
        window_diagnosis_file='longmemeval-g1-window-diagnosis.json',
        original_8k_rerank_timing='zero HTTP responses; expanded-vs-original inference timing causality unmeasured',
        original_8k_rerank_idle_resources=lm.read(SOURCE/'rerank-resources.json'))
    report['answer_timing'] = {}
    for arm, part in answers.items():
        by_question = {}
        for qid, reply in part.items():
            t = reply.get('timings') or {}
            components = {'answer_HTTP_seconds': reply['wall_seconds'], 'prefill_seconds': t.get('prompt_ms',0)/1000,
                'generation_seconds':t.get('predicted_ms',0)/1000,
                'assembly_preflight_seconds':prepared['inputs'][arm][qid]['assembly_seconds']}
            if arm == 'C-final-norw':
                components.update({k:v for k,v in timings[qid].items() if isinstance(v,(int,float))},
                                  query_embedding_seconds=embedding['timings'][qid]['query_seconds'])
                components['separately_measured_component_sum_seconds'] = sum(components[k] for k in
                    ('answer_HTTP_seconds','rerank_seconds','search_seconds','assembly_seconds','query_embedding_seconds'))
            by_question[qid] = components
        report['answer_timing'][arm] = {'by_question':by_question,
            'registered_question_means':{k:float(np.mean([r[k] for r in by_question.values()])) for k in next(iter(by_question.values()))},
            'scope':'separate staged components; ingestion excluded; swap reference, no causal8k timing comparison'}
    lm.write(LOCAL/'pending-final-report.json', report)
    return report


def run():
    p = st.frozen(PLAN); original.verify(p); ae.gate(); original.other_workers()
    if any((LOCAL/n).exists() for n in ('run-start.json','received.jsonl','inflight.json')):
        raise RuntimeError('continuation already started; no auto retry')
    replies = saved_ingestion()
    prepared, jobs = lm.read(SOURCE/'prepared.json'), lm.read(diagnosis.LOCAL/'rerank-jobs.json')
    if set(jobs) != set(p['question_ids']):
        raise RuntimeError('registered input cohort mismatch')
    LOCAL.mkdir(parents=True, exist_ok=True)
    lm.write(LOCAL/'run-start.json', {'time':lm.stamp(),'unix_time':time.time(), 'pid':os.getpid(),
                'pgid':os.getpgid(0),'plan_sha256':lm.file_hash(PLAN)})
    lm.write(LOCAL/'token-counts.json',lm.read(diagnosis.LOCAL/'token-counts.json'))
    start = time.monotonic()
    with runtime():
        manager = Servers(p)
        try:
            original.status('reuse completed ingestion',status='running',ingestion_sessions_received=1441,
                            ingestion_sessions_total=1441, reused_fact_failures=3)
            pid, now = manager.start(p['selected_rerank_context'], 'rerank')
            timings = rerank_and_preflight(prepared,jobs,pid,now,replies,p)
            answers = original.answers(prepared,manager,replies,p)
            judges = original.judge(prepared,answers,manager,replies)
            report = finish(p,prepared,answers,judges,timings,replies,start)
            new_records=[json.loads(s) for s in (LOCAL/'received.jsonl').read_text().splitlines()]
            if len(replies)!=1651 or len(new_records)!=210 or len({r['key'] for r in new_records})!=210 or (LOCAL/'inflight.json').exists():
                raise RuntimeError('registered journal incomplete; no final report')
            manager.start(131072,'original-restore',restore=True)
            original.verify(p)
            report['continuation_seconds_including_restore']=time.monotonic()-start
            machine_seconds = lm.read(SOURCE/'status.json')['elapsed_seconds']+report['continuation_seconds_including_restore']
            report['machine_time_component_sum_seconds_excluding_human_wait'] = machine_seconds
            report['time_sum_scope']='old run duration plus continuation duration; interruption/human wait excluded'
            report['allocation_records']=[json.loads(s)['reply'] for s in (LOCAL/'server-starts.jsonl').read_text().splitlines()]
            lm.write(lm.RESULTS/'longmemeval-g1-report.json',report)
            (lm.RESULTS/'longmemeval-g1-report.md').write_text(original.reporting.markdown(report)+
                '\n8k rerank設計は24/30入力で不成立。16kへ生成前登録逸脱、KV/OSピークはJSON allocation_records/rerank_resources。'
                '旧8kは0HTTPで因果的時間差は未測定。取り込み/失敗3件を再生成せず保持。\n')
            lm.write(lm.RESULTS/'longmemeval-g1-verification.json', {'original_hash_match':True,'source_hash_match':True,
                'source_ingestion_preserved':1441,'new_unique_records':210,'total_unique_records':1651,
                'no_unknown_inflight':True,'new_received_journal_sha256':lm.file_hash(LOCAL/'received.jsonl'),
                'old_received_journal_sha256':lm.file_hash(SOURCE/'received.jsonl'),
                'report_sha256':lm.file_hash(lm.RESULTS/'longmemeval-g1-report.json')})
            original.status('evaluation complete',status='complete',registered_records=1651,new_registered_records=210,
                            ingestion_sessions_received=1441,ingestion_sessions_total=1441,
                            continuation_seconds=time.monotonic()-start,machine_time_sum_seconds=machine_seconds)
            original.event('evaluation complete',registered_records=1651,new_registered_records=210)
            lm.write(ae.PAUSE,{'time':lm.stamp(),'runner':'longmemeval-g1-continuation','reason':'G1 complete; CPU report then human confirmation'})
        except BaseException as error:
            original.status('stopped',status='stopped',error=f'{type(error).__name__}: {error}',
                            registered_records=len(replies),unknown_inflight=(LOCAL/'inflight.json').exists())
            original.event('stopped',error=f'{type(error).__name__}: {error}',registered_records=len(replies))
            if not ae.PAUSE.exists():
                lm.write(ae.PAUSE,{'time':lm.stamp(),'runner':'longmemeval-g1-continuation','reason':'stopped; human review required; no auto retry'})
        finally:
            lm.write(LOCAL/'responses.json',replies)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['run','pause'])
    args=parser.parse_args(); {'run':run,'pause':original.pause}[args.command]()
