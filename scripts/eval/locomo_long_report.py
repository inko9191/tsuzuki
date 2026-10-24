"""CPU reporting of completed registered long-context conditions, without raw text."""
from collections import Counter, defaultdict

import numpy as np
import locomo_long_eval as runner
import locomo_memory_reporting as inference

lm, ae = runner.lm, runner.ae
STRATA = ('multiple_evidence', 'overall', 'single_fact', 'temporal')
METRICS = ('f1', 'em', 'abstention', 'failure', 'inserted_all', 'inserted_recall')


def distribution(values):
    values=[v for v in values if v is not None]
    if not values:return {'n':0,'mean':None,'median':None,'p90':None}
    a=np.asarray(values,float)
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.percentile(a,90))}


def aggregates(rows):
    return {kind:{'questions':len(part),**{m:float(np.mean([r[m] for r in part])) for m in METRICS}}
        for kind in STRATA if (part:=[r for r in rows if kind=='overall' or r['type']==kind])}


def timing(rows):
    return {m:distribution([r.get(m) for r in rows]) for m in
        ('wall_seconds','prefill_seconds','generation_seconds','input_tokens','cache_tokens','assembly_seconds')}


def condition_summary(p,prepared,n,arm,rows,aux,reranks,resource,elapsed):
    if len(rows)!=30 or len({r['id'] for r in rows})!=30:
        raise RuntimeError('condition summary requires all thirty distinct main questions')
    blocks=defaultdict(list)
    for row in rows:blocks[row['id'].split('/')[0]].append(row)
    costs=[]
    for row in rows:
        key=f'{n}|{arm}|{row["id"]}'
        rerank=reranks[key]['wall_seconds'] if arm.startswith('C') else 0.
        search=prepared['search_seconds'].get(key,0.)
        assembly=row['assembly_seconds']
        costs.append({'answer':row['wall_seconds'],'rerank':rerank,'search':search,'assembly':assembly,
            'known_sum':row['wall_seconds']+(rerank or 0)+search+assembly,
            'complete_sum':None if rerank is None else row['wall_seconds']+rerank+search+assembly})
    return {'provisional':True,'length':int(n),'condition':arm,'main_questions':30,'auxiliary_questions':len(aux),
        'metrics':aggregates(rows),'failure_counts':dict(Counter(r['failure_kind'] or 'none' for r in rows)),
        'HTTP_answer_attempts':sum(r['attempts'] for r in rows),'answer_time':timing(rows),
        'attempted_answer_time':timing([r for r in rows if r['attempts']]),
        'components_seconds':{k:distribution([c[k] for c in costs]) for k in costs[0]},
        'components_scope':'answer HTTP + rerank HTTP + cached-vector search + raw-input construction; '
            'no fresh embedding/ingestion; separate CPU preparation; sums are components, not end-to-end stopwatches; '
            'unattempted context failures have zero HTTP seconds; unknown reused HTTP not treated as zero in complete_sum',
        'cache_by_conversation':{g:{'first':timing(part[:1]),'warm':timing(part[1:]),'all':timing(part),
            'scores':aggregates(part)} for g,part in blocks.items()},
        'A_cache_off_auxiliary':{'time':timing(aux),'failures':sum(r['failure'] for r in aux),
            'paired_cached_time':timing([r for r in rows if r['id'] in {a['id'] for a in aux}]),
            'excluded_from_main_F1':True} if aux else None,
        'peak_memory':resource['peak'],'swap_affected':resource['swap_affected'],
        'initial_OS':resource['initial_OS'],'time_interpretation':resource['time_interpretation'],
        'condition_elapsed_seconds':elapsed,'resource_samples50ms':resource['samples50ms'],
        'limitations':'synthetic other-speaker distractors may ease retrieval; development30/ten conversations; '
            'no real-long-conversation, unseen-conversation, Japanese or semantic-correctness claim'}


def paired(left,right):
    left=sorted(left,key=lambda r:r['id']);right=sorted(right,key=lambda r:r['id'])
    result={}
    for kind in STRATA:
        a=[r for r in left if kind=='overall' or r['type']==kind]
        b=[r for r in right if kind=='overall' or r['type']==kind]
        result[kind]={}
        for metric in METRICS:
            stat=inference.paired(a,b,metric)
            interval=stat['conversation_ci95']
            stat['equivalent_within_0.03']=bool(interval and interval[0]>=-.03 and interval[1]<=.03)
            stat['practical_gain_lower_CI_above_0.05']=bool(interval and interval[0]>.05)
            result[kind][metric]=stat
    return result


def verify_complete(p,prepared,answers,reranks,completed):
    expected={f'{n}|{a}|{qid}' for n,arms in p['design']['conditions'].items() for a in arms for qid in prepared['schedule']}
    expected|={f'{n}|A|{qid}|cache-off' for n in p['design']['conditions'] for qid in
        (min([q for q in prepared['schedule'] if prepared['questions'][q]['group']==g],
            key=lambda q:runner.am.hashed(runner.builder.QUESTION_SEED,q))
         for g in {q['group'] for q in prepared['questions'].values()})}
    if set(answers)!=expected or set(reranks)!=set(prepared['rerank_jobs']):
        raise RuntimeError('final analysis requires all registered responses/failures, no extra questions')
    for key,reply in answers.items():
        if reply['attempts']==1:
            ae.checked_timings(reply)
            n,a,qid,*extra=key.split('|')
            item=lm.read(runner.LOCAL/f'{n}-{a}-inputs.json')[f'{n}|{a}|{qid}']
            body=ae.request(p['answer_plan'],item)
            if extra:body['cache_prompt']=False
            server=next(s['new'] for s in lm.read(runner.LOCAL/'context-switches.json') if s['label']==f'{n}-{a}')
            digest=lm.digest({'request':body,'plan':lm.file_hash(runner.PLAN),'server':server})
            if reply['digest']!=digest:raise RuntimeError('registered answer request digest mismatch')
        elif not reply.get('failure_kind'):raise RuntimeError('zero attempts requires recorded context failure')
    for key,reply in reranks.items():
        if not reply.get('failure_kind') and reply['digest']!=prepared['rerank_jobs'][key]['digest']:
            raise RuntimeError('registered rerank digest mismatch')
    for (n,a),result in completed.items():
        if {r['id'] for r in result['rows']}!=set(prepared['schedule']) or len(result['rows'])!=30:
            raise RuntimeError('completed main condition identities differ')
        if a=='A' and len(result['auxiliary'])!=10:raise RuntimeError('cache-off auxiliary incomplete')


def final(p,prepared):
    completed={(n,a):lm.read(runner.LOCAL/f'{n}-{a}-completed.json')
        for n,arms in p['design']['conditions'].items() for a in arms}
    answers=ae.recover(runner.LOCAL/'answer-responses.json',runner.LOCAL/'answer-received.jsonl')
    reranks=ae.recover(runner.LOCAL/'rerank-responses.json',runner.LOCAL/'rerank-received.jsonl')
    verify_complete(p,prepared,answers,reranks,completed)
    short=lm.read(runner.LOCAL/'short-controls.json')
    contrasts={}
    for n,arms in p['design']['conditions'].items():
        for arm in [a for a in arms if a.startswith('C')]:
            for reference in ('A','B1536'):
                contrasts[f'{n}:{arm}-{reference}']=paired(completed[n,arm]['rows'],completed[n,reference]['rows'])
    contrasts['64000:C-final-norw-C-final']=paired(completed['64000','C-final-norw']['rows'],completed['64000','C-final']['rows'])
    contrasts['128000-64000:C-final-norw']=paired(completed['128000','C-final-norw']['rows'],completed['64000','C-final-norw']['rows'])
    contrasts['64000:C-final-long-short']=paired(completed['64000','C-final']['rows'],short['C-final'])
    runner.norw.verify_originals(p)
    result={'time':lm.stamp(),'provisional':False,'plan_sha256':lm.file_hash(runner.PLAN),
        'design_sha256':p['design_sha256'],'all_registered_jobs_complete':True,'original_files_hash_match':True,
        'answer_jobs':len(answers),'rerank_jobs':len(reranks),
        'HTTP_answer_attempts':sum(r['attempts'] for r in answers.values()),
        'HTTP_rerank_attempts':sum(r['attempts'] for r in reranks.values()),
        'same_digest_rerank_reuse':sum(bool(r.get('source')) for r in reranks.values()),
        'conditions':{f'{n}|{a}':{**v['summary'],'provisional':False} for (n,a),v in completed.items()},
        'short_controls':{a:aggregates(rows) for a,rows in short.items()},'contrasts':contrasts,
        'inference':'10000 whole-conversation bootstrap,seed0; exact two-sided sign, exact zero ties excluded; '
            'pointwise unadjusted; equivalence requires whole CI inside±.03; practical+.05 unchanged',
        'runtime':lm.read(runner.LOCAL/'runtime.json'),
        'context_switches':lm.read(runner.LOCAL/'context-switches.json'),
        'embedding':'original cached document/query vectors reused; no fresh embedding timed or generated',
        'scope':'same development30 from ten LoCoMo conversations; synthetic concatenated memories only; '
            'no independent generalization/semantic correctness/real long history; NORW short control unavailable',
        'saved_hashes':{name:lm.file_hash(runner.LOCAL/name) for name in
            ('answer-responses.json','answer-received.jsonl','rerank-responses.json','rerank-received.jsonl')}}
    lm.write(lm.RESULTS/'long-context-report.json',result)
    lines=['# 長い合成会話の探索評価','',result['scope'],'',result['inference'],'',
        '| 長さ | 条件 | 種類 | n | 語彙F1 | 棄権率 | 失敗数 | 回答秒/問 | rerank秒/問 | 検索秒/問 | 成分合計秒/問 | footprint GiB | swap増加 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    def f(value):return '未記録' if value is None else f'{value:.4f}'
    for key,s in result['conditions'].items():
        for kind,m in s['metrics'].items():
            c=s['components_seconds']
            lines.append(f'| {s["length"]} | {s["condition"]} | {kind} | {m["questions"]} | {m["f1"]:.4f} | '
                f'{m["abstention"]:.4f} | {m["failure"]*m["questions"]:.0f} | {f(c["answer"]["mean"])} | '
                f'{f(c["rerank"]["mean"])} | {f(c["search"]["mean"])} | {f(c["complete_sum"]["mean"])} | '
                f'{s["peak_memory"]["physical_footprint_bytes"]/1024**3:.3f} | {s["swap_affected"]} |')
    lines+=['','時間列は条件全体の平均（種類別の平均ではない）。swap増加条件は時間参考値。',
        '成分合計は別時計の和。未知rerank時間を含む問は合計未記録、known_sumは既知成分の下限。',
        'A cache-off10問は主F1から除外。対応cached10問、各会話cold/warm、分布と初期swapはJSON参照。','']
    for name,tables in contrasts.items():
        lines += [f'## {name}','']
        for kind,metrics in tables.items():
            stat=metrics['f1'];groups=list(stat['per_conversation'])
            lines += [f'### {kind}: 語彙F1',
                f'許容差±.03内: {stat["equivalent_within_0.03"]}。',
                *inference.table({'比較':{'基準':stat}},'基準',groups),'']
    lines += ['## 確認できたこと','', '固定した入力・順序・採点で全登録要求の成功／失敗を記録。元ファイルhash一致。',
        '', '## 確認できないこと','', 'NORWの同じ30問の短文対照は未計測。話者の異なる妨害による易化、少数問の広い区間を排除できない。',
        '', '## 主張してよい範囲','',result['scope']]
    (lm.RESULTS/'long-context-report.md').write_text('\n'.join(lines)+'\n')
