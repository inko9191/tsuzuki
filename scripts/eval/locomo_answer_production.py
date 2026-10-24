#!/usr/bin/env python3
"""One frozen heldout answer run; checkpoints and a persistent human pause gate."""
from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error

import numpy as np

import locomo_answer_eval as ae
import locomo_answer_methods as am

lm, st, mm = ae.lm, ae.st, ae.mm
LOCAL = ae.LOCAL / 'production'
PLAN = lm.RESULTS / 'answer-plan.json'
CODE = ('locomo_answer_production.py', 'locomo_answer_production_report.py',
        'locomo_memory_pause.py', 'locomo_answer_report.py', 'locomo_memory_reporting.py')


def status(phase: str, **extra) -> None:
    lm.write(LOCAL / 'status.json', {'time': lm.stamp(), 'phase': phase, **extra})


def environment() -> dict:
    servers, other = [], []
    for line in subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True).splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        argv = fields[1].split()
        if not argv:
            continue
        if Path(argv[0]).name == 'llama-server':
            servers.append({'pid': int(fields[0]), 'command': fields[1]})
        elif Path(argv[0]).name.startswith(('python', 'ollama')) and any(
                tag in fields[1] for tag in ('ComfyUI', 'ollama runner', 'mlx_lm', 'mteb_code')):
            other.append({'pid': int(fields[0]), 'command': fields[1]})
    if len(servers) != 1 or other:
        raise RuntimeError('other GPU work or multiple/missing Qwen servers; stop without changing workloads')
    return {'time': lm.stamp(), 'qwen_servers': servers, 'other_known_gpu_work': other,
            'MTEB_Code': 'user reported paused; no active known evaluation process found'}


def checked() -> dict:
    pilot = ae.checked_plan()
    record = st.frozen(PLAN)
    p = record['plan']
    if lm.digest(p) != record['plan_sha256'] or p['pilot_plan_sha256'] != pilot['plan_sha256']:
        raise RuntimeError('production or inherited pilot registration differs')
    if p['answer'] != pilot['plan']:
        raise RuntimeError('answer rules differ from the registered development pilot')
    for name, digest in p['code_sha256'].items():
        if lm.file_hash(lm.ROOT / 'scripts/eval' / name) != digest:
            raise RuntimeError(f'production implementation changed: {name}')
    for name, digest in p['public_inputs_sha256'].items():
        if lm.file_hash(lm.RESULTS / name) != digest:
            raise RuntimeError(f'production planning input changed: {name}')
    if lm.file_hash(lm.ROOT / 'docs/MEMORY_ANSWER_EVAL.md') != p['document_sha256']:
        raise RuntimeError('production protocol document changed')
    return record


def register() -> None:
    pilot = ae.checked_plan()
    report = lm.read(lm.RESULTS / 'answer-pilot-report.json')
    costs = lm.read(lm.RESULTS / 'answer-pilot-costs.json')
    if report['responses'] != 840 or costs['plan_sha256'] != pilot['plan_sha256']:
        raise RuntimeError('development measurements incomplete')
    _, manifest = lm.load_dev()  # Registration cannot read heldout questions.
    p = {'version': 1, 'answer': pilot['plan'], 'pilot_plan_sha256': pilot['plan_sha256'],
         'split': 'existing heldout302; same ten conversations; no question or rule changes',
         'questions': 302, 'responses': 302 * len(set(pilot['plan']['aliases'].values())),
         'conversation_counts': manifest['splits']['holdout']['conversations'],
         'search_context': 8192, 'answer_context': ae.CONTEXT,
         'search_generation': st.Qwen.offline().settings,
         'search_generation_calls': 302, 'search_generation_kind': 'rerank',
         'search': 'frozen final pool and corpus generations; t1/p2/final ranks shared by all C budgets',
         'flow': '65536 full-A and all rerank input preflight, native original queries; planned8192 rerank; '
                 'planned65536 all-answer preflight; all2114 answers; only then scoring',
         'failures': 'registered output/parse failures score0; no response regeneration; identical transport '
                     'requests at most4 attempts with2/4/8s waits; uncertain inflight stops for review',
         'pause': 'persistent user-pause.json; SIGINT saves received answers; Qwen remains running; '
                  'resume requires explicit human instruction; check durable journal before any repeat',
         'ordering': pilot['plan']['ordering'],
         'scope': 'LoCoMoの同じ10会話の、調整に使わなかった質問への性能',
         'interpretation': pilot['plan']['interpretation'],
         'precision_forecast': report['precision_forecast'], 'costs': costs['time_plan'],
         'large_B_C_cost': costs['additional_large_B_C'],
         'code_sha256': {name: lm.file_hash(lm.ROOT / 'scripts/eval' / name) for name in CODE},
         'document_sha256': lm.file_hash(lm.ROOT / 'docs/MEMORY_ANSWER_EVAL.md'),
         'public_inputs_sha256': {name: lm.file_hash(lm.RESULTS / name) for name in
             ('answer-pilot-report.json', 'answer-pilot-costs.json', 'answer-authorization.json')}}
    if PLAN.exists():
        if lm.read(PLAN)['plan'] != p:
            raise RuntimeError('existing production plan differs; do not overwrite it')
        return
    lm.write(PLAN, {'created_at': lm.stamp(), 'plan': p, 'plan_sha256': lm.digest(p)})
    print('production plan registered; commit before opening heldout questions', flush=True)


def load_heldout(registration: dict) -> tuple[dict, dict]:
    _, manifest = lm.load_dev()
    path = lm.LOCAL / 'holdout.json'
    if lm.file_hash(path) != manifest['split_files_sha256']['holdout']:
        raise RuntimeError('heldout file differs from the existing split')
    data = lm.read(path)
    assignments = lm.read(lm.LOCAL / 'split-map.json')
    expected = {qid for qid, split in assignments.items() if split == 'holdout'}
    if set(data['qrels']) != expected or {q['id'] for q in data['queries']} != expected:
        raise RuntimeError('heldout membership differs')
    if len(expected) != registration['plan']['questions']:
        raise RuntimeError('heldout count differs')
    data['corpus'] = lm.read(lm.LOCAL / 'corpus.json')
    return data, manifest


def rerank_chat(query: dict, order: list[int], units: list[dict], turns: list[dict], cfg: dict) -> list[dict]:
    lookup = {t['id']: (n, t) for n, t in enumerate(turns)}
    lines = []
    for n, i in enumerate(order[:mm.RERANK_CANDIDATES], 1):
        ids = sorted(units[i]['turns'], key=lambda tid: lookup[tid][0])
        body = '\n'.join(lookup[tid][1]['text'] for tid in ids)
        passage = mm.dated(body, lookup[ids[0]][1]['date'], mm.uses_dates(cfg))
        lines.append(f'[{n}] {passage}')
    return mm.messages('rerank', question=query['text'], passages='\n\n'.join(lines))


def reranked_order(order: list[int], value) -> list[int]:
    candidates = order[:mm.RERANK_CANDIDATES]
    return mm.merge_rerank(candidates, [candidates[j] for j in value or []]) + order[mm.RERANK_CANDIDATES:]


def ensure_length(length: int, output: int, context: int, margin: int = 0) -> None:
    if length + output + margin > context:
        raise RuntimeError('input exceeds frozen context; stop without truncation or exclusions')


def prepare_search() -> None:
    reg = checked()
    plan = reg['plan']['answer']
    path = LOCAL / 'search-inputs.json'
    if path.exists():
        if lm.read(path)['plan_sha256'] != reg['plan_sha256']:
            raise RuntimeError('prepared search plan differs')
        return
    now = ae.server(plan)
    gpu = environment()
    if any(slot.get('is_processing') for slot in st.http_json(now['url'] + '/slots')):
        raise RuntimeError('Qwen is busy; do not overlap native embedding work')
    status('preparing search', status='running')
    started = time.perf_counter()
    data, manifest = load_heldout(reg)
    raw = st.load_raw(manifest)
    turns = mm.turn_table(data['corpus'], raw)
    refs = {f'{s["sample_id"]}/q{i}': am.reference(qa.get('answer')) for s in raw
            for i, qa in enumerate(s['qa']) if f'{s["sample_id"]}/q{i}' in data['qrels']}
    cfgs = plan['configs']
    qwen = st.Qwen.offline()
    outputs = {kind: st.generated(kind, data, turns, qwen)[0] for kind in ('rw', 'fact')}
    names = list(dict.fromkeys(s for cfg in (cfgs['C-final'], cfgs['C-t1'], cfgs['C-p2']) for s in cfg['units']))
    units = st.build_units(names, turns, outputs)
    from egemma_server.tokenizer import PromptTokenizer
    tokenizer = PromptTokenizer(ae.rp.V1_ST_DIR)
    identity = lm.fingerprint(st.VARIANT)
    if identity != lm.read(lm.RESULTS / 'final-config.json')['embedding_artifact_sha256']:
        raise RuntimeError('native embedding artifacts differ')
    vectors = np.zeros((len(units), 768), np.float32)

    def missing(_):
        raise RuntimeError('frozen corpus vector cache missing; no regeneration or alternate backend')

    for name in names:
        indices = [i for i, u in enumerate(units) if u['set'] == name]
        texts = ae.rp.fit_like_st(tokenizer, [lm.PROTOCOL['prompts']['document'] + units[i]['text'] for i in indices])
        side = 'd' if name == 't1' else f'u-{name}'
        vectors[indices], _ = lm.cached_vectors(st.VARIANT, side, texts, identity, missing)
    texts = ae.rp.fit_like_st(tokenizer, [lm.PROTOCOL['prompts']['query'] + q['text'] for q in data['queries']])
    args = argparse.Namespace(port=8398, model_dir=lm.ROOT / 'models/embeddinggemma-2', v1_dir=ae.rp.V1_ST_DIR)
    ae.gate()
    with lm.start_backend(st.VARIANT, args) as (_, embed):
        qvecs, query_timing = lm.cached_vectors(st.VARIANT, 'q-holdout-answer', texts, identity, embed)
    index = mm.Index(units, vectors)
    by_group = defaultdict(list)
    for t in turns:
        by_group[t['group']].append(t)
    count = ae.Tokenizer(now['url'])
    queries, jobs = {}, {}
    try:
        for qi, q in enumerate(data['queries']):
            ae.gate()
            orders = {arm: mm.distinct([i for i, _ in index.rank(cfg, q['group'], [(qvecs[qi], q['text'])])], units)
                      for arm, cfg in cfgs.items()}
            chat = rerank_chat(q, orders['C-final'], units, turns, cfgs['C-final'])
            prompt = st.http_json(now['url'] + '/apply-template', {'messages': chat,
                'chat_template_kwargs': qwen.settings['request']['chat_template_kwargs']})['prompt']
            ntokens = len(st.http_json(now['url'] + '/tokenize', {'content': prompt,
                'add_special': True, 'parse_special': True})['tokens'])
            ensure_length(ntokens, mm.PROMPTS['rerank']['max_tokens'], 8192)
            body = qwen.request('rerank', chat)
            jobs[q['id']] = {'body': body, 'digest': qwen.item_digest('rerank', chat), 'input_tokens': ntokens}
            ensure_length(count.input_count(am.messages(am.render(by_group[q['group']]), '', q['text'])),
                          am.GENERATION['max_tokens'], ae.CONTEXT, ae.MARGIN)
            queries[q['id']] = {**q, 'orders': orders, 'gold': refs[q['id']],
                'evidence': list(data['qrels'][q['id']]), 'type': lm.query_type(q, data['qrels'][q['id']])}
    finally:
        count.save()
    private = {'plan_sha256': reg['plan_sha256'], 'queries': queries, 'turns': turns, 'units': units,
               'jobs': jobs, 'schedule': list(dict.fromkeys(qid for qid, _ in am.schedule(data['queries'], plan['aliases'])))}
    lm.write(path, private)
    lm.write(lm.RESULTS / 'answer-search-preparation.json', {'time': lm.stamp(), 'questions': len(queries),
        'full_A_and_all_rerank_preflight_complete': True, 'seconds': time.perf_counter() - started,
        'query_embedding': query_timing, 'inputs_sha256': lm.file_hash(path),
        'rerank_max_input': max(j['input_tokens'] for j in jobs.values()), 'holdout_answers_generated': False,
        'environment': gpu})


def recover_pending(inflight: Path, replies: dict) -> None:
    if inflight.exists():
        pending = lm.read(inflight)
        if pending['key'] not in replies or replies[pending['key']]['digest'] != pending['digest']:
            raise RuntimeError('uncertain interrupted request; stop for human review before any repeat')
        inflight.unlink()


def generate_jobs(kind: str, jobs: dict, schedule: list[str], url: str, require_timings: bool) -> dict:
    path, journal = LOCAL / f'{kind}-responses.json', LOCAL / f'{kind}-received.jsonl'
    replies = ae.recover(path, journal)
    if set(replies) - set(jobs):
        raise RuntimeError('saved responses contain unregistered jobs')
    recover_pending(LOCAL / 'inflight.json', replies)
    for key, reply in replies.items():
        if reply['digest'] != jobs[key]['digest']:
            raise RuntimeError('saved successful request differs; never overwrite it')
        if require_timings:
            ae.checked_timings(reply)
    started = time.perf_counter()
    status(f'{kind} generation', status='running', completed=len(replies), total=len(jobs))
    previous = None
    try:
        for key in schedule:
            ae.gate()
            if key in replies:
                continue
            condition = key.split('|')[0] if '|' in key else 'rerank'
            if condition != previous:
                lm.write(path, replies)
                previous = condition
            job = jobs[key]
            for attempt in range(4):
                ae.gate()
                lm.write(LOCAL / 'inflight.json', {'key': key, 'digest': job['digest'], 'kind': kind,
                    'started_at': lm.stamp(), 'attempt': attempt + 1})
                tick = time.perf_counter()
                try:
                    response = st.http_json(url + '/v1/chat/completions', job['body'], timeout=900)
                    wall = time.perf_counter() - tick
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    if attempt == 3:
                        raise
                    time.sleep(2 ** (attempt + 1))
            replies[key] = {'digest': job['digest'], 'response': response, 'wall_seconds': wall,
                'finished_at': lm.stamp(), 'attempts': attempt + 1, 'timings': response.get('timings') or {},
                'input_tokens': job['input_tokens']}
            ae.received(journal, key, replies[key])
            (LOCAL / 'inflight.json').unlink()
            if len(replies) % 25 == 0:
                lm.write(path, replies)
                status(f'{kind} generation', status='running', completed=len(replies), total=len(jobs))
                print(f'{kind} {len(replies)}/{len(jobs)} saved', flush=True)
            if require_timings:
                ae.checked_timings(replies[key])
    finally:
        lm.write(path, replies)
        runs_path = LOCAL / f'{kind}-runs.json'
        runs = lm.read(runs_path) if runs_path.exists() else []
        runs.append({'finished_at': lm.stamp(), 'seconds': time.perf_counter() - started,
                     'received': len(replies), 'total': len(jobs)})
        lm.write(runs_path, runs)
    status(f'{kind} complete', status='complete', completed=len(replies), total=len(jobs),
           seconds_this_call=time.perf_counter() - started)
    return replies


def switch_context(context: int) -> None:
    reg = checked()
    plan = reg['plan']['answer']
    now = st.model_identity(plan['model']['url'], None)  # Failure here never starts a stopped server.
    ae.server(plan, now['n_ctx'])
    if now['n_ctx'] == context:
        return
    if {now['n_ctx'], context} != {8192, ae.CONTEXT}:
        raise RuntimeError('unexpected context change required')
    if any(slot.get('is_processing') for slot in st.http_json(now['url'] + '/slots')):
        raise RuntimeError('Qwen is busy; do not restart it')
    command = lm.read(lm.RESULTS / 'answer-pilot-server.json')['new_args']
    if lm.file_hash(Path(command[command.index('-m') + 1])) != now['model_sha256']:
        raise RuntimeError('fresh model hash differs before planned restart')
    expected = command.copy()
    expected[expected.index('-c') + 1] = str(now['n_ctx'])
    # Exact argv and executable identify only the registered server.
    candidates = []
    for line in subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True).splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and fields[1] == ' '.join(expected):
            candidates.append(int(fields[0]))
    if len(candidates) != 1:
        raise RuntimeError('cannot identify exactly one registered Qwen server')
    ae.gate()
    tick = time.perf_counter()
    os.kill(candidates[0], signal.SIGTERM)
    for _ in range(100):
        try:
            reaped, _ = os.waitpid(candidates[0], os.WNOHANG)
            if reaped:
                break
        except ChildProcessError:
            pass
        try:
            os.kill(candidates[0], 0)
        except ProcessLookupError:
            break
        time.sleep(.1)
    else:
        raise RuntimeError('planned server stop did not finish; no duplicate start')
    command[command.index('-c') + 1] = str(context)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL / f'qwen-{context}.log').open('a') as log:
        proc = subprocess.Popen(command, cwd=lm.ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(60):
        if proc.poll() is not None:
            raise RuntimeError('planned server restart failed; no automatic retry')
        try:
            identity = ae.server(plan, context)
            break
        except (urllib.error.URLError, ConnectionError):
            time.sleep(1)
    else:
        raise RuntimeError('planned server restart did not become ready')
    switches = LOCAL / 'context-switches.json'
    records = lm.read(switches) if switches.exists() else []
    records.append({'time': lm.stamp(), 'old': now, 'new': identity, 'new_pid': proc.pid,
                    'args': command, 'seconds': time.perf_counter() - tick, 'fresh_hash_checked': True})
    lm.write(switches, records)
    ae.gate()


def prepare_answers() -> None:
    reg = checked()
    plan = reg['plan']['answer']
    now = ae.server(plan)
    path = LOCAL / 'answer-inputs.json'
    if path.exists():
        old = lm.read(path)
        if old['plan_sha256'] != reg['plan_sha256'] or old['server'] != now:
            raise RuntimeError('prepared answer settings differ')
        return
    search = lm.read(LOCAL / 'search-inputs.json')
    replies = ae.recover(LOCAL / 'rerank-responses.json', LOCAL / 'rerank-received.jsonl')
    if set(replies) != set(search['jobs']):
        raise RuntimeError('all rerank requests must finish before answer preparation')
    by_group = defaultdict(list)
    for t in search['turns']:
        by_group[t['group']].append(t)
    rankings, fallback = {}, 0
    for qid, q in search['queries'].items():
        if replies[qid]['digest'] != search['jobs'][qid]['digest']:
            raise RuntimeError('rerank response request differs')
        content = replies[qid]['response']['choices'][0]['message'].get('content') or ''
        value = mm.parse('rerank', content, n_passages=len(q['orders']['C-final'][:mm.RERANK_CANDIDATES]))
        fallback += value is None
        orders = {**q['orders'], 'C-final': mm.distinct(reranked_order(q['orders']['C-final'], value), search['units'])}
        rankings[qid] = {arm: list(dict.fromkeys(t for i in order for t in search['units'][i]['turns']))
                         for arm, order in orders.items()}
    count = ae.Tokenizer(now['url'])
    items = {}
    started = time.perf_counter()
    try:
        for qid, arm in am.schedule(list(search['queries'].values()), plan['aliases']):
            ae.gate()
            q = search['queries'][qid]
            payload = am.payload(arm, by_group[q['group']], rankings[qid], count.count)
            messages = am.messages(payload.pop('conversation'), payload.pop('retrieved'), q['text'])
            ntokens = count.input_count(messages)
            ensure_length(ntokens, am.GENERATION['max_tokens'], ae.CONTEXT, ae.MARGIN)
            inserted = set(payload['recent_ids'] + payload['memory_ids'])
            evidence = set(q['evidence'])
            items[f'{arm}|{qid}'] = {'id': qid, 'condition': arm, 'category': q['category'], 'type': q['type'],
                'messages': messages, 'gold': q['gold'], 'input_tokens': ntokens, **payload,
                'inserted_all': float(evidence <= inserted), 'inserted_recall': len(evidence & inserted) / len(evidence)}
    finally:
        count.save()
    prepared = {'plan_sha256': reg['plan_sha256'], 'server': now, 'items': items, 'schedule': list(items)}
    lm.write(path, prepared)
    lm.write(lm.RESULTS / 'answer-preparation.json', {'time': lm.stamp(), 'questions': len(rankings),
        'requests': len(items), 'input_max': max(i['input_tokens'] for i in items.values()),
        'common_context': ae.CONTEXT, 'seconds': time.perf_counter() - started,
        'rerank_fallback': fallback, 'inputs_sha256': lm.file_hash(path), 'all_inputs_preflight_complete': True})


def generate_answers() -> None:
    reg = checked()
    plan = reg['plan']['answer']
    now = ae.server(plan)
    gpu = environment()
    lm.write(LOCAL / 'answer-environment.json', gpu)
    prepared = lm.read(LOCAL / 'answer-inputs.json')
    if prepared['plan_sha256'] != reg['plan_sha256'] or prepared['server'] != now:
        raise RuntimeError('prepared answer plan or server differs')
    jobs = {key: {'body': ae.request(plan, item), 'input_tokens': item['input_tokens'],
        'digest': lm.digest({'request': ae.request(plan, item), 'plan': reg['plan_sha256'], 'server': now})}
        for key, item in prepared['items'].items()}
    generate_jobs('answer', jobs, prepared['schedule'], now['url'], True)


def run() -> None:
    reg = checked()
    LOCAL.mkdir(parents=True, exist_ok=True)
    pending_path = LOCAL / 'inflight.json'
    if pending_path.exists():
        kind = lm.read(pending_path).get('kind')
        if kind not in ('answer', 'rerank'):
            raise RuntimeError('unknown interrupted generation stage; stop before changing the server')
        recover_pending(pending_path, ae.recover(LOCAL / f'{kind}-responses.json', LOCAL / f'{kind}-received.jsonl'))
    import locomo_memory_pause as pause
    processes = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True)
    for line in processes.splitlines():
        fields = line.strip().split(None, 1)
        if (len(fields) == 2 and int(fields[0]) != os.getpid()
                and 'locomo_answer_production.py' in fields[1]
                and pause.script(fields[1], lm.ROOT) == Path(__file__).resolve()):
            raise RuntimeError('another production runner is active')
    prepare_search()
    search = lm.read(LOCAL / 'search-inputs.json')
    replies = ae.recover(LOCAL / 'rerank-responses.json', LOCAL / 'rerank-received.jsonl')
    if set(replies) != set(search['jobs']):
        switch_context(8192)
        now = ae.server(reg['plan']['answer'], 8192)
        lm.write(LOCAL / 'rerank-environment.json', environment())
        generate_jobs('rerank', search['jobs'], search['schedule'], now['url'], False)
    switch_context(ae.CONTEXT)
    prepare_answers()
    generate_answers()
    from locomo_answer_production_report import report
    report()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register', 'run'))
    command = parser.parse_args().command
    try:
        {'register': register, 'run': run}[command]()
    except BaseException as error:
        status('stopped', status='stopped', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
