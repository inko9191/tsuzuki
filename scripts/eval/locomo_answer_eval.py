#!/usr/bin/env python3
"""Separately registered answer evaluation. Pilot commands never open holdout.json."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error

import numpy as np

import locomo_memory_stages as st
import locomo_answer_methods as am

lm, mm, rp = st.lm, st.mm, st.rp
LOCAL = lm.LOCAL / 'answer'
PLAN_PATH = lm.RESULTS / 'answer-pilot-plan.json'
PAUSE = lm.LOCAL / 'user-pause.json'
CONTEXT = 65536
MARGIN = 128
CODE = ['locomo_answer_methods.py', 'locomo_answer_eval.py']


def gate() -> None:
    if PAUSE.exists():
        raise RuntimeError('explicit user pause is active; do not automatically resume')


def status(phase: str, **extra) -> None:
    lm.write(LOCAL / 'status.json', {'time': lm.stamp(), 'phase': phase, **extra})


def configs() -> dict:
    frozen = st.frozen(lm.RESULTS / 'final-config.json')
    for name, expected in frozen['code_files_sha256'].items():
        if lm.file_hash(lm.ROOT / 'scripts/eval' / name) != expected:
            raise RuntimeError(f'frozen retrieval code changed: {name}')
    for name, expected in frozen['protocol_files_sha256'].items():
        if lm.file_hash(lm.RESULTS / name) != expected:
            raise RuntimeError(f'frozen retrieval protocol changed: {name}')
    return {'C-t1': mm.config(['t1']), 'C-p2': mm.config(['p2']),
            'C-final': frozen['selected_config']}


def dataset_and_pilot():
    data, manifest = lm.load_dev()
    types = {q['id']: lm.query_type(q, data['qrels'][q['id']]) for q in data['queries']}
    chosen = am.pilot(data['queries'], types)
    return data, manifest, chosen, types


def register() -> None:
    gate()
    data, manifest, chosen, types = dataset_and_pilot()
    cfg = configs()
    plan = {'version': 1, 'split': 'dev-pilot-only; no holdout execution authorized by this file',
            'question_ids': chosen, 'questions': 120,
            'pilot_selection': '10 conversations x 3 proxy types x 4; SHA256(locomo-answer-pilot-v2 + id)',
            'types': {qid: types[qid] for qid in chosen},
            'configs': cfg, 'aliases': am.aliases(cfg), 'conditions': am.CONDITIONS,
            'system': am.SYSTEM, 'schema': am.SCHEMA, 'generation': am.GENERATION,
            'context': CONTEXT, 'output_margin': MARGIN,
            'budgets': {'B1536': [1536, 0], 'C-t1': [1024, 512], 'C-p2': [1024, 512],
                       'C-final': [1024, 512], 'B8192': [8192, 0], 'C8192-final': [1024, 7168]},
            'ordering': 'SHA256(locomo-answer-v2 + group/id); conversation, rotated unique condition, question',
            'scoring': 'NFKC/casefold, ASCII punctuation to space, remove a/an/the, multiset token F1; EM; '
                       'canonical Insufficient evidence scores zero; parse/unfinished zero; no quality retry',
            'interpretation': {'equivalence_margin': .03, 'practical_improvement': .05,
                               'no_equivalence_claim_if_ci_not_inside_margin': True},
            'split_sha256': lm.file_hash(lm.RESULTS / 'split.json'),
            'search_freeze_sha256': lm.file_hash(lm.RESULTS / 'final-config.json'),
            'search_freeze_commit': subprocess.check_output(
                ['git', 'log', '-1', '--format=%H', '--', 'results/eval/memory/final-config.json'],
                cwd=lm.ROOT, text=True).strip(),
            'source_sha256': manifest['source']['files'][0]['sha256'],
            'draft_sha256': lm.file_hash(lm.ROOT / 'docs/MEMORY_ANSWER_EVAL_DRAFT.md'),
            'model': st.Qwen.offline().settings['server'],
            'code_sha256': {name: lm.file_hash(lm.ROOT / 'scripts/eval' / name) for name in CODE},
            'checkpoint': '25 successful responses, condition switches, finally; durable inflight log; user pause gate',
            'forecast': '500 within-conversation/type paired simulations; development type weights; '
                        '10000 whole-conversation bootstrap seed0; centered shifts 0:.005:.30',
            'holdout_evaluated': False}
    record = {'created_at': lm.stamp(), 'plan': plan, 'plan_sha256': lm.digest(plan)}
    if PLAN_PATH.exists():
        if lm.read(PLAN_PATH)['plan'] != plan:
            raise RuntimeError('pilot registration already exists with different conditions')
        return
    lm.write(PLAN_PATH, record)
    print('pilot registered; commit before preparing or generating', flush=True)


def checked_plan() -> dict:
    gate()
    record = st.frozen(PLAN_PATH)
    plan = record['plan']
    if lm.digest(plan) != record['plan_sha256']:
        raise RuntimeError('pilot plan digest mismatch')
    for name, digest in plan['code_sha256'].items():
        if lm.file_hash(lm.ROOT / 'scripts/eval' / name) != digest:
            raise RuntimeError(f'pilot implementation changed: {name}')
    if configs() != plan['configs'] or lm.file_hash(lm.RESULTS / 'split.json') != plan['split_sha256']:
        raise RuntimeError('frozen search or split changed')
    if lm.file_hash(lm.ROOT / 'docs/MEMORY_ANSWER_EVAL_DRAFT.md') != plan['draft_sha256']:
        raise RuntimeError('pilot draft changed')
    return record


def server(plan: dict, context: int = CONTEXT) -> dict:
    now = st.model_identity(plan['model']['url'], None)
    for key in ('model_sha256', 'model_id', 'build'):
        if now[key] != plan['model'][key]:
            raise RuntimeError(f'server identity differs: {key}')
    if now['n_ctx'] != context:
        raise RuntimeError(f'server context must be {context}; currently {now["n_ctx"]}')
    return now


class Tokenizer:
    """Only CPU tokenizer endpoints. Cache text hashes, never public licensed text."""

    def __init__(self, url: str):
        self.url = url
        self.path = LOCAL / 'token-counts.json'
        self.counts = lm.read(self.path) if self.path.exists() else {}

    def count(self, text: str) -> int:
        if not text:
            return 0
        key = lm.digest(text)
        if key not in self.counts:
            gate()
            self.counts[key] = len(st.http_json(self.url + '/tokenize', {
                'content': text, 'add_special': False, 'parse_special': False})['tokens'])
        return self.counts[key]

    def input_count(self, chat: list[dict]) -> int:
        gate()
        prompt = st.http_json(self.url + '/apply-template', {'messages': chat,
             'chat_template_kwargs': am.GENERATION['chat_template_kwargs']})['prompt']
        return len(st.http_json(self.url + '/tokenize', {'content': prompt, 'add_special': True,
                                                       'parse_special': True})['tokens'])

    def save(self):
        lm.write(self.path, self.counts)


def dev_rankings(data: dict, manifest: dict, chosen: list[str], cfgs: dict) -> tuple[list[dict], dict]:
    """Full registered pool reconstructed on CPU, with cached rerank only; never starts a GPU backend."""
    gate()
    turns = mm.turn_table(data['corpus'], st.load_raw(manifest))
    outputs = {kind: st.generated(kind, data, turns, st.Qwen.offline())[0] for kind in ('rw', 'fact')}
    set_names = list(dict.fromkeys(s for cfg in [cfgs['C-final'], cfgs['C-t1'], cfgs['C-p2']]
                                   for s in cfg['units']))
    units = st.build_units(set_names, turns, outputs)
    from egemma_server.tokenizer import PromptTokenizer
    tokenizer = PromptTokenizer(rp.V1_ST_DIR)
    identity = lm.fingerprint(st.VARIANT)
    if identity != lm.read(lm.RESULTS / 'final-config.json')['embedding_artifact_sha256']:
        raise RuntimeError('embedding artifacts changed after search freeze')

    def missing(_):
        raise RuntimeError('required native vector cache missing; stop without changing embedding backend')

    vectors = np.zeros((len(units), 768), np.float32)
    for name in set_names:
        indices = [i for i, u in enumerate(units) if u['set'] == name]
        texts = rp.fit_like_st(tokenizer, [lm.PROTOCOL['prompts']['document'] + units[i]['text'] for i in indices])
        side = 'd' if name == 't1' else f'u-{name}'
        vectors[indices], _ = lm.cached_vectors(st.VARIANT, side, texts, identity, missing)
    texts = rp.fit_like_st(tokenizer, [lm.PROTOCOL['prompts']['query'] + q['text'] for q in data['queries']])
    qvecs, _ = lm.cached_vectors(st.VARIANT, 'q-dev', texts, identity, missing)
    index = mm.Index(units, vectors)
    qwen = st.Qwen.offline()
    qwen.chat = lambda *args: missing(None)
    rerank, _ = st.reranker(qwen, cfgs['C-final'], units, turns, st.Store('rerank'))
    before = lm.read(lm.LOCAL / 'stage2-dev-rows.json')
    before.update(lm.read(lm.LOCAL / 'stage5-dev-rows.json'))
    references = {cid: {r['id']: r['ranking'] for r in rows} for cid, rows in before.items()}
    rankings = {}
    for i, q in enumerate(data['queries']):
        if q['id'] not in set(chosen):
            continue
        rankings[q['id']] = {}
        for arm, cfg in cfgs.items():
            gate()
            order = mm.distinct([u for u, _ in index.rank(cfg, q['group'], [(qvecs[i], q['text'])])], units)
            if cfg['rerank']:
                order = mm.distinct(rerank(q, order), units)
            if [units[u]['id'] for u in order[:20]] != references[cfg['id']][q['id']]:
                raise RuntimeError('reconstructed top20 differs from frozen search results')
            rankings[q['id']][arm] = list(dict.fromkeys(t for u in order for t in units[u]['turns']))
    return turns, rankings


def prepare() -> None:
    registered = checked_plan()
    plan = registered['plan']
    now = server(plan)
    data, manifest, chosen, types = dataset_and_pilot()
    if chosen != plan['question_ids']:
        raise RuntimeError('pilot membership changed')
    path = LOCAL / 'pilot-inputs.json'
    if path.exists():
        old = lm.read(path)
        if old['plan_sha256'] != registered['plan_sha256']:
            raise RuntimeError('prepared pilot plan mismatch')
        print('pilot input cache already complete', flush=True)
        return
    started = time.perf_counter()
    status('preparing pilot', status='running', holdout_evaluated=False)
    turns, rankings = dev_rankings(data, manifest, chosen, plan['configs'])
    by_group = defaultdict(list)
    for t in turns:
        by_group[t['group']].append(t)
    lookup = {q['id']: q for q in data['queries'] if q['id'] in set(chosen)}
    raw = st.load_raw(manifest)
    refs = {f'{sample["sample_id"]}/q{i}': am.reference(qa.get('answer'))
            for sample in raw for i, qa in enumerate(sample['qa'])
            if f'{sample["sample_id"]}/q{i}' in lookup}
    tokenizer = Tokenizer(now['url'])
    items = {}
    try:
        for n, (qid, arm) in enumerate(am.schedule(list(lookup.values()), plan['aliases']), 1):
            gate()
            q = lookup[qid]
            p = am.payload(arm, by_group[q['group']], rankings[qid], tokenizer.count)
            chat = am.messages(p.pop('conversation'), p.pop('retrieved'), q['text'])
            ntokens = tokenizer.input_count(chat)
            if ntokens + am.GENERATION['max_tokens'] + MARGIN > CONTEXT:
                raise RuntimeError('full prompt exceeds common context; no truncation permitted')
            evidence = set(data['qrels'][qid])
            inserted = set(p['recent_ids'] + p['memory_ids'])
            items[f'{arm}|{qid}'] = {'id': qid, 'condition': arm, 'category': q['category'],
                'type': types[qid], 'messages': chat, 'input_tokens': ntokens,
                'gold': refs[qid], 'inserted_all': float(evidence <= inserted),
                'inserted_recall': len(evidence & inserted) / len(evidence), **p}
            if n % 100 == 0:
                print(f'prepared {n} pilot inputs (no inference)', flush=True)
        private = {'plan_sha256': registered['plan_sha256'], 'server': now, 'items': items,
                   'schedule': list(items), 'prepared_at': lm.stamp()}
        lm.write(path, private)
        public = {'time': lm.stamp(), 'holdout_evaluated': False, 'questions': 120, 'requests': len(items),
                  'input_max': max(i['input_tokens'] for i in items.values()), 'common_context': CONTEXT,
                  'seconds': time.perf_counter() - started, 'inputs_sha256': lm.file_hash(path),
                  'ranking_reproduction': 'every pilot top20 equals frozen t1/p2/final; full same registered pool'}
        lm.write(lm.RESULTS / 'answer-pilot-preparation.json', public)
        status('pilot prepared', status='ready', **public)
    finally:
        tokenizer.save()


def request(plan: dict, item: dict) -> dict:
    return {'model': plan['model']['model_id'], 'messages': item['messages'], **am.GENERATION,
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': am.SCHEMA}}}


def recover(path: Path, journal: Path) -> dict:
    replies = lm.read(path) if path.exists() else {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            entry = json.loads(line)
            key, reply = entry['key'], entry['reply']
            if key in replies and replies[key] != reply:
                raise RuntimeError('successful-response journal conflicts with checkpoint')
            replies[key] = reply
    return replies


def received(journal: Path, key: str, reply: dict) -> None:
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open('a') as stream:
        stream.write(json.dumps({'key': key, 'reply': reply}, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def checked_timings(reply: dict) -> dict:
    timings = reply.get('timings') or {}
    if not all(k in timings for k in ('cache_n', 'prompt_n', 'prompt_ms', 'predicted_n', 'predicted_ms')):
        raise RuntimeError('successful response lacks required timing fields; saved, no further generation')
    return timings


def generate() -> None:
    registered = checked_plan()
    plan = registered['plan']
    now = server(plan)
    prepared = lm.read(LOCAL / 'pilot-inputs.json')
    if prepared['plan_sha256'] != registered['plan_sha256']:
        raise RuntimeError('prepared pilot digest differs')
    path = LOCAL / 'pilot-responses.json'
    journal = LOCAL / 'pilot-received.jsonl'
    replies = recover(path, journal)
    for reply in replies.values():
        checked_timings(reply)  # A resumed process must not silently skip a prior timing failure.
    inflight = LOCAL / 'inflight.json'
    if inflight.exists():
        pending = lm.read(inflight)
        if pending['key'] in replies and replies[pending['key']]['digest'] == pending['digest']:
            inflight.unlink()  # A response was durably received before the process ended.
        else:
            raise RuntimeError('uncertain interrupted request; stop for review before repeating it')
    last_condition, completed = None, 0
    started = time.perf_counter()
    status('pilot generation', status='running', completed=len(replies), total=len(prepared['items']))
    try:
        for key in prepared['schedule']:
            gate()
            item = prepared['items'][key]
            body = request(plan, item)
            digest = lm.digest({'request': body, 'plan': registered['plan_sha256'], 'server': now})
            if key in replies:
                if replies[key]['digest'] != digest:
                    raise RuntimeError('saved response has different inputs or model; never overwrite it')
                continue
            if item['condition'] != last_condition:
                lm.write(path, replies)
                last_condition = item['condition']
            for attempt in range(4):
                gate()
                lm.write(inflight, {'key': key, 'digest': digest, 'started_at': lm.stamp(), 'attempt': attempt + 1})
                t0 = time.perf_counter()
                try:
                    response = st.http_json(now['url'] + '/v1/chat/completions', body, timeout=900)
                    wall = time.perf_counter() - t0
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    if attempt == 3:
                        raise
                    time.sleep(2 ** (attempt + 1))
            timings = response.get('timings') or {}
            # Save a successful response before any diagnostic can stop the experiment.
            replies[key] = {'digest': digest, 'response': response, 'wall_seconds': wall,
                            'finished_at': lm.stamp(), 'attempts': attempt + 1,
                            'timings': timings, 'input_tokens': item['input_tokens']}
            received(journal, key, replies[key])
            completed += 1
            if completed % 25 == 0:
                lm.write(path, replies)
                status('pilot generation', status='running', completed=len(replies), total=len(prepared['items']))
                print(f'pilot {len(replies)}/{len(prepared["items"])} saved', flush=True)
            inflight.unlink()
            checked_timings(replies[key])
    finally:
        lm.write(path, replies)
    status('pilot complete', status='complete', completed=len(replies), total=len(prepared['items']),
           seconds_this_call=time.perf_counter() - started, holdout_evaluated=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['register', 'prepare-pilot', 'generate-pilot'])
    args = p.parse_args()
    try:
        {'register': register, 'prepare-pilot': prepare, 'generate-pilot': generate}[args.command]()
    except BaseException as error:
        status('stopped', status='stopped', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
