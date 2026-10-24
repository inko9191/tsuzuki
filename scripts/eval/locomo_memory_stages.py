#!/usr/bin/env python3
"""LoCoMo memory retrieval, stages 2-5 on the development split (stage 1: locomo_memory.py).

  register             freeze the stage plan in results/eval/memory/stages.json; commit it before any run
  run --stage 2        turn windows and context-only windows, with and without session dates (no LLM)
  run --stage 2b       BM25, dense+BM25 fusion, session cap and MMR on the stage 2 selection (no LLM)
  freeze-generation    record the local Qwen server and its model hash (generation.json); commit it next
  generate KIND        local Qwen outputs: rw, fact (from conversation text only), para, hyde, decomp
                       (from dev questions only); resumable, cached in build/eval/memory/gen/
  run --stage 3|4|5    query expansion / rewritten turns and session facts / Qwen rerank

Every stage starts from the configuration the previous stage selected (results/eval/memory/
stage<prev>-dev.json). Nothing here opens the holdout split, and there is no holdout command.
The methods themselves live in locomo_memory_methods.py.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import platform
import subprocess
import sys
import threading
import time
import urllib.error
from pathlib import Path

import numpy as np

import locomo_memory as lm
import locomo_memory_methods as mm

rp = lm.rp
from common import http_json  # noqa: E402  (scripts/stage2, put on sys.path by retrieval_public)

ROOT, LOCAL, RESULTS = lm.ROOT, lm.LOCAL, lm.RESULTS
GEN = LOCAL / 'gen'
VARIANT = 'v1-fp32-prompt'
STAGES = ('2', '2b', '3', '4', '5')
PREVIOUS = dict(zip(STAGES[1:], STAGES))
QWEN_URL = 'http://127.0.0.1:8080'
CONTEXT_TURNS = 6
CORPUS_KINDS, QUERY_GEN_KINDS = ('rw', 'fact'), ('para', 'hyde', 'decomp')

PLAN = {
    'version': 1,
    'split': 'development questions only (locomo_memory.load_dev); the holdout file is never opened',
    'embedding': f'{VARIANT}: EmbeddingGemma 300m fp32 on egemma-server, stage 1 prompts, 768 dims, 2048-token cut',
    'units': {
        'windows': {name: {'before': b, 'after': a, 'returns': 'centre turn only' if c else 'every turn in the window'}
                    for name, (b, a, c) in mm.WINDOWS.items()},
        'window_rules': 'one unit per centre turn, clipped at session boundaries; window text is its turns in '
                        'conversation order joined by newlines; returned turns ordered centre first, then by distance '
                        '(earlier first)',
        't1': 'identical to the stage 1 document (shares its embedding cache); stage 2 checks it reproduces stage 1',
        'date': '"+d" prepends "Date: <session date_time>" to the embedded text, and the payload pays that line once '
                'per session',
        'rw': 'each turn embedded through its local-LLM rewrite (stand-alone statement); returns the raw turn; '
              'a failed rewrite falls back to the raw turn text',
        'fact': 'facts the local LLM lists per session, each citing message ids of that session; returns only the '
                'cited raw turns; uncited or invalid citations are dropped',
    },
    'ranking': {
        'scope': 'units of the question\'s own conversation; dense = exact cosine; ties keep corpus order',
        'pool': mm.POOL,
        'bm25': {'k1': mm.BM25_K1, 'b': mm.BM25_B, 'idf': 'log(1 + (N - df + 0.5) / (df + 0.5)) per conversation',
                 'terms': 'NFKC, casefold, \\w+; each query term once; units with score 0 are not ranked',
                 'text': 'the embedded unit text without the embedding prompt; the raw question'},
        'fusion': f'reciprocal rank fusion (k {mm.RRF_K}) over every (unit set, ranker, query text) list of '
                  f'{mm.POOL}; a single list keeps its own scores; ties by unit order',
        'post': {f'cap{mm.SESSION_CAP}': f'at most {mm.SESSION_CAP} units per session first, the rest after in order',
                 f'mmr{mm.MMR_LAMBDA}': f'MMR over the pool, lambda {mm.MMR_LAMBDA}; relevance = list score min-max '
                                        'scaled; redundancy = max cosine of unit embeddings'},
        'distinct': 'a unit returning no turn not already returned is skipped and does not count toward k',
        'rerank': f'local Qwen sees the question and the payload of the first {mm.RERANK_CANDIDATES} units, returns '
                  f'up to {mm.RERANK_KEEP} passage numbers; those first, the rest in the original order; '
                  'invalid output keeps the original order',
    },
    'metrics': {
        '@k': f'evidence turns among the turns returned by the first k units, k in {list(mm.KS)}',
        f'@N{mm.TURN_BUDGET}': f'the first {mm.TURN_BUDGET} returned turns (equal turn budget)',
        '@T<b>': f'returned turns that fit in b payload tokens, b in {list(mm.TOKEN_BUDGETS)}: turns in returned '
                 'order, stop at the first that does not fit (equal token budget)',
        'payload': 'unique raw turns ("speaker: text" + image caption) plus date lines when dated; tokens counted '
                   'with the EmbeddingGemma 300m tokenizer, no special tokens',
        'recall': 'macro mean of found / gold evidence turns; all_evidence and hit as in stage 1',
    },
    'selection': {
        'metric': mm.PRIMARY,
        'rule': 'a configuration replaces the stage base only if its paired bootstrap 95% CI (10000 resamples of '
                f'dev questions, seed 0) of {mm.PRIMARY} minus base lies above 0; among those the larger '
                f'{mm.PRIMARY}, then larger recall@10, then fewer payload tokens@10, then config id',
        'why': 'units return different amounts of text, so recall@10 by units alone always favours the widest '
               f'window; {mm.PRIMARY} compares methods at the same small LLM input. This replaces the stage 1 '
               'selection line (recall@10 first) for stages 2-5 before any stage 2 score exists.',
    },
    'stages': {
        '2': {'base': 't1', 'configs': 'every window in units.windows, without and with +d'},
        '2b': {'bases': ['t1', 'stage 2 selection'], 'rankers': [['dense'], ['bm25'], ['dense', 'bm25']],
               'post': list(mm.POSTS), 'stage base': 'stage 2 selection'},
        '3': {'base': 'stage 2b selection', 'queries': [['orig'], ['orig', 'para'], ['orig', 'hyde'],
                                                       ['orig', 'decomp'], ['orig', 'para', 'hyde', 'decomp']],
              'query_rules': 'every generated text is one more query for every ranker; a failed generation adds none'},
        '4': {'base': 'stage 3 selection', 'units': ['base', 'rw', 'base+rw', 'base+fact', 'rw+fact', 'base+rw+fact'],
              'date': 'rw and fact take +d when the base units use dates'},
        '5': {'base': 'stage 4 selection', 'configs': ['base', 'base + Qwen rerank']},
    },
    'generation': {
        'system': mm.SYSTEM, 'prompts': mm.PROMPTS, 'temperature': 0, 'seed': 0, 'max_text_chars': mm.MAX_TEXT,
        'rw_context': f'up to {CONTEXT_TURNS} earlier turns of the same session, the session date and the turn',
        'fact_input': 'every turn of one session as "[dia_id] speaker: text", and the session date',
        'thinking': 'chat_template_kwargs enable_thinking false; any <think> block is stripped before parsing',
        'inputs': 'query prompts get the question text only; corpus prompts get conversation text only; '
                  'no answers, evidence, categories or retrieval results (rerank sees retrieved text, by design)',
        'holdout': 'no generation for holdout questions until a final configuration is selected and committed',
    },
}


# -- small helpers ----------------------------------------------------------------
def shown(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def committed(path: Path) -> bool:
    rel = shown(path)
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel], cwd=ROOT).returncode == 0
    return tracked and clean


def frozen(path: Path, expected=None, what='file') -> dict:
    if not path.exists():
        raise RuntimeError(f'{shown(path)} missing: run the {what} command first')
    value = lm.read(path)
    if not committed(path):
        raise RuntimeError(f'commit {shown(path)} before running (registration must precede scores)')
    return value


def check_plan() -> dict:
    registration = frozen(RESULTS / 'stages.json', what='register')
    if registration['plan'] != json.loads(json.dumps(PLAN)) or registration['plan_sha256'] != lm.digest(PLAN):
        raise ValueError('stage plan changed after registration')
    return registration


def register() -> None:
    path = RESULTS / 'stages.json'
    if path.exists():
        if lm.read(path)['plan_sha256'] == lm.digest(PLAN):
            print('stage plan already registered (unchanged)')
            return
        raise RuntimeError('a different stage plan is registered; do not overwrite it')
    lm.write(path, {'created_at': lm.stamp(), 'plan': PLAN, 'plan_sha256': lm.digest(PLAN)})
    print(f'registered {shown(path)}: commit it before `run --stage 2`')


def load_raw(manifest: dict) -> list[dict]:
    raw = ROOT / 'build/eval/raw/locomo/locomo10.json'
    if lm.file_hash(raw) != manifest['source']['files'][0]['sha256']:
        raise ValueError('raw LoCoMo checksum differs from the frozen split manifest')
    return lm.read(raw)


# -- local Qwen ---------------------------------------------------------------------
def model_identity(url: str, model_path: str | None) -> dict:
    props = http_json(f'{url}/props', timeout=10)
    models = http_json(f'{url}/v1/models', timeout=10)
    data = models.get('data') or models.get('models') or [{}]
    path = Path(model_path or props.get('model_path') or '')
    if not path.is_file():
        raise RuntimeError('cannot read the served model file; pass --model-path')
    stat = path.stat()
    memo_path = LOCAL / 'model-hash.json'
    memo = lm.read(memo_path) if memo_path.exists() else {}
    key = f'{path}|{stat.st_size}|{stat.st_mtime_ns}'
    if key not in memo:
        print(f'hashing {path} ({stat.st_size / 2**30:.1f} GiB)', flush=True)
        memo[key] = lm.file_hash(path)
        lm.write(memo_path, memo)
    return {'url': url, 'model_id': data[0].get('id') or data[0].get('model') or data[0].get('name'),
            'model_file': path.name, 'model_bytes': stat.st_size, 'model_sha256': memo[key],
            'n_ctx': (props.get('default_generation_settings') or {}).get('n_ctx'), 'build': props.get('build_info')}


def freeze_generation(url: str, model_path: str | None, json_schema: bool) -> None:
    check_plan()
    path = RESULTS / 'generation.json'
    if path.exists():
        raise RuntimeError('generation settings are already frozen; do not overwrite them')
    lm.write(path, {'created_at': lm.stamp(), 'server': model_identity(url, model_path),
                    'plan_generation_sha256': lm.digest(PLAN['generation']),
                    'request': {'temperature': 0, 'seed': 0, 'stream': False, 'json_schema': json_schema,
                                'chat_template_kwargs': {'enable_thinking': False}}})
    print(f'froze {shown(path)}: commit it before `generate`')


class Qwen:
    """The frozen local server; refuses to run if the served model differs from generation.json."""

    def __init__(self, model_path: str | None = None):
        self.settings = frozen(RESULTS / 'generation.json', what='freeze-generation')
        if self.settings['plan_generation_sha256'] != lm.digest(PLAN['generation']):
            raise ValueError('generation prompts changed after freezing')
        server = self.settings['server']
        now = model_identity(server['url'], model_path)
        if (now['model_sha256'], now['model_id']) != (server['model_sha256'], server['model_id']):
            raise RuntimeError('the served model differs from generation.json')
        self.url, self.model = server['url'], server['model_id']

    @classmethod
    def offline(cls) -> 'Qwen':
        """The frozen settings without contacting the server: enough to check stored outputs."""
        qwen = cls.__new__(cls)
        qwen.settings = frozen(RESULTS / 'generation.json', what='freeze-generation')
        qwen.url, qwen.model = qwen.settings['server']['url'], qwen.settings['server']['model_id']
        return qwen

    def request(self, kind: str, chat: list[dict]) -> dict:
        req = self.settings['request']
        body = {'model': self.model, 'messages': chat, 'temperature': req['temperature'], 'seed': req['seed'],
                'stream': False, 'max_tokens': mm.PROMPTS[kind]['max_tokens'], 'cache_prompt': True,
                'chat_template_kwargs': req['chat_template_kwargs']}
        if req['json_schema']:
            body['response_format'] = {'type': 'json_schema',
                                       'json_schema': {'name': kind, 'schema': mm.PROMPTS[kind]['schema']}}
        return body

    def chat(self, kind: str, chat: list[dict]) -> str:
        if kind == 'rerank':
            prompt = http_json(f'{self.url}/apply-template', {
                'messages': chat, 'chat_template_kwargs': self.settings['request']['chat_template_kwargs']})['prompt']
            tokens = http_json(f'{self.url}/tokenize', {
                'content': prompt, 'add_special': True, 'parse_special': True})['tokens']
            total = len(tokens) + mm.PROMPTS[kind]['max_tokens']
            if total > self.settings['server']['n_ctx']:
                raise RuntimeError(f'rerank input plus output reservation {total} exceeds frozen '
                                   f'context {self.settings["server"]["n_ctx"]}; stop without truncation')
        for attempt in range(4):
            try:
                reply = http_json(f'{self.url}/v1/chat/completions', self.request(kind, chat), timeout=900)
                return reply['choices'][0]['message'].get('content') or ''
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if attempt == 3:
                    raise RuntimeError(f'Qwen request failed: {error}') from error
                time.sleep(2 ** (attempt + 1))
        raise AssertionError('unreachable')

    def item_digest(self, kind: str, chat: list[dict]) -> str:
        return lm.digest({'kind': kind, 'request': self.request(kind, chat), 'model': self.settings['server']['model_sha256']})


class Store:
    """build/eval/memory/gen/<kind>.json: item id -> {digest, reply, value}; saved atomically as it fills."""

    def __init__(self, kind: str):
        self.path = GEN / f'{kind}.json'
        self.items = lm.read(self.path) if self.path.exists() else {}
        self.lock = threading.Lock()
        self.unsaved = 0

    def get(self, item_id: str, digest: str):
        entry = self.items.get(item_id)
        return entry if entry and entry['digest'] == digest else None

    def put(self, item_id: str, entry: dict) -> None:
        with self.lock:
            self.items[item_id] = entry
            self.unsaved += 1
            if self.unsaved >= 25:
                self.save()

    def save(self) -> None:
        lm.write(self.path, self.items)
        self.unsaved = 0


def generation_jobs(kind: str, dataset: dict, turns: list[dict]) -> list[tuple[str, list[dict], dict]]:
    """(item id, chat messages, parse arguments). Inputs are conversation text or the bare question only."""
    if kind in QUERY_GEN_KINDS:
        return [(q['id'], mm.messages(kind, question=q['text']), {}) for q in dataset['queries']]
    sessions = mm.sessions_of(turns)
    if kind == 'rw':
        jobs = []
        for turn in turns:
            earlier = sessions[turn['session']][max(0, turn['pos'] - CONTEXT_TURNS):turn['pos']]
            jobs.append((turn['id'], mm.messages('rw', date=turn['date'] or 'unknown',
                                                 context='\n'.join(t['text'] for t in earlier) or '(none)',
                                                 message=turn['text']), {}))
        return jobs
    if kind == 'fact':
        jobs = []
        for session, members in sessions.items():
            dia = [t['id'].split('/', 1)[1] for t in members]
            lines = '\n'.join(f'[{d}] {t["text"]}' for d, t in zip(dia, members))
            jobs.append((session, mm.messages('fact', date=members[0]['date'] or 'unknown', messages=lines),
                         {'valid_ids': dia}))
        return jobs
    raise ValueError(kind)


def generate(kind: str, workers: int, model_path: str | None, limit: int | None = None) -> None:
    check_plan()
    dataset, manifest = lm.load_dev()
    turns = mm.turn_table(dataset['corpus'], load_raw(manifest))
    qwen = Qwen(model_path)
    store = Store(kind)
    jobs = generation_jobs(kind, dataset, turns)[:limit]
    todo = [(item, chat, extra, qwen.item_digest(kind, chat)) for item, chat, extra in jobs]
    todo = [job for job in todo if store.get(job[0], job[3]) is None]
    print(f'{lm.stamp()} {kind}: {len(jobs)} items, {len(todo)} to generate with {workers} worker(s)', flush=True)
    started, done = time.perf_counter(), [0]

    def one(job):
        item, chat, extra, digest = job
        reply = qwen.chat(kind, chat)
        store.put(item, {'digest': digest, 'reply': reply, 'value': mm.parse(kind, reply, **extra)})
        done[0] += 1
        if done[0] % 50 == 0:
            rate = done[0] / (time.perf_counter() - started)
            print(f'  {kind}: {done[0]}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - done[0]) / rate / 60:.0f} min', flush=True)

    try:
        if workers == 1:
            # Don't queue the rest of a long run after a fatal API error.
            for job in todo:
                one(job)
        else:
            with ThreadPoolExecutor(workers) as pool:
                futures = [pool.submit(one, job) for job in todo]
                try:
                    for future in futures:
                        future.result()
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
    finally:
        store.save()
    summary_path = RESULTS / 'generation-outputs.json'
    summary = lm.read(summary_path) if summary_path.exists() else {}
    entries = {item: store.items[item] for item, _, _ in jobs}
    summary[kind] = {'finished_at': lm.stamp(), 'items': len(jobs), 'limit': limit,
                     'parsed': sum(e['value'] is not None for e in entries.values()),
                     'fallback': sum(e['value'] is None for e in entries.values()),
                     'replies_sha256': lm.digest({k: e['reply'] for k, e in sorted(entries.items())}),
                     'seconds_this_call': time.perf_counter() - started,
                     'server': {k: qwen.settings['server'][k] for k in ('model_id', 'model_sha256')}}
    lm.write(summary_path, summary)
    print(json.dumps(summary[kind], indent=2))


def generated(kind: str, dataset: dict, turns: list[dict], qwen) -> tuple[dict, dict]:
    """Outputs of one complete generation, keyed by item, with fallbacks applied, and their record.

    An item counts only if it was generated from exactly this prompt and the frozen request and model
    (its digest); the stored reply is parsed again here, so the parser in use is the one that scores.
    """
    path = GEN / f'{kind}.json'
    items = lm.read(path) if path.exists() else {}
    jobs = generation_jobs(kind, dataset, turns)
    stale = [item for item, chat, _ in jobs if (items.get(item) or {}).get('digest') != qwen.item_digest(kind, chat)]
    if stale:
        raise RuntimeError(f'{kind}: {len(stale)} of {len(jobs)} items not generated with the frozen settings; '
                           f'run `generate {kind}`')
    values = {item: mm.parse(kind, items[item]['reply'], **extra) for item, _, extra in jobs}
    record = {'items': len(jobs), 'fallback': sum(v is None for v in values.values()),
              'replies_sha256': lm.digest({item: items[item]['reply'] for item in sorted(values)})}
    if kind == 'rw':
        text = {t['id']: t['text'] for t in turns}
        return {i: v or text[i] for i, v in values.items()}, record
    if kind == 'fact':
        return {s: [{'fact': f['fact'], 'turns': [f'{s.split("#")[0]}/{d}' for d in f['turns']]} for f in v or []]
                for s, v in values.items()}, record
    if kind == 'hyde':
        return {i: [v] if v else [] for i, v in values.items()}, record
    return {i: v or [] for i, v in values.items()}, record


# -- stage plans ------------------------------------------------------------------
def stage_configs(stage: str, previous: dict | None) -> tuple[dict, list[dict]]:
    """(stage base, every configuration of the stage including the base)."""
    if stage == '2':
        base = mm.config(['t1'])
        return base, [mm.config([mm.unit_set(n, d)]) for d in (False, True) for n in mm.WINDOWS]
    p = previous
    if stage == '2b':
        configs = []
        for units in dict.fromkeys((('t1',), tuple(p['units']))):
            configs += [mm.config(units, r, p['queries'], post) for r in PLAN['stages']['2b']['rankers'] for post in mm.POSTS]
        return p, configs
    if stage == '3':
        return p, [mm.config(p['units'], p['rankers'], q, p['post']) for q in PLAN['stages']['3']['queries']]
    if stage == '4':
        date = mm.uses_dates(p)
        rw, fact, units = mm.unit_set('rw', date), mm.unit_set('fact', date), p['units']
        options = [units, [rw], units + [rw], units + [fact], [rw, fact], units + [rw, fact]]
        return p, [mm.config(u, p['rankers'], p['queries'], p['post']) for u in options]
    if stage == '5':
        return p, [p, mm.config(p['units'], p['rankers'], p['queries'], p['post'], rerank='qwen')]
    raise ValueError(stage)


def previous_selection(stage: str) -> dict | None:
    if stage == '2':
        return None
    path = RESULTS / f'stage{PREVIOUS[stage]}-dev.json'
    report = frozen(path, what=f'run --stage {PREVIOUS[stage]}')
    return report['selection']['selected_config']


# -- run ----------------------------------------------------------------------------
def build_units(set_names, turns, outputs) -> list[dict]:
    units = []
    for set_name in set_names:
        name, date = mm.parse_unit_set(set_name)
        if name in mm.WINDOWS:
            units += mm.window_units(turns, name, date)
        elif name == 'rw':
            units += mm.rewrite_units(turns, outputs['rw'], date)
        else:
            units += mm.fact_units(turns, outputs['fact'], date)
    return units


def reranker(qwen: Qwen, cfg: dict, units: list[dict], turns: list[dict], store: Store):
    dates = mm.uses_dates(cfg)
    turn = {t['id']: (n, t) for n, t in enumerate(turns)}
    failures, done, started = [0], [0], time.perf_counter()

    def passage(i):
        ordered = sorted(units[i]['turns'], key=lambda t: turn[t][0])  # conversation order, not centre first
        body = '\n'.join(turn[t][1]['text'] for t in ordered)
        return mm.dated(body, turn[ordered[0]][1]['date'], dates)

    def rerank(query: dict, order: list[int]) -> list[int]:
        candidates = order[:mm.RERANK_CANDIDATES]
        text = '\n\n'.join(f'[{n}] {passage(i)}' for n, i in enumerate(candidates, 1))
        chat = mm.messages('rerank', question=query['text'], passages=text)
        digest = qwen.item_digest('rerank', chat)
        entry = store.get(f'{cfg["id"]}|{query["id"]}', digest)
        if entry is None:
            reply = qwen.chat('rerank', chat)
            entry = {'digest': digest, 'reply': reply, 'value': mm.parse('rerank', reply, n_passages=len(candidates))}
            store.put(f'{cfg["id"]}|{query["id"]}', entry)
        failures[0] += entry['value'] is None
        done[0] += 1
        if done[0] % 100 == 0:
            print(f'  rerank: {done[0]} questions, {(time.perf_counter() - started) / done[0]:.1f} s/question', flush=True)
        picks = [candidates[j] for j in entry['value'] or []]
        return mm.merge_rerank(candidates, picks) + order[mm.RERANK_CANDIDATES:]

    return rerank, failures


def stage1_check(rows: list[dict]) -> dict:
    path = LOCAL / f'{VARIANT}-dev-rankings.json'
    if not path.exists():
        return {'checked': False, 'reason': 'stage 1 rankings not found'}
    before = {r['id']: r for r in lm.read(path)}
    differing = [r['id'] for r in rows if [u.split('|', 1)[1] for u in r['ranking'][:20]] != before[r['id']]['ranking']]
    worst = max(abs(r[f'recall@{k}'] - before[r['id']][f'recall@{k}']) for r in rows for k in mm.KS)
    return {'checked': True, 'queries_with_different_top20': len(differing), 'max_abs_recall_diff': worst}


def run(stage: str, port: int, model_path: str | None) -> None:
    registration = check_plan()
    dataset, manifest = lm.load_dev()
    turns = mm.turn_table(dataset['corpus'], load_raw(manifest))
    previous = previous_selection(stage)
    base, configs = stage_configs(stage, previous)
    print(f'{lm.stamp()} stage {stage}: {len(configs)} configurations, base {base["id"]}', flush=True)

    from egemma_server.tokenizer import PromptTokenizer
    tokenizer = PromptTokenizer(rp.V1_ST_DIR)
    prompts = lm.PROTOCOL['prompts']
    for side in ('query', 'document'):
        if tokenizer.prompt_for(side) != prompts[side]:
            raise ValueError('checkpoint prompts differ from the frozen protocol')

    def count(texts):
        return [len(e.ids) for e in tokenizer.tokenizer.encode_batch(texts, add_special_tokens=False)]
    turn_tokens = dict(zip([t['id'] for t in turns], count([t['text'] for t in turns])))
    first_date = {}
    for t in turns:
        first_date.setdefault(t['session'], t['date'])
    header_tokens = dict(zip(first_date, count([f'Date: {d}' for d in first_date.values()])))
    turn_session = {t['id']: t['session'] for t in turns}

    set_names = list(dict.fromkeys(s for cfg in configs for s in cfg['units']))
    kinds = list(dict.fromkeys(k for cfg in configs for k in cfg['queries'] if k != 'orig'))
    needed = [k for k in CORPUS_KINDS if any(mm.parse_unit_set(s)[0] == k for s in set_names)] + kinds
    outputs, generation = {}, {}
    for kind in needed:
        outputs[kind], generation[kind] = generated(kind, dataset, turns, Qwen.offline())

    units = build_units(set_names, turns, outputs)
    identity = lm.fingerprint(VARIANT)
    args = argparse.Namespace(port=port, model_dir=ROOT / 'models/embeddinggemma-2', v1_dir=rp.V1_ST_DIR)
    report = {'stage': stage, 'started_at': lm.stamp(), 'split': 'dev', 'holdout_evaluated': False,
              'plan_sha256': registration['plan_sha256'], 'split_manifest_sha256': lm.file_hash(RESULTS / 'split.json'),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'environment': {'platform': platform.platform(), 'python': sys.version, 'gpu_before': lm.gpu_state()},
              'previous_selection': previous, 'base': base['id'], 'embedding_artifact_sha256': identity,
              'generation': generation, 'units': {}, 'queries': {}}

    vectors = np.zeros((len(units), 768), np.float32)
    qtexts = {}
    with lm.start_backend(VARIANT, args) as (info, embed):
        report['embedding_server'] = info
        for set_name in set_names:
            index = [i for i, u in enumerate(units) if u['set'] == set_name]
            raw = [prompts['document'] + units[i]['text'] for i in index]
            fitted = rp.fit_like_st(tokenizer, raw)
            side = 'd' if set_name == 't1' else f'u-{set_name}'  # t1 is the stage 1 document: same cache entry
            vecs, timing = lm.cached_vectors(VARIANT, side, fitted, identity, embed)
            vectors[index] = vecs
            report['units'][set_name] = {'units': len(index), 'token_cuts': sum(a != b for a, b in zip(fitted, raw)),
                                         'embedding': timing}
        for kind in ['orig'] + kinds:
            owners, texts = [], []
            for qi, query in enumerate(dataset['queries']):
                for text in [query['text']] if kind == 'orig' else outputs[kind][query['id']]:
                    owners.append(qi)
                    texts.append(text)
            raw = [prompts['query'] + t for t in texts]
            fitted = rp.fit_like_st(tokenizer, raw)
            qtexts[kind] = [[] for _ in dataset['queries']]
            if not texts:  # every generation fell back: the kind adds no query
                report['queries'][kind] = {'texts': 0}
                continue
            vecs, timing = lm.cached_vectors(VARIANT, 'q-dev' if kind == 'orig' else f'q-{kind}', fitted, identity, embed)
            for owner, vec, text in zip(owners, vecs, texts):
                qtexts[kind][owner].append((vec, text))
            report['queries'][kind] = {'texts': len(texts), 'token_cuts': sum(a != b for a, b in zip(fitted, raw)),
                                       'embedding': timing}

    index = mm.Index(units, vectors)
    all_rows, report['configs'] = {}, {}
    for cfg in configs:
        rerank, failures = None, [0]
        if cfg['rerank']:
            store = Store('rerank')
            rerank, failures = reranker(Qwen(model_path), cfg, units, turns, store)
        started = time.perf_counter()
        try:
            rows = mm.score_config(index, cfg, dataset, qtexts, turn_session, turn_tokens, header_tokens,
                                   lambda q: lm.query_type(q, dataset['qrels'][q['id']]), rerank)
        finally:
            if cfg['rerank']:
                store.save()
        all_rows[cfg['id']] = rows
        report['configs'][cfg['id']] = {'config': cfg, 'seconds': time.perf_counter() - started,
                                        **({'rerank_fallback': failures[0]} if cfg['rerank'] else {}),
                                        **mm.aggregate(rows, lm.TYPES)}
        overall = report['configs'][cfg['id']]['overall']
        print(f'  {cfg["id"]:48s} R@10 {overall["recall@10"]:.4f}  {mm.PRIMARY} {overall[mm.PRIMARY]:.4f}', flush=True)
    if stage == '2':
        report['stage1_reproduction'] = stage1_check(all_rows[base['id']])
    selection = mm.select(base['id'], all_rows)
    selection['selected_config'] = report['configs'][selection['selected']]['config']
    report['selection'] = selection
    if stage in ('3', '4', '5'):
        import locomo_memory_reporting as reporting
        references = lm.read(LOCAL / 'stage2-dev-rows.json')
        report['reporting'] = reporting.comparisons(all_rows, references, base['id'])
        report['reporting']['reference_source'] = {
            'stage2_report_sha256': lm.file_hash(RESULTS / 'stage2-dev.json'),
            'stage2_rows_sha256': lm.file_hash(LOCAL / 'stage2-dev-rows.json'),
        }
    report['finished_at'] = lm.stamp()
    report['environment']['gpu_after'] = lm.gpu_state()
    lm.write(LOCAL / f'stage{stage}-dev-rows.json', all_rows)
    lm.write(RESULTS / f'stage{stage}-dev.json', report)
    write_report(report)
    print(f'selected: {selection["selected"]}')


def write_report(report: dict) -> None:
    if 'reporting' in report:
        from locomo_memory_reporting import markdown
        (RESULTS / f'stage{report["stage"]}-dev.md').write_text(markdown(report))
        return
    stage, base = report['stage'], report['base']
    vs = report['selection']['vs_base']
    lines = [f'# 外部記憶・検索評価：ステージ{stage}（開発用のみ）', '',
             f'開発用 {report["configs"][base]["overall"]["queries"]} 問。本番用は未計測。埋め込みは {VARIANT}。',
             f'基準設定は `{base}`。選択指標は {mm.PRIMARY}（返した原文を{mm.TOKEN_BUDGETS[1]}トークンまで詰めた時の Recall）。',
             'R@k は上位k単位が返した原文、R@N10 は最初の10発言、R@T は同じトークン予算での値。', '',
             f'| 設定 | R@5 | R@10 | R@20 | All@10 | R@N10 | R@T256 | R@T512 | R@T1024 | 発言数@10 | トークン@10 | Δ{mm.PRIMARY} [95% CI] |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for cid, entry in report['configs'].items():
        o = entry['overall']
        delta = '基準' if cid == base else f'{vs[cid]["mean_diff"]:+.4f} [{vs[cid]["ci95"][0]:+.4f}, {vs[cid]["ci95"][1]:+.4f}]'
        values = [o[m] for m in ('recall@5', 'recall@10', 'recall@20', 'all_evidence@10', f'recall@N{mm.TURN_BUDGET}',
                                 'recall@T256', 'recall@T512', 'recall@T1024')]
        lines.append(f'| `{cid}` | ' + ' | '.join(f'{v:.4f}' for v in values)
                     + f' | {o["payload_turns@10"]:.1f} | {o["payload_tokens@10"]:.0f} | {delta} |')
    selected = report['selection']['selected']
    lines += ['', f'**選択**：`{selected}`' + ('（基準より有意に良い設定なし）' if selected == base else ''), '',
              f'## 種類別 {mm.PRIMARY}（基準と選択）', '', '| 設定 | ' + ' | '.join(lm.TYPES) + ' |',
              '|---|' + '---:|' * len(lm.TYPES)]
    for cid in dict.fromkeys((base, selected)):
        by_type = report['configs'][cid]['by_type']
        lines.append(f'| `{cid}` | ' + ' | '.join(f'{by_type[t].get(mm.PRIMARY, float("nan")):.4f}' for t in lm.TYPES) + ' |')
    if 'stage1_reproduction' in report:
        lines += ['', f'ステージ1の再現確認（t1）：{json.dumps(report["stage1_reproduction"], ensure_ascii=False)}']
    if report['generation']:
        lines += ['', '生成物（失敗時は既定の代替）：' + ', '.join(
            f'{k} {v["items"]}件中代替{v["fallback"]}件' for k, v in report['generation'].items())]
    lines += ['', '設定IDは `単位集合/ランカー/クエリ/後処理[/rerank]`。+d はセッション日時付き。'
              'CI は質問単位の対応ありブートストラップで、会話間の相関は考慮しない。',
              '計画と規則は [stages.json](stages.json) と [docs/MEMORY_RETRIEVAL_STAGES.md](../../../docs/MEMORY_RETRIEVAL_STAGES.md)。', '']
    (RESULTS / f'stage{stage}-dev.md').write_text('\n'.join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('register')
    p = sub.add_parser('run')
    p.add_argument('--stage', choices=STAGES, required=True)
    p.add_argument('--port', type=int, default=8398)
    p.add_argument('--model-path', help='the GGUF file the Qwen server serves, if /props does not give it')
    p = sub.add_parser('freeze-generation')
    p.add_argument('--qwen-url', default=QWEN_URL)
    p.add_argument('--model-path')
    p.add_argument('--no-json-schema', action='store_true', help='if the server rejects response_format json_schema')
    p = sub.add_parser('generate')
    p.add_argument('kind', choices=CORPUS_KINDS + QUERY_GEN_KINDS)
    p.add_argument('--workers', type=int, default=1, help='parallel requests (match llama-server --parallel)')
    p.add_argument('--limit', type=int, help='only the first N items: a quick check of the output before the full run')
    p.add_argument('--model-path')
    args = parser.parse_args()
    if args.command == 'register':
        register()
    elif args.command == 'run':
        run(args.stage, args.port, args.model_path)
    elif args.command == 'freeze-generation':
        freeze_generation(args.qwen_url, args.model_path, not args.no_json_schema)
    else:
        generate(args.kind, args.workers, args.model_path, args.limit)


if __name__ == '__main__':
    main()
