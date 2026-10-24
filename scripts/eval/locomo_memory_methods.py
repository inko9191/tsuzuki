"""LoCoMo memory retrieval, stages 2-5: the retrieval methods themselves.

Pure functions only (no files, servers or models), so every rule here is unit-tested on Linux.
locomo_memory_stages.py does the I/O: embeddings, the local Qwen server, results files.

Vocabulary: a *turn* is one dialogue utterance (the stage 1 document). A *unit* is what gets
embedded and ranked (a turn, a window of turns, a rewritten turn, a fact from a session); every
unit lists the raw turns it returns, and only returned raw turns count as retrieved evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import re
import unicodedata

import numpy as np

KS = (5, 10, 20)
TOKEN_BUDGETS = (256, 512, 1024)
TURN_BUDGET = 10
POOL = 100        # candidates kept from every ranked list before fusion and post-processing
RRF_K = 60
BM25_K1, BM25_B = 1.2, 0.75
SESSION_CAP = 3
MMR_LAMBDA = 0.7
RERANK_CANDIDATES, RERANK_KEEP = 30, 10
PRIMARY = f'recall@T{TOKEN_BUDGETS[1]}'

# name -> (turns before, turns after, embedded context only: return just the centre turn)
WINDOWS = {'t1': (0, 0, False), 'c3': (1, 1, False), 'c5': (2, 2, False), 'c9': (4, 4, False),
           'p2': (1, 0, False), 'p3': (2, 0, False), 'x2': (1, 0, True), 'x3': (2, 0, True)}
GENERATED_UNITS = ('rw', 'fact')
QUERY_KINDS = ('orig', 'para', 'hyde', 'decomp')
RANKERS = ('dense', 'bm25')
POSTS = ('none', f'cap{SESSION_CAP}', f'mmr{MMR_LAMBDA}')


# -- turns and units ---------------------------------------------------------------
def turn_table(corpus: list[dict], raw: list[dict]) -> list[dict]:
    """Corpus turns with session number, position in the session and session date (from raw LoCoMo)."""
    meta = {}
    for sample in raw:
        cid, conv = sample['sample_id'], sample['conversation']
        for key in (k for k in conv if re.fullmatch(r'session_\d+', k)):
            for turn in conv[key]:
                meta[f'{cid}/{turn["dia_id"]}'] = (int(key.split('_')[1]), conv.get(f'{key}_date_time') or '')
    table, position, last = [], Counter(), {}
    for doc in corpus:
        session, date = meta[doc['id']]
        if last.get(doc['group'], 0) > session:
            raise ValueError('corpus turns are not in session order')
        last[doc['group']] = session
        table.append({'id': doc['id'], 'group': doc['group'], 'session': f'{doc["group"]}#{session}',
                      'pos': position[(doc['group'], session)], 'date': date, 'text': doc['text']})
        position[(doc['group'], session)] += 1
    return table


def sessions_of(turns: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for turn in turns:
        out[turn['session']].append(turn)
    return out


def dated(text: str, date: str, on: bool) -> str:
    return f'Date: {date}\n{text}' if on and date else text


def window_units(turns: list[dict], name: str, date: bool) -> list[dict]:
    """One unit per centre turn; the window never crosses a session boundary.

    Returned turns are ordered centre first, then by distance (earlier before later), so a turn
    budget keeps the centre and its nearest neighbours.
    """
    before, after, centre_only = WINDOWS[name]
    sessions = sessions_of(turns)
    units = []
    for turn in turns:
        session = sessions[turn['session']]
        i = turn['pos']
        span = session[max(0, i - before):i + after + 1]
        order = [turn] if centre_only else sorted(span, key=lambda t: (abs(t['pos'] - i), t['pos']))
        units.append({'id': f'{unit_set(name, date)}|{turn["id"]}', 'set': unit_set(name, date), 'group': turn['group'],
                      'session': turn['session'], 'text': dated('\n'.join(t['text'] for t in span), turn['date'], date),
                      'turns': [t['id'] for t in order]})
    return units


def rewrite_units(turns: list[dict], rewrites: dict[str, str], date: bool) -> list[dict]:
    """Each turn embedded through its stand-alone rewrite; it still returns only the raw turn."""
    name = unit_set('rw', date)
    return [{'id': f'{name}|{t["id"]}', 'set': name, 'group': t['group'], 'session': t['session'],
             'text': dated(rewrites[t['id']], t['date'], date), 'turns': [t['id']]} for t in turns]


def fact_units(turns: list[dict], facts: dict[str, list[dict]], date: bool) -> list[dict]:
    """Facts extracted per session; a fact returns exactly the raw turns it cites, never the whole session."""
    name = unit_set('fact', date)
    first = {}
    for t in turns:
        first.setdefault(t['session'], t)
    units = []
    for session, items in facts.items():
        head = first[session]
        for j, fact in enumerate(items):
            units.append({'id': f'{name}|{session}#{j}', 'set': name, 'group': head['group'], 'session': session,
                          'text': dated(fact['fact'], head['date'], date), 'turns': list(fact['turns'])})
    return units


def unit_set(name: str, date: bool) -> str:
    return name + ('+d' if date else '')


def parse_unit_set(set_name: str) -> tuple[str, bool]:
    name, date = (set_name[:-2], True) if set_name.endswith('+d') else (set_name, False)
    if name not in WINDOWS and name not in GENERATED_UNITS:
        raise ValueError(f'unknown unit set {set_name}')
    return name, date


# -- configurations ----------------------------------------------------------------
def config(units, rankers=('dense',), queries=('orig',), post='none', rerank=None) -> dict:
    units, rankers, queries = list(dict.fromkeys(units)), list(rankers), list(queries)
    for name in units:
        parse_unit_set(name)
    if not rankers or set(rankers) - set(RANKERS) or not queries or queries[0] != 'orig' or set(queries) - set(QUERY_KINDS):
        raise ValueError('invalid rankers or queries')
    if post not in POSTS or rerank not in (None, 'qwen'):
        raise ValueError('invalid post-processing')
    cid = '/'.join([','.join(units), '+'.join(rankers), '+'.join(queries), post] + ([f'rerank-{rerank}'] if rerank else []))
    return {'id': cid, 'units': units, 'rankers': rankers, 'queries': queries, 'post': post, 'rerank': rerank}


def uses_dates(cfg: dict) -> bool:
    return any(parse_unit_set(name)[1] for name in cfg['units'])


# -- rankers -----------------------------------------------------------------------
TERM = re.compile(r'\w+')


def terms(text: str) -> list[str]:
    return TERM.findall(unicodedata.normalize('NFKC', text).casefold())


class BM25:
    """Okapi BM25 (k1 1.2, b 0.75, Lucene idf) over one conversation's units; query terms counted once."""

    def __init__(self, texts: list[str]):
        docs = [Counter(terms(t)) for t in texts]
        self.n = len(docs)
        self.length = np.array([sum(d.values()) for d in docs], float)
        self.avg = float(self.length.mean()) if self.n else 0.0
        self.postings = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in doc.items():
                self.postings[term].append((i, tf))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n)
        norm = BM25_K1 * (1 - BM25_B + BM25_B * self.length / (self.avg or 1))
        for term in set(terms(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (self.n - len(posting) + .5) / (len(posting) + .5))
            for i, tf in posting:
                out[i] += idf * tf * (BM25_K1 + 1) / (tf + norm[i])
        return out


def top(indices: list[int], scores: np.ndarray, n: int = POOL) -> list[tuple[int, float]]:
    """Best n by score; ties keep corpus order."""
    order = np.argsort(-scores, kind='stable')[:n]
    return [(indices[j], float(scores[j])) for j in order]


def rrf(lists: list[list[tuple[int, float]]]) -> list[tuple[int, float]]:
    if len(lists) == 1:
        return lists[0]
    score = defaultdict(float)
    for items in lists:
        for rank, (item, _) in enumerate(items, 1):
            score[item] += 1 / (RRF_K + rank)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))


def session_cap(ranked: list[tuple[int, float]], session: list[str], cap: int = SESSION_CAP) -> list[tuple[int, float]]:
    """At most `cap` units per session come first; the rest keep their order after them."""
    kept, overflow, seen = [], [], Counter()
    for item in ranked:
        (kept if seen[session[item[0]]] < cap else overflow).append(item)
        seen[session[item[0]]] += 1
    return kept + overflow


def mmr(ranked: list[tuple[int, float]], vectors: np.ndarray, lam: float = MMR_LAMBDA) -> list[tuple[int, float]]:
    """Maximal marginal relevance over the pooled list.

    Relevance is the list's own score min-max scaled to [0, 1] (cosine for one dense list, RRF score
    after fusion); redundancy is the cosine between unit embeddings.
    """
    head, tail = ranked[:POOL], ranked[POOL:]
    if len(head) < 2:
        return ranked
    items = [i for i, _ in head]
    rel = np.array([s for _, s in head], float)
    rel = (rel - rel.min()) / (rel.max() - rel.min()) if rel.max() > rel.min() else np.ones(len(rel))
    sim = vectors[items] @ vectors[items].T
    redundancy = np.full(len(items), 0.0)
    remaining = np.ones(len(items), bool)
    out = []
    for step in range(len(items)):
        value = np.where(remaining, lam * rel - (1 - lam) * redundancy, -np.inf)
        j = int(np.argmax(value))  # first index wins ties
        out.append((items[j], float(rel[j])))
        remaining[j] = False
        redundancy = sim[j] if step == 0 else np.maximum(redundancy, sim[j])
    return out + tail


class Index:
    """All units of a run with their unit-normalized embeddings; BM25 built lazily per set and conversation."""

    def __init__(self, units: list[dict], vectors: np.ndarray):
        if len(units) != len(vectors):
            raise ValueError('one vector per unit')
        self.units, self.vectors = units, vectors
        self.session = [u['session'] for u in units]
        self.members = defaultdict(list)
        for i, unit in enumerate(units):
            self.members[(unit['set'], unit['group'])].append(i)
        self._bm25 = {}

    def ranked(self, set_name: str, group: str, ranker: str, vector: np.ndarray, text: str) -> list[tuple[int, float]]:
        members = self.members[(set_name, group)]
        if ranker == 'dense':
            return top(members, self.vectors[members] @ vector)
        key = (set_name, group)
        if key not in self._bm25:
            self._bm25[key] = BM25([self.units[i]['text'] for i in members])
        scores = self._bm25[key].scores(text)
        return [item for item in top(members, scores) if item[1] > 0]

    def rank(self, cfg: dict, group: str, queries: list[tuple[np.ndarray, str]]) -> list[tuple[int, float]]:
        """Every (unit set, ranker, query text) list fused with RRF, then the post-processing step."""
        lists = [self.ranked(s, group, r, v, t) for s in cfg['units'] for r in cfg['rankers'] for v, t in queries]
        lists = [items for items in lists if items] or [[]]
        fused = rrf(lists)
        if cfg['post'].startswith('cap'):
            return session_cap(fused, self.session)
        if cfg['post'].startswith('mmr'):
            return mmr(fused, self.vectors)
        return fused


def distinct(order: list[int], units: list[dict], limit: int = POOL) -> list[int]:
    """Units in rank order, skipping any unit that returns no turn not already returned."""
    seen, out = set(), []
    for i in order:
        new = [t for t in units[i]['turns'] if t not in seen]
        if new:
            out.append(i)
            seen.update(new)
            if len(out) == limit:
                break
    return out


def merge_rerank(candidates: list[int], chosen: list[int]) -> list[int]:
    """Reranker's picks first (valid, unique), then the remaining candidates in their original order."""
    picked = list(dict.fromkeys(i for i in chosen if i in set(candidates)))
    return picked + [i for i in candidates if i not in set(picked)]


# -- metrics -----------------------------------------------------------------------
def score(order: list[int], units: list[dict], relevant: set[str], turn_session: dict[str, str],
          turn_tokens: dict[str, int], header_tokens: dict[str, int], dates: bool) -> dict:
    """Evidence metrics of one ranked unit list (already filtered by `distinct`).

    @k counts the first k units; @N10 the first 10 returned turns; @T<b> the returned turns that fit in
    b payload tokens (turns in returned order, stopping at the first that does not fit; a session's
    date line is paid once when the configuration uses dates).
    """
    if not relevant:
        raise ValueError('Recall denominator must be positive')
    returned, seen, counted = [], set(), {}
    for n, i in enumerate(order, 1):
        for t in units[i]['turns']:
            if t not in seen:
                returned.append(t)
                seen.add(t)
        counted[n] = len(returned)
    def evidence(part, suffix):
        found = len(set(part) & relevant)
        return {f'recall@{suffix}': found / len(relevant), f'all_evidence@{suffix}': float(found == len(relevant)),
                f'hit@{suffix}': float(found > 0)}
    out = {}
    for k in KS:
        out.update(evidence(returned[:counted.get(min(k, len(order)), 0)], k))
    out.update(evidence(returned[:TURN_BUDGET], f'N{TURN_BUDGET}'))
    for budget in TOKEN_BUDGETS:
        used, paid, part = 0, set(), []
        for t in returned:
            cost = turn_tokens[t] + (header_tokens[turn_session[t]] if dates and turn_session[t] not in paid else 0)
            if used + cost > budget:
                break
            used += cost
            paid.add(turn_session[t])
            part.append(t)
        out.update(evidence(part, f'T{budget}'))
    payload = returned[:counted.get(min(10, len(order)), 0)]
    out['payload_turns@10'] = len(payload)
    out['payload_tokens@10'] = sum(turn_tokens[t] for t in payload) + (
        sum(header_tokens[s] for s in {turn_session[t] for t in payload}) if dates else 0)
    return out


def score_config(index: Index, cfg: dict, dataset: dict, qtexts: dict, turn_session: dict, turn_tokens: dict,
                 header_tokens: dict, qtype, rerank=None) -> list[dict]:
    """Per-question rows of one configuration. qtexts[kind][question index] = [(vector, text), ...]."""
    dates = uses_dates(cfg)
    rows = []
    for qi, query in enumerate(dataset['queries']):
        queries = [pair for kind in cfg['queries'] for pair in qtexts[kind][qi]]
        order = distinct([i for i, _ in index.rank(cfg, query['group'], queries)], index.units)
        if rerank is not None:
            order = distinct(rerank(query, order), index.units)
        rows.append({'id': query['id'], 'category': query['category'], 'type': qtype(query),
                     'ranking': [index.units[i]['id'] for i in order[:max(KS)]],
                     **score(order, index.units, set(dataset['qrels'][query['id']]), turn_session, turn_tokens,
                             header_tokens, dates)})
    return rows


METRICS = tuple(f'{m}@{k}' for m in ('recall', 'all_evidence', 'hit') for k in
                (*KS, f'N{TURN_BUDGET}', *(f'T{b}' for b in TOKEN_BUDGETS))) + ('payload_turns@10', 'payload_tokens@10')


def aggregate(rows: list[dict], types: tuple[str, ...]) -> dict:
    def mean(part):
        return {'queries': len(part), **{m: float(np.mean([r[m] for r in part])) for m in METRICS}} if part else {'queries': 0}
    return {'overall': mean(rows), 'by_type': {t: mean([r for r in rows if r['type'] == t]) for t in types},
            'by_category': {str(c): mean([r for r in rows if r['category'] == c]) for c in range(1, 5)}}


def paired(base: list[float], other: list[float], seed: int = 0, resamples: int = 10000) -> dict:
    """Mean of other - base over queries with a percentile bootstrap 95% CI (queries resampled together)."""
    diff = np.asarray(other, float) - np.asarray(base, float)
    rng = np.random.default_rng(seed)
    means = np.concatenate([diff[rng.integers(0, len(diff), (1000, len(diff)))].mean(axis=1)
                            for _ in range(resamples // 1000)])
    return {'mean_diff': float(diff.mean()), 'ci95': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def select(base_id: str, rows: dict[str, list[dict]]) -> dict:
    """Carry a configuration forward only when it beats the stage base on PRIMARY with a CI above 0.

    Among those: larger PRIMARY, then larger recall@10, then fewer payload tokens@10, then config id.
    Otherwise the base stays. Every comparison is reported, selected or not.
    """
    def mean(cid, metric):
        return float(np.mean([r[metric] for r in rows[cid]]))
    comparisons = {}
    for cid, part in rows.items():
        if cid != base_id:
            if [r['id'] for r in part] != [r['id'] for r in rows[base_id]]:
                raise ValueError('configurations must score the same queries in the same order')
            comparisons[cid] = paired([r[PRIMARY] for r in rows[base_id]], [r[PRIMARY] for r in part])
    better = [cid for cid, c in comparisons.items() if c['ci95'][0] > 0]
    chosen = min(better, key=lambda c: (-mean(c, PRIMARY), -mean(c, 'recall@10'), mean(c, 'payload_tokens@10'), c),
                 default=base_id)
    return {'metric': PRIMARY, 'base': base_id, 'selected': chosen, 'vs_base': comparisons}


# -- local LLM prompts and parsers ------------------------------------------------------
# Generators see only what is named: query prompts get the question alone; corpus prompts get
# conversation text alone (no question, answer, evidence or ranking).
PROMPTS = {
    'para': {
        'max_tokens': 200,
        'schema': {'type': 'object', 'properties': {'paraphrases': {'type': 'array', 'items': {'type': 'string'},
                                                                    'minItems': 2, 'maxItems': 2}},
                   'required': ['paraphrases']},
        'user': 'Question about a long chat between two friends:\n{question}\n\n'
                'Rewrite the question in two different ways that keep its exact meaning, using different words '
                'where possible. Return JSON: {{"paraphrases": ["...", "..."]}}',
    },
    'hyde': {
        'max_tokens': 200,
        'schema': {'type': 'object', 'properties': {'message': {'type': 'string'}}, 'required': ['message']},
        'user': 'Question about a long chat between two friends:\n{question}\n\n'
                'Write one short chat message, in the style of that chat, that one of the friends could have sent '
                'and that would answer this question. Invent plausible details where needed. '
                'Return JSON: {{"message": "..."}}',
    },
    'decomp': {
        'max_tokens': 200,
        'schema': {'type': 'object', 'properties': {'subquestions': {'type': 'array', 'items': {'type': 'string'},
                                                                     'minItems': 1, 'maxItems': 3}},
                   'required': ['subquestions']},
        'user': 'Question about a long chat between two friends:\n{question}\n\n'
                'Split the question into one to three simpler search questions, each asking for one fact that is '
                'needed to answer it. If it already asks for one fact, return it alone. '
                'Return JSON: {{"subquestions": ["..."]}}',
    },
    'rw': {
        'max_tokens': 200,
        'schema': {'type': 'object', 'properties': {'rewrite': {'type': 'string'}}, 'required': ['rewrite']},
        'user': 'Chat session date: {date}\n\nEarlier messages in the session:\n{context}\n\n'
                'Message to rewrite:\n{message}\n\n'
                'Rewrite the message as a self-contained statement: name the speaker, replace pronouns and vague '
                'references (it, that, there, he, she, they) with what they refer to, and turn relative times '
                '(yesterday, last week) into dates using the session date when possible. Keep every fact in the '
                'message, including what an image description shows. Do not add facts that are not in the '
                'messages. Return JSON: {{"rewrite": "..."}}',
    },
    'fact': {
        'max_tokens': 1500,
        'schema': {'type': 'object', 'properties': {'facts': {'type': 'array', 'items': {
            'type': 'object', 'properties': {'fact': {'type': 'string'}, 'ids': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['fact', 'ids']}}}, 'required': ['facts']},
        'user': 'Chat session date: {date}\n\nMessages (id, speaker: text):\n{messages}\n\n'
                'List the facts in this session that someone might later ask about: events, plans, activities, '
                'possessions, preferences, relationships, feelings and when things happened. Write each fact as one '
                'self-contained sentence that names the person, with dates resolved from the session date where '
                'possible. For each fact give the ids of the messages that state it. '
                'Return JSON: {{"facts": [{{"fact": "...", "ids": ["..."]}}]}}',
    },
    'rerank': {
        'max_tokens': 120,
        'schema': {'type': 'object', 'properties': {'ranking': {'type': 'array', 'items': {'type': 'integer'},
                                                                'maxItems': RERANK_KEEP}}, 'required': ['ranking']},
        'user': 'Question about a long chat between two friends:\n{question}\n\nRetrieved passages:\n{passages}\n\n'
                f'Choose up to {RERANK_KEEP} passages that are most useful for answering the question, most useful '
                'first. Passages that together answer it are all useful. Return JSON: {{"ranking": [numbers]}}',
    },
}
SYSTEM = 'You prepare text for a search index over personal chat logs. Answer with JSON only.'
MAX_TEXT = 600


def messages(kind: str, **fields) -> list[dict]:
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': PROMPTS[kind]['user'].format(**fields)}]


def json_object(text: str):
    """The JSON object in a reply, tolerating a thinking block or a code fence around it."""
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S).strip()
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def clean(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = ' '.join(value.split())
    return value[:MAX_TEXT] if value else None


def parse(kind: str, reply: str, valid_ids: list[str] | None = None, n_passages: int = 0):
    """Validated generator output, or None (the caller then falls back and counts the failure)."""
    value = json_object(reply)
    if value is None:
        return None
    if kind in ('para', 'decomp'):
        items = value.get('paraphrases' if kind == 'para' else 'subquestions')
        items = [clean(v) for v in items] if isinstance(items, list) else []
        items = list(dict.fromkeys(v for v in items if v))[:2 if kind == 'para' else 3]
        return items or None
    if kind in ('hyde', 'rw'):
        return clean(value.get('message' if kind == 'hyde' else 'rewrite'))
    if kind == 'fact':
        if not isinstance(value.get('facts'), list):
            return None
        allowed, facts = set(valid_ids or ()), []
        for item in value['facts'][:60]:
            if not isinstance(item, dict) or not clean(item.get('fact')) or not isinstance(item.get('ids'), list):
                continue
            ids = list(dict.fromkeys(i.strip() for i in item['ids'] if isinstance(i, str) and i.strip() in allowed))
            if ids:  # a fact that cites nothing real cannot return evidence
                facts.append({'fact': clean(item['fact']), 'turns': ids})
        return facts
    if kind == 'rerank':
        ranking = value.get('ranking')
        if not isinstance(ranking, list):
            return None
        picks = [int(v) - 1 for v in ranking if isinstance(v, (int, float)) and float(v).is_integer() and 1 <= v <= n_passages]
        return list(dict.fromkeys(picks))[:RERANK_KEEP]
    raise ValueError(kind)
