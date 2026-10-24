"""Pure U1 rules: time-aware answer input (T) and write-time update links (V). No model/network/file I/O."""
from __future__ import annotations

from datetime import datetime
import hashlib
import math

import locomo_answer_methods as am
import locomo_memory_methods as mm

SEED = 'longmemeval-u1-v1'
QUOTA = {'knowledge-update': 10, 'temporal-reasoning': 10, 'multi-session': 4,
         'single-session-user': 1, 'single-session-assistant': 1, 'single-session-preference': 1,
         'abstention': 3}
DATE_FORMAT = '%Y/%m/%d (%a) %H:%M'
TOP_EARLIER = 2        # earlier facts per new fact, by cosine of the frozen document vectors
MAX_EARLIER = 20       # earlier facts shown in one update check
MAX_NOTES = 2          # newest replacements noted under one outdated message
ARMS = ('C-final-norw', 'S', 'T', 'V')

SYSTEM_T = am.SYSTEM + (
    '\nEach group of messages starts with the date it was written. The current date is given at the top.'
    '\nIf messages from different dates give different information about the same thing, the most recent one '
    'is current, unless the question asks about an earlier time.'
    '\nA "Later update" line under a message summarizes a newer message that changed it.'
    '\nUse the dates to work out when things happened and how much time has passed.')

UPDATE = {
    'max_tokens': 300,
    'schema': {'type': 'object', 'properties': {'updates': {'type': 'array', 'maxItems': MAX_EARLIER, 'items': {
        'type': 'object', 'properties': {'old': {'type': 'string'}, 'new': {'type': 'string'}},
        'required': ['old', 'new'], 'additionalProperties': False}}},
        'required': ['updates'], 'additionalProperties': False},
    'user': 'Earlier facts from a personal chat history:\n{earlier}\n\n'
            'New facts from a later chat session dated {date}:\n{new}\n\n'
            'An earlier fact is outdated when a new fact changes any detail in it: where someone lives or works, '
            'a number, amount or count, a status, possession, preference, habit, schedule, plan or date. The detail '
            'can be about the user or anyone else, and it can be mentioned in passing. It is not outdated when the new '
            'fact only adds information, repeats it, or is about something else. List every outdated earlier fact with '
            'the new fact that changes it. '
            'Return JSON: {{"updates": [{{"old": "E1", "new": "N1"}}]}} with an empty list if none.',
}


def window(max_input: int, output: int, choices: tuple[int, ...]) -> int | None:
    """G1 rule (d2e7f4f): smallest registered window >= ceil(1.10 * (max input + output + 128))."""
    need = math.ceil(1.10 * (max_input + output + 128))
    return next((c for c in choices if c >= need), None)


# -- sampling (metadata only) --------------------------------------------------------

def layer(meta: dict) -> str:
    if meta['question_id'].endswith('_abs'):
        return 'abstention'
    return meta['question_type']


def stratum(meta: dict) -> str:
    kind = layer(meta)
    return 'single-session' if kind.startswith('single-session-') else kind


def rank(qid: str) -> str:
    return hashlib.sha256((SEED + '|' + qid).encode()).hexdigest()


def sample(metadata: list[dict], exclude: set[str]) -> list[str]:
    """IDs, types and the seed only; refuses short strata instead of substituting."""
    cells = {k: [] for k in QUOTA}
    for meta in metadata:
        if meta['question_id'] in exclude:
            continue
        kind = layer(meta)
        if kind in cells:
            cells[kind].append(meta['question_id'])
    chosen = []
    for kind, n in QUOTA.items():
        pool = sorted(cells[kind], key=lambda q: (rank(q), q))
        if len(pool) < n:
            raise ValueError('stratum has too few questions: ' + kind)
        chosen += pool[:n]
    return sorted(chosen, key=lambda q: (rank(q), q))


# -- time order -----------------------------------------------------------------------

def when(date: str) -> datetime:
    return datetime.strptime(date, DATE_FORMAT)


def session_order(turns: list[dict]) -> list[str]:
    """Sessions by date; equal dates keep the original history position."""
    first = {}
    for n, t in enumerate(turns):
        first.setdefault(t['session'], (when(t['date']), n))
    return sorted(first, key=lambda s: first[s])


def chronological(turns: list[dict]) -> list[dict]:
    position = {s: i for i, s in enumerate(session_order(turns))}
    return sorted(turns, key=lambda t: (position[t['session']], t['pos']))


def render(turns: list[dict], notes: dict[str, list[str]] | None = None) -> str:
    """Turns in date order; a Date line opens every session block; notes follow their message."""
    position = {s: i for i, s in enumerate(session_order(turns))} if turns else {}
    lines, current = [], None
    for t in sorted(turns, key=lambda t: (position[t['session']], t['pos'])):
        if t['session'] != current:
            lines.append(f'Date: {t["date"] or "unknown"}')
            current = t['session']
        lines.append(f'[{t["id"]}] {t["text"]}')
        lines += [f'  {n}' for n in (notes or {}).get(t['id'], [])]
    return '\n'.join(lines)


def recent(turns: list[dict], budget: int, count) -> tuple[str, list[str]]:
    """Frozen recent rule (whole messages from the end, stop at the first that does not fit) on dated order."""
    chosen = []
    for turn in reversed(chronological(turns)):
        trial = [turn] + chosen
        if count(render(trial)) > budget:
            break
        chosen = trial
    return render(chosen), [t['id'] for t in chosen]


def memory(order: list[str], lookup: dict[str, dict], exclude: list[str], budget: int, count,
           notes: dict[str, list[str]] | None = None) -> tuple[str, list[str]]:
    """Frozen selection (rank order, stop at the first message that does not fit); shown in date order."""
    seen, chosen = set(exclude), []
    for tid in order:
        if tid in seen:
            continue
        trial = chosen + [lookup[tid]]
        if count(render(trial, notes)) > budget:
            break
        chosen = trial
        seen.add(tid)
    return render(chosen, notes), [t['id'] for t in chosen]


def messages(conversation: str, retrieved: str, question: str, question_date: str) -> list[dict]:
    return [{'role': 'system', 'content': SYSTEM_T},
            {'role': 'user', 'content': f'Current date: {question_date}\n\n'
             f'Conversation (most recent messages, in date order):\n{conversation or "none"}\n\n'
             f'Retrieved conversation excerpts (in date order):\n{retrieved or "none"}\n\nQuestion:\n{question}'}]


# -- write-time update links ----------------------------------------------------------

def fact_table(turns: list[dict], facts: dict[str, list[dict]]) -> list[dict]:
    """Facts in date order of their sessions, keyed like the frozen fact units (`session#j`)."""
    dates = {t['session']: t['date'] for t in turns}
    order = session_order(turns)
    table = []
    for rank_, session in enumerate(order):
        for j, fact in enumerate(facts.get(session, [])):
            table.append({'key': f'{session}#{j}', 'session': session, 'rank': rank_, 'date': dates[session],
                          'text': fact['fact'], 'turns': list(fact['turns'])})
    return table


def update_jobs(table: list[dict], vectors: dict[str, list[float]]) -> list[dict]:
    """One check per later session: its facts against the most similar facts of earlier sessions."""
    import numpy as np
    by_session = {}
    for f in table:
        by_session.setdefault(f['rank'], []).append(f)
    jobs = []
    for r in sorted(by_session):
        new = by_session[r]
        earlier = [f for f in table if f['rank'] < r]
        if not earlier:
            continue
        E = np.asarray([vectors[f['key']] for f in earlier], np.float64)
        best = {}
        for f in new:
            sims = E @ np.asarray(vectors[f['key']], np.float64)
            for i in np.argsort(-sims, kind='stable')[:TOP_EARLIER]:
                best[i] = max(best.get(i, -2.), float(sims[i]))
        picked = sorted(best, key=lambda i: (-best[i], i))[:MAX_EARLIER]
        picked = sorted(picked, key=lambda i: (earlier[i]['rank'], i))
        jobs.append({'session': new[0]['session'], 'date': new[0]['date'],
                     'earlier': [earlier[i]['key'] for i in picked], 'new': [f['key'] for f in new]})
    return jobs


def update_messages(job: dict, facts: dict[str, dict]) -> list[dict]:
    earlier = '\n'.join(f'[E{i}] ({facts[k]["date"]}) {facts[k]["text"]}' for i, k in enumerate(job['earlier'], 1))
    new = '\n'.join(f'[N{i}] {facts[k]["text"]}' for i, k in enumerate(job['new'], 1))
    system = 'You prepare text for a search index over personal chat logs. Answer with JSON only.'
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': UPDATE['user'].format(earlier=earlier, date=job['date'], new=new)}]


def parse_updates(text: str, job: dict) -> list[dict] | None:
    """Valid label pairs only; None when the reply is not the registered JSON."""
    value = mm.json_object(text)
    if not isinstance(value, dict) or not isinstance(value.get('updates'), list):
        return None
    links, seen = [], set()
    for item in value['updates']:
        if not isinstance(item, dict):
            continue
        old, new = str(item.get('old', '')).strip(), str(item.get('new', '')).strip()
        if not (old[:1] == 'E' and new[:1] == 'N' and old[1:].isdigit() and new[1:].isdigit()):
            continue
        i, j = int(old[1:]), int(new[1:])
        if not (1 <= i <= len(job['earlier']) and 1 <= j <= len(job['new'])):
            continue
        pair = (job['earlier'][i-1], job['new'][j-1])
        if pair not in seen:
            seen.add(pair)
            links.append({'old': pair[0], 'new': pair[1]})
    return links


def replacements(links: list[dict], facts: dict[str, dict]) -> dict[str, list[str]]:
    """Each outdated fact -> its newest replacements, following chains to facts that are not outdated."""
    nxt = {}
    for link in links:
        nxt.setdefault(link['old'], []).append(link['new'])
    result = {}
    for old in nxt:
        frontier, seen, final = list(nxt[old]), {old}, []
        while frontier:
            k = frontier.pop(0)
            if k in seen:
                continue
            seen.add(k)
            if k in nxt:
                frontier += nxt[k]
            else:
                final.append(k)
        final = sorted(set(final), key=lambda k: (-facts[k]['rank'], k))[:MAX_NOTES]
        if final:
            result[old] = final
    return result


def versioned(raw_ids: list[str], table: list[dict], links: list[dict]) -> tuple[list[str], dict[str, list[str]]]:
    """Rank order with each outdated message followed by the messages of its replacements, plus notes."""
    facts = {f['key']: f for f in table}
    newer = replacements(links, facts)
    by_turn = {}
    for f in table:
        for tid in f['turns']:
            by_turn.setdefault(tid, []).append(f['key'])
    notes, order, emitted = {}, [], set()
    for tid in raw_ids:
        if tid not in emitted:
            order.append(tid); emitted.add(tid)
        repl = list(dict.fromkeys(r for k in by_turn.get(tid, []) for r in newer.get(k, [])))
        if not repl:
            continue
        notes[tid] = [f'Later update ({facts[r]["date"]}): {facts[r]["text"]}' for r in repl]
        for r in repl:
            for t in facts[r]['turns']:
                if t not in emitted:
                    order.append(t); emitted.add(t)
    return order, notes


def payload(arm: str, turns: list[dict], raw_ids: list[str], count, table=None, links=None) -> dict:
    """S, T and V answer inputs: recent 1024 + memory 512 (frozen budgets) in date order; dated notes for V.

    S and T share the same text; they differ only in the chat template (`messages` for T, frozen for S)."""
    if arm not in ('S', 'T', 'V'):
        raise ValueError(arm)
    lookup = {t['id']: t for t in turns}
    conversation, recent_ids = recent(turns, 1024, count)
    order, notes = (raw_ids, None) if arm in ('S', 'T') else versioned(raw_ids, table, links)
    retrieved, memory_ids = memory(order, lookup, recent_ids, 512, count, notes)
    shown = {k: v for k, v in (notes or {}).items() if k in memory_ids}
    return {'conversation': conversation, 'retrieved': retrieved, 'recent_ids': recent_ids,
            'memory_ids': memory_ids, 'conversation_tokens': count(conversation),
            'memory_tokens': count(retrieved), 'notes_shown': len(shown),
            'inserted_ids': [t for t in order if t not in raw_ids and t in memory_ids]}
