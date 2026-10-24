"""Conformance fixtures: run the evaluated Python pipeline on synthetic data; the TS port must match exactly.

CPU only, no models or servers. Usage (repo root): .venv/bin/python -I dsh-plugins/packages/tsuzuki/test/make_fixtures.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'scripts/eval'))
import locomo_answer_methods as am  # noqa: E402
import locomo_answer_production as prod  # noqa: E402
import locomo_memory_methods as mm  # noqa: E402
import longmemeval_u1_methods as um  # noqa: E402

OUT = Path(__file__).resolve().parent/'fixtures/conformance.json'


def turns_fixture():
    # Sessions listed out of date order on purpose (as in LongMemEval).
    sessions = [('g#1', '2023/05/20 (Sat) 10:00', ['user: I moved to Kyoto last week.', 'assistant: How is it?',
                                                    'user: Great, my commute is 20 minutes now.']),
                ('g#2', '2023/05/01 (Mon) 09:00', ['user: I live in Osaka and take the train for an hour.',
                                                    'assistant: That is a long commute.']),
                ('g#3', '2023/05/10 (Wed) 09:00', ['user: My cat Tama turned 3. 🐈 She loves tuna.',
                                                    'assistant: Happy birthday to Tama!', 'user: Thanks!'])]
    turns = []
    for session, date, texts in sessions:
        for pos, text in enumerate(texts):
            turns.append({'id': f'g/D{session[2:]}:{pos + 1}', 'group': 'g', 'session': session, 'pos': pos,
                          'date': date, 'text': text})
    return turns


def vectors(n, rng, dim=32):
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    turns = turns_fixture()
    facts = {'g#2': [{'fact': 'The user lives in Osaka.', 'turns': ['g/D2:1']},
                     {'fact': 'The user commutes one hour by train.', 'turns': ['g/D2:1', 'g/D2:2']}],
             'g#1': [{'fact': 'The user moved to Kyoto around May 13, 2023.', 'turns': ['g/D1:1']},
                     {'fact': 'The user commutes 20 minutes.', 'turns': ['g/D1:3']}],
             'g#3': [{'fact': 'The user has a cat named Tama who turned 3.', 'turns': ['g/D3:1']}]}
    units = mm.window_units(turns, 'p2', False) + mm.fact_units(turns, facts, False)

    cases = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        vec = vectors(len(units), rng)
        query = vectors(1, rng)[0]
        scores = vec @ query
        gaps = np.diff(np.sort(scores))
        if gaps.min() < 1e-4:
            continue
        index = mm.Index(units, vec)
        cfg = mm.config(['p2', 'fact'], rerank='qwen')
        fused = index.rank(cfg, 'g', [(query, 'q')])
        order = mm.distinct([i for i, _ in fused], units)
        chat = prod.rerank_chat({'text': 'Where do I live now?'}, order, units, turns, cfg)
        replies = ['{"ranking": [3, 1, 3, 99, 0, 2.0, true]}', 'not json', '<think>x</think>{"ranking": [2]}',
                   '```json\n{"ranking": []}\n```', '{"ranking": "1"}']
        reranks = []
        for reply in replies:
            value = mm.parse('rerank', reply, n_passages=min(len(order), mm.RERANK_CANDIDATES))
            final = mm.distinct(prod.reranked_order(order, value), units)
            raw = list(dict.fromkeys(t for i in final for t in units[i]['turns']))
            lookup = {t['id']: t for t in turns}
            words = lambda text: len(text.split())  # noqa: E731
            frozen = am.memory(raw, lookup, ['g/D1:3'], 25, words)
            dated = um.memory(raw, lookup, ['g/D1:3'], 25, words)
            reranks.append({'reply': reply, 'picks': value, 'final': final, 'raw': raw,
                            'frozen': {'text': frozen[0], 'ids': frozen[1]}, 'dated': {'text': dated[0], 'ids': dated[1]}})
        cases.append({'seed': seed, 'vectors': vec.tolist(), 'query': query.tolist(), 'fused': fused, 'order': order,
                      'rerank_messages': chat, 'reranks': reranks})
        if len(cases) == 12:
            break

    fact_replies = [
        '{"facts": [{"fact": "  The user   lives\\u3000in Osaka. ", "ids": ["D2:1", " D2:1 ", "D9:9", 3]},'
        ' {"fact": "", "ids": ["D2:1"]}, {"fact": "No ids", "ids": []}, "junk", {"fact": "x", "ids": "D2:1"}]}',
        '<think>plan</think>```json\n{"facts": [{"fact": "' + ('🐈' * 700) + '", "ids": ["D2:2"]}]}\n```',
        '{"facts": "none"}', 'garbage', '{"facts": []}',
    ]
    members = [t for t in turns if t['session'] == 'g#2']
    dia = [t['id'].split('/', 1)[1] for t in members]
    fact_chat = mm.messages('fact', date=members[0]['date'], messages='\n'.join(f'[{d}] {t["text"]}' for d, t in zip(dia, members)))
    fixtures = {
        'turns': turns, 'facts': facts, 'units': units,
        'fact_messages': fact_chat, 'fact_valid_ids': dia,
        'fact_parses': [{'reply': r, 'value': mm.parse('fact', r, valid_ids=dia)} for r in fact_replies],
        'clean': [{'input': s, 'value': mm.clean(s)} for s in ['a  b\tc\n', '\x1cx\x1fy', '', '   ', 'é' * 650, None]],
        'top_ties': {'indices': [5, 6, 7, 8], 'scores': [0.5, 0.9, 0.5, 0.9], 'value': mm.top([5, 6, 7, 8], np.array([0.5, 0.9, 0.5, 0.9]))},
        'session_order': um.session_order(turns),
        'render_frozen_mixed': am.render([turns[3], turns[0], turns[4], turns[1]]),
        'render_dated_mixed': um.render([turns[3], turns[0], turns[4], turns[1]]),
        'cases': cases,
    }
    # Write-time update links (U1 arm V).
    table = um.fact_table(turns, facts)
    rng = np.random.default_rng(7)
    fvec = {f['key']: v.tolist() for f, v in zip(table, vectors(len(table), rng, 16))}
    jobs = um.update_jobs(table, fvec)
    lookup = {f['key']: f for f in table}
    update_replies = ['{"updates": [{"old": "E1", "new": "N1"}, {"old": "E1", "new": "N1"}, {"old": "E9", "new": "N1"}, {"old": "N1", "new": "E1"}, {"old": 3, "new": "N1"}]}',
                      'nope', '{"updates": []}', '<think>t</think>{"updates": [{"old": " E2 ", "new": "N2"}]}']
    last = jobs[-1]
    links = [{'old': 'g#2#0', 'new': 'g#1#0'}, {'old': 'g#2#1', 'new': 'g#1#1'}]
    chain = [{'old': 'g#2#0', 'new': 'g#3#0'}, {'old': 'g#3#0', 'new': 'g#1#0'}]
    fixtures['updates'] = {
        'table': table, 'vectors': fvec, 'jobs': jobs,
        'messages': [um.update_messages(j, lookup) for j in jobs],
        'parses': [{'reply': r, 'value': um.parse_updates(r, last)} for r in update_replies], 'parse_job': last,
        'replacements': um.replacements(links, lookup), 'chain': um.replacements(chain, lookup),
        'versioned': dict(zip(('order', 'notes'), um.versioned(['g/D2:1', 'g/D3:1', 'g/D1:1'], table, links))),
        'session_order': um.session_order(turns),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixtures, ensure_ascii=False, indent=1) + '\n')
    print(f'{len(cases)} ranking cases, {len(fact_replies)} fact replies -> {OUT}')


if __name__ == '__main__':
    main()
