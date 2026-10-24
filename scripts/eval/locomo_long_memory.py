#!/usr/bin/env python3
"""Build seeded synthetic long memories after phase2 approval; no LLM generation.

Imports and unit tests perform no network, GPU, process control, or real-corpus build.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import urllib.request

QUESTION_SEED = 'locomo-long-questions-v1'
MEMORY_SEED = 'locomo-long-memory-v1'
LENGTHS = (64000, 128000)
TOLERANCE = .03


def hashed(seed, value):
    return hashlib.sha256((seed + value).encode()).hexdigest()


def sample_questions(queries, types, allowed_ids=None):
    groups = sorted({q['group'] for q in queries})
    if allowed_ids is not None:
        allowed_ids = set(allowed_ids)
        if not allowed_ids <= {q['id'] for q in queries}:
            raise ValueError('registered short-control pool must be development-only')
        queries = [q for q in queries if q['id'] in allowed_ids]
    cells = defaultdict(list)
    for q in queries:
        cells[(q['group'], types[q['id']])].append(q['id'])
    chosen = []
    for group in groups:
        for kind in ('multiple_evidence', 'single_fact', 'temporal'):
            cell = cells[(group, kind)]
            if not cell:
                raise ValueError('a development conversation/type cell is empty')
            chosen.append(min(cell, key=lambda q: hashed(QUESTION_SEED, q)))
    return chosen


def render(turns):
    paid, lines = set(), []
    for t in turns:
        if t['session'] not in paid:
            lines.append(f'Date: {t.get("date") or "unknown"}')
            paid.add(t['session'])
        lines.append(f'[{t["id"]}] {t["text"]}')
    return '\n'.join(lines)


class QwenCounter:
    """Future CPU-tokenizer endpoint only; never called in synthetic tests."""
    def __init__(self, url='http://127.0.0.1:8080'):
        self.url = url
        self.memo = {}

    def __call__(self, text):
        key = hashed('', text)
        if key not in self.memo:
            request = urllib.request.Request(self.url + '/tokenize',
                data=json.dumps({'content': text, 'add_special': False, 'parse_special': False}).encode(),
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=30) as response:
                self.memo[key] = len(json.load(response)['tokens'])
        return self.memo[key]


def sessions(turns):
    groups = {}
    for turn in turns:
        groups.setdefault(turn['session'], []).append(turn)
    return list(groups.values())


def validate(conversations, target):
    by_group = {c['group']: c['turns'] for c in conversations}
    if len(by_group) != len(conversations) or target not in by_group:
        raise ValueError('unique conversation groups and an existing target required')
    ids = [t['id'] for c in conversations for t in c['turns']]
    if len(ids) != len(set(ids)):
        raise ValueError('turn IDs must be unique; no duplicated distractors')
    for c in conversations:
        if any(t['group'] != c['group'] for t in c['turns']):
            raise ValueError('turn/source group mismatch')
    return by_group


def construct(conversations, target, token_count, target_tokens, tolerance=TOLERANCE,
              required_turn_ids=()):
    """Keep complete target, greedily add whole distractor sessions; seeded position."""
    by_group = validate(conversations, target)
    source_order = sorted((g for g in by_group if g != target),
        key=lambda g: hashed(MEMORY_SEED + target, g))
    slot = int(hashed(MEMORY_SEED, target), 16) % 3
    blocks = [b for g in source_order for b in sessions(by_group[g])]
    required = set(required_turn_ids) - {t['id'] for t in by_group[target]}
    selected = set()
    for i, block in enumerate(blocks):
        ids = {t['id'] for t in block}
        if ids & required:
            if not ids <= required:
                raise ValueError('required memory must contain complete sessions')
            selected.add(i)
    if not required <= {t['id'] for i in selected for t in blocks[i]}:
        raise ValueError('required memory contains unknown distractor IDs')

    def merged(blocks):
        # Adjacent sessions of a source stay in the original order. Place the target
        # before, halfway through, or after the distractor source blocks.
        sources = []
        for block in blocks:
            if sources and sources[-1][0]['group'] == block[0]['group']:
                sources[-1].extend(block)
            else:
                sources.append(list(block))
        where = 0 if slot == 0 else len(sources)//2 if slot == 1 else len(sources)
        sources.insert(where, by_group[target])
        return [t for block in sources for t in block]

    minimum = token_count(render(by_group[target]))
    if target_tokens <= 0 or not 0 <= tolerance < 1 or minimum > target_tokens:
        raise ValueError('target already exceeds requested memory; never truncate it')
    current = [b for i, b in enumerate(blocks) if i in selected]
    if token_count(render(merged(current))) > target_tokens:
        raise ValueError('required memory exceeds the new target')
    for i in range(len(blocks)):
        if i in selected:
            continue
        trial = [b for j, b in enumerate(blocks) if j in selected or j == i]
        if token_count(render(merged(trial))) <= target_tokens:
            selected.add(i)
    turns = merged([b for i, b in enumerate(blocks) if i in selected])
    count = token_count(render(turns))
    if count < target_tokens * (1-tolerance):
        raise ValueError('insufficient whole-session memory for registered length tolerance')
    chosen = {t['id'] for t in turns}
    return {'target_group': target, 'requested_tokens': target_tokens, 'memory_tokens': count,
        'target_position': ('start', 'middle', 'end')[slot], 'source_order': source_order,
        'turns': turns, 'selected_turn_ids': [t['id'] for t in turns],
        'selected_sessions': list(dict.fromkeys(t['session'] for t in turns)),
        'target_complete': all(t['id'] in chosen for t in by_group[target]),
        'rendered_memory_sha256': hashed('', render(turns)),
        'note': 'Synthetic concatenation; count is Qwen render length, not full chat prompt length. '
                'Full template/output/context preflight is required before phase2 generation.'}


def eligible_units(units, selected_turn_ids, synthetic_group):
    """Reuse original units/vector row IDs; exclude units with any absent raw turn."""
    selected = set(selected_turn_ids)
    return [{**u, 'group': synthetic_group, 'original_group': u['group'], 'original_vector_row': i}
        for i, u in enumerate(units) if u['turns'] and set(u['turns']) <= selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-phase2', action='store_true', help='only after the human phase1 checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.approved_phase2:
        parser.error('phase2 has not been confirmed; do not build real memories')
    # Lazy imports: CLI alone may inspect the immutable development corpus.
    import locomo_memory_stages as st
    lm, mm = st.lm, st.mm
    if (lm.LOCAL / 'user-pause.json').exists():
        raise RuntimeError('persistent user pause')
    if lm.LOCAL.resolve() not in args.output.resolve().parents:
        raise RuntimeError('licensed memory must stay in gitignored build/eval/memory')
    if args.output.exists():
        raise RuntimeError('no overwrite of a prior constructed memory')
    data, manifest = lm.load_dev()
    turns = mm.turn_table(data['corpus'], st.load_raw(manifest))
    by_group = defaultdict(list)
    for turn in turns:
        by_group[turn['group']].append(turn)
    conversations = [{'group': g, 'turns': part} for g, part in sorted(by_group.items())]
    types = {q['id']: lm.query_type(q, data['qrels'][q['id']]) for q in data['queries']}
    pilot_path = lm.ROOT / 'results/eval/memory/answer-pilot-plan.json'
    pilot = json.loads(pilot_path.read_text())
    question_ids = sample_questions(data['queries'], types, pilot['plan']['question_ids'])
    counter = QwenCounter()
    memories, previous = [], {}
    for n in LENGTHS:
        for group in sorted(by_group):
            built = construct(conversations, group, counter, n,
                              required_turn_ids=previous.get(group, ()))
            memories.append(built)
            previous[group] = built['selected_turn_ids']
    lm.write(args.output, {'created_at': lm.stamp(), 'development_question_ids': question_ids,
        'lengths': LENGTHS, 'seeds': [QUESTION_SEED, MEMORY_SEED], 'memories': memories,
        'dev_sha256': lm.file_hash(lm.LOCAL/'dev.json'), 'corpus_sha256': lm.file_hash(lm.LOCAL/'corpus.json'),
        'short_control_plan_sha256': lm.file_hash(pilot_path),
        'code_sha256': lm.file_hash(Path(__file__)), 'generation_executed': False,
        'embedding_recomputed': False})


if __name__ == '__main__':
    main()
