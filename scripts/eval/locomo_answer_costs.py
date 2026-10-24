#!/usr/bin/env python3
"""Finish planning costs using only the registered development pilot."""
from __future__ import annotations

import argparse
from collections import defaultdict
import subprocess
import sys
import time
from pathlib import Path

import locomo_answer_eval as ae

lm = ae.lm


def totals(report: dict, measured: dict) -> dict:
    answer = report['timing']['total_answer_seconds']
    auxiliary = report['auxiliary_costs']
    n = measured['target_questions']
    query = measured['query_seconds'] * n / measured['questions']
    query_scenario = measured['slowest_batch_seconds_per_query'] * n
    score = measured['scoring_reporting_save_seconds'] * n / measured['questions']
    other = (auxiliary['heldout_rerank_seconds_mean_projection']
             + auxiliary['heldout_embedding_search_and_input_preparation_seconds_projection']
             + auxiliary['two_planned_context_switches_seconds_projection']
             + measured['embedding_startup_seconds'])
    dev = report['development_preparation_actual_seconds'] + measured['measurement_wall_seconds']
    held = answer['mean_seconds'] + other + query + score
    scenario = answer['p90_components_seconds'] + other + query_scenario + score
    return {'development_actual_seconds': dev,
            'heldout_answer_seconds': answer,
            'heldout_query_embedding_seconds_mean_projection': query,
            'heldout_query_embedding_seconds_slowest_batch_projection': query_scenario,
            'heldout_scoring_reporting_save_seconds_projection': score,
            'heldout_total_seconds_mean_projection': held,
            'heldout_total_seconds_component_scenario': scenario,
            'all_422_questions_seconds_mean_projection': dev + held,
            'all_422_questions_seconds_component_scenario': dev + scenario,
            'all_422_questions_seconds_with_20_percent_margin': [1.2 * (dev + held), 1.2 * (dev + scenario)],
            'remaining_seconds_with_20_percent_margin': [1.2 * held, 1.2 * scenario]}


def main() -> None:
    registered = ae.checked_plan()
    plan = registered['plan']
    ae.server(plan)
    state = lm.read(ae.LOCAL / 'status.json')
    if state.get('phase') != 'pilot complete' or state.get('status') != 'complete':
        raise RuntimeError('pilot must finish before separate embedding measurement')
    report = lm.read(lm.RESULTS / 'answer-pilot-report.json')
    if report['plan_sha256'] != registered['plan_sha256']:
        raise RuntimeError('report refers to a different pilot')
    processes = subprocess.check_output(['ps', '-axo', 'command='], text=True).splitlines()
    if any(line.strip().startswith(('.venv/bin/python', sys.executable))
           and 'locomo_answer_eval.py generate-pilot' in line for line in processes):
        raise RuntimeError('pilot process still running')
    if any('egemma_server' in line and '-m egemma_server' in line for line in processes):
        raise RuntimeError('embedding backend already running; do not duplicate it')
    if any(slot.get('is_processing') for slot in ae.st.http_json(plan['model']['url'] + '/slots')):
        raise RuntimeError('Qwen is busy; do not add overlapping GPU work')
    measured_path = lm.RESULTS / 'answer-pilot-costs.json'
    if measured_path.exists():
        raise RuntimeError('cost measurement already recorded; do not overwrite it')
    started = time.perf_counter()
    data, manifest, chosen, _ = ae.dataset_and_pilot()
    from egemma_server.tokenizer import PromptTokenizer
    tokenizer = PromptTokenizer(ae.rp.V1_ST_DIR)
    selected = set(chosen)
    texts = ae.rp.fit_like_st(tokenizer, [lm.PROTOCOL['prompts']['query'] + q['text']
                                        for q in data['queries'] if q['id'] in selected])
    identity = lm.fingerprint(ae.st.VARIANT)
    if identity != lm.read(lm.RESULTS / 'final-config.json')['embedding_artifact_sha256']:
        raise RuntimeError('embedding artifacts differ from frozen search')
    ae.gate()
    boot = time.perf_counter()
    args = argparse.Namespace(port=8398, model_dir=lm.ROOT / 'models/embeddinggemma-2',
                              v1_dir=ae.rp.V1_ST_DIR)
    batches = []
    with lm.start_backend(ae.st.VARIANT, args) as (info, embed):
        startup = time.perf_counter() - boot
        for offset in range(0, len(texts), 32):
            ae.gate()
            batch = texts[offset:offset + 32]
            tick = time.perf_counter()
            lm.checked_vectors(embed(batch), len(batch))
            batches.append({'questions': len(batch), 'seconds': time.perf_counter() - tick})
    # Re-run only the deterministic development analysis, including public/private atomic saves.
    tick = time.perf_counter()
    subprocess.run([sys.executable, str(lm.ROOT / 'scripts/eval/locomo_answer_report.py')],
                   cwd=lm.ROOT, check=True)
    scoring_seconds = time.perf_counter() - tick
    record = {'created_at': lm.stamp(), 'split': 'development pilot only', 'holdout_evaluated': False,
              'plan_sha256': registered['plan_sha256'], 'code_sha256': lm.file_hash(Path(__file__)),
              'questions': len(texts), 'target_questions': sum(manifest['splits']['holdout']['conversations'].values()),
              'embedding_startup_seconds': startup, 'embedding_info': info,
              'query_seconds': sum(b['seconds'] for b in batches), 'query_batches': batches,
              'slowest_batch_seconds_per_query': max(b['seconds'] / b['questions'] for b in batches),
              'scoring_reporting_save_seconds': scoring_seconds,
              'measurement_wall_seconds': time.perf_counter() - started}
    record['time_plan'] = totals(report, record)
    replies = ae.recover(ae.LOCAL / 'pilot-responses.json', ae.LOCAL / 'pilot-received.jsonl')
    actual = defaultdict(float)
    for key, reply in replies.items():
        actual[key.split('|')[0]] += reply['wall_seconds']
    large = ('B8192', 'C8192-final')
    projected = report['timing']['per_condition_projection']
    record['additional_large_B_C'] = {
        'development_responses': 240, 'heldout_responses': 604, 'all_responses': 844,
        'development_answer_seconds': sum(actual[a] for a in large),
        'heldout_answer_seconds_mean_projection': sum(projected[a]['mean_seconds'] for a in large),
        'heldout_answer_seconds_component_scenario': sum(projected[a]['p90_components_seconds'] for a in large)}
    record['note'] = ('All costs measured on development only. Native query measurement uses fresh inputs, '
                      'no vector cache and no Qwen generation. Heldout quantities are projections, not bounds; '
                      'rerank uses stage5 development mean, reload uses the observed switch. The component '
                      'scenario substitutes answer component p90 and the slowest query batch, not a calibrated '
                      'total-time percentile. CPU preparation includes cached corpus/search/tokenization. '
                      'Scoring/save projection includes the full development analysis and precision forecast, '
                      'so it is a conservative planning allocation. Human/idle preparation time is excluded.')
    lm.write(measured_path, record)
    print('development-only cost measurements saved; no heldout questions opened')


if __name__ == '__main__':
    main()
