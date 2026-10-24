#!/usr/bin/env python3
"""Resume the registered retrieval stages, committing each completed stage; never opens holdout."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import locomo_memory_stages as st


def command(*args: str) -> None:
    subprocess.run([sys.executable, '-u', str(st.ROOT / 'scripts/eval/locomo_memory_stages.py'), *args],
                   cwd=st.ROOT, check=True)


def commit(paths: list[str], message: str) -> None:
    subprocess.run(['git', 'add', '--', *paths], cwd=st.ROOT, check=True)
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet', '--', *paths], cwd=st.ROOT)
    if changed.returncode == 1:
        # Commit exactly these files, preserving any other agent's staging area.
        subprocess.run(['git', 'commit', '--only', '-m', message, '--', *paths], cwd=st.ROOT, check=True)
    elif changed.returncode:
        raise RuntimeError('cannot inspect staged evaluation files')


def state(phase: str, status: str = 'running', **extra) -> None:
    st.lm.write(st.LOCAL / 'resume-status.json',
                {'time': st.lm.stamp(), 'phase': phase, 'status': status, **extra})
    print(f'{st.lm.stamp()} {status}: {phase}', flush=True)


def complete(stage: str) -> bool:
    path = st.RESULTS / f'stage{stage}-dev.json'
    if not path.exists() or not st.committed(path):
        return False
    report = st.lm.read(path)
    base, configs = st.stage_configs(stage, st.previous_selection(stage))
    if (report['plan_sha256'] != st.lm.digest(st.PLAN)
            or report['split_manifest_sha256'] != st.lm.file_hash(st.RESULTS / 'split.json')
            or report['holdout_evaluated'] or report['base'] != base['id']
            or set(report['configs']) != {c['id'] for c in configs} or 'reporting' not in report):
        raise RuntimeError(f'stage {stage} completion metadata differs; stop for review')
    if not (st.LOCAL / f'stage{stage}-dev-rows.json').exists():
        raise RuntimeError(f'stage {stage} local rows missing; stop for review')
    return True


def freeze_final() -> None:
    """Freeze a portable search configuration with its selection commit, not a new measurement."""
    report_path = st.RESULTS / 'stage5-dev.json'
    report = st.frozen(report_path, what='run --stage 5')
    selection_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=st.ROOT, text=True).strip()
    artifact = {
        'version': 1, 'frozen_at': st.lm.stamp(),
        'selection_commit': selection_commit, 'selected_config': report['selection']['selected_config'],
        'selection_report_sha256': st.lm.file_hash(report_path),
        'protocol_files_sha256': {name: st.lm.file_hash(st.RESULTS / name)
                                 for name in ('split.json', 'stages.json', 'generation.json', 'inference-policy.json')},
        'code_files_sha256': {name: st.lm.file_hash(st.ROOT / 'scripts/eval' / name) for name in
                              ('locomo_memory.py', 'locomo_memory_methods.py', 'locomo_memory_stages.py',
                               'locomo_memory_reporting.py', 'locomo_memory_resume.py')},
        'embedding_artifact_sha256': report['embedding_artifact_sha256'],
        'qwen_server': st.lm.read(st.RESULTS / 'generation.json')['server'],
        'effective_protocol': st.PLAN,
        'scope': 'selected on development questions of the same 10 LoCoMo conversations',
        'holdout_evaluated': False, 'answer_evaluation': 'draft only; not approved or run',
        'independent_conversation_evaluation': 'not run; requires a separate approved one-shot protocol',
    }
    path = st.RESULTS / 'final-config.json'
    if path.exists():
        old = st.frozen(path, what='freeze final retrieval configuration')
        if old['selected_config'] != artifact['selected_config']:
            raise RuntimeError('refuse to replace a frozen final configuration')
        return
    st.lm.write(path, artifact)
    lines = ['# 外部記憶の検索：最終設定の固定', '',
             f'選択設定：`{artifact["selected_config"]["id"]}`。',
             f'選択結果のコミット：`{selection_commit}`。',
             '設定とコード・モデルのhashは `results/eval/memory/final-config.json` に保存した。',
             '選択規則は変更していない。これは同じ10会話の開発用質問での選択であり、本番用は未計測。',
             '「LoCoMoの同じ10会話の、調整に使わなかった質問への性能」はまだ測っていない。',
             '未知の会話・日本語会話への一般化は評価していない。',
             't1・p2・全候補、会話別の差、会話単位95%区間、符号検定、複数根拠を含む種類別の結果は',
             '[ステージ3](../results/eval/memory/stage3-dev.md)、',
             '[ステージ4](../results/eval/memory/stage4-dev.md)、',
             '[ステージ5](../results/eval/memory/stage5-dev.md)を参照。',
             '別の会話への一度の独立評価には、この設定を再選択せず使う。データと実行手順は別途承認・固定する。',
             'A/B/C回答評価は [未固定の案](MEMORY_ANSWER_EVAL_DRAFT.md) のみで、回答生成は実行していない。', '']
    (st.ROOT / 'docs/MEMORY_RETRIEVAL_FINAL.md').write_text('\n'.join(lines))
    commit(['results/eval/memory/final-config.json', 'docs/MEMORY_RETRIEVAL_FINAL.md'],
           'Freeze the selected memory retrieval configuration for independent evaluation')


def resume(wait: bool) -> None:
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=st.ROOT, text=True).strip()
    if branch != 'dsh-plugin':
        raise RuntimeError('expected dsh-plugin; stop without changing branches')
    st.check_plan()
    st.Qwen()  # Check existing server; never start it.
    if wait:
        state('waiting for the already running query generation')
        while True:
            path = st.RESULTS / 'generation-outputs.json'
            summary = st.lm.read(path) if path.exists() else {}
            if all(summary.get(k, {}).get('items') == 1233 and summary[k].get('limit') is None
                   for k in st.QUERY_GEN_KINDS):
                break
            time.sleep(30)
    for stage in ('3', '4', '5'):
        if complete(stage):
            state(f'stage {stage} already committed; resume next stage')
            continue
        st.Qwen()
        if stage == '3':
            for kind in st.QUERY_GEN_KINDS:
                state(f'generate {kind}')
                command('generate', kind, '--workers', '1')
        elif stage == '4':
            for kind in st.CORPUS_KINDS:
                state(f'generate {kind}')
                command('generate', kind, '--workers', '1')
        state(f'run stage {stage}')
        command('run', '--stage', stage)
        paths = [f'results/eval/memory/stage{stage}-dev.{ext}' for ext in ('json', 'md')]
        paths.append('results/eval/memory/generation-outputs.json')
        commit(paths, f'Evaluate registered memory retrieval stage {stage} on development questions')
        report = st.lm.read(st.RESULTS / f'stage{stage}-dev.json')
        state(f'stage {stage} committed', selected=report['selection']['selected'])
    state('freeze final retrieval configuration')
    freeze_final()
    state('retrieval stages 3–5 complete', status='complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-query-generation', action='store_true',
                        help='wait for the initial sequential batch that is already running')
    args = parser.parse_args()
    try:
        resume(args.wait_for_query_generation)
    except BaseException as error:
        state('evaluation stopped', status='stopped', error=f'{type(error).__name__}: {error}')
        raise
