#!/usr/bin/env python3
"""LoCoMo memory retrieval, stage 1: frozen split and development-only turn baseline.

prepare mechanically partitions the pinned dataset; run opens only corpus/dev.
No holdout evaluation command is provided at this stage. Licensed text, vectors,
rankings and audit queues stay in gitignored build/eval/memory/.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import retrieval_public as rp

ROOT = rp.ROOT
LOCAL = ROOT / 'build/eval/memory'
RESULTS = ROOT / 'results/eval/memory'
SEED = 'egemma-memory-stage1-2026-10-12'
KS = (5, 10, 20)
VARIANTS = ('v1-fp32-prompt', 'v2-fp16-prompt', 'prime-onnx-q8-none')
TYPES = ('single_fact', 'multiple_evidence', 'temporal')
PROTOCOL = {
    'version': 1, 'seed': SEED, 'holdout_fraction': 0.2,
    'split': 'conversation/category-stratified SHA256 order; identical normalized questions bundled',
    'metrics': ['recall', 'all_evidence', 'hit'], 'ks': list(KS), 'dimensions': 768,
    'ranking': 'exact cosine within own conversation; stable corpus-order ties',
    'query_types': 'category 2 temporal; else category 1 or >1 evidence multiple; else single',
    'document_format': 'speaker: text + optional BLIP caption; no session timestamp',
    'prompts': {'query': rp.V1_PROMPTS['query'], 'document': rp.V1_PROMPTS['document']},
    'engine_max_tokens': 2048, 'prime_max_chars': 5000,
    'stage1_variants': list(VARIANTS), 'stage1_unit': 'one dialogue utterance',
    'selection': 'after all dev stages: largest macro recall@10, then fewer mean payload tokens@10, then config id',
    'holdout': 'one selected configuration only, after committed selection; no stage1 holdout execution',
    'audit': 'first 20 incomplete v1 recall@10 queries in seeded SHA256 order; manual multi-label audit',
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    rp.write_json(path, value)


def stamp() -> str:
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()


def query_type(query: dict, evidence: dict) -> str:
    if query['category'] == 2:
        return 'temporal'
    return 'multiple_evidence' if query['category'] == 1 or len(evidence) > 1 else 'single_fact'


def split_questions(queries: list[dict]) -> dict[str, str]:
    """No answers/qrels or retrieval scores enter the split. Bundle duplicates first."""
    bundles = defaultdict(list)
    for query in queries:
        text = unicodedata.normalize('NFKC', query['text']).casefold()
        text = re.sub(r'[^\w\s]', '', text)
        key = (query['group'], ' '.join(text.split()))
        bundles[key].append(query)
    strata = defaultdict(list)
    for key, bundle in bundles.items():
        # A duplicate with disagreeing category labels still stays in one split.
        strata[(key[0], min(q['category'] for q in bundle))].append(bundle)
    assignments = {}
    for bundles_in_stratum in strata.values():
        ordered = sorted(bundles_in_stratum, key=lambda b: digest([SEED, sorted(q['id'] for q in b)]))
        n = len(ordered)
        n_holdout = min(n - 1, max(1, int(n * 0.2 + 0.5))) if n > 1 else 0
        for i, bundle in enumerate(ordered):
            for query in bundle:
                assignments[query['id']] = 'holdout' if i < n_holdout else 'dev'
    if len(assignments) != len(queries):
        raise ValueError('duplicate query ids')
    return assignments


def counts(queries: list[dict], qrels: dict) -> dict:
    return {'queries': len(queries),
            'categories': dict(sorted(Counter(str(q['category']) for q in queries).items())),
            'types': dict(Counter(query_type(q, qrels[q['id']]) for q in queries)),
            'conversations': dict(sorted(Counter(q['group'] for q in queries).items()))}


def prepare() -> None:
    manifest_path = RESULTS / 'split.json'
    if manifest_path.exists():
        raise RuntimeError('split is already frozen; do not overwrite it')
    source = rp.DATA / 'locomo.json'
    pinned = read(ROOT / 'results/eval/datasets.json')['datasets']['locomo']
    raw = ROOT / 'build/eval/raw/locomo/locomo10.json'
    if file_hash(raw) != pinned['files'][0]['sha256']:
        raise ValueError('raw LoCoMo checksum differs from pinned manifest')
    dataset = read(source)
    from fetch_data import locomo
    canonical = locomo([])  # Uses the checksum-verified local raw file; no download.
    if any(dataset[key] != canonical[key] for key in ('corpus', 'queries', 'qrels', 'skipped')):
        raise ValueError('normalized LoCoMo differs from pinned raw data')
    corpus_ids = {d['id'] for d in dataset['corpus']}
    if len(corpus_ids) != pinned['corpus'] or len(dataset['queries']) != pinned['queries']:
        raise ValueError('dataset counts differ from pinned manifest')
    for query in dataset['queries']:
        evidence = dataset['qrels'][query['id']]
        if not evidence or not set(evidence) <= corpus_ids:
            raise ValueError('invalid evidence ids')
        if any(not did.startswith(query['group'] + '/') for did in evidence):
            raise ValueError('cross-conversation evidence')
    assignments = split_questions(dataset['queries'])
    paths = {}
    splits = {}
    write(LOCAL / 'split-map.json', assignments)
    write(LOCAL / 'corpus.json', dataset['corpus'])
    for split in ('dev', 'holdout'):
        queries = [q for q in dataset['queries'] if assignments[q['id']] == split]
        qrels = {q['id']: dataset['qrels'][q['id']] for q in queries}
        path = LOCAL / f'{split}.json'
        write(path, {'queries': queries, 'qrels': qrels})
        paths[split] = file_hash(path)
        splits[split] = counts(queries, qrels)
    unknown = Counter()
    affected = Counter()
    for sample in read(raw):
        cid = sample['sample_id']
        for i, qa in enumerate(sample['qa']):
            qid = f'{cid}/q{i}'
            if qid not in assignments:
                continue
            refs = {e for entry in qa.get('evidence', []) for e in re.split(r'[;,\s]+', entry) if e}
            n_unknown = sum(f'{cid}/{e}' not in corpus_ids for e in refs)
            unknown[assignments[qid]] += n_unknown
            affected[assignments[qid]] += int(n_unknown > 0)
    manifest = {'created_at': stamp(), 'protocol': PROTOCOL, 'protocol_sha256': digest(PROTOCOL),
                'source': pinned, 'normalized_sha256': file_hash(source),
                'split_map_sha256': file_hash(LOCAL / 'split-map.json'),
                'corpus_sha256': file_hash(LOCAL / 'corpus.json'), 'split_files_sha256': paths,
                'splits': splits, 'unknown_evidence_in_included_questions': dict(unknown),
                'questions_with_partial_unknown_evidence': dict(affected),
                'prior_exposure': 'All 1535 questions were scored in retrieval_public before this split. '
                'This is a prospective development holdout, not previously unseen data.'}
    write(manifest_path, manifest)
    print(json.dumps({'splits': splits, 'protocol_sha256': manifest['protocol_sha256']}, indent=2))


def load_dev() -> tuple[dict, dict]:
    manifest = read(RESULTS / 'split.json')
    if manifest['protocol'] != PROTOCOL or manifest['protocol_sha256'] != digest(PROTOCOL):
        raise ValueError('protocol changed after split freeze')
    for name, expected in [('corpus', manifest['corpus_sha256']), ('dev', manifest['split_files_sha256']['dev']),
                           ('split-map', manifest['split_map_sha256'])]:
        if file_hash(LOCAL / f'{name}.json') != expected:
            raise ValueError(f'{name} changed after split freeze')
    dev = read(LOCAL / 'dev.json')
    assignments = read(LOCAL / 'split-map.json')
    actual = {q['id'] for q in dev['queries']}
    expected = {qid for qid, split in assignments.items() if split == 'dev'}
    if actual != expected or set(dev['qrels']) != expected:
        raise ValueError('dev membership differs from frozen split')
    dev['corpus'] = read(LOCAL / 'corpus.json')
    return dev, manifest


def evidence_metrics(ranking: list[str], relevant: set[str]) -> dict:
    if not relevant:
        raise ValueError('Recall denominator must be positive')
    result = {}
    for k in KS:
        found = len(set(ranking[:k]) & relevant)
        result[f'recall@{k}'] = found / len(relevant)
        result[f'all_evidence@{k}'] = float(found == len(relevant))
        result[f'hit@{k}'] = float(found > 0)
    return result


def evaluate(dataset: dict, queries: np.ndarray, documents: np.ndarray) -> list[dict]:
    by_group = defaultdict(list)
    for i, document in enumerate(dataset['corpus']):
        by_group[document['group']].append(i)
    output = []
    for i, query in enumerate(dataset['queries']):
        candidates = by_group[query['group']]
        scores = documents[candidates] @ queries[i]
        order = np.argsort(-scores, kind='stable')[:max(KS)]
        ranking = [dataset['corpus'][candidates[j]]['id'] for j in order]
        relevant = set(dataset['qrels'][query['id']])
        output.append({'id': query['id'], 'category': query['category'], 'type': query_type(query, dataset['qrels'][query['id']]),
                       'ranking': ranking, 'missing@10': sorted(relevant - set(ranking[:10])),
                       **evidence_metrics(ranking, relevant)})
    return output


def aggregate(rows: list[dict]) -> dict:
    metrics = [f'{metric}@{k}' for metric in ('recall', 'all_evidence', 'hit') for k in KS]
    def mean(part):
        return {'queries': len(part), **{m: float(np.mean([r[m] for r in part])) for m in metrics}} if part else {'queries': 0}
    return {'overall': mean(rows), 'by_type': {t: mean([r for r in rows if r['type'] == t]) for t in TYPES},
            'by_category': {str(c): mean([r for r in rows if r['category'] == c]) for c in range(1, 5)}}


def checked_vectors(vectors, n: int) -> np.ndarray:
    vectors = np.asarray(vectors, np.float32)
    if vectors.shape != (n, 768) or not np.isfinite(vectors).all():
        raise ValueError('invalid embeddings: expected finite (n,768)')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(np.abs(norms - 1) > 1e-4):
        raise ValueError('embeddings are not normalized')
    return vectors / norms


def fingerprint(variant: str) -> dict:
    """Cache invalidates on model/runtime changes, not just text changes."""
    if variant.startswith('prime'):
        setup = read(ROOT / 'build/stage2/onnx-setup.json')
        model_dir = Path(setup['modelDir'])
        paths = [model_dir / f['path'] for f in setup['model']['files']]
        paths += [Path(setup['worker']), ROOT / 'scripts/stage2/onnx_prime_worker.mjs']
        runtime = Path(setup['runtimeDir']) / 'node_modules'
        paths += [runtime / '@huggingface/transformers/package.json', runtime / 'onnxruntime-node/package.json']
    else:
        model_dir = rp.V1_ST_DIR if variant.startswith('v1') else ROOT / 'models/embeddinggemma-2'
        paths = [model_dir / name for name in ('model.safetensors', 'tokenizer.json', 'config.json',
                                               'config_sentence_transformers.json', 'tokenizer_config.json')]
        paths += sorted((model_dir / '2_Dense').glob('*.safetensors')) + sorted((model_dir / '3_Dense').glob('*.safetensors'))
        paths += sorted((ROOT / 'build').glob('_egemma_native.cpython-*.so'))
        paths += sorted((ROOT / 'server/egemma_server').glob('*.py'))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def cached_vectors(variant: str, side: str, texts: list[str], identity: dict, compute) -> tuple[np.ndarray, dict]:
    key = digest({'variant': variant, 'side': side, 'texts': texts, 'identity': identity, 'protocol': PROTOCOL})
    path = LOCAL / 'cache' / f'{variant}-{side}-{key}.npy'
    started = time.perf_counter()
    hit = path.exists()
    if hit:
        vectors = checked_vectors(np.load(path, allow_pickle=False), len(texts))
    else:
        vectors = checked_vectors(compute(texts), len(texts))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
    return vectors, {'cache_hit': hit, 'texts': len(texts), 'seconds': time.perf_counter() - started,
                     'sha256': file_hash(path)}


def gpu_state() -> dict:
    check = subprocess.run(['bash', str(ROOT / 'scripts/gpu_quiet_check.sh')], capture_output=True, text=True)
    return {'time': stamp(), 'quiet': check.returncode == 0, 'record': check.stdout + check.stderr}


@contextmanager
def start_backend(variant: str, args):
    from bench import HttpBackend, JsonlBackend, default_specs
    log = LOCAL / 'logs' / f'{variant}.log'
    if variant.startswith('prime'):
        backend = JsonlBackend(default_specs(args.model_dir, sys.executable)['onnx-prime'], log, 1800)
    else:
        model_dir = args.v1_dir if variant.startswith('v1') else args.model_dir
        model_name = 'embeddinggemma-300m' if variant.startswith('v1') else 'embeddinggemma-2'
        dtype = 'fp32' if variant.startswith('v1') else 'fp16'
        spec = {'kind': 'http', 'port': args.port, 'env': {'PYTHONPATH': str(ROOT / 'server')},
                'command': [sys.executable, '-m', 'egemma_server', '--port', str(args.port), '--dtype', dtype,
                            '--matmul', 'auto', '--model-dir', str(model_dir), '--model-name', model_name]}
        backend = HttpBackend(spec, log, 600)
    try:
        info = backend.wait_ready(300)
        def embed(texts):
            vectors = []
            batch_size = 16 if variant.startswith('prime') else 32
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                if variant.startswith('prime'):
                    reply = backend.call({'op': 'embed', 'texts': batch, 'prefix': '', 'return_vectors': True})
                    vectors.extend(reply['vectors'])
                else:
                    reply = backend._request('POST', '/v1/embeddings', {'model': model_name, 'input': batch,
                                                                      'task': 'none', 'dimensions': 768})
                    vectors.extend(d['embedding'] for d in sorted(reply['data'], key=lambda d: d['index']))
                if len(vectors) % 512 == 0:
                    print(f'  {variant}: {len(vectors)}/{len(texts)} embedded', flush=True)
            return vectors
        yield info, embed
    finally:
        backend.stop()


def run(port: int) -> None:
    dataset, manifest = load_dev()  # Never opens holdout.json or old all-query caches/results.
    args = argparse.Namespace(port=port, model_dir=ROOT / 'models/embeddinggemma-2', v1_dir=rp.V1_ST_DIR)
    from egemma_server.tokenizer import PromptTokenizer
    report = {'started_at': stamp(), 'split': 'dev', 'holdout_evaluated': False,
              'protocol_sha256': manifest['protocol_sha256'], 'split_manifest_sha256': file_hash(RESULTS / 'split.json'),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'environment': {'platform': platform.platform(), 'python': sys.version, 'gpu_before': gpu_state()},
              'variants': {}}
    for variant in VARIANTS:
        print(f'{stamp()} {variant}: {len(dataset["queries"])} dev questions / {len(dataset["corpus"])} turns', flush=True)
        identity = fingerprint(variant)
        raw_queries = [q['text'] for q in dataset['queries']]
        raw_docs = [d['text'] for d in dataset['corpus']]
        cuts = {}
        if variant.startswith('prime'):
            q_in, d_in = raw_queries, raw_docs
            cuts['chars_queries'] = sum(len(t) > 5000 for t in q_in)
            cuts['chars_docs'] = sum(len(t) > 5000 for t in d_in)
        else:
            tokenizer = PromptTokenizer(args.v1_dir if variant.startswith('v1') else args.model_dir)
            for side in ('query', 'document'):
                if tokenizer.prompt_for(side) != PROTOCOL['prompts'][side]:
                    raise ValueError('checkpoint prompts differ from frozen protocol')
            # Engine paths retain the whole prompt and do not apply prime's 5000-char limit.
            q_in = rp.fit_like_st(tokenizer, [PROTOCOL['prompts']['query'] + t for t in raw_queries])
            d_in = rp.fit_like_st(tokenizer, [PROTOCOL['prompts']['document'] + t for t in raw_docs])
            cuts['tokens_queries'] = sum(a != PROTOCOL['prompts']['query'] + b for a, b in zip(q_in, raw_queries))
            cuts['tokens_docs'] = sum(a != PROTOCOL['prompts']['document'] + b for a, b in zip(d_in, raw_docs))
        with start_backend(variant, args) as (info, embed):
            q_vec, q_timing = cached_vectors(variant, 'q-dev', q_in, identity, embed)
            d_vec, d_timing = cached_vectors(variant, 'd', d_in, identity, embed)
            rows = evaluate(dataset, q_vec, d_vec)
            write(LOCAL / f'{variant}-dev-rankings.json', rows)
            report['variants'][variant] = {'info': info, 'artifact_sha256': identity, 'cuts': cuts,
                                           'embedding': {'queries': q_timing, 'documents': d_timing}, **aggregate(rows)}
            write(RESULTS / 'stage1-dev.json', report)
            print(json.dumps(report['variants'][variant]['overall']), flush=True)
            if variant.startswith('v1'):
                write_audit_queue(dataset, rows)
    report['finished_at'] = stamp()
    report['environment']['gpu_after'] = gpu_state()
    write(RESULTS / 'stage1-dev.json', report)
    write_report(report)


def write_audit_queue(dataset: dict, rows: list[dict]) -> None:
    queries = {q['id']: q for q in dataset['queries']}
    docs = {d['id']: d['text'] for d in dataset['corpus']}
    misses = sorted((r for r in rows if r['missing@10']), key=lambda r: digest([SEED, 'audit', r['id']]))
    queue = []
    for row in misses:
        qid = row['id']
        queue.append({**row, 'question': queries[qid]['text'],
                      'evidence': {did: docs[did] for did in dataset['qrels'][qid]},
                      'top5': {did: docs[did] for did in row['ranking'][:5]}, 'manual_labels': [], 'note': ''})
    write(LOCAL / 'dev-audit-queue.json', {'sample_ids': [r['id'] for r in misses[:20]], 'misses': queue})


def write_report(report: dict) -> None:
    manifest = read(RESULTS / 'split.json')
    lines = ['# 外部記憶・検索評価：ステージ1（開発用のみ）', '',
             f'開発用 {manifest["splits"]["dev"]["queries"]} 問。本番用 {manifest["splits"]["holdout"]["queries"]} 問は未計測。',
             '発言1件を1文書として同じ会話内の厳密 cos 検索。768次元。Recall は各問の回収根拠数 / 正解根拠数の平均。',
             'All は全根拠を回収できた質問の割合。Hit は根拠1件以上を回収できた割合。', '',
             '| 埋め込み | 種類 | 問数 | Recall@5 | Recall@10 | Recall@20 | All@10 | Hit@10 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for variant, entry in report['variants'].items():
        for name, row in [('all', entry['overall']), *entry['by_type'].items()]:
            if row['queries']:
                values = [row[m] for m in ('recall@5', 'recall@10', 'recall@20', 'all_evidence@10', 'hit@10')]
                lines.append(f'| {variant} | {name} | {row["queries"]} | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', '種類はスコアを見る前に固定した代理分類：カテゴリ2は temporal、その他でカテゴリ1または根拠が複数なら multiple_evidence、残りは single_fact。',
              'カテゴリ3（open-domain）も残してあるため、single_fact は厳密な単一事実ラベルではない。元カテゴリ別の値は JSON に保存。', '',
              'v1 は fp32＋プレフィックス、v2 は fp16＋プレフィックス。prime は実物の 0.22.1 worker、ONNX q8 CPU、プレフィックスなし・Denseなし。',
              'エンジンは2048トークン、primeは5000文字を上限とする。切り詰め件数とモデル・実行コードの SHA256 は JSON に保存。',
              '今回は LLM を使っていない。fresh cache を使い、既存の全質問のベクトル・結果は読み込んでいない。', '',
              '**既存評価との重複**：分割前に1535問すべてを公開ベンチマークで評価済み。今回の本番用は将来の調整から隔離した分割であり、完全な未使用データではない。',
              '**実行環境**：開発用の品質評価では Qwen の待機サーバーを停止していない。前後の GPU 状態を JSON に記録。処理時間を単独性能の値として比較しない。',
              '**次のステージ**：窓検索、複数クエリ、要約、利用可能ならローカル再ランキングを開発用だけで比較。本番評価・最終設定選択はまだ行わない。', '']
    (RESULTS / 'stage1-dev.md').write_text('\n'.join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=('prepare', 'run'))
    parser.add_argument('--port', type=int, default=8398)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        run(args.port)


if __name__ == '__main__':
    main()
