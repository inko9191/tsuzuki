"""Requested diagnostic only; does not write official caches, results, or selection."""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
import time
import numpy as np
import locomo_memory_stages as st
import locomo_memory_reporting as reporting
from egemma_server.tokenizer import PromptTokenizer

lm, mm, rp = st.lm, st.mm, st.rp
started = lm.stamp()
st.check_plan()
dataset, manifest = lm.load_dev()
turns = mm.turn_table(dataset['corpus'], st.load_raw(manifest))
base = st.previous_selection('3')
assert base['id'] == reporting.P2
para, generation = st.generated('para', dataset, turns, st.Qwen.offline())
tokenizer = PromptTokenizer(rp.V1_ST_DIR)
prompts = lm.PROTOCOL['prompts']
assert all(tokenizer.prompt_for(s) == prompts[s] for s in ('query', 'document'))
identity = lm.fingerprint(st.VARIANT)

def cache(side, texts):
    def missing(_):
        raise RuntimeError('Required official cache missing; no GPU backend will be started')
    return lm.cached_vectors(st.VARIANT, side, texts, identity, missing)

units = st.build_units(base['units'], turns, {})
doc_texts = rp.fit_like_st(tokenizer, [prompts['document'] + u['text'] for u in units])
vectors, doc_record = cache('u-p2', doc_texts)
orig_texts = rp.fit_like_st(tokenizer, [prompts['query'] + q['text'] for q in dataset['queries']])
orig_vecs, orig_record = cache('q-dev', orig_texts)
owners, texts = [], []
for qi, q in enumerate(dataset['queries']):
    for text in para[q['id']]:
        owners.append(qi)
        texts.append(text)
fitted = rp.fit_like_st(tokenizer, [prompts['query'] + t for t in texts])

import torch
import sentence_transformers
from sentence_transformers import SentenceTransformer
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
model = SentenceTransformer(str(rp.V1_ST_DIR), device='cpu', local_files_only=True).float()
assert all(p.device.type == 'cpu' and (not p.is_floating_point() or p.dtype == torch.float32)
           for p in model.parameters())
assert model.max_seq_length == rp.MAX_TOKENS
probe_ids = list(dict.fromkeys(next(i for i, q in enumerate(dataset['queries']) if q['group'] == g)
                               for g in sorted({q['group'] for q in dataset['queries']})))
probe = lm.checked_vectors(model.encode([orig_texts[i] for i in probe_ids], batch_size=8,
    prompt='', normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False), len(probe_ids))
ref = orig_vecs[probe_ids]
validation = {'questions': len(probe_ids), 'cosine_min': float(np.sum(probe * ref, axis=1).min()),
              'max_abs_diff': float(np.abs(probe - ref).max())}
assert validation['cosine_min'] > .99999, validation
print('CPU vs native cached query validation', validation, flush=True)

# Separate diagnostic cache: never impersonate the registered Metal implementation.
key = lm.digest({'texts': fitted, 'identity': identity, 'backend': 'sentence-transformers-cpu-fp32',
                 'torch': torch.__version__, 'sentence_transformers': sentence_transformers.__version__})
prefix = lm.LOCAL / f'preview-para-cpu-{key}'
array_path, state_path = prefix.with_suffix('.npy'), prefix.with_suffix('.json')
vecs = np.zeros((len(fitted), 768), np.float32)
done = 0
if array_path.exists() and state_path.exists():
    state = lm.read(state_path)
    assert state['key'] == key
    vecs = np.load(array_path, allow_pickle=False)
    assert vecs.shape == (len(fitted), 768)
    done = state['done']
t0 = time.perf_counter()
for offset in range(done, len(fitted), 64):
    batch = fitted[offset:offset + 64]
    vecs[offset:offset + len(batch)] = lm.checked_vectors(model.encode(batch, batch_size=8, prompt='',
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False), len(batch))
    np.save(array_path, vecs)
    lm.write(state_path, {'key': key, 'done': offset + len(batch), 'total': len(fitted)})
    print(f'CPU para embeddings {offset + len(batch)}/{len(fitted)} ({time.perf_counter() - t0:.1f}s)', flush=True)
vecs = lm.checked_vectors(vecs, len(fitted))
qtexts = {'orig': [[(v, q['text'])] for v, q in zip(orig_vecs, dataset['queries'])],
          'para': [[] for _ in dataset['queries']]}
for owner, vec, text in zip(owners, vecs, texts):
    qtexts['para'][owner].append((vec, text))
count = lambda ts: [len(e.ids) for e in tokenizer.tokenizer.encode_batch(ts, add_special_tokens=False)]
turn_tokens = dict(zip([t['id'] for t in turns], count([t['text'] for t in turns])))
dates = {t['session']: t['date'] for t in turns}
header_tokens = dict(zip(dates, count([f'Date: {d}' for d in dates.values()])))
turn_session = {t['id']: t['session'] for t in turns}
index = mm.Index(units, vectors)
candidate = mm.config(base['units'], base['rankers'], ['orig', 'para'], base['post'])
rows = {}
for cfg in (base, candidate):
    rows[cfg['id']] = mm.score_config(index, cfg, dataset, qtexts, turn_session, turn_tokens, header_tokens,
        lambda q: lm.query_type(q, dataset['qrels'][q['id']]))
references = lm.read(lm.LOCAL / 'stage2-dev-rows.json')
stats = reporting.comparisons(rows, references, base['id'])
record = {'kind': 'requested interim CPU-reference diagnostic; not an official stage result or selection',
          'started_at': started, 'finished_at': lm.stamp(), 'split': 'dev', 'holdout_evaluated': False,
          'official_pipeline_changed': False, 'selection_performed': False,
          'scope': 'same 10 LoCoMo conversations, development questions only',
          'embedding': {'model': 'embeddinggemma-300m', 'dtype': 'float32', 'device': 'cpu', 'threads': 2,
             'batch_size': 8, 'dimensions': 768, 'max_tokens': rp.MAX_TOKENS,
             'new_vectors': 'para queries only; document and original query vectors reused from native cache',
             'torch': torch.__version__, 'sentence_transformers': sentence_transformers.__version__,
             'artifact_sha256': identity, 'validation': validation},
          'generation': generation, 'candidate': candidate, 'base': base,
          'protocol_sha256': lm.digest(st.PLAN), 'split_sha256': lm.file_hash(lm.RESULTS / 'split.json'),
          'preview_code_sha256': lm.file_hash(Path(__file__)),
          'doc_cache': doc_record, 'orig_cache': orig_record, 'preview_vectors_sha256': lm.file_hash(array_path),
          'rows_sha256': lm.digest(rows), 'reporting': stats}
lm.write(lm.LOCAL / 'preview-stage3-para-cpu-rows.json', rows)
lm.write(lm.RESULTS / 'stage3-para-cpu-preview.json', record)
lines = ['# 言い換え検索：ユーザー依頼のCPU参照・暫定結果', '',
    '正式なステージ3結果ではない。生成・正式評価は停止せず、条件・選択規則を変更していない。',
    '全1233問の保存済み言い換えを使用。新しい言い換えクエリの埋め込みだけを同じモデル・fp32・'
    'プレフィックスのSentenceTransformers CPU参照実装で計算し、原質問・p2文書は正式Metalキャッシュを再利用した。',
    '正式なMetal経路との結果の一致は未確認。この暫定値を選択には使わない。',
    f'10会話の原質問でCPU/nativeベクトルを点検：{validation}。全クエリの一致を証明するものではない。',
    '同じ10会話の開発用質問のみ。本番用は未計測。調整に使わなかった質問への性能や、'
    '未知の会話・日本語への一般化は主張しない。',
    '区間は対応あり会話単位10000回bootstrap、seed 0、質問macro平均。符号検定は会話平均差の'
    '厳密な両側検定。個別95%区間・未補正p値。分類は固定の代理分類。', '']
for stratum in ('type:multiple_evidence', 'overall', 'type:single_fact', 'type:temporal'):
    lines += [f'## {stratum}', '']
    for metric in reporting.METRICS:
        for reference in (reporting.T1, reporting.P2):
            lines += [f'### {metric} / {reference}', '']
            lines += reporting.table(stats['strata'][stratum][metric], reference, stats['groups']) + ['']
(lm.RESULTS / 'stage3-para-cpu-preview.md').write_text('\n'.join(lines))
print('Preview written; official pipeline untouched', flush=True)
