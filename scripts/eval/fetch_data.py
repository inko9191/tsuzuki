#!/usr/bin/env python3
"""Download the public retrieval benchmarks at pinned revisions and normalize them.

Run with a data-prep venv (pyarrow for the JMTEB parquet files; .venv itself is left unchanged):
  .venv/bin/python -m venv build/eval/venv-data
  build/eval/venv-data/bin/pip install pyarrow==21.0.0 huggingface_hub==1.33.0
  build/eval/venv-data/bin/python scripts/eval/fetch_data.py

Japanese (JMTEB, sbintuitions/JMTEB, test split):
  jagovfaqs_22k, nlp_journal_title_abs, nlp_journal_abs_intro
English (MTEB / BEIR, mteb/<name>, test qrels):
  scifact, nfcorpus, arguana
Long conversation memory:
  LoCoMo (snap-research/locomo data/locomo10.json): one corpus per conversation (its dialogue turns),
  queries are the QA questions of categories 1-4 with their evidence turns as the relevant documents
  (category 5 is adversarial "not mentioned" questions and is left out).

Everything goes under build/eval/ (gitignored); the data is not committed (CC BY-SA / CC BY-NC
licenses). Output: build/eval/data/<name>.json with {name, lang, source, corpus: [{id, text}],
queries: [{id, text, group?}], qrels: {qid: {docid: score}}} plus build/eval/data/manifest.json
(revisions, sha256 of every downloaded file, counts), copied to results/eval/datasets.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'build/eval/raw'
DATA = ROOT / 'build/eval/data'
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')  # the xet client fails on this Mac (File exists, os error 17)

JMTEB_REVISION = '6064d6d00e3eccca0f9673621022b409f79d1a18'
MTEB_REVISIONS = {
    'scifact': 'cf10ab6856b15b0e670ef8ae5dae4e266c12d035',
    'nfcorpus': '52ac3f19d3449632d9f00aab0ad34a110fc03816',
    'arguana': '6c1bcf74b13dfd823aff056b79d4d93e702f19c7',
}
LOCOMO_COMMIT = '3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376'
JMTEB_TASKS = {'jagovfaqs_22k': 'jagovfaqs_22k', 'nlp_journal_title_abs': 'nlp_journal_title_abs',
               'nlp_journal_abs_intro': 'nlp_journal_abs_intro'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def hf_file(repo: str, filename: str, revision: str, files: list) -> Path:
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(repo, filename, repo_type='dataset', revision=revision,
                                local_dir=str(RAW / repo.replace('/', '__')), token=False))
    files.append({'repo': repo, 'revision': revision, 'file': filename, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    return path


def jmteb(task: str, files: list) -> dict:
    import pyarrow.parquet as pq
    corpus_rows = pq.read_table(hf_file('sbintuitions/JMTEB', f'data/{task}-corpus/corpus.parquet', JMTEB_REVISION, files)).to_pylist()
    query_rows = pq.read_table(hf_file('sbintuitions/JMTEB', f'data/{task}-query/test.parquet', JMTEB_REVISION, files)).to_pylist()
    corpus = [{'id': str(r['docid']), 'text': r['text']} for r in corpus_rows]
    queries, qrels = [], {}
    for i, r in enumerate(query_rows):
        qid = str(r.get('qid', i))
        relevant = r['relevant_docs']
        relevant = relevant if isinstance(relevant, list) else [relevant]
        queries.append({'id': qid, 'text': r['query']})
        qrels[qid] = {str(d): 1 for d in relevant}
    return {'lang': 'ja', 'source': f'sbintuitions/JMTEB@{JMTEB_REVISION[:7]} {task} (test)',
            'corpus': corpus, 'queries': queries, 'qrels': qrels}


def mteb(name: str, files: list) -> dict:
    repo, revision = f'mteb/{name}', MTEB_REVISIONS[name]
    qrels: dict[str, dict[str, int]] = {}
    with open(hf_file(repo, 'qrels/test.tsv', revision, files)) as f:
        next(f)  # query-id  corpus-id  score
        for line in f:
            qid, did, score = line.rstrip('\n').split('\t')
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    corpus = []
    with open(hf_file(repo, 'corpus.jsonl', revision, files)) as f:
        for line in f:
            r = json.loads(line)
            # MTEB's retrieval evaluation embeds "title text" (title first, space separated).
            corpus.append({'id': str(r['_id']), 'text': (r.get('title', '') + ' ' + r['text']).strip()})
    queries = []
    with open(hf_file(repo, 'queries.jsonl', revision, files)) as f:
        for line in f:
            r = json.loads(line)
            if str(r['_id']) in qrels:
                queries.append({'id': str(r['_id']), 'text': r['text']})
    return {'lang': 'en', 'source': f'{repo}@{revision[:7]} (test)', 'corpus': corpus, 'queries': queries, 'qrels': qrels}


def locomo(files: list) -> dict:
    url = f'https://raw.githubusercontent.com/snap-research/locomo/{LOCOMO_COMMIT}/data/locomo10.json'
    path = RAW / 'locomo' / 'locomo10.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response, open(path, 'wb') as out:
            shutil.copyfileobj(response, out)
    files.append({'repo': 'snap-research/locomo', 'revision': LOCOMO_COMMIT, 'file': 'data/locomo10.json',
                  'bytes': path.stat().st_size, 'sha256': sha256(path)})
    corpus, queries, qrels, skipped = [], [], {}, {'category5': 0, 'no_evidence': 0, 'unknown_evidence': 0}
    for sample in json.loads(path.read_text()):
        conv, cid = sample['conversation'], sample['sample_id']
        turns = {}
        sessions = sorted((k for k in conv if re.fullmatch(r'session_\d+', k)), key=lambda k: int(k.split('_')[1]))
        for key in sessions:
            for turn in conv[key]:
                text = f"{turn['speaker']}: {turn['text']}"
                if turn.get('blip_caption'):
                    text += f" [shares an image: {turn['blip_caption']}]"
                turns[turn['dia_id']] = text
        for did, text in turns.items():
            corpus.append({'id': f'{cid}/{did}', 'text': text, 'group': cid})
        for i, qa in enumerate(sample['qa']):
            if qa.get('category') == 5:
                skipped['category5'] += 1
                continue
            evidence = [e for raw in qa.get('evidence', []) for e in re.split(r'[;,\s]+', raw) if e]
            relevant = {f'{cid}/{e}': 1 for e in evidence if e in turns}
            if not evidence:
                skipped['no_evidence'] += 1
                continue
            if not relevant:
                skipped['unknown_evidence'] += 1
                continue
            qid = f'{cid}/q{i}'
            queries.append({'id': qid, 'text': qa['question'], 'group': cid, 'category': qa.get('category')})
            qrels[qid] = relevant
    return {'lang': 'en', 'source': f'snap-research/locomo@{LOCOMO_COMMIT[:7]} locomo10.json (categories 1-4)',
            'corpus': corpus, 'queries': queries, 'qrels': qrels, 'skipped': skipped,
            'note': 'retrieval is per conversation (documents of other conversations are never candidates)'}


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {'datasets': {}}
    builders = {**{f'jmteb-{k}': (lambda files, t=t: jmteb(t, files)) for k, t in JMTEB_TASKS.items()},
                **{f'mteb-{n}': (lambda files, n=n: mteb(n, files)) for n in MTEB_REVISIONS},
                'locomo': locomo}
    for name, build in builders.items():
        files: list = []
        data = build(files)
        data['name'] = name
        (DATA / f'{name}.json').write_text(json.dumps(data, ensure_ascii=False))
        manifest['datasets'][name] = {'lang': data['lang'], 'source': data['source'], 'files': files,
                                      'corpus': len(data['corpus']), 'queries': len(data['queries']),
                                      'qrels': sum(len(v) for v in data['qrels'].values()),
                                      **({'skipped': data['skipped']} if 'skipped' in data else {})}
        print(f'{name:28s} corpus {len(data["corpus"]):6d}  queries {len(data["queries"]):5d}  {data["source"]}', flush=True)
    (DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    out = ROOT / 'results/eval/datasets.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / 'manifest.json', out)


if __name__ == '__main__':
    main()
