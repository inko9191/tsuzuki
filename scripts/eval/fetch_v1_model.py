#!/usr/bin/env python3
"""Fetch google/embeddinggemma-300m (v1) for the sentence-transformers quality path and prove it is the official files.

The official repository is gated, so the files come from a public copy distributed under the Gemma terms
(unsloth/embeddinggemma-300m at MIRROR_REVISION) and are checked against the official revision without
downloading from it: the official file listing is public and gives each file's git blob id. LFS files: the
local sha256 must equal the copy's LFS sha256 and the copy's LFS pointer blob id must equal the official one
(the pointer holds the sha256 and the size); other files: the local git blob id must equal the official one.
Output: models/embeddinggemma-300m/ (gitignored) and results/eval/v1_model.json. Exit 1 if any file differs.
Usage: .venv/bin/python scripts/eval/fetch_v1_model.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'models/embeddinggemma-300m'
OFFICIAL, OFFICIAL_REVISION = 'google/embeddinggemma-300m', '57c266a740f537b4dc058e1b0cda161fd15afa75'
MIRROR, MIRROR_REVISION = 'unsloth/embeddinggemma-300m', 'bfa3c846ac738e62aa61806ef9112d34acb1dc5a'
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')


def tree(repo: str, revision: str) -> dict:
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true', timeout=30) as r:
        return {f['path']: f for f in json.load(r) if f['type'] == 'file'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def main() -> int:
    from huggingface_hub import snapshot_download
    snapshot_download(MIRROR, revision=MIRROR_REVISION, local_dir=str(OUT), token=False)
    official, mirror = tree(OFFICIAL, OFFICIAL_REVISION), tree(MIRROR, MIRROR_REVISION)
    files, ok = [], True
    for path, entry in sorted(official.items()):
        local = OUT / path
        if mirror[path].get('lfs'):
            match = (mirror[path]['lfs']['oid'] == sha256(local) and mirror[path]['oid'] == entry['oid']
                     and local.stat().st_size == entry['size'])
        else:
            match = git_blob(local) == entry['oid']
        ok &= match
        files.append({'path': path, 'bytes': local.stat().st_size, 'sha256': sha256(local), 'official_git_oid': entry['oid'],
                      'matches_official': match})
    record = {'model': OFFICIAL, 'official_revision': OFFICIAL_REVISION, 'downloaded_from': f'{MIRROR}@{MIRROR_REVISION}',
              'how_verified': "LFS files: local sha256 equals the mirror's LFS sha256, and the mirror's LFS pointer blob id "
                              "equals the official one (same sha256 and size); other files: git blob id equals the official one",
              'all_match': ok, 'files': files}
    (ROOT / 'results/eval').mkdir(parents=True, exist_ok=True)
    (ROOT / 'results/eval/v1_model.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'{len(files)} files, all match the official revision: {ok}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
