#!/usr/bin/env python3
"""Check models/embeddinggemma-2 against google/embeddinggemma-2 at the pinned revision on Hugging Face.

The repository is not gated, so its file listing gives each LFS file's sha256 and each small file's git blob id.
Every file of the revision must exist locally with the same sha256 (LFS) or git blob id (other files).
Output: results/eval/review/v2_model.json. Exit 1 on any mismatch.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'models/embeddinggemma-2'
REPO, REVISION = 'google/embeddinggemma-2', '914f7f89142e33e77833254d9c9b90c3cef7303b'


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
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true', timeout=60) as r:
        listing = [f for f in json.load(r) if f['type'] == 'file']
    files, ok = [], True
    for entry in sorted(listing, key=lambda f: f['path']):
        local = MODEL / entry['path']
        row = {'path': entry['path'], 'bytes': entry.get('size'), 'present': local.exists()}
        if not local.exists():
            ok = False
            files.append(row | {'match': False})
            continue
        if entry.get('lfs'):
            row |= {'kind': 'lfs', 'expected_sha256': entry['lfs']['oid'], 'local_sha256': sha256(local)}
            row['match'] = row['expected_sha256'] == row['local_sha256'] and local.stat().st_size == entry['size']
        else:
            row |= {'kind': 'git', 'expected_git_oid': entry['oid'], 'local_git_oid': git_blob(local)}
            row['match'] = row['expected_git_oid'] == row['local_git_oid']
        ok &= row['match']
        files.append(row)
    extra = sorted(str(p.relative_to(MODEL)) for p in MODEL.rglob('*')
                   if p.is_file() and '.cache' not in p.parts and str(p.relative_to(MODEL)) not in {f['path'] for f in listing})
    report = {'repo': REPO, 'revision': REVISION, 'all_match': ok, 'files': files, 'local_files_not_in_revision': extra}
    out = ROOT / 'results/eval/review/v2_model.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(files)} files, all match: {ok}; extra local files: {extra}')
    for f in files:
        if not f['match']:
            print('MISMATCH', f)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
