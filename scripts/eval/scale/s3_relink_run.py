#!/usr/bin/env python3
"""S3 amendment: re-run the update checks lost to dropped connections (scripts/eval/scale/s3_relink.mjs) on Gemma 4 E2B
(same file and server arguments as the build, 4 slots)."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts/eval'))
import longmemeval_e1_e2b as e1  # noqa: E402
import longmemeval_u2_run as u2  # noqa: E402

WORK = ROOT/'build/eval/memory/s3'

if __name__ == '__main__':
    small = e1.E2BServer(u2.Ctx(WORK/'build'))
    try:
        small.start(8192, 4, 'relink')
        subprocess.run([shutil.which('node'), 'scripts/eval/scale/s3_relink.mjs', '--db', str(WORK/'fillers-5000.sqlite'),
                        '--log', str(WORK/'build-5000.log'), '--chat', 'http://127.0.0.1:8080/v1'], cwd=ROOT, check=True,
                       stderr=open(WORK/'relink-5000.log', 'a'))
    finally:
        small.stop()
