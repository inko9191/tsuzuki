#!/usr/bin/env python3
"""S3 step 1: build the filler stores with facts (scripts/eval/scale/s3_build.mjs) on Gemma 4 E2B (E1's model file and
server arguments, 8 slots of 8192 for throughput). Sizes 50 and 5000. Resumable.

    .venv/bin/python -I scripts/eval/scale/s3_build_run.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts/eval'))
import longmemeval_e1_e2b as e1  # noqa: E402
import longmemeval_u2_run as u2  # noqa: E402

WORK = ROOT/'build/eval/memory/s3'
SLOTS = 8

if __name__ == '__main__':
    WORK.mkdir(parents=True, exist_ok=True)
    ctx = u2.Ctx(WORK/'build')
    small = e1.E2BServer(ctx)
    try:
        small.start(8192, SLOTS, 'facts')
        for size in (50, 5000):
            ctx.status('build', size=size)
            subprocess.run([shutil.which('node'), 'scripts/eval/scale/s3_build.mjs', '--size', str(size), '--db', str(WORK/f'fillers-{size}.sqlite'),
                            '--chat', 'http://127.0.0.1:8080/v1', '--workers', str(SLOTS)], cwd=ROOT, check=True,
                           stderr=open(WORK/f'build-{size}.log', 'a'))
        ctx.status('built')
    except BaseException as error:
        ctx.status('stopped', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        small.stop()
