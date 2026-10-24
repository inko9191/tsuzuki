#!/usr/bin/env python3
"""Interrupt this checkout's memory evaluation, preserving its existing finally checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
MARKER = ROOT / 'build/eval/memory/user-pause.json'
RUNNERS = {'locomo_memory_resume.py', 'locomo_memory_stages.py', 'locomo_answer_eval.py',
           'locomo_answer_production.py'}


def script(argv: str, cwd: Path) -> Path | None:
    """Require an actual Python command and an exact script path in this checkout."""
    try:
        args = shlex.split(argv)
    except ValueError:
        return None
    if not args or not Path(args[0]).name.startswith('python'):
        return None
    args = args[1:]
    while args and args[0] in ('-u', '-B', '-O', '-OO'):
        args.pop(0)
    if not args or args[0].startswith('-'):
        return None
    path = (cwd / args[0]).resolve()
    return path if path.parent == ROOT / 'scripts/eval' and path.name in RUNNERS else None


def inventory() -> list[dict]:
    result = subprocess.check_output(['ps', '-axo', 'pid=,pgid=,command='], text=True)
    processes = []
    for line in result.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        pid, pgid, argv = int(parts[0]), int(parts[1]), parts[2]
        if not any(name in argv for name in RUNNERS):
            continue
        cwd_result = subprocess.run(['lsof', '-a', '-p', str(pid), '-d', 'cwd', '-Fn'],
                                    capture_output=True, text=True)
        cwd_lines = [line[1:] for line in cwd_result.stdout.splitlines() if line.startswith('n')]
        if not cwd_lines:
            continue
        target = script(argv, Path(cwd_lines[0]))
        if target:
            processes.append({'pid': pid, 'pgid': pgid, 'script': target.name})
    return processes


def pause(processes: list[dict], dry_run: bool = False) -> dict:
    groups = sorted({p['pgid'] for p in processes})
    if os.getpgrp() in groups:
        raise RuntimeError('evaluation shares the caller process group; use the dedicated launcher')
    record = {'requested_at': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
              'reason': 'explicit user pause; do not automatically resume',
              'processes': processes, 'signal': 'SIGINT',
              'server_stopped': False, 'dry_run': dry_run}
    if dry_run:
        return record
    # The marker precedes signals so the heartbeat cannot mistake this for an accidental interruption.
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    temporary = MARKER.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(MARKER)
    for pgid in groups:
        try:
            os.killpg(pgid, signal.SIGINT)
        except ProcessLookupError:
            pass
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='inspect targets; do not signal or record a pause')
    args = parser.parse_args()
    print(json.dumps(pause(inventory(), args.dry_run), ensure_ascii=False, indent=2))
