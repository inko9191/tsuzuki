#!/bin/zsh
set -euo pipefail
task_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$task_root"
exec /usr/bin/python3 scripts/eval/locomo_memory_pause.py
