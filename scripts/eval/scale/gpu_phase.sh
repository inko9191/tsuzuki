#!/bin/zsh
# GPU part of the memory-egemma scale test. Run it only when nothing else uses the GPU: it refuses to start while
# ports 8091 (llama-server) or 8298 (another egemma-server) listen. Starts its own egemma-server on 8296 and always
# stops it on exit.
#
#   zsh scripts/eval/scale/gpu_phase.sh            # all steps
#   STEPS="bench accuracy" zsh scripts/eval/scale/gpu_phase.sh
set -u
ROOT=${0:A:h}/../../..
cd "$ROOT" || exit 1
OUT=build/scale
LOG=$OUT/gpu_phase.log
STEPS=${STEPS:-"embed build bench accuracy fix"}
PORT=8296
log() { print -r -- "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG" }

for p in 8091 8298; do
  if lsof -nP -iTCP:$p -sTCP:LISTEN >/dev/null 2>&1; then log "port $p is still listening; not starting"; exit 2; fi
done
if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then log "port $PORT is busy; not starting"; exit 2; fi

PYTHONPATH=server .venv/bin/python -m egemma_server --model-dir models/embeddinggemma-300m \
  --model-name embeddinggemma-300m --port $PORT >"$OUT/egemma-$PORT.log" 2>&1 &
SERVER=$!
print $SERVER >"$OUT/egemma-$PORT.pid"
cleanup() {
  kill $SERVER 2>/dev/null
  wait $SERVER 2>/dev/null
  rm -f "$OUT/egemma-$PORT.pid"
  log "egemma-server (pid $SERVER) stopped"
}
trap cleanup EXIT INT TERM
log "egemma-server pid $SERVER starting on $PORT (steps: $STEPS)"
ready=""
for i in {1..300}; do
  st=$(curl -s --max-time 2 http://127.0.0.1:$PORT/health | .venv/bin/python -I -c 'import sys, json; print(json.load(sys.stdin).get("status", ""))' 2>/dev/null)
  if [[ $st == ready ]]; then ready=1; break; fi
  if ! kill -0 $SERVER 2>/dev/null; then break; fi
  sleep 1
done
if [[ -z $ready ]]; then log "egemma-server did not become ready"; exit 3; fi
log "egemma-server ready"

run() {
  log "step: $*"
  if ! "$@" >>"$LOG" 2>&1; then log "FAILED: $*"; exit 4; fi
}
for step in ${=STEPS}; do
  case $step in
    embed) run node --no-warnings scripts/eval/scale/embed_windows.mjs --sequential 100 --concurrency 4 ;;
    build) run node --no-warnings scripts/eval/scale/build_stores.mjs --vectors real --sizes 50,500,2000,5000 ;;
    bench) run node --no-warnings scripts/eval/scale/run_bench.mjs --vectors real --sizes 50,500,2000,5000 --ingest 120 ;;
    accuracy) run node --no-warnings scripts/eval/scale/accuracy.mjs --vectors real --sizes 50,500,2000,5000 --modes fixed,shipped --int8 --validate 50 ;;
    fix)
      mkdir -p $OUT/fix-real
      run node --no-warnings --expose-gc scripts/eval/scale/fix_prototype.mjs --store $OUT/stores-real/s5000 --vectors real --lean --out $OUT/fix-real/s5000-lean.json ;;
  esac
done
log "gpu phase done"
