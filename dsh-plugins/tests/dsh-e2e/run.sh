#!/bin/bash
# Real-composition check: install the packed bundles into a fresh profile under an
# isolated DSH_HOME, enable the engine provider (dsh-text-embeddings-tsuzuki), boot dsh with a probe overlay.
# Usage: run.sh <dsh-version> <port> [engine|openai]
#   openai: start egemma-server outside dsh and reach it through dsh-text-embeddings-openai.
set -euo pipefail
version="$1"; port="$2"; provider="${3:-engine}"
here="$(cd "$(dirname "$0")" && pwd)"
plugins="$(cd "$here/../.." && pwd)"
engine="$(cd "$plugins/.." && pwd)"
dsh="$engine/build/dsh-runtime/$version/node_modules/.bin/dsh"
export DSH_HOME="$engine/build/dsh-home-e2e/$version"
export DSH_TELEMETRY_DISABLED=1
out="$engine/results/dsh/e2e_${version}_$provider"
rm -rf "$DSH_HOME"; mkdir -p "$DSH_HOME" "$out"
"$dsh" plugin --profile embtest add "$plugins/dist/dsh-text-embeddings-0.1.0.tgz" \
  "$plugins/dist/dsh-text-embeddings-tsuzuki-0.1.0.tgz" "$plugins/dist/dsh-text-embeddings-openai-0.1.0.tgz" > "$out/plugin-add.txt" 2>&1
if [ "$provider" = engine ]; then
cat > "$DSH_HOME/profiles/embtest/cordis.patch.yml" <<YAML
- id: text-embeddings-tsuzuki
  disabled: false
  config:
    engineRoot: $engine
    port: $port
YAML
else
PYTHONPATH="$engine/server" "$engine/.venv/bin/python" -m egemma_server --port "$port" > "$out/daemon.txt" 2>&1 & daemon=$!
trap 'kill $daemon 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$port/health" > /dev/null; do kill -0 $daemon || exit 1; sleep 0.2; done
cat > "$DSH_HOME/profiles/embtest/cordis.patch.yml" <<YAML
- id: text-embeddings-openai
  disabled: false
  config:
    baseUrl: http://127.0.0.1:$port/v1
    model: embeddinggemma-2
    dimensions: 768
    allowedDimensions: [512, 256, 128]
    sendDimensions: true
    taskPrefixes:
      query: 'task: search result | query: '
      document: 'title: none | text: '
YAML
fi
cat > "$out/probe.patch.yml" <<YAML
- insert:
    - id: text-embeddings-probe
      name: $here/probe.mjs
      config:
        out: $out/probe.json
YAML
"$dsh" --profile embtest --dump-config > "$out/dump-config.yml" 2>&1
start=$(date +%s)
"$dsh" --profile embtest --patch "$out/probe.patch.yml" > "$out/boot.txt" 2>&1 & pid=$!
for i in $(seq 1 300); do kill -0 $pid 2>/dev/null || break; sleep 1; done
if kill -0 $pid 2>/dev/null; then kill -TERM $pid; echo "dsh did not exit" >> "$out/boot.txt"; fi
wait $pid; status=$?
echo "{\"version\":\"$version\",\"exit\":$status,\"seconds\":$(( $(date +%s) - start ))}" > "$out/run.json"
sleep 1
if [ "$provider" = engine ] && curl -s "http://127.0.0.1:$port/health" > /dev/null; then echo "daemon still running after dsh exit" >> "$out/boot.txt"; status=9; fi
cat "$out/run.json"; cat "$out/probe.json" 2>/dev/null || tail -40 "$out/boot.txt"
exit $status
