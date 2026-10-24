#!/bin/bash
# Working memory in a real dsh composition, CPU only (fake coding model + fake embeddings).
# The fake agent runs bash commands whose outputs overflow a 16k window, so dsh compacts; the summary drops the
# marker printed by the first command. Arm S (summary only) must lose it; arm SR (summary + working memory) must
# get it back in a <memory> block after the compaction.
# Usage: run-working-memory.sh [dsh-version] [port]
# Needs the dsh runtime unpacked at build/dsh-runtime/<version> (node_modules/.bin/dsh). Writes its report and logs to
# build/dsh-e2e/working_memory_e2e_<version>; set E2E_OUT_DIR=<dir> to write them elsewhere (E2E_OUT_DIR=results/dsh updates
# the committed results).
set -euo pipefail
version="${1:-0.2.1-alpha.2}"; port="${2:-8456}"
here="$(cd "$(dirname "$0")" && pwd)"
plugins="$(cd "$here/../.." && pwd)"
engine="$(cd "$plugins/.." && pwd)"
dsh="$engine/build/dsh-runtime/$version/node_modules/.bin/dsh"
if [ ! -x "$dsh" ]; then
  echo "dsh runtime $version not found: $dsh is missing. Unpack dsh $version into build/dsh-runtime/$version first." >&2
  exit 2
fi
mkdir -p "$engine/build/dsh-home-working"
export DSH_HOME; DSH_HOME="$(mktemp -d "$engine/build/dsh-home-working/run.XXXXXX")"
export DSH_TELEMETRY_DISABLED=1 FAKE_KEY=fake-key FAKE_STEPS=12
out_dir="${E2E_OUT_DIR:-$engine/build/dsh-e2e}"
mkdir -p "$out_dir"
out="$(cd "$out_dir" && pwd)/working_memory_e2e_$version"
rm -rf "$out"; mkdir -p "$out"
(cd "$plugins" && npx -y pnpm@10.26.1 -r --filter './packages/*' pack --pack-destination "$plugins/dist") > "$out/pack.txt" 2>&1

for arm in S SR; do
  export FAKE_LOG="$out/$arm.jsonl"
  node "$here/../fixtures/fake-coder.mjs" "$port" > "$out/$arm.fake.txt" 2>&1 & fake=$!
  until curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null; do kill -0 $fake || exit 1; sleep 0.1; done
  profile="work$arm"
  "$dsh" --profile $profile --from-default-profile headless --dump-config > /dev/null 2>&1
  "$dsh" plugin --profile $profile add "$plugins/dist/dsh-text-embeddings-0.1.0.tgz" \
    "$plugins/dist/dsh-text-embeddings-openai-0.1.0.tgz" "$plugins/dist/dsh-tsuzuki-0.1.0.tgz" > "$out/$arm.plugin-add.txt" 2>&1
  working=false; [ $arm = SR ] && working=true
  cat > "$DSH_HOME/profiles/$profile/cordis.patch.yml" <<YAML
- id: llm-pi-ai
  config:
    providers:
      fake:
        displayName: Fake Coder
        apiKeyEnv: FAKE_KEY
        api: openai-completions
        baseURL: http://127.0.0.1:$port/v1
        models:
          - id: fake-coder
            name: Fake Coder
            contextWindow: 16384
            maxTokens: 1024
- id: agent-default-model
  config:
    provider: fake
    model: fake-coder
- id: compaction-basic
  config:
    thresholdRatio: 0.8
    headroomTokens: 1
    retainTokens: 3000
- id: text-embeddings-openai
  disabled: false
  config:
    baseUrl: http://127.0.0.1:$port/v1
    model: fake-embed
    dimensions: 64
    taskPrefixes:
      query: 'task: search result | query: '
      document: 'title: none | text: '
- id: tsuzuki
  config:
    dataDir: $DSH_HOME/memory-$arm
    llm:
      baseUrl: ''
    facts:
      enabled: false
    working:
      enabled: $working
YAML
  work="$DSH_HOME/work-$arm"; mkdir -p "$work"
  start=$(date +%s)
  (cd "$work" && "$dsh" --profile $profile --json "Find out why the widget build fails. Run the diagnostics." > "$out/$arm.events.jsonl" 2> "$out/$arm.stderr") \
    || { echo "$arm failed"; tail -30 "$out/$arm.stderr"; kill $fake; exit 1; }
  echo "$arm: $(( $(date +%s) - start ))s"
  kill $fake; wait $fake 2>/dev/null || true
done

node --input-type=module - "$out" <<'JS'
import { readFileSync, writeFileSync } from 'node:fs'
const out = process.argv[2]
const read = arm => readFileSync(`${out}/${arm}.jsonl`, 'utf8').trim().split('\n').map(l => JSON.parse(l))
const arm = name => {
  const lines = read(name)
  const agent = lines.filter(l => l.kind === 'agent').map(l => l.body)
  const summaries = lines.filter(l => l.kind === 'summary').map(l => l.body)
  const afterSummary = agent.filter(a => a.hasSummary)
  return {
    agentRequests: agent.length, summaryRequests: summaries.length, maxTokens: Math.max(...agent.map(a => a.tokens)),
    tools: agent[0]?.tools, compacted: afterSummary.length > 0,
    markerAfterCompaction: afterSummary.some(a => a.hasMarker), memoryAfterCompaction: afterSummary.some(a => a.hasMemory),
  }
}
const report = { S: arm('S'), SR: arm('SR') }
report.ok = report.S.compacted && report.SR.compacted && !report.S.markerAfterCompaction
  && report.SR.markerAfterCompaction && report.SR.memoryAfterCompaction
writeFileSync(`${out}/report.json`, JSON.stringify(report, null, 2))
console.log(JSON.stringify(report, null, 2))
process.exit(report.ok ? 0 : 1)
JS
