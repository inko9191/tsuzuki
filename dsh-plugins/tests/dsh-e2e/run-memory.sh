#!/bin/bash
# Real dsh composition, CPU only (fake embeddings and fake chat model; no GPU, no network model).
# Run 1 tells the agent something; run 2, a new session, must receive it back as a <memory> recall.
# Run 3 checks that facts extracted from the closed segments are indexed and searchable.
# Run 4 asks to forget everything: the fake model calls memory_forget {all: true}. Run 5 must recall nothing of
# runs 1-3, and the store file must no longer contain their text.
# Usage: run-memory.sh [dsh-version] [port]
# Needs the dsh runtime unpacked at build/dsh-runtime/<version> (node_modules/.bin/dsh). Writes its report and logs to
# build/dsh-e2e/memory_e2e_<version>; set E2E_OUT_DIR=<dir> to write them elsewhere (E2E_OUT_DIR=results/dsh updates
# the committed results).
set -euo pipefail
version="${1:-0.2.1-alpha.2}"; port="${2:-8455}"
here="$(cd "$(dirname "$0")" && pwd)"
plugins="$(cd "$here/../.." && pwd)"
engine="$(cd "$plugins/.." && pwd)"
dsh="$engine/build/dsh-runtime/$version/node_modules/.bin/dsh"
if [ ! -x "$dsh" ]; then
  echo "dsh runtime $version not found: $dsh is missing. Unpack dsh $version into build/dsh-runtime/$version first." >&2
  exit 2
fi
mkdir -p "$engine/build/dsh-home-memory"
export DSH_HOME; DSH_HOME="$(mktemp -d "$engine/build/dsh-home-memory/run.XXXXXX")"
export DSH_TELEMETRY_DISABLED=1 FAKE_QWEN_KEY=fake-key
out_dir="${E2E_OUT_DIR:-$engine/build/dsh-e2e}"
mkdir -p "$out_dir"
out="$(cd "$out_dir" && pwd)/memory_e2e_$version"
rm -rf "$out"; mkdir -p "$out"
export FAKE_LOG="$out/fake.jsonl"
node "$here/../fixtures/fake-openai.mjs" "$port" > "$out/fake.txt" 2>&1 & fake=$!
trap 'kill $fake 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null; do kill -0 $fake || exit 1; sleep 0.1; done

(cd "$plugins" && npx -y pnpm@10.26.1 -r --filter './packages/*' pack --pack-destination "$plugins/dist") > "$out/pack.txt" 2>&1
"$dsh" --profile memtest --from-default-profile headless --dump-config > /dev/null 2>&1
"$dsh" plugin --profile memtest add "$plugins/dist/dsh-text-embeddings-0.1.0.tgz" \
  "$plugins/dist/dsh-text-embeddings-openai-0.1.0.tgz" "$plugins/dist/dsh-tsuzuki-0.1.0.tgz" > "$out/plugin-add.txt" 2>&1
cat > "$DSH_HOME/profiles/memtest/cordis.patch.yml" <<YAML
- id: llm-pi-ai
  config:
    providers:
      fakeqwen:
        displayName: Fake Qwen
        apiKeyEnv: FAKE_QWEN_KEY
        api: openai-completions
        baseURL: http://127.0.0.1:$port/v1
        models:
          - id: fake-qwen
            name: Fake Qwen
            contextWindow: 32768
            maxTokens: 1024
- id: agent-default-model
  config:
    provider: fakeqwen
    model: fake-qwen
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
    llm:
      baseUrl: http://127.0.0.1:$port/v1
    facts:
      segmentTokens: 2500
      intervalSeconds: 5
      updateLinks: true
YAML
"$dsh" --profile memtest --dump-config > "$out/dump-config.yml" 2>&1

ask() {  # $1 label, $2 task
  local start; start=$(date +%s)
  "$dsh" --profile memtest "$2" > "$out/$1.stdout" 2> "$out/$1.stderr" || { echo "$1 failed"; tail -30 "$out/$1.stderr"; exit 1; }
  echo "$1: $(( $(date +%s) - start ))s, answer: $(cat "$out/$1.stdout")"
}
ask run1 "I moved to Kyoto last week and my cat Tama loves the new garden."
ask run2 "Remind me where I moved recently?"
sleep 1
ask run3 "What is the name of my cat?"

# The store after runs 1-3, and whether their text is in the file (memory.db and its WAL, raw bytes).
store() {
  node --input-type=module - "$DSH_HOME/tsuzuki" <<'JS'
import { readdirSync, readFileSync } from 'node:fs'
import { DatabaseSync } from 'node:sqlite'
const dir = process.argv[2]
const db = new DatabaseSync(`${dir}/memory.db`, { readOnly: true })
const n = sql => db.prepare(sql).get().n
const files = readdirSync(dir).filter(f => f.startsWith('memory.db')).map(f => readFileSync(`${dir}/${f}`))
console.log(JSON.stringify({ turns: n('SELECT COUNT(*) AS n FROM turns'), units: n('SELECT COUNT(*) AS n FROM units'),
  factUnits: n("SELECT COUNT(*) AS n FROM units WHERE kind = 'fact'"), segments: n('SELECT COUNT(*) AS n FROM segments'),
  pending: n("SELECT COUNT(*) AS n FROM segments WHERE status = 'pending'"), links: n('SELECT COUNT(*) AS n FROM links'),
  kyotoOrTamaOnDisk: files.some(b => b.includes('Kyoto') || b.includes('Tama')) }))
JS
}
store > "$out/store-before-forget.json"
ask run4 "Please forget everything you remember about me."
ask run5 "Remind me where I moved recently?"
store > "$out/store-after-forget.json"

node --input-type=module - "$out" <<'JS'
import { readFileSync, writeFileSync } from 'node:fs'
const [out] = process.argv.slice(2)
const lines = readFileSync(`${out}/fake.jsonl`, 'utf8').trim().split('\n').map(l => JSON.parse(l))
const agent = lines.filter(l => l.kind === 'agent')
const requests = agent.map(l => JSON.stringify(l.body.messages))
// One agent request per run (two in run 4: the tool call, then the answer after its result); session-title requests
// are separate and never carry recall.
const mainLines = agent.filter(l => !JSON.stringify(l.body.messages).includes('Generate the session title'))
const main = mainLines.map(l => JSON.stringify(l.body.messages))
const MARK = 'Excerpts from earlier conversations'
const recalls = main.map(m => m.includes(MARK))
const before = JSON.parse(readFileSync(`${out}/store-before-forget.json`, 'utf8'))
const after = JSON.parse(readFileSync(`${out}/store-after-forget.json`, 'utf8'))
const toolResult = JSON.stringify(mainLines[4]?.body.messages.find(m => m.role === 'tool')?.content ?? '')
const report = {
  mainRequests: main.length,
  recallPerRun: recalls,
  run1HasNoMemory: !recalls[0],
  run2RecallsKyoto: recalls[1] === true && main[1].includes('Kyoto'),
  run3RecallsTama: recalls[2] === true && main[2].includes('Tama'),
  titleRequestsWithoutRecall: requests.filter(m => m.includes('Generate the session title')).every(m => !m.includes(MARK)),
  factCalls: lines.filter(l => l.kind === 'fact').length,
  updateCalls: lines.filter(l => l.kind === 'updates').length,
  laterUpdateNoteInRun3: main[2]?.includes('Later update') ?? false,
  store: before,
  forget: {
    toolsOffered: ['memory_search', 'memory_forget'].every(t => mainLines[0]?.body.tools?.includes(t)),
    calledInRun4: mainLines[3]?.body.call === 'memory_forget',
    toolResult: toolResult.slice(0, 300),
    run5RecallsNothingOfRuns1to3: !(main[5] ?? 'Kyoto').includes('Kyoto') && !(main[5] ?? 'Tama').includes('Tama'),
    storeAfter: after,
  },
}
report.ok = main.length === 6 && report.run1HasNoMemory && report.run2RecallsKyoto && report.run3RecallsTama
  && report.titleRequestsWithoutRecall && before.turns >= 6 && before.factUnits > 0 && before.kyotoOrTamaOnDisk
  && report.forget.toolsOffered && report.forget.calledInRun4 && /Forgot all memory/.test(toolResult)
  && report.forget.run5RecallsNothingOfRuns1to3 && !after.kyotoOrTamaOnDisk
writeFileSync(`${out}/report.json`, JSON.stringify(report, null, 2))
console.log(JSON.stringify(report, null, 2))
process.exit(report.ok ? 0 : 1)
JS
