#!/bin/bash
# Install the memory plugin into the DeepSeek Harness desktop app profile (~/.dsh/profiles/desktop).
# The app manages that profile itself: quit the app first; this script refuses while it runs.
#
#   bash dsh-plugins/scripts/install-desktop.sh [engine|ollama]
#
#   engine (default): embeddings from this repository's EmbeddingGemma 300m engine (Metal, fp32; evaluated path).
#   ollama:           embeddings from Ollama's `embeddinggemma` (run `ollama pull embeddinggemma` first;
#                     EmbeddingGemma 300m is under the Gemma Terms of Use: ai.google.dev/gemma/terms).
# Facts and rerank use the desktop agent's local OpenAI-compatible server (default http://127.0.0.1:8080/v1).
set -euo pipefail
backend="${1:-engine}"
here="$(cd "$(dirname "$0")" && pwd)"
plugins="$(cd "$here/.." && pwd)"
engine="$(cd "$plugins/.." && pwd)"
app="/Applications/DeepSeek Harness.app"
dsh="$app/Contents/Resources/runtime/cli/bin/dsh"
profile="$HOME/.dsh/profiles/desktop"
if pgrep -f "$app/Contents/MacOS/DeepSeek Harness" > /dev/null; then echo "Quit DeepSeek Harness first." >&2; exit 1; fi
[ -x "$dsh" ] || { echo "app CLI not found: $dsh" >&2; exit 1; }
[ -d "$profile" ] || { echo "desktop profile not found: $profile" >&2; exit 1; }

(cd "$plugins" && npx -y pnpm@10.26.1 run build > /dev/null && npx -y pnpm@10.26.1 -r --filter './packages/*' pack --pack-destination "$plugins/dist" > /dev/null)
stamp="$(date +%Y%m%d-%H%M%S)"
cp "$profile/cordis.patch.yml" "$profile/cordis.patch.yml.bak-memory-$stamp"
cp "$profile/package.json" "$profile/package.json.bak-memory-$stamp"
provider="$plugins/dist/dsh-text-embeddings-tsuzuki-0.1.0.tgz"
[ "$backend" = ollama ] && provider="$plugins/dist/dsh-text-embeddings-openai-0.1.0.tgz"
"$dsh" plugin --profile desktop add "$plugins/dist/dsh-text-embeddings-0.1.0.tgz" "$provider" "$plugins/dist/dsh-tsuzuki-0.1.0.tgz"

if [ "$backend" = ollama ]; then
  embedding_row="- id: text-embeddings-openai
  disabled: false
  config:
    baseUrl: http://127.0.0.1:11434/v1
    model: embeddinggemma
    dimensions: 768
    taskPrefixes:
      query: 'task: search result | query: '
      document: 'title: none | text: '"
else
  embedding_row="- id: text-embeddings-tsuzuki
  disabled: false
  config:
    engineRoot: '$engine'
    modelDir: '$engine/models/embeddinggemma-300m'
    modelName: embeddinggemma-300m
    port: 8299"
fi
cat >> "$profile/cordis.patch.yml" <<YAML

# --- memory (dsh-tsuzuki), added $stamp; backup: cordis.patch.yml.bak-memory-$stamp ---
$embedding_row
- id: tsuzuki
  config:
    llm:
      baseUrl: http://127.0.0.1:8080/v1
YAML
echo "Installed. Start DeepSeek Harness; memory data goes to ~/.dsh/tsuzuki."
echo "Undo: restore $profile/cordis.patch.yml.bak-memory-$stamp and package.json.bak-memory-$stamp, then '$dsh plugin --profile desktop remove dsh-tsuzuki'."
