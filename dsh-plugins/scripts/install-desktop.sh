#!/bin/bash
# Install the memory plugin into the DeepSeek Harness desktop app profile (~/.dsh/profiles/desktop).
# The app manages that profile itself: quit the app first; this script refuses while it runs.
#
#   bash dsh-plugins/scripts/install-desktop.sh [engine|ollama|llamacpp [URL]]
#
#   engine (default): embeddings from this repository's EmbeddingGemma 300m engine (Metal, fp32; evaluated path).
#   ollama:           embeddings from Ollama's `embeddinggemma` (run `ollama pull embeddinggemma` first;
#                     EmbeddingGemma 300m is under the Gemma Terms of Use: ai.google.dev/gemma/terms).
#   llamacpp [URL]:   embeddings from llama.cpp's `llama-server --embedding` with the EmbeddingGemma 300m GGUF
#                     (default URL http://127.0.0.1:8081/v1; see the README).
# Facts and rerank use the desktop agent's local OpenAI-compatible server (default http://127.0.0.1:8080/v1).
set -euo pipefail
backend="${1:-engine}"
embed_url="${2:-}"
case "$backend" in engine|ollama|llamacpp) ;; *) echo "unknown backend: $backend (engine, ollama or llamacpp)" >&2; exit 1;; esac
[ "$backend" = ollama ] && embed_url="${embed_url:-http://127.0.0.1:11434/v1}"
[ "$backend" = llamacpp ] && embed_url="${embed_url:-http://127.0.0.1:8081/v1}"
case "$embed_url" in ""|http://127.0.0.1:*|http://localhost:*|http://\[::1\]:*) ;; *) echo "the embedding URL must be on this machine: $embed_url" >&2; exit 1;; esac
here="$(cd "$(dirname "$0")" && pwd)"
plugins="$(cd "$here/.." && pwd)"
engine="$(cd "$plugins/.." && pwd)"
app="/Applications/DeepSeek Harness.app"
dsh="$app/Contents/Resources/runtime/cli/bin/dsh"
profile="${DSH_HOME:-$HOME/.dsh}/profiles/desktop"
if pgrep -f "$app/Contents/MacOS/DeepSeek Harness" > /dev/null; then echo "Quit DeepSeek Harness first." >&2; exit 1; fi
[ -x "$dsh" ] || { echo "app CLI not found: $dsh" >&2; exit 1; }
[ -d "$profile" ] || { echo "desktop profile not found: $profile" >&2; exit 1; }
if grep -q '^- id: tsuzuki' "$profile/cordis.patch.yml" 2>/dev/null; then
  echo "Already installed (cordis.patch.yml has a tsuzuki row). To reinstall, restore a cordis.patch.yml.bak-memory-* backup or delete the rows this script added, then run it again." >&2
  exit 1
fi
case "$engine" in *"'"*) echo "the repository path contains a quote, which this script cannot write into YAML: $engine" >&2; exit 1;; esac

(cd "$plugins" && npx -y pnpm@10.26.1 run build > /dev/null && npx -y pnpm@10.26.1 -r --filter './packages/*' pack --pack-destination "$plugins/dist" > /dev/null)
stamp="$(date +%Y%m%d-%H%M%S)"
cp "$profile/cordis.patch.yml" "$profile/cordis.patch.yml.bak-memory-$stamp"
cp "$profile/package.json" "$profile/package.json.bak-memory-$stamp"
provider="$plugins/dist/dsh-text-embeddings-tsuzuki-0.1.0.tgz"
[ "$backend" != engine ] && provider="$plugins/dist/dsh-text-embeddings-openai-0.1.0.tgz"
"$dsh" plugin --profile desktop add "$plugins/dist/dsh-text-embeddings-0.1.0.tgz" "$provider" "$plugins/dist/dsh-tsuzuki-0.1.0.tgz"

if [ "$backend" != engine ]; then
  embedding_row="- id: text-embeddings-openai
  disabled: false
  config:
    baseUrl: $embed_url
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
# A new profile's patch file is a flow-style empty list (`[]`); block rows cannot follow it.
sed -i '' '/^[[:space:]]*\[\][[:space:]]*$/d' "$profile/cordis.patch.yml"
cat >> "$profile/cordis.patch.yml" <<YAML

# --- memory (dsh-tsuzuki), added $stamp; backup: cordis.patch.yml.bak-memory-$stamp ---
$embedding_row
- id: tsuzuki
  config:
    llm:
      baseUrl: http://127.0.0.1:8080/v1
YAML
echo "Installed. Start DeepSeek Harness; memory data goes to ~/.dsh/tsuzuki."
echo "Undo: restore $profile/cordis.patch.yml.bak-memory-$stamp and package.json.bak-memory-$stamp, then '$dsh plugin --profile desktop remove dsh-tsuzuki dsh-text-embeddings $(basename "$provider" -0.1.0.tgz)'."
