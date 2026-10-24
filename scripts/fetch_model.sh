#!/usr/bin/env bash
# Download google/embeddinggemma-2 into models/embeddinggemma-2.
# Gemma checkpoints are gated: accept the license on the model page and `hf auth login` first.
set -euo pipefail
cd "$(dirname "$0")/.."
hf download google/embeddinggemma-2 --local-dir models/embeddinggemma-2
ls -la models/embeddinggemma-2
