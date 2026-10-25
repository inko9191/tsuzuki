# tsuzuki: local long-term memory for agents

**Your agent remembers earlier conversations, and nothing leaves your machine.**

[日本語の README](README.ja.md)

- tsuzuki stores what you and your agent said in past sessions.
- Before each new message, it hands back the relevant past messages, with their dates: a **recall of at most 2,048 tokens instead of the full history**.
- Everything runs locally:
  - EmbeddingGemma 300m for search (through this repository's engine on a Mac, or llama.cpp / Ollama elsewhere);
  - your local chat model to pull facts out of past sessions (tested: Qwen 27B and Gemma 4 E2B on llama.cpp).

It works with [DeepSeek Harness (dsh)](https://github.com/deepseek-ai/deepseek-harness) as a plugin, with Hermes Agent as a memory provider, and with anything else through a small HTTP sidecar on `127.0.0.1`.

The repository also contains **an unofficial inference engine for EmbeddingGemma models (not affiliated with Google)**, written in C++/Metal for Apple silicon.

## Quick start (macOS, Linux, Windows)

You need **Node.js 22 (22.19 or later) or 24+** and **llama.cpp** (its `llama-server`). For Hermes Agent, nothing else (the installer uses Hermes' own Python):

| | Node.js | llama.cpp |
|---|---|---|
| macOS | `brew install node` | `brew install llama.cpp` |
| Windows | `winget install OpenJS.NodeJS.LTS` | `winget install llama.cpp`, or a build from [its releases](https://github.com/ggml-org/llama.cpp/releases) |
| Linux | [nodejs.org](https://nodejs.org) (distribution packages are often too old) | a build from [its releases](https://github.com/ggml-org/llama.cpp/releases), or `brew install llama.cpp` |

Then:

```bash
git clone https://github.com/inko9191/tsuzuki
cd tsuzuki
node scripts/setup.mjs
```

The script:
1. builds the plugins;
2. downloads two models, asking first: **EmbeddingGemma 300m** (334 MB, for search; under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)) and **Gemma 4 E2B** (3.35 GB, Apache 2.0; optional, for pulling facts out of past conversations, which keeps recall better as conversations pile up);
3. connects the memory to **Hermes Agent** (quit Hermes first), or prepares the **HTTP sidecar** for any other app;
4. checks, in a throwaway store, that a message from one conversation is recalled in another and that facts are extracted.

With Hermes there is nothing else to start: the memory and its models start by themselves when Hermes first needs them. For another app, start the memory server with the command setup prints at the end and keep it running. Either way, the memory starts llama.cpp itself and leaves it running for next time (about 0.5 GB of memory for search, about 4 GB more with facts); `node scripts/setup.mjs --stop` stops it. Facts appear about a minute after a conversation (at once after `POST /v1/memory/flush`), and they run on your machine even when your agent uses a cloud model.

Models go to `~/.tsuzuki/models`, the memory server's data to `~/.tsuzuki/memory` (Hermes: `<profile>/tsuzuki`). `node scripts/setup.mjs --help` lists the options (`--for hermes|app`, `--models`, `--llama-server`, `--no-facts`, `--yes`, ...); running it again is safe and continues an interrupted download.

For DeepSeek Harness, your own model servers, Ollama, or this repository's Mac engine, see [Install (manual)](#install-manual). The setup script is written for Windows too (there, file permissions are those of your user folder), but so far it has been run only on macOS.

## What is in this repository

| Part | Path | What it does |
|---|---|---|
| Memory | `dsh-plugins/packages/tsuzuki` (npm name `dsh-tsuzuki`) | Stores the messages, extracts facts, searches, builds the recall block. One code base serves the dsh plugin and the sidecar (`tsuzuki-memory` command). |
| Embedding engine | `native/`, `server/`, `reference/` | Runs EmbeddingGemma (300m and 2) in C++/Metal and serves it through an OpenAI-compatible `/v1/embeddings` daemon (`egemma-server`). **Apple silicon only.** |
| dsh embedding providers | `dsh-plugins/packages/text-embeddings*` | `ctx.textEmbeddings` for dsh. `dsh-text-embeddings-tsuzuki` runs this engine; `dsh-text-embeddings-openai` uses any OpenAI-compatible endpoint, such as Ollama. |
| Hermes provider | `integrations/hermes/` | A Hermes Agent `MemoryProvider` (id `tsuzuki`) that talks to the sidecar. |
| Evaluation | `docs/MEMORY_*`, `scripts/eval/`, `results/eval/memory/` | Pre-registrations, scripts and aggregate results. |

### How memory works

1. **Capture.** User and assistant messages go into a SQLite file. Subagent sessions are skipped. Secrets are redacted first (see [Privacy](#privacy)).
2. **Index.** Each message is embedded together with the message before it.
3. **Facts.** Each session is cut into segments of about 2,500 tokens, or wherever 30 minutes pass with no message. For each segment, a local chat model runs in the background:
   - it lists the facts the segment states, and each fact cites the messages it came from;
   - it adds update links, which mark a newer fact as replacing an older one.

   Facts are searched too, but only the **original messages** they cite are ever returned. The background work waits while the agent or the llama.cpp server is busy.
4. **Recall.** Before each new user message:
   - the message is embedded;
   - message windows and facts are ranked and fused (RRF);
   - the best messages from *other* sessions are selected, up to 2,048 tokens.

   They are shown in date order, under the current date and the rule "the most recent statement is current". There is no reranking by default. A message's text cannot close or open the `<memory>` block, and its second and later lines are indented, so a stored message cannot pose as instructions or as another excerpt (the evaluations rendered multi-line messages without that indentation).
5. **Tools.** `memory_search` (Hermes: `tsuzuki_search`) searches every session, including the current one (no rerank unless you turn it on). `memory_forget` (Hermes: `tsuzuki_forget`) deletes memory; it is **off by default** (see [Privacy](#privacy)).

## Results (pre-registered)

Every decision rule was fixed in writing before any answer was generated. The numbers below are copied from the report files in `results/eval/memory/`.

The plugin's defaults are the tested setting **P**. They were chosen in an earlier exploratory round (U1, 30 questions) and then confirmed in U2 on questions not used before.

**In short**
- **With a small local model** (Gemma 4 E2B), the memory gives better answers than putting the whole history in the context: English +0.150 (E1, at least equivalent), Japanese +0.275 (J1, better).
- **With a large local model** (Qwen 27B): English +0.225 over the whole ~116k-token history with the same template (U2-AT, better); on ~45k-token Japanese histories, the same accuracy within noise at 1/14 of the input and ~17× faster (J1-Q, inconclusive).
- **With thousands of stored sessions** accuracy falls. Facts (on whenever a local chat model is configured) win back part of it; a larger budget and reranking by a small model do not (S2, S3).

**Setup**
- **Data.** [LongMemEval](https://github.com/xiaowu0162/LongMemEval)-S, cleaned release. 40 new questions, drawn by ID and type only with a fixed seed: knowledge-update 10, temporal-reasoning 10, multi-session 10, single-session 6, abstention 4. That is 1,862 history sessions. The question IDs are in `results/eval/memory/longmemeval-u2-plan.json`.
- **Model.** Qwen3.8-27B Q4_K_M (GGUF) on llama.cpp `llama-server` (build b10964), thinking off. The same model answered, extracted the facts and checked the updates, and was the primary judge (LongMemEval's per-type judge prompts).
- **Embeddings.** EmbeddingGemma 300m, fp32, on this repository's Metal engine.
- **Hardware.** One Mac with Apple M5 Max and 64 GB of memory. The U2 and E1 files record the model hashes, the llama.cpp build and the server flags.
- **Statistics.** Paired percentile bootstrap over questions: 10,000 resamples, seed 0.

### U2: memory against the full history (Qwen 27B)

| Setting | What the model is given | Accuracy | Abstained | Input tokens (mean) | Answer time (mean) |
|---|---|---:|---:|---:|---:|
| **P** (tsuzuki's defaults) | recent history + 2,048-token date-aware recall | **0.675** | 0.250 | **2,913** | **5.7 s** |
| C512 (earlier frozen version) | recent history + 512-token recall, reranked | 0.475 | 0.525 | 1,218 | 3.0 s |
| A: full history | every session in a 131,072-token window, earlier frozen template (no question date) | 0.325 | 0.675 | 116,166 | 426.7 s |
| AT: full history, same template as P (U2-AT) | every session in date order, with the question date and P's rules | 0.450 | 0.425 | 116,289 | 431.2 s |

All 40 questions fit in the full-history window.

| Pre-registered comparison | Difference | 95% interval | Verdict | Wins / losses / ties |
|---|---:|---|---|---|
| P − C512 | +0.200 | [+0.075, +0.350] | improved (改善) | 9 / 1 / 30 |
| P − A | +0.350 | [+0.200, +0.500] | better (上回る) | 14 / 0 / 26 |
| P − AT (follow-up U2-AT) | +0.225 | [+0.100, +0.350] | better (上回る) | 9 / 0 / 31 |

So, on these 40 questions with this local model, the 2,048-token memory gave more correct answers than the whole ~116k-token history in the context, with about 40× fewer input tokens and about 75× less time per answer. The full history did better with P's date-aware template (AT − A: +0.125 [+0.025, +0.225], trend only), but still worse than P.

**Second judge (Claude).** Claude graded the U2 answers on sheets with the setting names hidden (Claude also designed the experiment, which is why the names were hidden):
- accuracy P 0.725, C512 0.475, A 0.350; agreement with Qwen 0.960, Cohen's κ 0.918;
- P − C512 +0.250 [+0.100, +0.400]; P − A +0.375 [+0.225, +0.525].

Claude graded the AT answers alone, so that setting could not be hidden: AT 0.525; P − AT +0.200 [+0.050, +0.350] (上回る); agreement with Qwen 0.925, κ 0.85.

**By question type** (Qwen judge; 4–10 questions each, so trends only)

| Setting | knowledge-update | multi-session | single-session | temporal-reasoning | abstention |
|---|---:|---:|---:|---:|---:|
| P | 0.80 | 0.70 | 0.50 | 0.50 | 1.00 |
| C512 | 0.60 | 0.40 | 0.50 | 0.20 | 1.00 |
| A | 0.50 | 0.10 | 0.50 | 0.00 | 1.00 |
| AT | 0.70 | 0.40 | 0.50 | 0.00 | 1.00 |

### E1: a small model (Gemma 4 E2B) as the reader

Same 40 questions ([`docs/MEMORY_LONGMEMEVAL_E1_PREREG.md`](docs/MEMORY_LONGMEMEVAL_E1_PREREG.md)). Gemma 4 E2B (Google's QAT q4_0 GGUF, 131,072-token window) answered. **E2B-P received exactly the messages Qwen's P received in U2**: the memory itself was built by the Qwen 27B pipeline; only the reader changed. All 40 E2B-AT inputs fit the window. Primary judge: Qwen.

| Setting | Accuracy | Abstained | Input tokens | Answer time |
|---|---:|---:|---:|---:|
| E2B-P (Gemma 4 E2B + memory) | **0.525** | 0.025 | 2,942 | 1.0 s |
| E2B-AT (Gemma 4 E2B + full history, same template) | 0.375 | 0.075 | 117,101 | 98.3 s |
| reference: Qwen 27B P (U2) | 0.675 | | | |
| reference: Qwen 27B AT (U2-AT) | 0.450 | | | |

- **Product bar** (E2B-P ≥ 0.50): **met**.
- **Primary, E2B-P − Qwen-AT** (small model + memory against the large model + full history): +0.075 [−0.100, +0.250], **inconclusive** (判定不能), 8 / 5 / 27.
- E2B-P − E2B-AT: +0.150 [−0.025, +0.300], at least equivalent (同等以上（非劣性）), 9 / 3 / 28.
- E2B-P − Qwen-P: −0.150 [−0.325, +0.000], trend only: the cost of the smaller reader.
- Second judge (Claude, blinded, graded before opening the Qwen labels): E2B-P 0.650, E2B-AT 0.475, Qwen-AT 0.525; E2B-P − Qwen-AT +0.125 [−0.025, +0.275] and E2B-P − E2B-AT +0.175 [+0.000, +0.350], both at least equivalent (同等以上（非劣性）); agreement with Qwen 0.887, κ 0.78. The primary verdict stays the Qwen judge's: inconclusive.

### J1: Japanese, everything on Gemma 4 E2B

A self-made Japanese test ([`docs/MEMORY_JA_J1_PREREG.md`](docs/MEMORY_JA_J1_PREREG.md)): no public Japanese long-term memory benchmark was found, so 40 questions were written with their answer sessions and hard negatives (answers fixed by construction), mixed into ~47 generic Japanese chats per question (~48k tokens). **Everything a small machine would run is Gemma 4 E2B**: fact extraction, update links and reading. Only the judge is Qwen 27B.

| Setting | Accuracy | Input tokens | Answer time |
|---|---:|---:|---:|
| E2B-P (memory, all E2B) | **0.550** | 3,190 | 0.9 s |
| E2B-AT (full history, same template) | 0.275 | 47,689 | 20.2 s |

- **Primary, E2B-P − E2B-AT**: +0.275 [+0.150, +0.425], **beats full context** (上回る), 11 / 0 / 29. Product bar met.
- The memory contained every answer session for all 40 questions (coverage 1.00): retrieval held in Japanese; the misses are reading and reasoning (temporal 0.12, multi-session counting 0.25).
- Second judge (Claude, blinded): 0.700 vs 0.425, +0.275 [+0.125, +0.425], beats full context; agreement 0.850, κ 0.71.
- A self-made test is cleaner than real chats; read it as a check that the method carries over to Japanese, not as a benchmark score.

### J1-Q: the same Japanese test, everything on Qwen 27B

The follow-up that J1's rule triggered ([`docs/MEMORY_JA_J1Q_PREREG.md`](docs/MEMORY_JA_J1Q_PREREG.md); run log [`docs/MEMORY_JA_J1Q_RUNLOG.md`](docs/MEMORY_JA_J1Q_RUNLOG.md)): the memory built and read by Qwen3.8 27B.

| Setting | Accuracy | Input tokens | Answer time |
|---|---:|---:|---:|
| Qwen-P (memory) | 0.725 | 3,205 | 5.8 s |
| Qwen-AT (full history, same template) | 0.750 | 45,311 | 99.4 s |

- **Primary, Qwen-P − Qwen-AT**: −0.025 [−0.100, +0.050], **inconclusive** (判定不能), 1 / 2 / 37.
- A large model reads ~45k tokens of Japanese history about as well as it reads the memory. Here the memory's gain is cost (1/14 of the input, ~17× faster), not accuracy; with the small model (J1) the memory was clearly more accurate.
- Second judge (Claude, blinded): 0.825 vs 0.850, −0.025 [−0.100, +0.050], inconclusive; agreement 0.900, κ 0.71.

### S2 and S3: 5,000 stored sessions, answered by Gemma 4 E2B

The scale test (below) showed recall falling as unrelated history piles up. S2 and S3 measured **answers** at 5,000 stored sessions (~52k messages) with Gemma 4 E2B as the reader, on 60 LongMemEval-S questions not used before (10 per non-abstention type), each with its answer sessions at their own dates among the filler sessions ([`docs/MEMORY_SCALE_S2_PREREG.md`](docs/MEMORY_SCALE_S2_PREREG.md), [`docs/MEMORY_SCALE_S3_PREREG.md`](docs/MEMORY_SCALE_S3_PREREG.md), run log [`docs/MEMORY_SCALE_S3_RUNLOG.md`](docs/MEMORY_SCALE_S3_RUNLOG.md)).

| Recall at 5,000 sessions | Accuracy | Answer message in the recall | Against windows only |
|---|---:|---:|---|
| message windows only (what dsh recalls without a chat model) | 0.317 | 0.483 | |
| budget 8,192 | 0.267 | 0.733 | −0.050 [−0.133, +0.033], inconclusive |
| E2B reranks the top 30 windows (the shipped option) | 0.267 | 0.417 | −0.050 [−0.133, +0.017], inconclusive |
| E2B reranks the top 100 windows | 0.250 | 0.350 | −0.067 [−0.133, −0.017], **worse** |
| date filter read from the question | 0.317 | 0.483 | ±0 (it applied to 4 of 60 questions) |
| **facts: setting P (S3)** | **0.400** | 0.667 | +0.083 [−0.017, +0.183], inconclusive, 7 / 2 |
| reference: windows only at 50 sessions | 0.433 | 0.850 | |

- More text does not help a small reader: the 8,192 budget put the answer in front of E2B more often, but it answered worse. As a reranker, E2B chose worse passages than the dense order, so **rerank is off by default**, for `memory_search` too.
- Facts (setting P, with the facts extracted and update-checked by E2B itself) pointed the same way in every comparison: +0.083 here, +0.050 on a second set of 60 questions, +0.067 at 50 sessions (19 wins, 7 losses in all), but no interval excludes zero. At 5,000 sessions P (0.400) is close to windows-only at 50 sessions (0.433).
- Building the facts of 5,000 sessions took E2B about 3.7 hours on the M5 Max (7,074 segments). In normal use this runs in the background after each conversation: about 10–14 s per conversation.
- Second judge (Claude, blinded samples of 90 answers): agreement 0.978 (κ 0.95) in S2 and 0.944 (κ 0.89) in S3.

Full tables, with the V1024R and T2048 ablations:
- [`results/eval/memory/longmemeval-u2-report.md`](results/eval/memory/longmemeval-u2-report.md), [`longmemeval-u2-at-report.md`](results/eval/memory/longmemeval-u2-at-report.md), [`longmemeval-e1-report.md`](results/eval/memory/longmemeval-e1-report.md), [`j1-report.md`](results/eval/memory/j1-report.md), [`j1q-report.md`](results/eval/memory/j1q-report.md), [`s2-report.md`](results/eval/memory/s2-report.md), [`s3-report.md`](results/eval/memory/s3-report.md); exploratory: [`scale-diagnosis.md`](results/eval/memory/scale-diagnosis.md), [`scr1-report.md`](results/eval/memory/scr1-report.md)
- [`longmemeval-u2-claude-judge.md`](results/eval/memory/longmemeval-u2-claude-judge.md), [`longmemeval-u2-at-claude-judge.md`](results/eval/memory/longmemeval-u2-at-claude-judge.md), [`longmemeval-e1-claude-judge.md`](results/eval/memory/longmemeval-e1-claude-judge.md), [`j1-claude-judge.md`](results/eval/memory/j1-claude-judge.md), [`j1q-claude-judge.md`](results/eval/memory/j1q-claude-judge.md), [`s2-claude-judge.md`](results/eval/memory/s2-claude-judge.md), [`s3-claude-judge.md`](results/eval/memory/s3-claude-judge.md)
- Pre-registrations: [`docs/MEMORY_LONGMEMEVAL_U2_PREREG.md`](docs/MEMORY_LONGMEMEVAL_U2_PREREG.md), [`docs/MEMORY_LONGMEMEVAL_U2_AT_PREREG.md`](docs/MEMORY_LONGMEMEVAL_U2_AT_PREREG.md), [`docs/MEMORY_LONGMEMEVAL_E1_PREREG.md`](docs/MEMORY_LONGMEMEVAL_E1_PREREG.md) (the documents are in Japanese)

### About the pre-registrations

This public repository starts from a fresh snapshot (its first commit); later commits are listed in [`docs/SANITIZED.md`](docs/SANITIZED.md). The pre-registrations were committed **before** the results in the author's private development repository; that history is not part of this repository, so it cannot show the order here. What you can check here: each `*-plan.json` in `results/eval/memory/` records the sha256 of its pre-registration document and of the code it ran, and these files are byte-identical to the private repository, except the few listed with both hashes in [`docs/SANITIZED.md`](docs/SANITIZED.md). The original commit history is available on request.

## What is **not** shown

- **Coding and working memory.** `working.enabled` stores tool calls and results and, after dsh compacts the context, gives back the details the summary dropped. It is **experimental and off by default**. The pre-registered test C1 ([`docs/MEMORY_CODING_C1_PREREG.md`](docs/MEMORY_CODING_C1_PREREG.md)) was **stopped without a verdict**: dsh's own summaries were unreliable with the local Qwen 27B at a 32k window, so it would have measured the summaries, not the memory.
- **Japanese beyond a self-made test.** J1 and J1-Q (above) use a small self-made test; no real Japanese user conversations were measured.
- **Thousands of sessions.** In a retrieval-only scale test ([`results/eval/memory/scale-report.md`](results/eval/memory/scale-report.md), in Japanese; LongMemEval-S sessions, message windows only, no facts, no rerank), the share of 60 questions whose answer message was in the recall fell from 0.883 at 50 stored sessions to 0.717 at 500, 0.667 at 2,000 and 0.517 at 5,000 (5,000 − 50: −0.367 [−0.483, −0.250]). **Accuracy falls as unrelated history accumulates.** S2 and S3 (above) measured answers at 5,000 sessions: facts win back part of it; a larger budget and reranking by a small model do not. Nothing beyond 5,000 sessions was measured.
- **Real long-term use.** These tests use benchmark histories, not months of real conversations.
- **Other hardware and the non-Mac path.** Everything was measured on one Mac. The llama.cpp embedding path (EmbeddingGemma 300m, Q8_0 GGUF) was checked against the fp32 engine on that Mac ([`scripts/eval/scale/llamacpp_parity.mjs`](scripts/eval/scale/llamacpp_parity.mjs)): cosine ≥ 0.9997 on 200 message windows and 60 questions, top-10 overlap 0.992, and the same scale-test recall at 500 sessions (0.717 and 0.717) and at 5,000 sessions (0.517 and 0.517). It has **not** been run on Linux or Windows yet. The Ollama path has **not** been measured; Ollama's build of EmbeddingGemma 300m may differ numerically from the fp32 engine. EmbeddingGemma 2 has not been measured in setting P (an exploratory LoCoMo check put its retrieval 0.03 recall@10 below 300m).
- **Other models.** In U2 the answering model, the fact extractor and the judge were all the same local Qwen3.8-27B Q4_K_M. E1 changed only the reader. Cloud models with stronger long-context ability were not tested; such a model could narrow or reverse the gap against the full history.
- **Write-time cost.** The answer times do not include the background work when messages are stored (fact extraction and update checks with the chat model, plus embedding; with Gemma 4 E2B on the M5 Max about 10–14 s per conversation in S3). The full-history times were measured with llama.cpp's host prompt cache off.
- **Comparison with published LongMemEval scores.** Not comparable: published scores use a GPT-4o judge and other models.
- **Precision.** 40 questions give wide intervals. In U2 the primary judge is the same model that answered; the second judge is there to check that.

## Requirements

- **Node.js** `^22.19` or `>=24`. dsh and the sidecar use `node:sqlite`.
- **A chat endpoint for facts.** OpenAI-compatible, with JSON-schema output. Tested: llama.cpp `llama-server` with Qwen3.8-27B Q4_K_M. Without one, memory still works on message windows only; that is **not** the tested setting P.
- **Embeddings.** One of:
  - **Mac with Apple silicon** (the tested path): this repository's engine, running EmbeddingGemma 300m in fp32. You need Xcode, CMake, Python 3.13 and [uv](https://github.com/astral-sh/uv).
  - **Any other machine**: llama.cpp's `llama-server` with the EmbeddingGemma 300m GGUF (checked against the engine), or [Ollama](https://ollama.com) with `embeddinggemma` (not measured).
- **For Hermes Agent**: Hermes' bundled Node (`~/.hermes/node/bin/node`) or another Node as above.

## Install (manual)

### 1. Build the plugins (all hosts)

```bash
cd dsh-plugins
npx -y pnpm@10.26.1 install --frozen-lockfile
npx -y pnpm@10.26.1 run build
npx -y pnpm@10.26.1 -r --filter './packages/*' pack --pack-destination dist
```

This writes the dsh packages to `dsh-plugins/dist/*.tgz`, and the sidecar to `dsh-plugins/packages/tsuzuki/lib/server.js`. The packages are not on npm: always install the tarballs by **absolute path**, never by package name (a package of that name on npm would be installed instead).

### 2. Embeddings

**Mac (tested path).** From the repository root:

```bash
uv venv .venv --python 3.13   # uv downloads Python 3.13 if it is not installed
uv pip install --python .venv/bin/python -r requirements-lock.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build -j 8
```

- `requirements-lock.txt` gives the exact versions used for the results. It also installs the validation stack (PyTorch, transformers).
- EmbeddingGemma 300m is gated under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Accept them on the [model page](https://huggingface.co/google/embeddinggemma-300m), log in with `.venv/bin/hf auth login`, then download the evaluated revision:

```bash
.venv/bin/hf download google/embeddinggemma-300m --revision 57c266a740f537b4dc058e1b0cda161fd15afa75 --local-dir models/embeddinggemma-300m
```

You do not need to start the engine yourself: the dsh provider and the sidecar start it on `127.0.0.1:8299` when they need it.

**Linux, Windows, or a Mac without the engine: llama.cpp** (checked against the engine, see "What is not shown"). Download the GGUF (this also accepts the Gemma Terms of Use; sha256 `b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63`, 334 MB) and start an embedding server on port 8081:

```bash
hf download ggml-org/embeddinggemma-300M-GGUF embeddinggemma-300M-Q8_0.gguf --local-dir models
llama-server -m models/embeddinggemma-300M-Q8_0.gguf --embedding -c 8192 -np 4 -b 2048 -ub 2048 --host 127.0.0.1 --port 8081
```

`-c 8192 -np 4` gives each of the 4 slots 2,048 tokens, EmbeddingGemma's input limit, and `-ub 2048` lets one input fill a slot. (`hf` comes with `pip install huggingface_hub`; or download the file from the model page.) Port 8081 keeps 8080 free for your chat server.

**Or Ollama** (not measured against the engine; pulling also accepts the Gemma Terms of Use):

```bash
ollama pull embeddinggemma
```

### 3a. DeepSeek Harness (CLI profile)

Install into a profile that already runs, or one of the shipped templates (`web`, `headless`, `acp`, `sdk`, `sdk-minimal`). Any other new name gives an empty profile that loads only `dsh-base`, so create a new profile from a template first:

```bash
dsh --profile <name> --from-default-profile headless --dump-config > /dev/null   # only for a new name; or web, acp, sdk, sdk-minimal
```

`dsh plugin` runs pnpm inside the profile directory, so pass **absolute** paths. From the repository root:

```bash
dist="$PWD/dsh-plugins/dist"
dsh plugin --profile <name> add "$dist/dsh-text-embeddings-0.1.0.tgz" "$dist/dsh-text-embeddings-tsuzuki-0.1.0.tgz" "$dist/dsh-tsuzuki-0.1.0.tgz"
```

Then add rows to `$DSH_HOME/profiles/<name>/cordis.patch.yml` (`~/.dsh/profiles/<name>/` unless you set `DSH_HOME`). In a new profile that file holds only `[]` (an empty list): replace the `[]` with the rows, or dsh cannot read the file. A patch replaces the row's whole `config`, so write every setting you need:

```yaml
- id: text-embeddings-tsuzuki
  disabled: false
  config:
    engineRoot: /absolute/path/to/this/repository
    modelDir: /absolute/path/to/this/repository/models/embeddinggemma-300m
    modelName: embeddinggemma-300m   # dtype "auto" -> fp32 for 300m
    port: 8299
- id: tsuzuki
  config:
    llm: { baseUrl: http://127.0.0.1:8080/v1 }   # your local chat server
```

**With llama.cpp or Ollama instead**, add `dsh-text-embeddings-openai-0.1.0.tgz` instead of `dsh-text-embeddings-tsuzuki-0.1.0.tgz`, and use this row (`baseUrl` `http://127.0.0.1:8081/v1` for llama.cpp as above, `http://127.0.0.1:11434/v1` for Ollama):

```yaml
- id: text-embeddings-openai
  disabled: false
  config:
    baseUrl: http://127.0.0.1:8081/v1   # llama.cpp; Ollama: http://127.0.0.1:11434/v1
    model: embeddinggemma
    dimensions: 768
    taskPrefixes:
      query: 'task: search result | query: '
      document: 'title: none | text: '
```

**Turn off other memory.** If the profile has another memory plugin, disable its row (`- id: <its row id>` with `disabled: true`) so that memory is not recalled twice.

Memory is stored in `$DSH_HOME/tsuzuki`.

**Expected warnings** (harmless): pnpm reports "Issues with peer dependencies found" for the `@deepseek-ai/*` packages, and dsh says `dsh-text-embeddings` "declares no dsh.bundle" (dsh supplies those packages, and that package has no row of its own). At run time, Node prints `ExperimentalWarning: SQLite is an experimental feature` once.

**Check that it works.** Start two sessions: tell the agent something in the first, ask about it in the second. The second request carries a `<memory>` block. The store is `$DSH_HOME/tsuzuki/memory.db`.

### 3b. DeepSeek Harness desktop app

Quit the app first, then run `bash dsh-plugins/scripts/install-desktop.sh` (add `llamacpp` or `ollama` as an argument to use llama.cpp on port 8081 or Ollama for embeddings; a second argument sets another local URL). The script backs up the profile files and prints how to undo the change. Memory is stored in `$DSH_HOME/tsuzuki` (`~/.dsh/tsuzuki` by default).

### 3c. Hermes Agent

Before you start:
- **Build the plugins** (step 1). The sidecar runs on Hermes' bundled Node (`~/.hermes/node/bin/node`); without it, on the `node` on your `PATH`, or set `memory.tsuzuki.node` in the profile's `config.yaml`.
- **Embeddings.** With the Mac engine built (step 2), the installer uses it. Otherwise it writes an Ollama configuration and says so: make sure Ollama runs on `127.0.0.1:11434` with `embeddinggemma` pulled. For llama.cpp (step 2), pass `--embedding llamacpp` (port 8081; `--embedding-url` for another local port).
- **Facts.** The installer uses the profile's own model endpoint (`model.base_url`) for fact extraction, but only when it is on this machine (see [Privacy](#privacy)). For a remote endpoint it stops and changes nothing: re-run with `--no-chat` (message windows only, not the tested setting P), or with `--allow-remote-chat` if you accept that conversation text is sent there.

Quit Hermes, then run (the profile must already have a `config.yaml`; `default` is the Hermes home itself):

```bash
python3 integrations/hermes/install.py <profile>            # --home <dir> when Hermes is not in ~/.hermes
```

For each profile, the script backs up `config.yaml`, copies the provider to `<profile>/plugins/tsuzuki`, sets `memory.provider: tsuzuki`, and writes `<profile>/tsuzuki/config.json` (kept if it exists). Each profile gets its own sidecar port (`8297` + a hash of the data directory, so `8297`–`8796`), so profiles do not share memory. Restart Hermes, then run `hermes memory status`.

**Check that memory really works.** `hermes memory status` only checks that Node and the sidecar script exist. After one conversation, `curl -s -H "Authorization: Bearer <token>" http://127.0.0.1:<port>/health` (the port and the token are in `config.json`) must show `embedding.ok: true`, `stats.turns` above 0, and `stats.unindexed` back at 0 a few seconds later.

**Changing `config.json` later.** The sidecar keeps running after Hermes quits and reads `config.json` only when it starts. Stop it (`kill $(lsof -t -iTCP:<port> -sTCP:LISTEN)`), then start Hermes again.

Details (in Japanese): [`integrations/hermes/README.md`](integrations/hermes/README.md).

### 3d. Anything else: the HTTP sidecar

```bash
node dsh-plugins/packages/tsuzuki/lib/server.js --config <dir>/config.json
```

(The packed `dsh-tsuzuki` package installs the same server as the `tsuzuki-memory` command; without `--config` it uses `~/.tsuzuki/config.json`.)

On the first start, the sidecar writes a default `config.json` next to the data: port `8297`, embeddings from Ollama `embeddinggemma`, **no chat model**. For the tested setting, set `chat.baseUrl` and `chat.model` (use `chat.server: "openai"` for servers other than llama.cpp). For llama.cpp embeddings (step 2), set `"embedding": {"baseUrl": "http://127.0.0.1:8081/v1", "model": "embeddinggemma"}`. To use the Mac engine instead, add:

```json
"embedding": {"engine": {"engineRoot": "/abs/repo", "modelDir": "/abs/repo/models/embeddinggemma-300m", "modelName": "embeddinggemma-300m", "port": 8299}}
```

The sidecar reads `config.json` only when it starts: after a change, stop it and start it again.

| Endpoint | Request | Response |
|---|---|---|
| `GET /health` | | `{status, version, dataDir, stats, embedding, chat}`: `embedding` and `chat` are `{url, ok, lastError, lastErrorAt}`; `stats.unindexed` counts what is not embedded yet |
| `POST /v1/memory/messages` | `{session, messages: [{role, text, id?, at?}]}` | `{stored}`, as soon as the messages are stored |
| `POST /v1/memory/recall` | `{query, session?, budget_tokens?}`: current session excluded, no rerank | `{text, ids, reranked}`; `text` is the `<memory>` block or `""`. `503 {error, cause}` when the query cannot be embedded |
| `POST /v1/memory/search` | `{query, budget_tokens?, rerank?}`: all sessions | same |
| `POST /v1/memory/flush` | `{session?}` | `{ok}` |
| `POST /v1/memory/forget` | exactly one of `{session}`, `{before}`, `{all: true}` | `{turns, units, facts, segments, links, vacuumed}` |

Send `Content-Type: application/json` and, when `config.json` has a `token` (a config the sidecar or the Hermes installer writes gets a random one), `Authorization: Bearer <token>`. A body that is not a JSON object is a `400`; one over 8 MB is a `413`.

**When a model hangs or is down**, nothing waits long: a recall gives up after about 3.5 s (the query embedding has 3 s, and after a timeout further queries fail at once for 30 s), a search's rerank, when turned on, after 15 s (at once when the chat server does not answer at all, keeping the dense order), and messages are stored even while the embedding model is down; they are indexed as soon as it answers again.

## Main settings

| Setting (dsh name / sidecar name) | Default | Meaning |
|---|---|---|
| `recall.budgetTokens` | `2048` | Size of the memory block per turn (setting P) |
| `recall.rerank` | `false` | Rerank recall with the chat model. Costs about 16 s per query on a local 27B model. |
| `tool.rerank` / `search.rerank` | `false` | Rerank `memory_search` / `POST /v1/memory/search` with the chat model. Off: with Gemma 4 E2B it lowered answers (S2), and setting P has no rerank. |
| `embedding.queryTimeoutMs` | `3000` | Limit of one query embedding (recall, search) |
| `llm.rerankTimeoutMs` / `chat.rerankTimeoutMs` | `15000` | Limit of one rerank; then the dense order is used |
| `llm.baseUrl` / `chat.baseUrl` | dsh: `http://127.0.0.1:8080/v1`; sidecar: empty | Chat endpoint for facts and rerank. Empty turns both off. |
| `facts.updateLinks` | `true` | Write-time update links |
| `capture.redactSecrets` | `true` | Redact secrets before storing (and in queries) |
| `tool.forget` (Hermes: `memory.tsuzuki.forget` in `config.yaml`) | `false` | Offer the forget tool to the model |
| `working.enabled` | `false` | Working memory for long coding tasks (experimental) |

Full list: [`dsh-plugins/packages/tsuzuki/README.md`](dsh-plugins/packages/tsuzuki/README.md).

## Privacy

- **Local by default.** Messages are kept in a local SQLite file. Embeddings come from `127.0.0.1` (the engine, llama.cpp or Ollama); the installers accept only local embedding URLs. Facts come from the chat endpoint you configure. There is no telemetry.
- **Your text leaves the machine if you point an endpoint at a cloud service.** If `llm.baseUrl` / `chat.baseUrl` or the embedding `baseUrl` is a remote API, stored messages and fact segments are sent there.
  - The Hermes installer uses the profile's model endpoint only when it is on this machine; for a remote one it stops unless you pass `--allow-remote-chat` (or `--no-chat`).
  - dsh's `llm.baseUrl` defaults to `http://127.0.0.1:8080/v1`. Make sure that port is your local chat server, or set it.
- **Secret redaction** is on by default. Before anything is stored, embedded or sent to the fact model (queries too), these become `[REDACTED:KIND]`: API keys and tokens (OpenAI, Anthropic, GitHub, GitLab, Hugging Face, AWS, Google, Slack and Slack webhooks, Stripe, npm, SendGrid, Discord), JWTs, bearer and Basic credentials, credentials inside URLs, `password:`/`OPENAI_API_KEY=`/`token:`-style values (including Japanese labels such as パスワード, and "my password is …" when the value has a digit or a symbol), and Luhn-valid card numbers. It is pattern-based and favours precision. **Unusual secrets can slip through.** Do not paste secrets into chats and count on it.
- **The sidecar.** It listens on `127.0.0.1` only and rejects requests that are not JSON, that carry a browser `Origin`, or that use a `Host` other than loopback (so web pages cannot use it). A config it writes itself, or the Hermes installer writes, has a random `token`: programs that cannot read that file (other users, sandboxes) are refused. A config without a `token` (written by hand, or before this version) lets any program on the machine use it. Another `host` than loopback requires a `token`.
- **Files.** The memory folder is created readable by you only (0700), the SQLite file and its WAL as 0600, `config.json` as 0600. The embedding engine (`egemma-server`) applies the same Origin, Host and JSON checks.
- **Forgetting.** The tools `memory_forget` (dsh) and `tsuzuki_forget` (Hermes) are **off by default**: the tool tells the model to state the exact scope and call it only after the user has confirmed it, but that is an instruction to the model, which a web page or pasted text could override. Turn them on with `tool.forget: true` (dsh) or `memory.tsuzuki.forget: true` in the Hermes profile's `config.yaml`. Other hosts use `POST /v1/memory/forget`. Three scopes: one session, everything captured before a date, or everything. The messages, the windows and facts made from them and their update links are deleted, and the SQLite file and its WAL are rewritten, so the text is gone from the file too. There is no per-message delete. Not covered: the host's own session history, copies or backups of the store, and what the local servers have already seen. To erase everything by hand, stop the host and delete the memory directory: dsh `$DSH_HOME/tsuzuki`; Hermes `<profile>/tsuzuki`; sidecar the `dataDir` of its `config.json` (setup: `~/.tsuzuki/memory`; the models in `~/.tsuzuki/models` are separate).

## Model licenses

No model weights are included in this repository or its packages.

- **EmbeddingGemma 300m** (`google/embeddinggemma-300m`), the model behind every number above, is under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). It is **not bundled**: you accept the terms when you download it from Hugging Face (the original or the GGUF) or with `ollama pull`. The sidecar's automatic Ollama pull refuses to fetch it unless you set `embedding.acceptModelLicense: true`.
- **EmbeddingGemma 2** (`google/embeddinggemma-2`): Apache 2.0, according to its model card, which also points to the Gemma Prohibited Use Policy. The engine supports it (fp16 by default); it has not been measured in setting P.
- **Gemma 4 E2B** (`google/gemma-4-E2B-it-qat-q4_0-gguf`, the reader in E1 and the model in J1): Apache 2.0.
- **Chat model**: your choice. Check its own license.

## License

Apache-2.0 (`LICENSE`, `NOTICE`). You may use, modify and ship it, including in commercial products; the license includes an explicit patent grant. Model weights are not included and keep their own licenses (see "Model licenses").

## Names

This project was developed under the working name "egemma". Everything a user types or reads was renamed to tsuzuki:

| Before | Now |
|---|---|
| `dsh-memory-egemma` (npm), row `memory-egemma`, folder `dsh-plugins/packages/memory-egemma` | `dsh-tsuzuki`, row `tsuzuki`, `dsh-plugins/packages/tsuzuki` |
| `dsh-text-embeddings-egemma` (npm), row `text-embeddings-egemma` | `dsh-text-embeddings-tsuzuki`, row `text-embeddings-tsuzuki` |
| command `egemma-memory`; default config `~/.egemma-memory/config.json` | `tsuzuki-memory`; `~/.tsuzuki/config.json` |
| dsh data `$DSH_HOME/memory-egemma`; Hermes data `<profile>/egemma-memory` | `$DSH_HOME/tsuzuki`; `<profile>/tsuzuki` |
| Hermes provider and plugin folder `egemma_memory`; config block `memory.egemma_memory` | `tsuzuki`; `memory.tsuzuki` |
| Hermes tools `egemma_memory_search`, `egemma_memory_forget` | `tsuzuki_search`, `tsuzuki_forget` |
| `EGEMMA_MEMORY_NODE`, `EGEMMA_MEMORY_SERVER_JS` | `TSUZUKI_NODE`, `TSUZUKI_SERVER_JS` |
| sidecar config `embedding.egemma`; `install-desktop.sh egemma` | `embedding.engine`; `install-desktop.sh engine` |

**Kept under the old name, and why:**
- **The embedding engine's internal names**: the Python package `egemma_server` (`python -m egemma_server`, process name `egemma-server`, log `egemma-server.log`), the native extension `_egemma_native` and `native/egemma_native.py`, the CLI `egemma-cli`, the C++ namespace `egemma`, the CMake project and targets, the Python distribution name `egemma-server` (`server/pyproject.toml`), the reference implementation `reference/egemma_ref`, the engine's environment variables (`EGEMMA_PORT`, `EGEMMA_API_KEY`, `EGEMMA_MATMUL`, ...) and build option `EGEMMA_ENABLE_MPP`, and the value `egemma` that the engine provider reports as `describe().provider` (it is part of the embedding-model key stored with each vector). The evaluation scripts start and import these modules, and their hashes are recorded in the plan files; renaming them would break both.
- **The evaluation record**: the pre-registrations (`docs/MEMORY_*.md`), `scripts/eval/` and `results/eval/memory/` keep the names they used when they ran (for example `dsh-memory-egemma`, or the seed `egemma-memory-stage1-2026-10-12`), so that their sha256 still match the plan files.

"EmbeddingGemma" and "Gemma" appear only where they name Google's models. This project is independent: it is not affiliated with or endorsed by Google, the makers of DeepSeek Harness or Hermes Agent, or Qwen.

## Third-party material

- dsh packages (`@deepseek-ai/*`, MIT) are peer dependencies. They are not bundled.
- LongMemEval's judge prompts (MIT, © 2024 Di Wu) are used for grading. The datasets (LongMemEval, LoCoMo, MTEB, JMTEB) are not redistributed; only IDs, hashes and aggregate scores are included.
- The engine re-implements EmbeddingGemma from the Hugging Face Transformers and vLLM sources (Apache-2.0), with citations. See [`NOTICE`](NOTICE).

## Development notes

- Tests (CPU only): `cd dsh-plugins && npx -y pnpm@10.26.1 test` (tests that need a downloaded model or the built engine are skipped without them); Hermes provider: see [`integrations/hermes/README.md`](integrations/hermes/README.md); dsh end-to-end runs: `dsh-plugins/tests/dsh-e2e/run-memory.sh` and `run-working-memory.sh` (they need a dsh runtime unpacked at `build/dsh-runtime/<version>`).
- Only the result files behind the claims above are included. Some older documents (`docs/DSH_PLUGIN.md`, `docs/NATIVE.md`, `docs/MEMORY_RETRIEVAL_*`) link to benchmark logs and results that stayed in the private development repository.
