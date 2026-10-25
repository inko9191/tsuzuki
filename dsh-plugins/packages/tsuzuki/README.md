# dsh-tsuzuki

**tsuzuki** for DeepSeek Harness: long-term conversation memory, built on the retrieval pipeline this repository evaluated on
LoCoMo and LongMemEval (`docs/MEMORY_RETRIEVAL_STAGES.md`, `docs/MEMORY_LONGMEMEVAL_G1_PREREG.md`): it remembers
what was said in earlier sessions and hands the relevant messages back, with their dates, before the next user
message.

## How it works

1. **Capture.** Committed user and assistant messages (`session/event`) are stored in a SQLite file
   (`node:sqlite`, no native addon). Subagent sessions are skipped.
2. **Index.** Each message is embedded together with the message before it (the `p2` window) through
   `ctx.textEmbeddings`, with EmbeddingGemma's `document` prompt.
3. **Facts.** A session is cut into segments (about 2,500 tokens, or 30 idle minutes). In the background, an
   OpenAI-compatible chat model (the local llama.cpp Qwen server by default) lists the facts each segment states,
   each citing the messages it came from. Facts are embedded and searched too, but only cited raw messages are ever
   returned. Background work waits while the agent is running or the llama.cpp server is busy (`GET /slots`),
   so the agent's own request is never queued behind it.
4. **Recall.** Before a step that carries a new user message (`agent/pre-step`), the message is embedded with the
   `query` prompt; windows and facts are ranked by cosine, fused with RRF (k=60), deduplicated, optionally reranked
   by the chat model, and the best raw messages from **other** sessions are selected within a token budget. They are
   prepended as a `recall` message:

   ```
   <memory>
   Current date: 2026/10/22 (Thu) 12:26
   Excerpts from earlier conversations with this user, in date order. ...
   Date: 2026/10/15 (Thu) 09:10
   [0f904653-8] user: I moved to Kyoto last week ...
   </memory>
   ```

   The system prompt only gets a static guide, so the prefix cache is not invalidated each turn.
5. **Tools.** `memory_search` searches all sessions (including the current one). It can rerank with the chat model (`tool.rerank`), off by default.
   `memory_forget` deletes memory (below; off unless `tool.forget: true`).

The ranking, parsing and selection code (`src/pipeline.ts`) is a port of the evaluated Python;
`test/conformance.test.mjs` replays fixtures produced by the Python itself (`test/make_fixtures.py`) and must match
exactly.

## Differences from the evaluated setup

| Evaluated (LongMemEval U2, arm P) | Here | Why |
|---|---|---|
| A benchmark session is the fact-extraction unit | A segment of a dsh session (≈2,500 tokens or 30 idle min) | dsh sessions can last days |
| Recent 1,024 tokens of history + 2,048 of memory | The live session is already in context; only other sessions are recalled | — |
| Dense order, no rerank (P) | Same; rerank is opt-in | Rerank costs ~16 s per query on a local 27B model, and with Gemma 4 E2B it lowered recall and answers at 5,000 sessions (S2) |
| Qwen tokenizer counts | Estimated tokens (ASCII ≈ 4 chars, other characters ≈ 1) | No tokenizer in the plugin; never undercounts Japanese |
| One date per session | A date line per session and day | A dsh session can span days |
| Messages as written | Each message clipped to ~600 estimated tokens; rerank passages to ~250; secrets redacted | Fit one 2,048-token embedding input and the rerank window |

## Measured (pre-registered, LongMemEval-S, 40 new questions, local Qwen 27B Q4, Qwen judge)

The defaults are U2's arm **P**: date-aware recall (current date, "most recent is current"), write-time update links,
memory budget 2048, dense order without rerank.

| | accuracy | input tokens | answer time |
|---|---:|---:|---:|
| P (this plugin's defaults) | **0.675** | 2,913 | 5.7 s |
| frozen C-final (memory 512, rerank) | 0.475 | 1,218 | 3.0 s |
| full history in the context | 0.325 | 116,166 | 426.7 s |

P − frozen: +0.200 (9/1/30, 95% CI [+0.075, +0.350]); P − full history: +0.350 (14/0/26, [+0.200, +0.500]).
A blinded second judge (Claude) agreed (κ 0.92). English synthetic histories only; Japanese and real long-term use
are not verified yet. Details: `results/eval/memory/longmemeval-u2-report.md`, `docs/MEMORY_LONGMEMEVAL_U2_PREREG.md`.

## Requirements

- dsh `0.2.0-rc.2` or `0.2.1-alpha.2` (Node ≥ 22.19).
- One `ctx.textEmbeddings` provider: `dsh-text-embeddings-tsuzuki` (local engine) or `dsh-text-embeddings-openai`.
  The task names `query` and `document` must exist (EmbeddingGemma's prompts; for the OpenAI-compatible provider
  set `taskPrefixes`).
- For facts and rerank: an OpenAI-compatible chat endpoint with JSON-schema output. Without one (`llm.baseUrl: ''`),
  memory still works on windows only.

## Install

The packages are not on npm. Build and pack them in `dsh-plugins/` (`npx -y pnpm@10.26.1 install --frozen-lockfile`,
`npx -y pnpm@10.26.1 run build`, `npx -y pnpm@10.26.1 -r --filter './packages/*' pack --pack-destination dist`), then add
the tarballs by **absolute** path: `dsh plugin` runs pnpm inside the profile directory, where a relative path is not
found. Never add them by package name: that would fetch whatever is published under that name on npm.

Install into a profile that already runs, or one of the shipped templates (`web`, `headless`, `acp`, `sdk`,
`sdk-minimal`). Any other new name gets an empty profile that loads only `dsh-base` and does nothing when started, so
create it from a template first:

```sh
dsh --profile <name> --from-default-profile headless --dump-config > /dev/null   # only for a new name; or web, acp, sdk
dist="$PWD/dsh-plugins/dist"                                                 # from the repository root
dsh plugin --profile <name> add "$dist/dsh-text-embeddings-0.1.0.tgz" "$dist/dsh-text-embeddings-tsuzuki-0.1.0.tgz" \
  "$dist/dsh-tsuzuki-0.1.0.tgz"
```

Use `dsh-text-embeddings-openai-0.1.0.tgz` instead of the engine provider (`dsh-text-embeddings-tsuzuki`) for Ollama. pnpm then reports "Issues with
peer dependencies found" for the `@deepseek-ai/*` packages, and dsh says `dsh-text-embeddings` "declares no
dsh.bundle"; both are expected (dsh supplies those packages, and that package has no row of its own). While the plugin
runs, Node prints `ExperimentalWarning: SQLite is an experimental feature` once; it is harmless (`node:sqlite`).

The row is enabled by the bundle. If the profile has another memory plugin, disable its row (`- id: <its row id>` with
`disabled: true`) to avoid double recall. Override settings in `$DSH_HOME/profiles/<name>/cordis.patch.yml`
(`~/.dsh/profiles/<name>` by default; a patch replaces the row's whole `config`):

```yaml
- id: tsuzuki
  config:
    llm: { baseUrl: http://127.0.0.1:8080/v1 }
    recall: { rerank: false, budgetTokens: 512 }
```

| Config | Default | Meaning |
|---|---|---|
| `dataDir` | `$DSH_HOME/tsuzuki` | Store directory, shared by all profiles |
| `capture.enabled` / `skipSubagents` | `true` / `true` | |
| `recall.enabled` | `true` | Prepend excerpts before each new user message |
| `recall.budgetTokens` | `2048` | Excerpt budget (U2 arm P) |
| `recall.rerank` | `false` | Rerank automatic recall with the chat model |
| `recall.timeoutMs` | `4000` | Skip recall for the turn when retrieval is slower (its requests are aborted) |
| `embedding.queryTimeoutMs` | `3000` | Limit of one query embedding (recall, `memory_search`); storing uses the provider's own limit. After a timeout, queries fail at once for 30 s or until the model answers again |
| `llm.rerankTimeoutMs` | `15000` | Limit of one rerank; then the dense order is used. A chat server that does not answer `GET /models` within 1.5 s is not asked |
| `tool.enabled` / `rerank` / `budgetTokens` | `true` / `false` / `1024` | `memory_search` |
| `tool.forget` | `false` | Offer `memory_forget` (only while `tool.enabled`). Off by default: a model steered by a web page or pasted text could call it, and asking the user first is only an instruction to the model |
| `llm.baseUrl` | `http://127.0.0.1:8080/v1` | Chat endpoint for facts and rerank; empty disables both |
| `llm.server` | `llama.cpp` | Adds `cache_prompt` and thinking-off, enables the `/slots` busy check |
| `facts.segmentTokens` / `segmentIdleMinutes` | `2500` / `30` | Segment size for fact extraction |
| `facts.updateLinks` | `true` | Write-time update links (U1 arm V) |
| `capture.redactSecrets` | `true` | Replace API keys, tokens, passwords, private keys and card numbers before storing |
| `embedding.queryTask` / `documentTask` | `query` / `document` | Provider task names |
| `working.enabled` | `false` | Working memory for long tasks (below) |
| `working.budgetTokens` | `2048` | Excerpt budget of one working recall |
| `working.toolResultChars` / `callArgsChars` | `1600` / `400` | Characters stored per tool result (head and tail) / tool-call arguments |

### Working memory (long tasks; experimental)

A coding task needs its working set verbatim — the files being edited, the latest errors, the plan, the recent
turns — and memory cannot replace that. What grows without bound is the old part of the history, which dsh summarizes
when the context fills up; the summary drops details such as an error message from fifty steps ago. With
`working.enabled`, the plugin also stores tool calls and results, and when a compaction replaces part of the session
(`compaction/summary`), the next step gets, once, the summarized-away messages that fit it (query: the task, the
last assistant message and the last tool result), verbatim, within `working.budgetTokens`. Messages still in the
context are never repeated, and `memory_search` reaches the same range. File contents are not served from memory;
the block tells the model to re-read files. Whether this helps is **not shown**: the pre-registered test C1
(`docs/MEMORY_CODING_C1_PREREG.md`) was stopped without a verdict, because dsh's own summaries were unreliable with
the local model at a 32k window.

## Forgetting

Memory can be deleted in three scopes: one session, every message captured before a date, or everything. In dsh the
model does it with the tool `memory_forget` (`session: "current"` or a session id, `before: "2026-10-01"`, or
`all: true`); its description tells the model to say exactly what will be deleted and to call it only after the user
has confirmed that scope. Subagents cannot call it. Other hosts use `POST /v1/memory/forget` on the sidecar (below).
Exactly one scope must be given; anything else is refused without deleting.

- **What goes.** The messages, every window and fact that cites one of them, and all facts of a segment that held
  one (a fact may draw on any message of its segment). A segment that keeps some messages is narrowed to them and
  its facts are extracted again. Update links through a deleted fact are removed (old -> deleted -> new becomes
  old -> new). A kept message whose window showed a deleted one is embedded again without it right after.
- **From the file, not only from queries.** The store is rewritten (`VACUUM`) and its WAL emptied, so the text is no
  longer in `memory.db` or `memory.db-wal`; the tests check the raw bytes. If another process is reading the file at
  that moment, the rows are deleted at once (their cells zeroed, `secure_delete`) but the rewrite has to wait: the
  result says `vacuumed: false`, and the next background backfill (within 10 minutes) rewrites the file. Another dsh
  process sharing the store drops its caches, and fact extraction or embedding that was in flight during the forget
  stores nothing.
- **Not covered.** dsh's own session history and Hermes' session store (forget them there), copies or backups of
  the store (Time Machine, sync folders), and what the local embedding and chat servers have already seen. Deleting
  rows is not a secure erase of the disk.
- `before` is an ISO 8601 date or date-time; a date alone means local midnight (the dates in excerpts are local
  time). Messages are dated by when they were captured.

## Desktop app (DeepSeek Harness.app)

The app runs plugins on its Electron Node 24.18.1 (`node:sqlite` works) and manages `~/.dsh/profiles/desktop` itself,
so install with the app's own CLI while the app is quit — `scripts/install-desktop.sh` does that, backs up the
profile files and appends the rows. The app ships no `ctx.textEmbeddings` provider, so the script also installs
`dsh-text-embeddings` and a provider (this repository's engine, or Ollama).

## Sidecar (other hosts: Hermes Agent, scripts)

`node lib/server.js --config <dir>/config.json` (or the `tsuzuki-memory --config …` command of the installed package)
serves the same memory on `127.0.0.1:8297` by default (a default config is written on first start; the Hermes provider
gives each profile its own port, `8297` + a hash of the data directory). The config is read only at start: after
changing `config.json`, stop the running sidecar (its pid is in its log, or `lsof -t -iTCP:<port> -sTCP:LISTEN`) and
start it again. Every route is refused for browsers (any `Origin`), for a foreign `Host`, for a POST that is not
`application/json`, and without the bearer token when `token` is set. A config the sidecar writes itself (first start)
gets a random `token`, and the file is readable by the user only (0600); send `Authorization: Bearer <token>`. A body
that is not a JSON object is a `400`; one over 8 MB is a `413`. Recalled message text cannot close or open the
`<memory>` block, and its continuation lines are indented, so a stored message cannot pose as another excerpt.
Instead of an embedding or chat `baseUrl`, the sidecar can start llama.cpp itself: `"embedding": {"llamacpp": {"model": "<EmbeddingGemma GGUF>"}}` and `"chat": {"llamacpp": {"model": "<chat GGUF>"}}` (optional `binary`, default `llama-server` on `PATH`; `port`, default 8291 and 8292; `args`). It starts each one when it is first needed (`--alias tsuzuki-embeddinggemma` / `tsuzuki-chat`, so another server on the port is never used) and leaves it running for the next start. `node scripts/setup.mjs` writes these settings.

JSON API:

| | request | response |
|---|---|---|
| `GET /health` | | `{status, version, dataDir, stats, embedding, chat}`: `status` is the process; `embedding` and `chat` are `{url, ok, lastError, lastErrorAt}` (`ok` is `null` before the first request; `chat` is `null` without a chat model); `stats.unindexed` counts messages and facts not yet embedded |
| `POST /v1/memory/messages` | `{session, messages: [{role, text, id?, at?}]}` | `{stored}`, once the messages are in the store (they are embedded afterwards) |
| `POST /v1/memory/recall` | `{query, session?, budget_tokens?}` (current session excluded, no rerank) | `{text, ids, reranked}` — `text` is the `<memory>` block or `""`; `503 {error, cause}` when the query cannot be embedded (`cause`: `embedding-timeout` or `embedding-unavailable`) |
| `POST /v1/memory/search` | `{query, budget_tokens?, rerank?}` (all sessions) | same |
| `POST /v1/memory/flush` | `{session?}` | `{ok}` |
| `POST /v1/memory/forget` | exactly one of `{session}`, `{before}` (ISO 8601 date or date-time; a date alone is local midnight), `{all: true}`; anything else is `400` | `{turns, units, facts, segments, links, vacuumed}` — rows deleted; `vacuumed: false` when another process kept the file from being rewritten (see Forgetting) |

Embeddings come from any OpenAI-compatible endpoint (default Ollama `embeddinggemma` on `:11434`) with the
EmbeddingGemma task prompts added client-side, or from this repository's engine (`embedding.engine`, started on
demand; when it cannot start, for example without `.venv/bin/python`, the reason is in `/health` and the log and the
sidecar keeps running). With `embedding.autoPull` the server asks Ollama to download a missing model. For
EmbeddingGemma 300m (Gemma Terms of Use) it only does so after `embedding.acceptModelLicense: true`, and the notice is
shown only when the endpoint is an Ollama server without the model; EmbeddingGemma 2 (Apache 2.0) needs no acceptance.
A `MODEL-NOTICE.txt` is written next to the data. The Hermes memory provider lives in `integrations/hermes/`.

**When a model is slow, hung or down.** The sidecar never makes its client wait long: messages are answered once
stored; a recall waits at most 0.5 s for pending indexing and `embedding.queryTimeoutMs` (3 s) for the query embedding
(after a query timed out, further queries fail at once for 30 s or until the model answers again);
a search's rerank stops after `chat.rerankTimeoutMs` (15 s), or at once when the chat server does not answer at all,
and returns the dense order. An embedding outage is logged once with the failing URL; while it lasts, the sidecar
asks the model every 30 s, and the first answer indexes everything stored meanwhile.

**Check that it works.** After one conversation, `curl -s http://127.0.0.1:<port>/health` shows `embedding.ok: true`,
`stats.turns` above 0 and `stats.unindexed` at 0 (a few seconds after the last message). `embedding.ok: false` comes
with `lastError`, which names the URL that failed.

## Tests

```sh
pnpm run build && node --test packages/tsuzuki/test/*.test.mjs   # conformance + orchestration, CPU only
bash tests/dsh-e2e/run-memory.sh 0.2.1-alpha.2 [port]                    # real dsh, fake model and embeddings, CPU only
bash tests/dsh-e2e/run-working-memory.sh 0.2.1-alpha.2 [port]            # working memory across a real compaction
```

The two end-to-end scripts need the dsh runtime unpacked at `build/dsh-runtime/<version>` (they stop with the
missing path otherwise) and write their report to `build/dsh-e2e/`; `E2E_OUT_DIR=results/dsh` updates the committed
results.

The end-to-end run boots dsh headless five times in an isolated `DSH_HOME`: the second and third sessions must
receive the first session's messages as `<memory>`, session-title requests must not, and facts must be indexed. In the
fourth the user asks to forget everything and the fake model calls `memory_forget {all: true}`; the fifth must recall
nothing of the first three, and the store file must no longer contain their text.
The working-memory run drives a fake coding model whose shell outputs overflow a 16k window: after dsh compacts,
the summary-only profile has lost the first command's error message, and the working-memory profile gets it back.

## Model licenses

The plugin is Apache-2.0 and ships no weights; users (or their Ollama) download them. EmbeddingGemma 300m — the
model behind every number above — is under the Gemma Terms of Use (ai.google.dev/gemma/terms): use and redistribution
are allowed with conditions, and the user accepts them when downloading. EmbeddingGemma 2 is Apache-2.0; in an
exploratory LoCoMo check its retrieval trailed 300m by 0.03 recall@10 inside this pipeline (it is not yet measured as
the P setting).
