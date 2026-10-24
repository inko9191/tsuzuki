# dsh-text-embeddings-tsuzuki

`ctx.textEmbeddings` provider for DeepSeek Harness backed by **EmbeddingGemma** (2 by default, or 300m) running on the
local [tsuzuki](../../..) C++/Metal engine (Apple Silicon). The plugin supervises the engine's resident daemon,
`egemma-server`, which serves an OpenAI-compatible `POST /v1/embeddings` on loopback — so other tools (and memory
plugins that take an OpenAI-compatible endpoint) can share the same daemon.

## Requirements

- macOS on Apple Silicon. fp16 with the Metal Performance Primitives matmul path on M5-class GPUs; every other Mac
  (and fp32) uses the engine's own SIMD-group kernels.
- An egemma-engine checkout with the native extension built (`build/_egemma_native*.so`), a Python environment with
  `numpy` and `tokenizers`, and a checkpoint: EmbeddingGemma 2 (`scripts/fetch_model.sh`) or EmbeddingGemma 300m (the
  model of the memory results; see the repository README). Weights are not bundled.

## Install and enable

The packages are not on npm. Build and pack them in `dsh-plugins/` (see the repository README), then add the tarballs by
**absolute** path (`dsh plugin` runs pnpm inside the profile directory, so a relative path is not found), into a profile
that already runs (a new name needs `dsh --profile <name> --from-default-profile headless --dump-config` first, or
another shipped template such as `web`):

```sh
dist="$PWD/dsh-plugins/dist"   # from the repository root
dsh plugin --profile <name> add "$dist/dsh-text-embeddings-0.1.0.tgz" "$dist/dsh-text-embeddings-tsuzuki-0.1.0.tgz"
```

The bundle registers the row **disabled**. Enable it in the profile's `cordis.patch.yml`:

```yaml
- id: text-embeddings-tsuzuki
  disabled: false
  config:
    engineRoot: /absolute/path/to/tsuzuki
```

For EmbeddingGemma 300m (the evaluated memory setting), set both `modelDir` and `modelName`; `dtype: auto` then
runs it in fp32:

```yaml
    modelDir: /absolute/path/to/tsuzuki/models/embeddinggemma-300m
    modelName: embeddinggemma-300m
```

| Config | Default | Meaning |
|---|---|---|
| `engineRoot` | required | Absolute path of the tsuzuki checkout |
| `modelDir` / `python` / `buildDir` | under `engineRoot` | `models/embeddinggemma-2`, `.venv/bin/python`, `build` |
| `modelName` | `embeddinggemma-2` | Name the daemon serves; `embeddinggemma-300m` together with the 300m `modelDir` |
| `dtype` | `auto` | `auto` (fp16 for EmbeddingGemma 2, fp32 for 300m), `fp16`, `bf16` (faster, less exact 300m), `fp32` |
| `matmul` | `auto` | `auto` (MPP on M5-class GPUs), `native`, `mpp` |
| `dimensions` | `768` | Default width; callers may request 768/512/256/128 |
| `defaultTask` | `none` | Prompt name applied when a call names none (model card default: none) |
| `host` / `port` | `127.0.0.1` / `8299` | Daemon address |
| `autoStart` | `true` | Spawn and supervise the daemon; `false` attaches to one started elsewhere |
| `apiKey` | `''` | Bearer token for the daemon (required by the daemon on non-loopback hosts) |
| `maxTokens` | `2048` | Per-input limit including prompt and BOS/EOS; longer inputs fail with `input-too-long` |
| `startupTimeoutMs`, `requestTimeoutMs`, `readyWaitMs` | 180 s, 120 s, 60 s | Start, call, and wait-while-restarting bounds |
| `healthIntervalMs`, `healthFailures` | 5 s, 3 | Hung-daemon detection |
| `maxRestarts`, `restartWindowMs`, `backoffMs`, `maxBackoffMs` | 5, 10 min, 1 s, 30 s | Restart policy |
| `extraArgs` | `[]` | Extra `egemma-server` arguments (batching limits etc.) |

Tasks are the checkpoint's prompt names (`query`, `document`, `STS`, `Classification`, `Clustering`, …).

## Lifecycle and failures

- With `autoStart`, the fiber becomes active only after the daemon prints its ready line; a first-start failure fails
  the row with the daemon's error and stderr tail.
- If an egemma-server serving the same `modelName` already answers on the port (for example one started by the memory
  sidecar for Hermes), the plugin shares it instead of starting a second copy. Should that daemon go away, the next call
  starts the plugin's own daemon on the freed port, which it then supervises and stops on unload. A port held by anything
  else fails the row.
- A crash, or `healthFailures` consecutive failed health checks, restarts the daemon with exponential backoff. While it
  restarts, calls wait up to `readyWaitMs`, then fail with `unavailable`. After `maxRestarts` within `restartWindowMs`
  the provider stays `failed`; `describe().status` and `lastError` carry the exit status and stderr tail, and every call
  fails with `unavailable` until the row is reloaded.
- Unloading the row terminates the owned daemon. An attached daemon is never managed.

## Using the daemon from other tools

`curl http://127.0.0.1:8299/v1/embeddings -d '{"model":"embeddinggemma-2:query:256","input":["…"]}'` —
`model` may carry `:<task>[:<dimensions>]` for clients that can only send `{model, input}`; `task`, `prompt` and
`dimensions` fields take precedence. `GET /stats` reports cold start, latency percentiles, batching, and memory.

## Known Limitations and Deferred Work

- macOS/Metal only; one daemon per port.
- Requires a local engine checkout; there is no prebuilt binary distribution yet.

License: Apache-2.0. No weights are bundled. EmbeddingGemma 2 is Apache-2.0 according to its model card (which refers
to the Gemma Prohibited Use Policy); EmbeddingGemma 300m is under the Gemma Terms of Use
(ai.google.dev/gemma/terms), accepted when you download it.
