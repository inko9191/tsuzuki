# dsh-text-embeddings-openai

Generic `ctx.textEmbeddings` provider for DeepSeek Harness that calls any OpenAI-compatible `POST {baseUrl}/embeddings`
service (OpenAI, SiliconFlow, Jina, a self-hosted server, or an `egemma-server` started outside dsh).

The packages are not on npm. Build and pack them in `dsh-plugins/` (see the repository README), then add the tarballs
by **absolute** path (`dsh plugin` runs pnpm inside the profile directory):

```sh
dist="$PWD/dsh-plugins/dist"   # from the repository root
dsh plugin --profile <name> add "$dist/dsh-text-embeddings-0.1.0.tgz" "$dist/dsh-text-embeddings-openai-0.1.0.tgz"
```

The bundle registers the row **disabled**. Enable it in the profile's `cordis.patch.yml`:

```yaml
- id: text-embeddings-openai
  disabled: false
  config:
    baseUrl: https://api.example.com/v1
    apiKey: !!js process.env.EMBEDDINGS_API_KEY
    model: text-embedding-3-small
    dimensions: 1536
```

| Config | Default | Meaning |
|---|---|---|
| `baseUrl` | required | Base URL including the version path |
| `apiKey` | `''` | Bearer token (secret); empty sends no Authorization header |
| `model` | required | Model name sent with every request |
| `dimensions` | required | Native output width |
| `allowedDimensions` | `[]` | Extra Matryoshka widths callers may request |
| `sendDimensions` | `false` | Send `dimensions` to the server; otherwise truncate and renormalize locally |
| `taskPrefixes` | `{}` | Task name → text prefix (the OpenAI wire format has no task field) |
| `defaultTask` | `'none'` | Task applied when a call names none |
| `timeoutMs` | `60000` | Per HTTP request |
| `maxInputsPerCall` | `256` | Larger calls are split |

Invalid configuration (bad URL, unknown default task, widths out of range) fails the row at load.

## Known Limitations and Deferred Work

- Local truncation is only meaningful for Matryoshka-trained models; list only such widths in `allowedDimensions`.
- No retry: transient failures surface as `unavailable`/`timeout` for the caller to retry.

License: Apache-2.0.
