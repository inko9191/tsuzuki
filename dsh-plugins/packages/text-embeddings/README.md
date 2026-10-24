# dsh-text-embeddings

Service Definition for text embeddings in [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (dsh):
the abstract `ctx.textEmbeddings` service, its request/result types, typed failures, the
`text-embeddings/settled` observation event, and the OpenAI-compatible `/embeddings` client shared by HTTP providers.

Install a provider — [`dsh-text-embeddings-tsuzuki`](../text-embeddings-tsuzuki) (local EmbeddingGemma 2 on Apple GPUs)
or [`dsh-text-embeddings-openai`](../text-embeddings-openai) (any OpenAI-compatible endpoint) — and enable exactly one.
This package itself contributes no Loader row.

## Consumer API

```ts
import type {} from 'dsh-text-embeddings'

export const name = 'my-memory'
export const inject = ['textEmbeddings']

export function apply(ctx) {
  const info = ctx.textEmbeddings.describe()   // provider, model, dimensions, tasks, status
  const { vectors } = await ctx.textEmbeddings.embed({
    input: ['Which planet is known as the Red Planet?'],
    task: 'query',        // provider task name; 'none' = no prompt; omitted = provider default
    dimensions: 256,      // one of info.dimensions; omitted = info.defaultDimensions
  })
}
```

- `embed()` = `resolve()` (defaults and validation) + `execute()`, then emits `text-embeddings/settled` (mode `emit`).
- Vectors are L2-normalized `Float32Array`s in input order.
- Every failure is a `TextEmbeddingError` with `kind`: `invalid-request`, `input-too-long`, `unavailable`, `timeout`,
  `aborted`, or `provider`. Use `TextEmbeddingError.is(e)` (brand check, safe across package copies).

## Provider API

Subclass `TextEmbeddings`, implement `describe()`, `resolve()` (use `resolveAgainst(request, info)`), and `execute()`,
and default-export the class. `postOpenAIEmbeddings()` performs one `POST {baseUrl}/embeddings` call and maps HTTP
status / OpenAI error codes onto the failure kinds.

## Compatibility

Peer: `@deepseek-ai/cordis` `~4.0.4 || ~4.0.5-alpha.1` (dsh 0.2.0-rc.2 and 0.2.1-alpha.2). dsh is a developer preview;
breaking changes are expected.

## Known Limitations and Deferred Work

- One provider per composition: the service key is global.
- No streaming or token-array inputs.

License: Apache-2.0.
