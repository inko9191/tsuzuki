# DeepSeek Harness（dsh）埋め込みプロバイダ: ステージ0 調査と設計案（2026-10-10）

> **公開版の注**：開発時の記録。公開版ではパッケージと行 ID を改名した（`dsh-text-embeddings-egemma` → `dsh-text-embeddings-tsuzuki`、`text-embeddings-egemma` → `text-embeddings-tsuzuki`）。下の 6 節の使い方は新しい名前に直した。ステージ0の設計案（4・5 節）の名前は当時のまま。エンジンのデーモン（`egemma-server`、`server/egemma_server/`）は内部名のまま。リンク先の `results/dsh/` などは公開版に含めていない。

## 1. 対象バージョン（固定）

| 項目 | 値 |
|---|---|
| GitHub最新リリース | `dsh-v0.2.1-alpha.2`（2026-10-09、prerelease）。タグオブジェクト `6851e49`、コミット `d743267388641bc76f17c45ce8b4c231aed1d32c` |
| npm dist-tags（`@deepseek-ai/dsh`） | `latest` = `0.2.0-rc.2`、`alpha` = `0.2.1-alpha.2` |
| 検証に使ったDesktopアプリ | DeepSeek Harness.app `0.2.0-rc.2` |
| ライセンス | MIT（`deepseek-ai/deepseek-harness`） |
| 読んだソース | `build/dsh`（gitignore対象、上記タグの浅いclone） |

すべてのリリースがprereleaseで、READMEにも互換性を壊す変更があり得ると明記されている。
プラグインの `peerDependencies` は、検証した版に固定する。

## 2. プラグインの仕組み（要点と出典）

- **Cordis**（vendored）: プラグインは `apply(ctx, config)` を持つモジュール、または `Service` のサブクラス。
  `super(ctx, 'name')` で `ctx.name` を名前空間に登録し、利用側は `inject: ['name']` で待ち合わせる。
  提供側がunloadされると、依存する側も自動でunloadされ、提供側が戻ると再loadされる。
  登録はすべて `ctx.effect()` / `ctx.on()` の効果として扱われ、解除できる（`docs/cordis-primer.md`、`docs/cordis-tutorial/03-services.md`）。
- **設定**: `export const Config = Schema.object(...)`（Schemastery）。不正な値はload時に失敗する。
  `.volatile()` を付けた項目は再mountなしで更新できる（`docs/cordis-tutorial/05-config.md`）。
- **イベント**: `emit` / `waterfall` / `parallel` / `serial` / `bail` の5種。新しいイベントには `@mode` を明記する（`docs/cordis-primer.md`）。
- **3役割の能力分割**: Service Definition（型とサービス鍵）、Service Provider（実装）、Consumer（利用、例: ツール）。
  例はBash（`dsh-shell` / `dsh-bash-local` / `dsh-tool-bash`）。provider は `cordis.yml` の行を差し替えて交換する（`docs/user/develop/practice/index.md`）。
- **配布単位**: `package.json` に `"dsh": {"bundle": {"patch": "./cordis.patch.yml"}}` を持つnpmパッケージ（bundle）。
  `dsh plugin --profile <name> add <pkg>` はprofileディレクトリでpnpmを実行し、`dsh.profile.bundles` に追記する。
  パッチの適用順は bundle → profileの `cordis.patch.yml` → `$DSH_HOME/cordis.patch.yml` → `--patch`。
  dshパッケージは `peerDependencies` と `devDependencies` の両方に宣言する。
  git経由の導入は `prepare` によるビルドと `allowBuilds` の許可が必要なので、npmには `lib/` をビルド済みで公開する（`docs/user/develop/basic/publish.md`）。
- **関係するサービス**:
  - `ctx.subprocess`（`@deepseek-ai/dsh-subprocess`）: 子プロセス管理の能力分割。pipe、inherit、collect の各出力モードがある。
  - `ctx.webServer`: ブラウザ向けのHTTP経路だけを担い、headless・ACP・SDKのプロファイルにはない。外部向けAPIの置き場には使えない（`docs/subsystems/web-server.md`）。
- **埋め込みについて**: dsh本体に埋め込みのサービスやイベントはない。`ctx.llm` はchatのストリーミングだけ。
  本体の拡張表でも Memory は「system prompt section + tool」とされている（`docs/cookbook/extension-cookbook.md`）。

## 3. 既存の記憶・埋め込みプラグインの埋め込み呼び出し

awesome-dsh-plugin の `data/capabilities.json`（3844件）から名前で114件を抽出し、上位20件を読み取り専用で確認した（コードは実行していない）。

| プラグイン（commit） | 埋め込みの経路 | 差し替え |
|---|---|---|
| drscrewdriver/dsh-prime-memory 0.22.1（`9909abc`） | オフ／**リモート: OpenAI互換 `POST {baseUrl}/embeddings`、送るのは `{model, input}` だけ**／ローカル: onnxruntime-node（CPU、`onnx-community/embeddinggemma-300m-ONNX` を含むカタログ） | 設定 `embedding.baseUrl/apiKey(任意)/model/dimensions` だけで切り替え可能。返る次元が `dimensions` と違うとエラー |
| JunNanLYS/dsh-layered-memory 0.11.0（`69028fb`） | OpenAI互換 `/embeddings`、またはtransformers.js/ONNX（embeddinggemma、bge-m3など） | 設定 `embedding.{enabled, baseUrl, apiKey, model, dimensions}` で切り替え可能 |
| FuRongJun-1999/dsh-memory 0.8.1（`1aae0ba`） | Pythonから `POST /embeddings`（`base_url`、`embedding_model`） | 設定で切り替え可能（Python側） |
| wly8691-jpg/knowlp-rag 3.0.11（`21a5e57`） | `base_url`、`EMBEDDING_MODEL`、ollama | 設定で切り替え可能 |
| Noelune/unified-agent-memory 0.6.0（`3840806`） | Pythonから `/v1/embeddings`（SiliconFlowが既定） | URLは定数（`core/unified_memory/embed.py:33`） |
| Aik358/dsh-auto-memory 3.2.14（`a712169`、BSD-3） | Pythonでbge-m3（transformers / ONNX int8） | 差し替え口なし |
| kenz1117/dsh-engram 0.7.15（`757c6c7`） | transformers.jsで `Xenova/bge-small-zh-v1.5` | 差し替え口なし |
| 残り（mem0、hindsight、OpenViking など） | 外部の記憶サーバに委譲し、埋め込みはサーバ側 | 対象外 |

わかったこと:
- **dsh内に共有の埋め込みサービスはない**。各プラグインが、自前のONNX推論かOpenAI互換の `/embeddings` を個別に持っている。
- 差し替え可能なものは、事実上すべて **OpenAI互換 `/embeddings`**（`{model, input}`、キー任意）。`dimensions` やタスク指定は送らない。
- EmbeddingGemmaのタスクプロンプト（query/document）を付けるプラグインは見当たらない。
- dsh-prime-memory の変更履歴によれば、ホスト内でのONNX推論（embeddinggemma-300m、CPUで1件あたり0.3〜1.3秒）が
  dshのイベントループを数秒止める事故があり、ワーカースレッドへ移した。**推論はdshのプロセス外で行う方がよい**という裏付けになる。

## 4. 設計案

想定案（デーモン + 管理プラグイン）を基本にし、dshの3役割分割に合わせて **サービス定義を独立させる** ことを提案する。

```
┌───────────────────────── dsh host (Node) ─────────────────────────┐
│ dsh-embeddings            (Service Definition: ctx.embeddings の型)   │
│ dsh-embeddings-egemma     (Provider: デーモン管理 + ctx.embeddings 実装) │──HTTP──┐
│ 将来の記憶プラグイン       (Consumer: inject ['embeddings'])            │        │
└───────────────────────────────────────────────────────────────────┘        ▼
既存の記憶プラグイン（prime-memory 等）── OpenAI互換 /v1/embeddings ──▶ egemma-server（localhost 常駐）
他のツール（LangChain、curl 等）       ──────────────────────────────▶   Python + pybind11 + Metal engine
```

### 4.1 デーモン `egemma-server`
- 中身はPython + 既存のpybind11拡張。トークナイザはHFの `tokenizers` を使う（378入力で参照実装とID一致を確認済みの経路）。
  C++だけで作るにはSentencePiece/HFトークナイザの移植が要り、検証済みの経路から外れるため採らない。
- `POST /v1/embeddings`（OpenAI互換）:
  - `input`: 文字列または文字列配列。
  - `model`: 別名で、タスクと次元も指定できる。例: `embeddinggemma-2`、`embeddinggemma-2-query`、`embeddinggemma-2-document-256`。
  - `dimensions`: 768/512/256/128。
  - `encoding_format`: `float` / `base64`。
  - 拡張フィールド: `task`（STの20種のプロンプト名）、`prompt`（任意の文字列）。
  - 既定のプロンプトはモデルカードどおり「なし」（`default_prompt_name: null`）。ただし起動設定で既定タスクを変えられる。
  - `{model, input}` しか送れない既存プラグインは、`model` の別名でタスクと次元を選ぶ。
- バッチング: 受け付けキューから、待ち時間の上限（例: 2 ms）とトークン数の上限まで束ね、長さが近い順に詰めて1回のencodeで処理する。
- 最大入力長: 起動設定 `max_tokens`（既定2048）。超えた場合は400で `{"error":{"code":"context_length_exceeded",...}}` を返す。黙って切り詰めることはしない。
- 精度と経路: 既定はfp16 + `matmul=auto`（M5ならMPP、それ以外は自作経路）。fp32も選べる。
- `GET /health`、`GET /v1/models`、`GET /stats`。`/stats` は起動時間の内訳、レイテンシのp50/p95、バッチ統計、
  physical footprint、Metal確保量、選ばれた行列積経路を返す。
- 待ち受けはloopbackのみ。トークンは任意。
- 既存の精度ゲートと同じ基準（fp32 ≥0.9999、fp16 ≥0.999）で、デーモン経由とエンジン単体・STを比較する。

### 4.2 dshプラグイン（TypeScript、Apache-2.0、npm公開可能な構成）
1. **`dsh-embeddings`（Service Definition）**
   - `ctx.embeddings.embed({ input: string[], task?, dimensions?, signal? }) → { vectors: Float32Array[], model, dimensions, tokens }` と `describe()`。
   - 型付きの失敗: `input-too-long`、`unavailable`、`invalid-request`。
   - 提供側に依存しない汎用の定義にする。
2. **`dsh-embeddings-egemma`（Service Provider）**
   - 設定: `modelDir`、`dtype`（fp16）、`dimensions`（768）、`port`、`autoStart`、`matmul`（auto）、`maxTokens`、`python`/`serverCommand`、`restart`（回数・バックオフ）。
   - `autoStart` のとき、`ctx.subprocess` でデーモンを起動し、`/health` を待ってから `ctx.embeddings` を登録する。
     `autoStart: false` のときは、外部で動いているデーモンに接続する。
   - 落ちたとき: 指数バックオフで再起動する。上限を超えたらサービス登録を外し、Cordisの依存解除で利用側も止める。
     ログと型付きエラーで原因（終了コードと標準エラーの末尾）を伝える。再起動中の要求は、待機上限まで待つか `unavailable` を返す。
   - プラグインのunloadで、デーモンを停止する（効果として登録する）。
3. （任意）**`dsh-embeddings-openai`**: 任意のOpenAI互換エンドポイント向けの提供側。外部で動かすegemmaデーモンや他社APIを同じ `ctx.embeddings` で使えるようにし、差し替え可能であることを実際に示す。

既存の記憶プラグイン（prime-memory、layered-memory）はサービスを使わない。代わりに、それらの設定
`embedding.baseUrl = http://127.0.0.1:<port>/v1`、`model = embeddinggemma-2[-<task>][-<dim>]`、`dimensions = <dim>` だけで切り替える。
次の段階の自作記憶プラグインは、`inject: ['embeddings']` で使う。

### 4.3 想定案からの変更点と根拠
- サービス定義を独立パッケージにする: dshの3役割の規約（`docs/user/develop/practice/index.md`）に合わせるため。
  次の記憶プラグインが、egemmaに直接依存せずに済む。
- 埋め込みAPIはdshの `webServer` に載せず、独立デーモンにする: `webServer` はブラウザ向けで、headless・ACPにはない。
  また、ホスト内での推論がイベントループを止めた前例がある（prime-memory の CHANGELOG）。
- 起動は `ctx.subprocess` 経由にする: dshの子プロセス規約（`DSH_*` 環境変数の扱い、sandbox）に従うため。
- タスクと次元はmodelの別名でも選べるようにする: 既存プラグインが `{model, input}` しか送らないため。

## 5. 決めてほしい点
1. 固定する版: `0.2.1-alpha.2`（GitHub最新）か `0.2.0-rc.2`（npm latest、このMacのDesktopと同じ）か。
   推奨は、開発とテストを `0.2.1-alpha.2` で行い（独立した `DSH_HOME` で `npx @deepseek-ai/dsh@0.2.1-alpha.2`）、
   Desktop 0.2.0-rc.2 でも読み込みを確認すること。
2. サービス鍵 `embeddings`: dsh本体は平たい名前を予約していく方針なので、将来公式の `embeddings` と衝突するおそれがある。
   `embeddings` のまま進めるか、`textEmbeddings` などに変えるか。
3. 任意の `dsh-embeddings-openai` 提供側を、今回の実装に含めるか。
4. 差し替えの動作確認に使う既存の記憶プラグイン: dsh-prime-memory（MIT、利用者が最多）を推奨。profileへの導入を伴う。
5. デーモンをPython（tokenizers + pybind11）で作ることでよいか。

## 6. ステージ1 実装結果（2026-10-10）

上のステージ0は調査時点の設計案。以下が決定事項を反映した実装・検証結果であり、サービス鍵、
パッケージ名、model別名、再起動上限後の挙動はこの節のとおり。

### 6.1 決定事項と構成

- 開発版は dsh `0.2.1-alpha.2`。Desktopと同じ `0.2.0-rc.2` でも読み込みを確認した。
- サービス鍵は **`textEmbeddings`**。`dsh-text-embeddings` が型・抽象サービス・型付きエラー・
  `text-embeddings/settled` 観測イベントと共有HTTPクライアントを定義する。
- **`dsh-text-embeddings-tsuzuki`** が `ctx.subprocess` でデーモンを管理し、HTTP経由でサービスを提供する。
  **`dsh-text-embeddings-openai`** も実装済みで、任意のOpenAI互換エンドポイントを同じサービスに接続する。
- 両提供者のbundleは `text-embeddings-tsuzuki` / `text-embeddings-openai` の行を **`disabled: true` の休眠状態**で
  挿入する。利用する提供者を1つだけprofileの裸patchで有効にする。定義パッケージにはbundle行がない。
- デーモンは `server/egemma_server/` のPython実装（tokenizers + 既存pybind11拡張 + C++/Metalエンジン）。
  推論はdshプロセス外で行う。既存の記憶プラグインは `ctx.textEmbeddings` を使わず、直接HTTP APIを呼ぶ。
- Apache-2.0。モデルの重みはリポジトリ・tarballに同梱しない。精度基準・既定値は変更していない。

デーモンは既定 `127.0.0.1:8299`、fp16 / `matmul=auto` / 768次元 / タスクなし / 最大2048トークン。
M5ではautoがMPPを選び、fp32および他の対応GPUでは自作経路を使う。入力超過は黙って切り詰めず400を返す。
長さ別のキューとバッチ処理を持ち、`request_queue_size=128`、待ち受けは `SO_REUSEADDR` のみ。
読み込み中（起動から約3秒）に届いた `/v1/embeddings` は 503 にせず、準備完了まで最大15秒保留する（`--load-wait`、
2026-10-12 追加）。dsh と同時に起動した利用者が起動直後に 503 を受け、prime-memory が60秒間 FTS のみに
切り替わる問題への対策（README「dshプラグインとしての性能」）。`/health` は読み込み中も 503（`loading`）のまま。

### 6.2 起動・設定例

ネイティブ拡張、`.venv`（numpy/tokenizers）、モデルを用意したリポジトリのルートから：

```sh
PYTHONPATH=server .venv/bin/python -m egemma_server --port 8299
```

`POST /v1/embeddings` の `input` は文字列または文字列配列。`model` の別名は
**`embeddinggemma-2[:<task>[:<dim>]]`**（コロン区切り）で、例は `embeddinggemma-2:query:256`。
`task` / `prompt` / `dimensions` の明示フィールドは別名より優先され、別名は起動時の既定値より優先される。
次元は768/512/256/128、`encoding_format` はfloat/base64。タスク名はチェックポイントの20種のプロンプト名を使う。
`GET /health`、`GET /v1/models`、`GET /stats` を提供する。`/stats` には要求数・エラー数・起動時間の内訳・
レイテンシ分位・バッチ統計・physical footprint・Metal確保量・選択した行列積経路が含まれる。

パッケージはまだnpmに公開していない。ローカルtarballをビルドする：

```sh
cd dsh-plugins
pnpm run build
for p in packages/*; do (cd "$p" && pnpm pack --pack-destination ../../dist); done
cd ..
export DSH_HOME="$PWD/build/dsh-home-prime"
export DSH_TELEMETRY_DISABLED=1
build/dsh-runtime/0.2.0-rc.2/node_modules/.bin/dsh plugin --profile memtest add \
  dsh-prime-memory@0.22.1 "$PWD"/dsh-plugins/dist/*.tgz
```

独立したprofileの `cordis.patch.yml` に次を指定する（`engineRoot` は実際の絶対パスに置換）：

```yaml
- id: text-embeddings-tsuzuki
  disabled: false
  config:
    engineRoot: /absolute/path/to/tsuzuki
    port: 8321
- id: dsh-memory
  config:
    embedding:
      enabled: true
      baseUrl: http://127.0.0.1:8321/v1
      apiKey: ''
      model: embeddinggemma-2
      dimensions: 768
    recall:
      strategy: hybrid
```

検索戦略のキーは **`recall.strategy`**（`retrieval.strategy` ではない）。npm配布物に `src/config.ts` は
含まれないため、0.22.1の `dist/config.js` と `README.ja.md` で確認した。既存IDは `insert:` で包まず
裸patchで上書きする。`config` は行ごと置換されるので、必要な設定はすべて書き直す。
prime-memoryは `llm` / `tools` / `systemPrompt` に加えて `connection` / `webServer` を要求する。
baseだけの試験profileには、この2サービスの実装を追加する必要がある（実際の構成は
[`profile.patch.yml`](../results/dsh/prime_memory_swap/profile.patch.yml)）。

別プロセスで起動したデーモンには `text-embeddings-tsuzuki` 提供者の `autoStart: false` で接続できる。
OpenAI互換提供者を選ぶ場合は `text-embeddings-tsuzuki` の行を無効にし、次の行を有効にする：

```yaml
- id: text-embeddings-openai
  disabled: false
  config:
    baseUrl: http://127.0.0.1:8299/v1
    apiKey: ''
    model: embeddinggemma-2
    dimensions: 768
    allowedDimensions: [512, 256, 128]
    sendDimensions: true
```

新しい利用者は `inject: ['textEmbeddings']` を宣言し、
`ctx.textEmbeddings.embed({ input: ['…'], task: 'query', dimensions: 256 })` を呼ぶ。
結果は入力順の `Float32Array[]`、model、dimensions、task、promptTokensを持つ。

### 6.3 エラーと再起動方針

`TextEmbeddingError.is(error)` で識別し、型付き `kind` で分岐する。HTTP status、code、causeは
`details` に保持する。成功・失敗とも `text-embeddings/settled` に種別、件数、次元、時間を通知する。

| `kind` | 意味と対応 |
|---|---|
| `invalid-request` | 空入力、未対応タスク・次元、400/404/422。入力・設定を修正する |
| `input-too-long` | `context_length_exceeded`。短縮・分割する |
| `unavailable` | 接続不能、502/503、デーモン待機上限・再起動上限。状態と原因を確認する |
| `timeout` | 要求の時間制限、408/504。必要に応じて利用者側で再試行する |
| `aborted` | 利用者のAbortSignalによるキャンセル |
| `provider` | 認証、その他のHTTPエラー、不正な応答・ベクトル数/幅不一致など |

初回起動はready通知を待ってから有効になり、失敗時は行のloadを失敗させて原因・stderr末尾を報告する。
クラッシュまたは5秒間隔のhealthチェックが3回連続失敗すると、所有するデーモンを停止し、
1秒から最大30秒の指数バックオフで再起動する。再起動中の要求は最大60秒待ち、待機失敗は `unavailable`。
10分間に5回の再起動上限に達すると **サービス登録を残して `failed` 状態**になる（ステージ0案の登録解除とは異なる）。
その後の要求は即座に `unavailable` となり、行の再loadまで復旧しない。
`describe().status` は `starting` / `ready` / `restarting` / `failed`、`lastError` は終了状態とstderr末尾を伝える。
`autoStart: false` の接続モードでは監視・再起動を行わず、`describe().status` はreadyを返すため、接続の健全性は
実際のHTTP要求で確認する。unload時は所有するデーモンだけを停止する。
OpenAI互換提供者はHTTP要求を自動再試行せず、型付きエラーを利用者へ返す。

### 6.4 検証結果

| 対象 | 結果・証拠 |
|---|---|
| デーモンのトークナイズ・精度・並列 | 426件のトークンが参照実装・transformers経路と完全一致。fp16-auto（MPP）・fp16-native・fp32がPASS。参照・エンジン単体・STとの最小cosはfp16 ≥0.9999946759、fp32 ≥0.999999999992。16並列の混在要求もPASS。[検証ログ](../results/dsh_server_validation.txt)、[詳細JSON](../results/dsh/server_validation.json) |
| TypeScript定義・2提供者 | `cd dsh-plugins && pnpm test`：20件PASS（本物のデーモンを起動・停止する統合試験を含む） |
| dsh 0.2.1-alpha.2 | egemma・openai両提供者PASS。[egemma](../results/dsh/e2e_0.2.1-alpha.2_egemma/probe.json)、[openai](../results/dsh/e2e_0.2.1-alpha.2_openai/probe.json) |
| dsh 0.2.0-rc.2 | egemma・openai両提供者PASS。[egemma](../results/dsh/e2e_0.2.0-rc.2_egemma/probe.json)、[openai](../results/dsh/e2e_0.2.0-rc.2_openai/probe.json) |
| prime-memory 0.22.1の差し替え（今回） | dsh 0.2.0-rc.2、キー不要の追加・hybrid検索がPASS。詳細は次項 |

デーモン・4構成のdsh検証は前コミットの証拠を確認した。今回の追加試験と `pnpm test` は再実行した。
精度基準は引き続きfp32 ≥0.9999、fp16 ≥0.999。起動時間は既存検証で約2.8秒（エンジン読み込み約2.5秒）。

### 6.5 prime-memory の差し替え結果

再実行：`bash dsh-plugins/tests/dsh-e2e/run-prime-memory.sh`（tarballのビルド・pack後）。
実行ごとに `build/dsh-home-prime/run.XXXXXX` の新しい **独立した `DSH_HOME`** を作り、
`DSH_TELEMETRY_DISABLED=1`、profile名 `memtest`、デーモン8321番ポートを使う。Desktopの `~/.dsh` には触れない。
設定・ログ・要求前後のstats・結果は [`results/dsh/prime_memory_swap/`](../results/dsh/prime_memory_swap/) に保存する。

- `embedding.enabled: true`、`baseUrl: http://127.0.0.1:8321/v1`、`model: embeddinggemma-2`、
  `dimensions: 768`、`apiKey: ''`、`recall.strategy: hybrid`。タスク・プレフィックスの比較は行っていない。
- 自動捕捉・L1/L2/L3蒸留を試験profileだけで無効にし、`live.memoryMutate: true` で手動追加を許可した。
  実物のwebServer（loopback、OS割当ポート）・connectionを追加し、probeから `ctx.tools.execute` を呼んだ。
  LLMモデルルートの解決ログはあるが、モデル呼び出し・APIキー設定は行っていない。
- `memory_add` で日英混在の火星についての記憶を1件追加し、`memory_search` が同じID・本文を返した。
  `/stats.requests` は **0 → 1（追加）→ 2（検索）**、`errors` は0。
- prime-memoryログは `FTS=true 向量=true` を示し、リモート埋め込みエラーはない。
  dshはexit 0、終了後はデーモンのhealthに接続できず、egemma-serverプロセスが残っていないことも確認した。

証拠：[`probe.json`](../results/dsh/prime_memory_swap/probe.json)、[`run.json`](../results/dsh/prime_memory_swap/run.json)、
[`stats-before.json`](../results/dsh/prime_memory_swap/stats-before.json)、
[`stats-after-add.json`](../results/dsh/prime_memory_swap/stats-after-add.json)、
[`stats-after-search.json`](../results/dsh/prime_memory_swap/stats-after-search.json)、
[`memory.log`](../results/dsh/prime_memory_swap/memory.log)、[`boot.txt`](../results/dsh/prime_memory_swap/boot.txt)。
`embedding-errors.txt` は該当ログが0件の空ファイル。

試験の制限：settings UIを持たない構成の警告と、agentを渡さない直接ツール呼び出しの全族検索警告がある。
NodeのSQLite実験機能警告、profile導入時のpeer依存警告も保存した。実ホストでのload・向量検索は成功しているが、
この試験はUI操作、agentごとの隔離、蒸留、検索品質を検証しない。
ステージ2の同居ベンチ、MLX比較、ローカルONNXとの速度比較、プレフィックス有無の品質比較は未着手。
npm公開・pushも行っていない。

### 6.6 流用・参照したコードとライセンス

一覧と詳細は [`NOTICE`](../NOTICE) のとおり：

- Hugging Face Transformers v5.19.0（Apache-2.0）：設定導出、forward、RoPE、特殊トークン規則を
  行番号の引用つきで独立再実装。ソースファイルのコピーはない。
- vLLM commit `1dfe9fb`（Apache-2.0）：PLE・attentionの第二の参照元。
- FlashAttention論文（Dao et al., 2022）：タイル化online-softmaxのアルゴリズムを参照。コードコピーはない。
- Apple MPP（Apple SDK）：SDKの `mpp::tensor_ops::matmul2d` を呼ぶ。OS提供のフレームワークを再配布しない。
- DeepSeek Harness / Cordis `dsh-v0.2.1-alpha.2`（MIT）：Service Definition/Providerとdsh-shellの
  request/spec分離パターンに従う。Cordis・Schemastery・dsh-subprocessはホスト供給のpeer依存で、同梱しない。

再配布しないもの：モデルの重み（EmbeddingGemma 300mはGemma Terms of Use、EmbeddingGemma 2はモデルカードのとおりApache-2.0で、
Gemmaの禁止用途ポリシーに従う）、評価データ（LongMemEvalの採点プロンプトはMIT、LoCoMoはCC BY-NCで本文を含めない）、
任意の実行環境のOllama（MIT）、比較用のllama.cppとnlohmann/json（MIT）、
レジストリから別途取得するnumpy（BSD-3）、tokenizers（Apache-2.0）、pybind11/PyTorch（BSD-3）、
検証用sentence-transformers・ビルド用TypeScript（Apache-2.0）。
