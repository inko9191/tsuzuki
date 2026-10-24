# tsuzuki：エージェントのための、ローカルで動く長期記憶

**エージェントが前の会話を覚えている。しかも、何もマシンの外に出ない。**

[English README](README.md)

- tsuzuki は、あなたとエージェントの過去の会話を保存する。
- 新しい発言のたびに、関係する過去の発言を日付つきで渡す。渡すのは**最大 2,048 トークンの記憶で、全履歴ではない**。
- すべてローカルで動く。
  - 検索は EmbeddingGemma 300m（Mac ではこのリポジトリのエンジン、ほかでは Ollama）。
  - 過去の会話からの事実の抽出は、手元のチャットモデル（検証は llama.cpp 上の Qwen）。

[DeepSeek Harness（dsh）](https://github.com/deepseek-ai/deepseek-harness)のプラグイン、Hermes Agent の記憶プロバイダ、`127.0.0.1` の小さな HTTP サイドカーの3つの形で使える。

このリポジトリには、**EmbeddingGemma のモデルを動かす非公式の推論エンジン（Google とは無関係）**も入っている。Apple シリコン向けの C++/Metal 実装。

## 中身

| 部品 | 場所 | すること |
|---|---|---|
| 記憶 | `dsh-plugins/packages/tsuzuki`（npm の名前は `dsh-tsuzuki`） | 発言の保存、事実の抽出、検索、記憶のブロックづくり。dsh のプラグインとサイドカー（コマンド `tsuzuki-memory`）は同じコード。 |
| 埋め込みエンジン | `native/`、`server/`、`reference/` | EmbeddingGemma（300m と 2）を C++/Metal で動かし、OpenAI 互換の `/v1/embeddings` のデーモン（`egemma-server`）で提供する。**Apple シリコン専用。** |
| dsh の埋め込みプロバイダ | `dsh-plugins/packages/text-embeddings*` | dsh の `ctx.textEmbeddings`。`dsh-text-embeddings-tsuzuki` はこのエンジンを、`dsh-text-embeddings-openai` は Ollama などの OpenAI 互換の接続先を使う。 |
| Hermes のプロバイダ | `integrations/hermes/` | サイドカーと話す Hermes Agent の `MemoryProvider`（id は `tsuzuki`）。 |
| 評価 | `docs/MEMORY_*`、`scripts/eval/`、`results/eval/memory/` | 事前登録、スクリプト、集計した結果。 |

### 記憶のしくみ

1. **保存**：ユーザーとアシスタントの発言を SQLite のファイルに入れる。サブエージェントのセッションは入れない。先に秘密情報を伏せ字にする（「プライバシー」）。
2. **索引**：発言を、1つ前の発言と組にして埋め込む。
3. **事実**：セッションを、約 2,500 トークンか、30 分発言のない所で区切る。区切りごとに、手元のチャットモデルが裏で次を行う。
   - その区切りに書かれた事実を挙げる。事実には、もとになった発言を付ける。
   - 更新のリンクを付ける（新しい事実が古い事実を置き換える、という印）。

   事実も検索するが、返すのは、事実が指す**元の発言**だけ。裏の作業は、エージェントか llama.cpp のサーバーが忙しい間は待つ。
4. **想起**：新しいユーザー発言の前に、
   - その発言を埋め込む。
   - 発言の窓と事実を順位付けし、RRF でまとめる。
   - *ほかの*セッションの発言から、良いものを 2,048 トークンまで選ぶ。

   日付の順に並べ、今日の日付と「いちばん新しい発言が今の事実」という規則を添えて渡す。既定では並べ替え（rerank）はしない。
5. **ツール**：`memory_search`（Hermes では `tsuzuki_search`）は、今のセッションも含めて全セッションを検索する。`memory_forget`（Hermes では `tsuzuki_forget`）は記憶を消す（「プライバシー」）。

## 結果（事前登録）

判定の規則は、回答を1つも作る前に文書で固定した。下の数字は `results/eval/memory/` の報告ファイルから写した。

プラグインの既定値は、評価した設定 **P**。探索の回（U1、30 問）で決め、U2 で、まだ使っていない問題で確かめた。

**条件**
- **データ**：[LongMemEval](https://github.com/xiaowu0162/LongMemEval)-S（cleaned 版）。新しい 40 問を、ID と種類だけを見て、固定の seed で選んだ（knowledge-update 10、temporal-reasoning 10、multi-session 10、single-session 6、abstention 4）。履歴は合計 1,862 セッション。問題の ID は `results/eval/memory/longmemeval-u2-plan.json`。
- **モデル**：Qwen3.8-27B Q4_K_M（GGUF）、llama.cpp の `llama-server`（b10964）、thinking なし。同じモデルが、回答、事実の抽出と更新の判定、主判定（LongMemEval の種類別の採点プロンプト）を行った。
- **埋め込み**：EmbeddingGemma 300m、fp32、このリポジトリの Metal エンジン。
- **機械**：Apple M5 Max・メモリ 64 GB の Mac 1 台。U2 と E1 のファイルには、モデルのハッシュ、llama.cpp の版、サーバーの引数を記録した。
- **統計**：質問単位の対応ありパーセンタイル・ブートストラップ（10,000 回、seed 0）。

### U2：記憶と全履歴（Qwen 27B）

| 設定 | モデルに渡すもの | 正答率 | 棄権率 | 入力トークン（平均） | 回答時間（平均） |
|---|---|---:|---:|---:|---:|
| **P**（tsuzuki の既定） | 直近の履歴 ＋ 日付つきの記憶 2,048 トークン | **0.675** | 0.250 | **2,913** | **5.7 秒** |
| C512（前に固定した版） | 直近の履歴 ＋ 記憶 512 トークン、rerank あり | 0.475 | 0.525 | 1,218 | 3.0 秒 |
| A：全履歴 | 全セッションを窓 131,072 トークンに入れる。前に固定した書式（質問の日付なし） | 0.325 | 0.675 | 116,166 | 426.7 秒 |
| AT：全履歴、P と同じ書式（U2-AT） | 全セッションを日付順に、質問の日付と P の規則つきで | 0.450 | 0.425 | 116,289 | 431.2 秒 |

40 問すべてが全履歴の窓に入った。

| 事前登録した比較 | 点差 | 95% 区間 | 判定 | 勝 / 負 / 同 |
|---|---:|---|---|---|
| P − C512 | +0.200 | [+0.075, +0.350] | 改善 | 9 / 1 / 30 |
| P − A | +0.350 | [+0.200, +0.500] | 上回る | 14 / 0 / 26 |
| P − AT（追試 U2-AT） | +0.225 | [+0.100, +0.350] | 上回る | 9 / 0 / 31 |

つまり、この 40 問とこのローカルモデルでは、2,048 トークンの記憶のほうが、約 11.6 万トークンの全履歴を文脈に入れるより多く正解した。入力は約 40 分の 1、1 回答の時間は約 75 分の 1。全履歴も P と同じ日付つきの書式にすると良くなった（AT − A：+0.125 [+0.025, +0.225]、傾向参考）が、P には届かなかった。

**副次の採点（Claude）**：Claude が、設定の名前を伏せたシートで U2 の回答を採点した（Claude は実験の設計者でもあるので、名前を伏せた）。
- 正答率：P 0.725、C512 0.475、A 0.350。Qwen との一致率 0.960、Cohen の κ 0.918。
- P − C512：+0.250 [+0.100, +0.400]。P − A：+0.375 [+0.225, +0.525]。

AT の回答は Claude が単独で採点したので、設定は伏せられなかった。AT 0.525。P − AT：+0.200 [+0.050, +0.350]（上回る）。Qwen との一致率 0.925、κ 0.85。

**種類別**（Qwen の採点。1 種類 4〜10 問なので傾向のみ）

| 設定 | knowledge-update | multi-session | single-session | temporal-reasoning | abstention |
|---|---:|---:|---:|---:|---:|
| P | 0.80 | 0.70 | 0.50 | 0.50 | 1.00 |
| C512 | 0.60 | 0.40 | 0.50 | 0.20 | 1.00 |
| A | 0.50 | 0.10 | 0.50 | 0.00 | 1.00 |
| AT | 0.70 | 0.40 | 0.50 | 0.00 | 1.00 |

### E1：小さいモデル（Gemma 4 E2B）が答える場合

同じ 40 問（[`docs/MEMORY_LONGMEMEVAL_E1_PREREG.md`](docs/MEMORY_LONGMEMEVAL_E1_PREREG.md)）。答える役を Gemma 4 E2B（Google 公式の QAT q4_0 の GGUF、窓 131,072 トークン）に替えた。**E2B-P には、U2 で Qwen の P が受け取ったのとまったく同じメッセージを渡した**。記憶そのものは Qwen 27B のパイプラインで作ったもので、替えたのは答える役だけ。E2B-AT は 40 問すべてが窓に入った。主判定は Qwen。

| 設定 | 正答率 | 棄権率 | 入力トークン | 回答時間 |
|---|---:|---:|---:|---:|
| E2B-P（Gemma 4 E2B ＋ 記憶） | **0.525** | 0.025 | 2,942 | 1.0 秒 |
| E2B-AT（Gemma 4 E2B ＋ 全履歴、同じ書式） | 0.375 | 0.075 | 117,101 | 98.3 秒 |
| 参考：Qwen 27B の P（U2） | 0.675 | | | |
| 参考：Qwen 27B の AT（U2-AT） | 0.450 | | | |

- **製品の目安**（E2B-P ≥ 0.50）：**満たす**。
- **主：E2B-P − Qwen-AT**（小さいモデル＋記憶 対 大きいモデル＋全履歴）：+0.075 [−0.100, +0.250]、**判定不能**、8 / 5 / 27。
- E2B-P − E2B-AT：+0.150 [−0.025, +0.300]、同等以上（非劣性）、9 / 3 / 28。
- E2B-P − Qwen-P：−0.150 [−0.325, +0.000]、傾向参考（答える役を小さくした分の損）。
- 副次の採点（Claude、名前を伏せ、Qwen の採点を開く前）：E2B-P 0.650、E2B-AT 0.475、Qwen-AT 0.525。E2B-P − Qwen-AT は +0.125 [−0.025, +0.275]、E2B-P − E2B-AT は +0.175 [+0.000, +0.350] で、どちらも同等以上（非劣性）。Qwen との一致率 0.887、κ 0.78。主判定は Qwen の採点のまま、判定不能。

### J1：日本語、全部 Gemma 4 E2B

自作の日本語テスト（[`docs/MEMORY_JA_J1_PREREG.md`](docs/MEMORY_JA_J1_PREREG.md)）。公開されている日本語の長期記憶のテストが見つからなかったので、答えの入った会話と紛らわしい会話を書いて 40 問を作り（正解は作った時点で決まっている）、1 問あたり約 47 本の一般的な日本語の雑談（約 4.8 万トークン）に混ぜた。**小さいマシンで動く部分は全部 Gemma 4 E2B**（事実の抜き出し、更新リンク、回答）。採点だけ Qwen 27B。

| 条件 | 正答率 | 入力トークン | 回答時間 |
|---|---:|---:|---:|
| E2B-P（記憶、全部 E2B） | **0.550** | 3,190 | 0.9 秒 |
| E2B-AT（全履歴、同じ書式） | 0.275 | 47,689 | 20.2 秒 |

- **主：E2B-P − E2B-AT**：+0.275 [+0.150, +0.425]、**上回る**（11 勝 / 0 敗 / 29 同）。製品の目安を満たす。
- 記憶は 40 問すべてで答えの会話を含んでいた（1.00）。日本語でも検索は崩れず、外したのは読み取りと推論（日付の計算 0.12、複数の会話の数え上げ 0.25）。
- 副次の採点者（Claude、条件名を伏せて）：0.700 対 0.425、+0.275 [+0.125, +0.425]、上回る。一致 0.850、κ 0.71。
- 自作のテストは本物の会話より整っている。方法が日本語にも通じるかの確認として読むこと（ベンチマークの点数ではない）。同じテストを Qwen 27B で全部行う J1-Q（[`docs/MEMORY_JA_J1Q_PREREG.md`](docs/MEMORY_JA_J1Q_PREREG.md)）を実行中。

全部の表（V1024R・T2048 の比較つき）：
- [`results/eval/memory/longmemeval-u2-report.md`](results/eval/memory/longmemeval-u2-report.md)、[`longmemeval-u2-at-report.md`](results/eval/memory/longmemeval-u2-at-report.md)、[`longmemeval-e1-report.md`](results/eval/memory/longmemeval-e1-report.md)
- [`longmemeval-u2-claude-judge.md`](results/eval/memory/longmemeval-u2-claude-judge.md)、[`longmemeval-u2-at-claude-judge.md`](results/eval/memory/longmemeval-u2-at-claude-judge.md)、[`longmemeval-e1-claude-judge.md`](results/eval/memory/longmemeval-e1-claude-judge.md)
- 事前登録：[`docs/MEMORY_LONGMEMEVAL_U2_PREREG.md`](docs/MEMORY_LONGMEMEVAL_U2_PREREG.md)、[`docs/MEMORY_LONGMEMEVAL_U2_AT_PREREG.md`](docs/MEMORY_LONGMEMEVAL_U2_AT_PREREG.md)、[`docs/MEMORY_LONGMEMEVAL_E1_PREREG.md`](docs/MEMORY_LONGMEMEVAL_E1_PREREG.md)

### 事前登録について

この公開リポジトリは、1 回のコミットで作った新しいスナップショット。事前登録は、作者の非公開の開発リポジトリで、結果より**前に**コミットした。その履歴はここには入っていないので、ここでは順番を示せない。ここで確かめられること：`results/eval/memory/` の各 `*-plan.json` に、事前登録の文書と、実行したコードの sha256 が記録されている。これらのファイルは非公開のリポジトリとバイト単位で同じ。例外は、[`docs/SANITIZED.md`](docs/SANITIZED.md) に元と今の両方のハッシュを並べた数ファイルだけ。元のコミット履歴は、求められれば見せる。

## まだ示していないこと

- **コーディングと作業中の記憶**：`working.enabled` は、ツールの呼び出しと結果を保存し、dsh が文脈を要約したあとで、要約から落ちた細部を戻す。**実験的で、既定はオフ**。事前登録した試験 C1（[`docs/MEMORY_CODING_C1_PREREG.md`](docs/MEMORY_CODING_C1_PREREG.md)）は、**判定を出さずに打ち切った**。dsh 自身の要約が、ローカルの Qwen 27B と 32k の窓では安定せず、記憶ではなく要約を測ることになるため。
- **自作テスト以外の日本語**：J1（上）は小さな自作テストで、本物の日本語の利用者の会話は測っていない。同じテストの Qwen 27B 版（J1-Q）は実行中。
- **数千セッション**：取り出しだけの規模試験（[`results/eval/memory/scale-report.md`](results/eval/memory/scale-report.md)。LongMemEval-S のセッション、発言の窓だけ、事実なし、rerank なし）で、60 問のうち答えの発言が想起に入った割合は、保存したセッションが 50 で 0.883、500 で 0.717、2,000 で 0.667、5,000 で 0.517 だった（5,000 − 50：−0.367 [−0.483, −0.250]）。**関係のない履歴が増えるほど精度は落ちる。** 事実・rerank・枠の拡大で戻るかは測っていない。
- **本当の長期利用**：試験はベンチマークの履歴で、何か月分もの本物の会話ではない。
- **ほかのハードウェアと Mac 以外の経路**：すべて Mac 1 台で測った。Linux や Windows で使う Ollama の埋め込みの経路は、**測っていない**。Ollama の EmbeddingGemma 300m は、fp32 のエンジンと数値が違う可能性がある。EmbeddingGemma 2 は設定 P で測っていない（LoCoMo の探索では、取り出しが 300m より recall@10 で 0.03 低かった）。
- **ほかのモデル**：U2 では、回答・事実の抽出・採点がすべて同じローカルの Qwen3.8-27B Q4_K_M。E1 は答える役だけを替えた。長い文脈に強いクラウドのモデルは試していない。そうしたモデルなら、全履歴との差は縮むか、逆になるかもしれない。
- **保存のときのコスト**：回答時間には、保存のときの裏の作業（チャットモデルによる事実の抽出と更新の判定、埋め込み）を含めていない。全履歴の時間は、llama.cpp のホストのプロンプトキャッシュを切って測った。
- **公表されている LongMemEval の点との比較**：比べられない。公表値は GPT-4o の採点と別のモデルを使っている。
- **精度**：40 問なので区間が広い。U2 の主判定は、答えたのと同じモデル。副次の採点はそれを確かめるためのもの。

## 必要なもの

- **Node.js** `^22.19` か `>=24`。dsh とサイドカーは `node:sqlite` を使う。
- **事実の抽出に使うチャットの接続先**：OpenAI 互換で、JSON スキーマの出力ができるもの。検証したのは llama.cpp の `llama-server` と Qwen3.8-27B Q4_K_M。無くても発言の窓だけで動くが、それは評価した設定 P **ではない**。
- **埋め込み**：次のどちらか。
  - **Apple シリコンの Mac**（評価した経路）：このリポジトリのエンジンで EmbeddingGemma 300m を fp32 で動かす。Xcode、CMake、Python 3.13、[uv](https://github.com/astral-sh/uv) が要る。
  - **ほかのマシン**：[Ollama](https://ollama.com) の `embeddinggemma`。
- **Hermes Agent で使うとき**：Hermes に同梱の Node（`~/.hermes/node/bin/node`）か、上の条件を満たす Node。

## 入れ方

### 1. プラグインをビルドする（どの使い方でも）

```bash
cd dsh-plugins
npx -y pnpm@10.26.1 install --frozen-lockfile
npx -y pnpm@10.26.1 run build
npx -y pnpm@10.26.1 -r --filter './packages/*' pack --pack-destination dist
```

dsh 用のパッケージが `dsh-plugins/dist/*.tgz` に、サイドカーが `dsh-plugins/packages/tsuzuki/lib/server.js` にできる。パッケージは npm に公開していない。tarball は必ず**絶対パス**で入れる。パッケージ名で入れてはいけない（npm に同じ名前のパッケージがあれば、そちらが入る）。

### 2. 埋め込み

**Mac（評価した経路）**：リポジトリのルートで実行する。

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r requirements-lock.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build -j 8
```

- `requirements-lock.txt` は、結果を出したときの版そのもの。検証用の PyTorch と transformers も入る。
- EmbeddingGemma 300m は、[Gemma の利用規約（Gemma Terms of Use）](https://ai.google.dev/gemma/terms)に同意しないと取得できない。[モデルのページ](https://huggingface.co/google/embeddinggemma-300m)で同意し、`.venv/bin/hf auth login` でログインしてから、評価した版を取得する。

```bash
.venv/bin/hf download google/embeddinggemma-300m --revision 57c266a740f537b4dc058e1b0cda161fd15afa75 --local-dir models/embeddinggemma-300m
```

エンジンを自分で起動する必要はない。dsh のプロバイダとサイドカーが、必要なときに `127.0.0.1:8299` で起動する。

**Linux、Windows、エンジンを使わない Mac**：Ollama でモデルを取得する（これでも Gemma の利用規約に同意することになる）。

```bash
ollama pull embeddinggemma
```

### 3a. DeepSeek Harness（CLI のプロファイル）

すでに動いているプロファイルか、同梱のテンプレート（`web`、`headless`、`acp`、`sdk`、`sdk-minimal`）に入れる。ほかの新しい名前だと、`dsh-base` しか読み込まない空のプロファイルになるので、先にテンプレートから作る。

```bash
dsh --profile <名前> --from-default-profile headless --dump-config > /dev/null   # 新しい名前のときだけ。web、acp、sdk でもよい
```

`dsh plugin` はプロファイルのフォルダの中で pnpm を動かすので、**絶対パス**で渡す。リポジトリのルートで：

```bash
dist="$PWD/dsh-plugins/dist"
dsh plugin --profile <名前> add "$dist/dsh-text-embeddings-0.1.0.tgz" "$dist/dsh-text-embeddings-tsuzuki-0.1.0.tgz" "$dist/dsh-tsuzuki-0.1.0.tgz"
```

次に、`$DSH_HOME/profiles/<名前>/cordis.patch.yml`（`DSH_HOME` を設定していなければ `~/.dsh/profiles/<名前>/`）に行を足す。patch は行の `config` を丸ごと置き換えるので、要る設定はすべて書く。

```yaml
- id: text-embeddings-tsuzuki
  disabled: false
  config:
    engineRoot: /このリポジトリの/絶対パス
    modelDir: /このリポジトリの/絶対パス/models/embeddinggemma-300m
    modelName: embeddinggemma-300m   # dtype "auto" は 300m では fp32
    port: 8299
- id: tsuzuki
  config:
    llm: { baseUrl: http://127.0.0.1:8080/v1 }   # 手元のチャットサーバー
```

**Ollama を使うとき**は、`dsh-text-embeddings-tsuzuki-0.1.0.tgz` の代わりに `dsh-text-embeddings-openai-0.1.0.tgz` を入れ、次の行を使う。

```yaml
- id: text-embeddings-openai
  disabled: false
  config:
    baseUrl: http://127.0.0.1:11434/v1
    model: embeddinggemma
    dimensions: 768
    taskPrefixes:
      query: 'task: search result | query: '
      document: 'title: none | text: '
```

**ほかの記憶を止める**：プロファイルにほかの記憶プラグインがあれば、その行を `disabled: true` にする（記憶が二重に渡らないように）。

記憶は `$DSH_HOME/tsuzuki` に保存される。

**出てよい警告**：pnpm が `@deepseek-ai/*` について "Issues with peer dependencies found" を出し、dsh が `dsh-text-embeddings` について "declares no dsh.bundle" と出す（それらは dsh が提供し、そのパッケージには自分の行がない）。動かすと、Node が `ExperimentalWarning: SQLite is an experimental feature` を1回出す。

**動いているかの確かめ方**：2つのセッションを始める。1つ目でエージェントに何かを伝え、2つ目でそれを聞く。2つ目の要求に `<memory>` のブロックが付く。保存先は `$DSH_HOME/tsuzuki/memory.db`。

### 3b. DeepSeek Harness のデスクトップアプリ

アプリを終了してから `bash dsh-plugins/scripts/install-desktop.sh` を実行する（埋め込みに Ollama を使うときは引数に `ollama`）。スクリプトはプロファイルのファイルをバックアップし、元に戻す方法を表示する。記憶は `$DSH_HOME/tsuzuki`（既定は `~/.dsh/tsuzuki`）に保存される。

### 3c. Hermes Agent

先に：
- **プラグインをビルドする**（手順 1）。サイドカーは Hermes に同梱の Node（`~/.hermes/node/bin/node`）で動く。無ければ `PATH` の `node`。変えるときは、プロファイルの `config.yaml` の `memory.tsuzuki.node`。
- **埋め込み**：Mac のエンジンをビルドしてあれば（手順 2）、導入スクリプトはそれを使う。なければ Ollama の設定を書き、そう表示する。Ollama が `127.0.0.1:11434` で動き、`embeddinggemma` を取得済みであること。
- **事実の抽出**：導入スクリプトは、プロファイル自身のモデルの接続先（`model.base_url`）を使う。ただし、このマシンの上にあるときだけ（「プライバシー」）。それ以外なら、何も変えずに止まる。`--no-chat`（発言の窓だけ。評価した設定 P ではない）か、会話の本文がそこへ送られてよければ `--allow-remote-chat` を付けて実行し直す。

Hermes を終了してから実行する（そのプロファイルの `config.yaml` が先にあること。`default` は Hermes のホームそのもの）。

```bash
python3 integrations/hermes/install.py <プロファイル>            # Hermes が ~/.hermes に無いときは --home <フォルダ>
```

スクリプトは、プロファイルごとに `config.yaml` をバックアップし、プロバイダを `<プロファイル>/plugins/tsuzuki` にコピーし、`memory.provider: tsuzuki` を設定し、`<プロファイル>/tsuzuki/config.json` を書く（あれば残す）。サイドカーのポートはプロファイルごとに違う（`8297` ＋ データフォルダのハッシュ。`8297`〜`8796`）ので、記憶は混ざらない。Hermes を起動し直して、`hermes memory status` で確かめる。

**本当に動いているかの確かめ方**：`hermes memory status` は、Node とサイドカーのファイルがあるかしか見ない。1 回会話したあとで、`curl -s http://127.0.0.1:<ポート>/health`（ポートは `config.json`）が、`embedding.ok: true`、`stats.turns` が 0 より大きい、数秒後に `stats.unindexed` が 0、を示すこと。

**あとで `config.json` を変えたとき**：サイドカーは Hermes を終了しても動き続け、起動したときにだけ `config.json` を読む。止めてから（pid は `<プロファイル>/tsuzuki/sidecar.log`、または `lsof -t -iTCP:<ポート> -sTCP:LISTEN`）、Hermes を起動し直す。

詳しくは [`integrations/hermes/README.md`](integrations/hermes/README.md)。

### 3d. そのほか：HTTP サイドカー

```bash
node dsh-plugins/packages/tsuzuki/lib/server.js --config <フォルダ>/config.json
```

（pack した `dsh-tsuzuki` を入れると、同じサーバーが `tsuzuki-memory` コマンドになる。`--config` が無ければ `~/.tsuzuki/config.json` を使う。）

初回に、データの横へ既定の `config.json` を書く。ポート `8297`、埋め込みは Ollama の `embeddinggemma`、**チャットモデルなし**。評価した設定にするには、`chat.baseUrl` と `chat.model` を書く（llama.cpp 以外のサーバーなら `chat.server: "openai"`）。Ollama の代わりに Mac のエンジンを使うときは、次を足す。

```json
"embedding": {"engine": {"engineRoot": "/abs/repo", "modelDir": "/abs/repo/models/embeddinggemma-300m", "modelName": "embeddinggemma-300m", "port": 8299}}
```

サイドカーは、起動したときにだけ `config.json` を読む。変えたら、止めて起動し直す。

| エンドポイント | 要求 | 応答 |
|---|---|---|
| `GET /health` | | `{status, version, dataDir, stats, embedding, chat}`。`embedding` と `chat` は `{url, ok, lastError, lastErrorAt}`。`stats.unindexed` は、まだ埋め込んでいない数 |
| `POST /v1/memory/messages` | `{session, messages: [{role, text, id?, at?}]}` | `{stored}`。保存した時点で答える |
| `POST /v1/memory/recall` | `{query, session?, budget_tokens?}`。今のセッションは除く。rerank なし | `{text, ids, reranked}`。`text` は `<memory>` のブロックか `""`。クエリを埋め込めないときは `503 {error, cause}` |
| `POST /v1/memory/search` | `{query, budget_tokens?, rerank?}`。全セッション | 同じ |
| `POST /v1/memory/flush` | `{session?}` | `{ok}` |
| `POST /v1/memory/forget` | `{session}`、`{before}`、`{all: true}` のどれか1つ | `{turns, units, facts, segments, links, vacuumed}` |

`Content-Type: application/json` で送る。JSON のオブジェクトでない本文は `400`。

**モデルが止まった・応答しないとき**も、長くは待たせない。想起は約 3.5 秒であきらめる（クエリの埋め込みに 3 秒。時間切れのあと 30 秒は、クエリをすぐ失敗にする）。検索の rerank は 15 秒で打ち切る（チャットサーバーがまったく応答しないときはすぐに省き、密な検索の順で返す）。埋め込みモデルが止まっていても発言は保存し、戻りしだい索引する。

## 主な設定

| 設定（dsh の名前 / サイドカーの名前） | 既定 | 意味 |
|---|---|---|
| `recall.budgetTokens` | `2048` | 1 回に渡す記憶の大きさ（設定 P） |
| `recall.rerank` | `false` | 想起をチャットモデルで並べ替える。ローカルの 27B では 1 回約 16 秒かかる |
| `embedding.queryTimeoutMs` | `3000` | クエリの埋め込み 1 回の上限（想起、検索） |
| `llm.rerankTimeoutMs` / `chat.rerankTimeoutMs` | `15000` | rerank 1 回の上限。超えたら密な検索の順を使う |
| `llm.baseUrl` / `chat.baseUrl` | dsh：`http://127.0.0.1:8080/v1`、サイドカー：空 | 事実の抽出と rerank に使うチャットの接続先。空なら両方とも使わない |
| `facts.updateLinks` | `true` | 保存のときに更新のリンクを付ける |
| `capture.redactSecrets` | `true` | 保存の前に秘密情報を伏せ字にする |
| `working.enabled` | `false` | 長いコーディング作業のための作業中の記憶（実験的） |

全部の一覧：[`dsh-plugins/packages/tsuzuki/README.md`](dsh-plugins/packages/tsuzuki/README.md)（英語）。

## プライバシー

- **既定ではローカルだけ**：発言はローカルの SQLite のファイルに保存する。埋め込みは `127.0.0.1`（エンジンか Ollama）。事実は、自分で設定したチャットの接続先が作る。テレメトリーは無い。
- **接続先をクラウドにすると、本文はマシンの外に出る**：`llm.baseUrl` / `chat.baseUrl` や埋め込みの `baseUrl` をリモートの API にすると、保存した発言と事実の抽出の区切りがそこへ送られる。
  - Hermes の導入スクリプトは、プロファイルのモデルの接続先を、このマシンの上にあるときだけ使う。リモートなら、`--allow-remote-chat`（か `--no-chat`）を付けない限り止まる。
  - dsh の `llm.baseUrl` の既定は `http://127.0.0.1:8080/v1`。そのポートが手元のチャットサーバーであることを確かめるか、設定する。
- **秘密情報の伏せ字**（既定でオン）：保存・埋め込み・事実のモデルへの送信の前に、次を `[REDACTED:種類]` にする。API キー（OpenAI、Anthropic、GitHub、Hugging Face、AWS、Google、Slack、Stripe）、JWT と bearer トークン、URL の中の認証情報、`password:` の形の値（パスワード などの日本語のラベルも）、Luhn の検査に通るカード番号。パターンによるもので、誤検出を避ける側に寄せている。**珍しい形の秘密は漏れることがある。** 会話に秘密を貼って、これに頼らないこと。
- **サイドカーにはログインが無い**：`127.0.0.1` だけで待ち受ける。このマシンのどのプログラムからも読める。`host` は `127.0.0.1` のままにする（ほかにするにはトークンが要る）。JSON でない要求、ブラウザの `Origin` 付きの要求、ループバック以外の `Host` の要求は拒む。
- **忘れる**：エージェントは、ツール `memory_forget`（dsh）か `tsuzuki_forget`（Hermes）で記憶を消せる。ツールの説明で、消す範囲をユーザーに示し、確認を取ってから呼ぶようモデルに求めている。ほかのホストは `POST /v1/memory/forget`。範囲は、1つのセッション、ある日時より前に保存したものすべて、すべて、の3つ。消えるのは、発言、そこから作った窓と事実、更新のリンク。SQLite のファイルと WAL も書き直すので、ファイルからも本文が消える。発言を1件ずつ消す機能は無い。対象外：ホスト自身のセッション履歴、保存先のコピーやバックアップ、ローカルのサーバーがすでに見たもの。手ですべて消すときは、ホストを止めて記憶のフォルダを消す（dsh は `$DSH_HOME/tsuzuki`、Hermes は `<プロファイル>/tsuzuki`、サイドカーは `config.json` のあるフォルダ）。

## モデルのライセンス

このリポジトリにも、そのパッケージにも、モデルの重みは入っていない。

- **EmbeddingGemma 300m**（`google/embeddinggemma-300m`）：上の数字はすべてこのモデルによる。[Gemma の利用規約（Gemma Terms of Use）](https://ai.google.dev/gemma/terms)のもとで配布されている。**同梱しない**。Hugging Face か `ollama pull` で取得するときに、利用者が規約に同意する。サイドカーが Ollama に自動で取りに行くのは、`embedding.acceptModelLicense: true` にしたときだけ。
- **EmbeddingGemma 2**（`google/embeddinggemma-2`）：モデルカードのとおり Apache 2.0（Gemma の禁止用途ポリシーにも従う）。エンジンは対応している（既定は fp16）が、設定 P では測っていない。
- **Gemma 4 E2B**（`google/gemma-4-E2B-it-qat-q4_0-gguf`。E1 の答える役、J1 のモデル）：Apache 2.0。
- **チャットモデル**：自分で選ぶ。そのモデルのライセンスを確かめる。

## ライセンス

Apache-2.0（`LICENSE`、`NOTICE`）。商用の製品も含めて、使う・変える・配ることができる。特許の許諾も明記されている。モデルの重みは含まれず、それぞれのライセンスに従う（「モデルのライセンス」）。

## 名前

開発中の仮の名前は「egemma」だった。利用者が打つ・読む名前は、すべて tsuzuki に変えた。

| 前 | 今 |
|---|---|
| `dsh-memory-egemma`（npm）、行 `memory-egemma`、フォルダ `dsh-plugins/packages/memory-egemma` | `dsh-tsuzuki`、行 `tsuzuki`、`dsh-plugins/packages/tsuzuki` |
| `dsh-text-embeddings-egemma`（npm）、行 `text-embeddings-egemma` | `dsh-text-embeddings-tsuzuki`、行 `text-embeddings-tsuzuki` |
| コマンド `egemma-memory`、既定の設定 `~/.egemma-memory/config.json` | `tsuzuki-memory`、`~/.tsuzuki/config.json` |
| dsh のデータ `$DSH_HOME/memory-egemma`、Hermes のデータ `<プロファイル>/egemma-memory` | `$DSH_HOME/tsuzuki`、`<プロファイル>/tsuzuki` |
| Hermes のプロバイダとプラグインのフォルダ `egemma_memory`、設定 `memory.egemma_memory` | `tsuzuki`、`memory.tsuzuki` |
| Hermes のツール `egemma_memory_search`、`egemma_memory_forget` | `tsuzuki_search`、`tsuzuki_forget` |
| `EGEMMA_MEMORY_NODE`、`EGEMMA_MEMORY_SERVER_JS` | `TSUZUKI_NODE`、`TSUZUKI_SERVER_JS` |
| サイドカーの設定 `embedding.egemma`、`install-desktop.sh egemma` | `embedding.engine`、`install-desktop.sh engine` |

**古い名前のまま残したものと理由**：
- **埋め込みエンジンの内部の名前**：Python のパッケージ `egemma_server`（`python -m egemma_server`、プロセス名 `egemma-server`、ログ `egemma-server.log`）、ネイティブ拡張 `_egemma_native` と `native/egemma_native.py`、CLI の `egemma-cli`、C++ の名前空間 `egemma`、CMake のプロジェクトとターゲット、Python の配布名 `egemma-server`（`server/pyproject.toml`）、参照実装 `reference/egemma_ref`、エンジンの環境変数（`EGEMMA_PORT`、`EGEMMA_API_KEY`、`EGEMMA_MATMUL` など）とビルドの設定 `EGEMMA_ENABLE_MPP`、エンジンのプロバイダが `describe().provider` で返す値 `egemma`（ベクトルと一緒に保存する埋め込みモデルの鍵の一部）。評価のスクリプトがこれらを起動・読み込みしていて、そのハッシュが計画ファイルに記録されている。名前を変えると両方が壊れる。
- **評価の記録**：事前登録（`docs/MEMORY_*.md`）、`scripts/eval/`、`results/eval/memory/` は、実行したときの名前のまま（例：`dsh-memory-egemma`、seed の `egemma-memory-stage1-2026-10-12`）。sha256 が計画ファイルと合うようにするため。

「EmbeddingGemma」「Gemma」は、Google のモデルの名前としてだけ使っている。このプロジェクトは独立したもので、Google、DeepSeek Harness や Hermes Agent の作り手、Qwen とは関係がなく、承認も受けていない。

## 第三者のもの

- dsh のパッケージ（`@deepseek-ai/*`、MIT）はピア依存。同梱しない。
- 採点に LongMemEval の採点プロンプト（MIT、© 2024 Di Wu）を使う。データセット（LongMemEval、LoCoMo、MTEB、JMTEB）は再配布しない。入っているのは ID・ハッシュ・集計した点だけ。
- エンジンは、Hugging Face Transformers と vLLM のソース（Apache-2.0）をもとに、行番号の引用つきで EmbeddingGemma を作り直したもの。[`NOTICE`](NOTICE) を参照。

## 開発者向け

- テスト（CPU だけ）：`cd dsh-plugins && npx -y pnpm@10.26.1 test`（ダウンロードしたモデルやビルドしたエンジンが要るテストは、それが無ければ飛ばす）。Hermes のプロバイダは [`integrations/hermes/README.md`](integrations/hermes/README.md)。dsh の通しの試験は `dsh-plugins/tests/dsh-e2e/run-memory.sh` と `run-working-memory.sh`（`build/dsh-runtime/<版>` に dsh の実行環境が要る）。
- 入れた結果のファイルは、上の主張の根拠になるものだけ。古い文書（`docs/DSH_PLUGIN.md`、`docs/NATIVE.md`、`docs/MEMORY_RETRIEVAL_*`）には、非公開の開発リポジトリに残したベンチマークのログや結果へのリンクがある。
