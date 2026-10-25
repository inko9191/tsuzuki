# tsuzuki（Hermes の記憶プロバイダ）

Hermes Agent の会話を、ローカルの記憶サイドカー（`http://127.0.0.1:<ポート>`。ポートはプロファイルごとに決まる）に保存・検索させるプラグイン。
Hermes 側は HTTP で呼ぶだけで、埋め込みと検索はサイドカーが行う。

## 動き

- **各ターンの前**：今のユーザー発言で `POST /v1/memory/recall` を呼ぶ（今のセッションの発言は除かれる）。結果は Hermes が `<memory-context>` で囲んで、ユーザー発言に付ける。4.5 秒で打ち切る（Hermes 側の上限は 8 秒）。サイドカーは、埋め込みモデルが応答しなくても約 3.5 秒で答える。
- **想起が時間切れになったとき**：そのあと 60 秒は、ターンの前の想起を待たずに飛ばす（裏の想起は続け、成功したらすぐ戻す）。止まった埋め込みモデルのせいで、毎ターン待たされることはない。
- **各ターンの後**：ユーザーの発言と応答を `POST /v1/memory/messages` で送る。裏のスレッドで、順番どおりに送る。サイドカーは保存した時点で答え、埋め込みはそのあとで行う。サイドカーが止まっていて届かなかった発言は、最大 256 件まで取っておき、サイドカーを起動し直してから順番どおりに送り直す。
- **ツール `tsuzuki_search`**：全セッションを対象に `POST /v1/memory/search` を呼ぶ。rerank は既定でオフ（サイドカーの `search.rerank`）。オンのときは 15 秒で打ち切り、チャットモデルがまったく応答しないときはすぐに省いて、密な検索の順で返す。失敗したときは、原因（サイドカーが動いていない、応答がない、埋め込みモデルが使えない）をそのまま返す。
- **ツール `tsuzuki_forget`**（既定でオフ。`config.yaml` の `memory.tsuzuki.forget: true` で出る。ウェブページや貼り付けた文に操られたモデルが呼べてしまい、「ユーザーに確認してから」はモデルへのお願いでしかないため）：記憶を消す。範囲は、1つのセッション（`session`。`"current"` で今の会話）、ある日時より前の発言すべて（`before`。ISO 8601 の日付か日時。日付だけなら、その日のローカル時刻 0 時）、すべて（`all: true`）のどれか1つ。`POST /v1/memory/forget` を呼ぶ前に、送信待ちの発言を送り終える。ツールの説明で、モデルに「消す範囲をユーザーに示し、その範囲で確認を取ってから呼ぶ」よう求めている。消えるのは、発言と、そこから作った索引・事実・更新リンクで、SQLite のファイル（WAL も）からも消える（`VACUUM`）。Hermes 自身のセッション履歴は消えない。サブエージェント・cron・flush の実行からは使えない。
- **セッションの終わりと切り替え**（`/new`、圧縮など）：`POST /v1/memory/flush` を呼ぶ。
- **サブエージェント・cron・flush の実行**：読むだけ。保存も flush も forget もしない。
- **サイドカーの起動**：止まっていれば、初期化のときに切り離して起動し、最大 15 秒待つ。途中で止まったときは、次の書き込みで起動し直す。Hermes を終了しても、サイドカーは止めない（CLI・gateway・Desktop で共有するため）。止めるときは、`sidecar.log` に書かれた pid を kill する。
- **埋め込みモデルにつながらないとき**：Hermes のログに WARNING を1回だけ出す（初期化のあと、または想起・検索が失敗したとき）。

## 入れ方（導入スクリプト）

Hermes を終了してから、リポジトリのルートで実行する。先に `dsh-plugins` をビルドしておく（`README.ja.md` の「入れ方」の手順 1）。

```sh
python3 integrations/hermes/install.py <プロファイル>
```

- `<プロファイル>` は、`default`（Hermes のホームそのもの）か、`<ホーム>/profiles` の下のプロファイル名。そのプロファイルの `config.yaml` が先にある必要がある（そのプロファイルで Hermes を1回起動しておく）。
- Hermes のホームが `~/.hermes` でなければ `--home <フォルダ>` を付ける。
- スクリプトがすること：`config.yaml` のバックアップ、プラグインのコピー（`<プロファイル>/plugins/tsuzuki`）、`memory.provider: tsuzuki` とサイドカーのパスの追加、`<プロファイル>/tsuzuki/config.json` の作成（あれば残す）。node は、Hermes に同梱の `<ホーム>/node/bin/node` があるときだけ書く。
- **埋め込み**：このリポジトリのエンジンがビルドされていれば（`README.ja.md` の「入れ方」の手順 2）、それを使う（EmbeddingGemma 300m、必要なときにポート 8299 で起動）。なければ Ollama の `embeddinggemma`（`127.0.0.1:11434`）を書き、そのことを表示する。
- **事実の抽出**：そのプロファイルのモデルの接続先（`model.base_url`）を使う。ただし、このマシンの上（ループバック）にあるときだけ。それ以外なら、何も変えずに止まる。`--no-chat`（事実の抽出なし。評価した設定 P ではない）か、会話の本文がそこへ送られてよければ `--allow-remote-chat` を付けて実行し直す。
- `config.yaml` は `yaml.safe_load` で読む（PyYAML が無い Python では、簡単な読み取りに切り替える）。値の引用符が `config.json` に残ることはない。

Hermes を起動し直して、`hermes memory status` で確認する。

## 入れ方（手で入れる）

1. **プラグインをコピーする**。コピー先のフォルダ名は `tsuzuki` にする。

   ```sh
   mkdir -p ~/.hermes/profiles/<プロファイル>/plugins
   cp -R integrations/hermes/tsuzuki ~/.hermes/profiles/<プロファイル>/plugins/tsuzuki
   ```

   - 既定のプロファイルなら、コピー先は `~/.hermes/plugins/tsuzuki`。
   - `plugins` フォルダが無いまま `cp -R … plugins/` を実行すると、`plugins` そのものがプラグインの中身になる。終了コードは 0 のままで、Hermes はプロバイダを見つけられない。
   - Desktop アプリも同じプロファイルを使う。いま Desktop が使っているプロファイルは、`~/Library/Application Support/Hermes/active-profile.json` の `profile` に書かれている。

2. **サイドカーの場所（ビルド済みの JS）を指定する**。次のどれか1つでよい。上ほど優先される。
   - 環境変数 `TSUZUKI_SERVER_JS`
   - そのプロファイルの `config.yaml` の `memory.tsuzuki.server_js`
   - コピー先の `plugin.yaml` の `server_js:`（絶対パス。相対パスはプラグインのフォルダから数える）

   node は、既定で Hermes に同梱の `~/.hermes/node/bin/node` を使い、無ければ `PATH` の `node` を使う（`^22.19` か `>=24`）。変えるときは `TSUZUKI_NODE` か、`node:` の設定を使う。

3. **そのプロファイルの `config.yaml` で有効にする**。`memory:` に `provider` の行を足す。

   ```yaml
   memory:
     provider: tsuzuki
     # 既存の memory_enabled などは、そのまま残す
   ```

4. **新しいセッションから有効になる**。`hermes memory status` で確認できる。

5. **事実の抽出を使うとき**（評価した設定 P）は、`<プロファイル>/tsuzuki/config.json` の `chat` に、このマシンで動くチャットサーバーを書く。書かないと、記憶は発言の窓だけで動く。

   ```json
   "chat": {"baseUrl": "http://127.0.0.1:8080/v1", "model": "", "server": "llama.cpp"}
   ```

   llama.cpp 以外のサーバーなら `"server": "openai"` にする。クラウドの API は書かない（会話の本文が送られる）。

6. **動いているか確かめる**（下の「動いているかの確かめ方」）。

## 設定ファイル（config.json）

初回に `<プロファイル>/tsuzuki/config.json` が既定値で作られる（導入スクリプトを使ったときは、スクリプトが作る）。
- ポート：`8297` に、データフォルダのパスのハッシュを 500 で割った余りを足した値（`8297`〜`8796`）。プロファイルごとに違うので、記憶は混ざらない。
- 埋め込み：Ollama の `embeddinggemma`（`127.0.0.1:11434`）。
- チャット：なし（事実の抽出をしない）。

ログは同じフォルダの `sidecar.log` に出る。

**`config.json` を変えたとき**は、動いているサイドカーを止めてから Hermes を起動し直す。サイドカーは、起動したときにだけ `config.json` を読む。止め方：`sidecar.log` に書かれた pid を kill する（または `lsof -t -iTCP:<ポート> -sTCP:LISTEN` で pid を調べる）。

まれに、ほかのプログラムがそのポートを使っていることがある（例：エンジンの `8299`）。そのときはサイドカーが起動できず、記憶は使われない（`sidecar.log` に `EADDRINUSE` が出る）。`config.json` の `port` を空いている番号に変える。

## 動いているかの確かめ方

`hermes memory status` は、node とサイドカーのファイルがあるかしか見ない。1回会話したあとで、次を確かめる（ポートは `config.json` の `port`）。

```sh
curl -s http://127.0.0.1:<ポート>/health
```

- `embedding.ok` が `true`。`false` なら、`embedding.lastError` に失敗した URL と理由が出る（例：Ollama が動いていない、エンジンの `.venv/bin/python` が無い）。
- `stats.turns` が 0 より大きい。
- `stats.unindexed` が、最後の発言から数秒後に 0 になる。埋め込みモデルが止まっている間は増え、戻るとすぐ 0 に戻る。
- 事実の抽出を使うなら、`chat.ok` が `true`（`null` は、まだ一度も呼んでいない）。

## 埋め込みモデルとライセンス

既定の埋め込みモデルは EmbeddingGemma 300m（Ollama の `embeddinggemma`、またはこのリポジトリのエンジン）。
このモデルは [Gemma の利用規約（Gemma Terms of Use）](https://ai.google.dev/gemma/terms)のもとで配布されている。
このリポジトリにはモデルを含めない。使う人が規約を確認し、自分でダウンロードする（例：`ollama pull embeddinggemma`）。
サイドカーが Ollama にモデルを取りに行くのは、`config.json` で `embedding.acceptModelLicense: true` にしたときだけ。
EmbeddingGemma 2 は、モデルカードのとおり Apache 2.0（Gemma の禁止用途ポリシーに従う）。既定では使わない。

## テスト

Hermes に同梱の Python で実行する（`<版>` は `ls ~/.hermes/tools` で確かめる。どの版でもよい）。

```sh
HPY="$(ls -d "$HOME"/.hermes/tools/python-*/bin/python3 | head -1)"
"$HPY" -I integrations/hermes/tests/test_tsuzuki.py -v   # 偽のサイドカーを相手に
"$HPY" -I integrations/hermes/tests/test_real_sidecar.py -v    # 本物のサイドカー（要ビルド）と偽の埋め込み
python3 -I integrations/hermes/tests/test_install.py -v        # 導入スクリプト（どの Python でもよい）
```

偽のサイドカー（標準ライブラリの HTTP サーバ）と一時ポートを使う。
Hermes の `MemoryProvider`・MemoryManager・プラグインローダーは `~/.hermes/hermes-agent` から読み込むだけで、`~/.hermes` には何も書かない。
