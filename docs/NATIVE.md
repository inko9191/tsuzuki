# フェーズ2: C++ / Metal エンジン

フェーズ1の `reference/egemma_ref` を正解とし、同じ実重みを safetensors から直接読む。
計算はすべて Metal。Python/PyTorch のフォワードへのフォールバックはない。
フェーズ2を完了した時点のブランチは `phase2-native`（作者の非公開の開発リポジトリ）。

> 公開版の注：エンジンの内部名（`egemma_native`・`egemma-cli`・`egemma_server`・`EGEMMA_*` の環境変数など）は開発時の名前のまま。tsuzuki の README の「Names」を参照。

## ビルド

リポジトリのルート（以下 `<repo>`）で作業する。重み・`.venv`・ビルド・ログもこのフォルダに置く（`models/`・`.venv/`・`build/` は git の対象外）。
Xcode と CMake が必要。pybind11 は拡張モジュールを作るためだけの依存で、実行時には不要。
Apple の Foundation / Metal 以外の C++ ランタイムライブラリは追加していない。

```bash
cd <repo>
uv pip install --python .venv/bin/python -r requirements-native.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build -j 8
build/egemma-cli --info
```

`native/kernels.metal` をビルド時にライブラリへ埋め込み、起動時に Metal API でコンパイルする。
このMacでは `metallib` コマンドが見つからなかったため、実行時コンパイルを採用した。
`MTLDevice.newLibraryWithSource:options:error:` は SDK `MTLDevice.h` L783-786 に定義されており、実機で成功確認済み。
演算は safe/precise を指定（SDK `MTLLibrary.h` L311-320）。OS の deployment target は macOS 15.0。
拡張モジュールはビルドに使った Python バージョンに対応するので、Python を変更したら再ビルドする。

## CLI

CLI は BOS/EOS を含む**トークンID**の JSONL を読み、入力系列ごとに1行のベクトル配列を出す。
自身ではトークナイズや特殊トークン追加をしない。

```bash
printf '%s\n' '[2,123,1]' '{"input_ids":[[2,124,1],[2,125,126,1]]}' |
  build/egemma-cli --model-dir models/embeddinggemma-2 --dtype fp32 --dim 512
```

上の例は説明用の整数ID。1行は単一系列の配列、または `input_ids` を持つオブジェクト。
オブジェクトは単一系列にもバッチにも対応する。可変長の行は右側に0を埋め、内部でパディングマスクを作る。
明示マスクを渡す場合は `attention_mask` を同じ行構造・長さで指定する。

```json
{"input_ids":[[2,123,1,0],[0,2,124,1]],"attention_mask":[[1,1,1,0],[0,1,1,1]]}
```

`--dtype` は `fp32` / `fp16`、`--dim` は既定768（通常は768/512/256/128）。
入力エラーは stderr と終了コード1で通知する。標準出力には成功した系列のベクトルのみ出力する。

## Python バインディングとトークナイズ

```python
import sys
sys.path.insert(0, "native")
from egemma_native import NativePipeline

pipeline = NativePipeline("models/embeddinggemma-2", dtype="fp16")
vectors = pipeline.encode(
    ["Which planet is known as the Red Planet?", "赤い惑星は何ですか？"],
    prompt_name="query", batch_size=8, dim=512,
)
```

トークナイザは Python の AutoTokenizer。実チェックポイントのテンプレートを文字列に展開してから、
Processor と同じ条件で BOS/EOS を付ける。展開結果が BOS で始まる場合のみ追加を無効にする。
これは `reference/egemma_ref/pipeline.py` L131-149 と同じ規則であり、全378入力および空文字・空白・BOSの
追加プローブで参照実装とのID一致を確認した。system にプロンプト、user に本文を渡し、両者を連結する。
プロンプトは実チェックポイントの `config_sentence_transformers.json` を読む。

トークン配列を直接渡す場合:

```python
ids, mask = pipeline.batch([
    pipeline.tokenize_one("Mars", pipeline.prompts["query"]),
    pipeline.tokenize_one("火星", pipeline.prompts["query"]),
])
# ids/mask: C contiguous int32 numpy arrays [B,T]; output: float32 [B,dim]
vectors = pipeline.engine.encode(ids, mask, 768)
```

空バッチ、空系列、範囲外ID、0/1以外のマスク、不一致形状・dtype、不正な出力次元を拒否する。
同じエンジンの呼び出しは mutex で直列化し、計算中は Python GIL を解放する。

## 数値仕様と実装上の判断

- safetensors のヘッダ・オフセット・形状・dtype を確認し、テキスト側のプレフィックスを自動検出。
  元の bf16 を選択した fp32/fp16 に変換する。実重みは `language_model.` の413テンソル、tower963を除外。
  学習済み `embedding_projection`、PLE、24個の `layer_scalar` をすべて使用する。
  根拠: `reference/egemma_ref/weights.py` L33-62, L88-120。
- 行列積は16×16タイルの素朴なカーネル。QK と PV の積は fp32 累積、活性化と線形出力は選択dtype。
  注意点はスケーリングが **1.0** であること、GQA の対応が `head / (heads/kv)` であること。
  根拠: `reference/egemma_ref/model.py` L141-147。
- スライディング窓は両方向の **`|q-k| <= 512`**（両端含む）。キーのパディングだけをマスクし、
  全マスク行も有限の最小値を使って softmax の NaN を防ぐ。位置番号はパディング込み。
  根拠: `reference/egemma_ref/model.py` L101-118。
- QK-norm、重みなし V-norm、RMSNorm は fp32 計算から選択dtypeへ戻す。RMSNorm の重みはそのまま掛ける。
  RoPE は NeoX の前半/後半方式、default と proportional をサポート。
  根拠: `reference/egemma_ref/model.py` L20-26, L37-60, L136-140。
- GELU は実チェックポイント指定の tanh 近似を実装。未知の活性化・RoPE・数値設定・STモジュールはエラー。
  ST の追加 Dense 層を無視することはない。現在の Transformer → mean Pooling → Normalize の構成を確認する。
  根拠: `reference/egemma_ref/model.py` L29-34、実 `modules.json` L1-20。
- 射影は参照と同じくトークン単位で行う。マスク付き平均を求めた後、MRL の先頭d次元を fp32 でL2正規化する。
  全次元正規化→先頭切り出し→再正規化の共通スカラーは相殺されるため、先頭を直接正規化する。
  丸めの違いは全入力・全次元の実試験で確認した。根拠: `reference/egemma_ref/model.py` L168-187。
- **fp16 の平均プーリングも和・除算を fp32 で保持する**。初回は和を fp16 へ戻していたため、
  1025トークンの入力で和80049.215がfp16の上限65504を超え、出力が非有限になった。
  バックボーンのキャプチャはすべて有限で、平均は最大78.097だった。
  `native/kernels.metal` L169-177 を修正し、基準を変えず全ケースを再試験した。
  診断: [`results/phase2_fp16_diagnosis.txt`](../results/phase2_fp16_diagnosis.txt)。

## 検証

```bash
export HF_HOME="$PWD/models/.hf-home"
export HF_HUB_OFFLINE=1
.venv/bin/python scripts/validate_native.py --model-dir models/embeddinggemma-2
.venv/bin/python scripts/selftest_native.py
.venv/bin/python scripts/benchmark_native.py --model-dir models/embeddinggemma-2
```

受け入れ試験は正解をフェーズ1の MPS/fp32 に固定し、6プロンプト × 日英63文 × 4次元 × fp32/fp16。
全次元をネイティブAPIに直接指定して比較する。平均でなく**すべての文の最小 cos**で合否を判定する。
fp32 ≥ 0.9999、fp16 ≥ 0.999。系列長3〜1357、生成長文3本はすべて512を超える。
左右パディング、マスクの穴、全マスク行、不正入力7種、CLIとバインディングの完全一致も検証する。
別の小規模ランダム重み試験で短い窓・GQA・異なるhead_dim・partial RoPE・attention bias・layer_scalarを確認し、
追加Denseと欠落射影がエラーになることも確認する。

層別診断は `pipeline.engine.encode(ids, mask, 768, "build/captures")` で76点の中間出力を保存する。
`captures.json` に形状、各 `.f32` に fp32 の値を記録。検証スクリプトは失敗バッチを再実行してこれを参照と比較する。

結果とベンチマーク表は [README](../README.md) に掲載。

## 制限

現在は精度を優先した初期実装で、STより遅い。1層ごとに同期し、注意行列を明示的に作るため、
メモリは系列長に対して二次で増える。Metalのバッファ・32bitインデックス制限を超える入力は拒否する。
切り詰め、任意の追加STモジュール、exact GELU、量子化、画像・音声・動画は未対応。
このMac・記載のバージョンで検証済み。以上はフェーズ2完了時点の実行経路・制限。

## フェーズ3の最適化とプロファイリング

最適化の計測・各コミットの精度・速度差は `docs/OPTIMIZATION.md` を参照。
現在の経路はM=B*T<256なら従来16×16行列積、それ以上ならSIMD-group matrixの32×32タイル。
通常推論は1 command bufferで実行し、層間のscratch bufferを再利用する。
キャプチャ指定時は層ごとの待機を保持する。
T>=256かつhead_dim<=512ならオンラインsoftmaxの融合アテンションを使い、
長さの二乗に比例するscores/probs bufferを確保しない。

```python
import json
from egemma_native import load_extension
engine = load_extension().Engine("models/embeddinggemma-2", "fp16", profile=True)
vectors = engine.encode(ids, mask)
print(json.loads(engine.profile()))
```

`profile()` は最後に成功したencodeのdispatch数、GPU完了待機回数、GPU内barrier数、
command bufferのGPU時間、CPU待機時間、Metalの割り当て観測ピークをJSONで返す。
`profile=True` のときだけカーネル別GPUタイムスタンプを採る。
対応sampling pointはデバイスに問い合わせ、dispatch boundaryが未対応なら
compute encoderを分割してstage boundaryで測る。計測による性能影響があるので、
スループットには必ずprofile=Falseのwall timeを使う。
今回のMacはstage boundary対応、dispatch boundary非対応。
通常経路はT<256が607 dispatch、融合経路は559 dispatch、GPU完了待機はともに1回。

## フェーズ3.5の実行経路と行列積バックエンド

```python
engine = load_extension().Engine("models/embeddinggemma-2", "fp16", matmul="auto")  # "native" / "mpp"
```

```bash
build/egemma-cli --model-dir models/embeddinggemma-2 --dtype fp16 --matmul native < ids.jsonl
EGEMMA_MATMUL=native .venv/bin/python scripts/validate_native.py --model-dir models/embeddinggemma-2
```

- `native`: 自作のSIMD-groupカーネルだけを使う。どのMacでも動く既定のフォールバック。
- `mpp`: fp16で行数（B×T）が64以上の線形層に、Metal Performance Primitivesの `matmul2d` を使う。
  使えない環境ではエラーになる。fp32ではMPPの方が遅かったため、`mpp` を指定しても自作経路になる。
- `auto`（既定）: Metal 4、Apple10ファミリ（M5世代）、MPPライブラリのコンパイル成功の3条件がそろえば `mpp`、
  そうでなければ `native`。`auto` のときだけ、環境変数 `EGEMMA_MATMUL` で上書きできる。
  `engine.info()` の `matmul_backend` / `mpp_status` / `mpp_min_rows` で、実際に選ばれた状態を確認できる。
- ビルド時に `-DEGEMMA_ENABLE_MPP=OFF` を指定すると、MPPのコードを含めずにビルドする。
- MPS、MPSGraph、Accelerateは使わない。MPPは、シェーダ内で呼ぶ `matmul2d` だけを使う。

経路の要点（詳細と測定は `docs/OPTIMIZATION.md` の施策1〜4）:
- N・Kが32の倍数の線形層の重みは、読み込み時に転置 [K,N] で保存する。
  q|k|vは1本に連結し、gate/upは16列ずつ交互に並べてまとめる。
- 線形層には、K分割のSIMD-groupタイル（またはMPP）を使う。epilogueでバイアス、GELU×up、GELU×PLEまで計算する。
- QKV射影の後は `qkv_norm_rope`、残差normの連鎖は `norm_chain` で処理する。1層あたり11 dispatch。
- アテンションは、T≥128で `flash_*`（Kは転置配置）、T<128で従来のqk/softmax/pvを使う。
- 活性化は、encode中だけ行数を32の倍数に切り上げて確保する。層をまたぐ出力は、層の偶奇で切り替える専用bufferに置く。
- 精度ゲートは `scripts/validate_phase35.sh <stage>`。6×63×4次元を両dtypeで実行し、
  長文・混在長・padding・mask holes・全マスク行、ランダム重みT40/T257も検証する。
