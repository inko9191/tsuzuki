# EmbeddingGemma 2 テキストエンコーダ仕様（一次情報からの洗い出し）

本書は推論エンジン実装の根拠となる仕様メモ。すべての記述に一次情報の出典（ファイル名と行番号）を付ける。

## 参照した一次情報

| 略号 | ソース | バージョン |
|---|---|---|
| **HF** | `transformers/models/embedding_gemma2/modeling_embedding_gemma2.py` | transformers **v5.19.0**（tag `b39d97c`、PyPI wheel 同梱版） |
| **HFcfg** | `transformers/models/embedding_gemma2/configuration_embedding_gemma2.py` | 同上 |
| **HFmask** | `transformers/masking_utils.py` | 同上 |
| **HFrope** | `transformers/modeling_rope_utils.py` | 同上 |
| **HFhet** | `transformers/integrations/heterogeneity/configuration_utils.py` | 同上 |
| **HFdoc** | `docs/source/en/model_doc/embedding_gemma2.md` | transformers v5.19.0 |
| **vLLM** | `vllm/model_executor/models/embedding_gemma2.py` | vLLM **main @ `1dfe9fb`**（v0.31.0 には未収録） |
| **ST** | `sentence_transformers/...` | sentence-transformers **6.1.0** |
| **HFproc** | `transformers/processing_utils.py` | transformers **5.19.0** |
| **HFtok** | `transformers/tokenization_utils_base.py` | 同上 |
| **Model** | [google/embeddinggemma-2 の実チェックポイント](https://huggingface.co/google/embeddinggemma-2/tree/914f7f89142e33e77833254d9c9b90c3cef7303b) | revision **`914f7f89142e33e77833254d9c9b90c3cef7303b`** |

**2026-10-10 Mac で実チェックポイントを確認済み**。Model のファイルはローカルの
`models/embeddinggemma-2/` に保存。設定ファイルの内容・revision・SHA256 は
[`results/phase1_checkpoint.json`](../results/phase1_checkpoint.json)、重みキー/dtype の確認は
[`results/phase1_validation.json`](../results/phase1_validation.json) に記録した。
参照実装は引き続き設定ファイルを実行時に読み込む。

---

## 1. 全体構成

```
input_ids ──► embed_tokens × sqrt(H) ──┬──────────────────────────────► x
                                       └─► PLE: Linear(H→L·P)×H^-0.5 → reshape[L,P] → RMSNorm(P)  ─► ple[l]
x ─► [EncoderLayer × L]  (各層: Attn ブロック → MLP ブロック → PLE ブロック → × layer_scalar)
  ─► RMSNorm(H) ─► embedding_projection: Linear(H→768, bias なし)  （トークンごと）
  ─► マスク付き平均プーリング ─► L2 正規化(fp32) ─►（MRL: 先頭 d 次元を切り出して再正規化）
```

- 双方向（エンコーダ）: `is_causal=False`（HF L331）。KV キャッシュなし（HF L313-314）。
- **学習済み射影層 `embedding_projection` はバックボーン内部**にあり（HF L504-505, L560 / vLLM L372-379, L405）、
  `last_hidden_state` が既に 768 次元。HF のコメント「Projecting per token is equivalent to projecting after mean
  pooling」（HF L504）の通り、線形かつバイアスなしなのでプーリング前後どちらで掛けても数学的に等価。
  **実チェックポイントに ST 側 Dense 層はない**。Transformer → Pooling → Normalize
  （Model `modules.json` L1-20）。参照実装は `modules.json` を読み、Dense があれば必ず適用、
  未知のモジュールがあればエラーで停止する（`reference/egemma_ref/pipeline.py`）。
- パラメータ数: テキストのみ 271M、+vision 439M、+audio 744M（HFdoc L565）。
  ユーザ指定の「約7.4億」はマルチモーダル全体の値。テキストエンコーダは 271M。
  ライブラリ既定 config（下表）でモデルを構築すると **271.0M（埋め込み 134.2M + 非埋め込み 136.8M）** となり、
  Gemma4 vision(167.4M)/audio(304.8M) タワーを足すと 438.4M / 743.2M で HFdoc の値と一致する
  （実チェックポイントのテキスト設定もこの形状と一致。Model `config.json` L54-129）。

## 2. ハイパーパラメータ（HFcfg L71-91 の既定値、Model `config.json` L54-129 で一致確認）

| 項目 | 既定値 | 備考 |
|---|---|---|
| vocab_size | 262,144 | |
| hidden_size (H) | 512 | |
| intermediate_size | 2048 | GeGLU |
| num_hidden_layers (L) | 24 | |
| num_attention_heads | 4 | 全層共通 |
| num_key_value_heads | 2 | スライディング層。グローバル層は 1（HFcfg L121-128） |
| head_dim | 256 | スライディング層。グローバル層は **512**（`global_head_dim`, HFcfg L121） |
| hidden_activation | gelu_pytorch_tanh | |
| rms_norm_eps | 1e-6 | |
| sliding_window | 512 | **片側半径・両端含む**（HFcfg L39-43） |
| hidden_size_per_layer_input (P) | 512 | PLE 次元 |
| embedding_dim | 768 | 出力次元 |
| attention_bias | False | |
| pad/eos/bos | 0 / 1 / 2 | |

## 3. ローカル/グローバル層の配置

- `layer_types` 未指定時: `i` 層目は `(i+1) % sliding_window_pattern != 0` ならスライディング、そうでなければフル
  （HFcfg L101-105、`sliding_window_pattern` 既定 6, L97）。→ 層 5, 11, 17, 23 がフル、他 20 層がスライディング（5:1）。
- **最終層は必ずフル**（違えば強制上書き, HFcfg L107-111）。
- per-layer override: フル層だけ `head_dim=512, num_key_value_heads=1`（HFcfg L123-128）。
  `config.per_layer_config[layer_idx]` は全体 config のコピーに override を当てたもの（HFhet `_get_layer_config`, `_PerLayerConfigView.__getitem__` L218-268）。
- q_proj: H→4·head_dim、k/v_proj: H→kv·head_dim、o_proj: 4·head_dim→H（HF L333-344）。
  したがってスライディング層は q:512→1024, k/v:512→512、フル層は q:512→2048, k/v:512→512, o:2048→512。

## 4. アテンション（HF L310-389, vLLM L90-182）

1. `q = q_proj(h)`, `k = k_proj(h)`, `v = v_proj(h)` を `[B, heads, T, head_dim]` に整形（HF L360-362）
2. **QK-norm**: `q = RMSNorm(q)·q_norm.w`, `k = RMSNorm(k)·k_norm.w`（head_dim 次元, HF L364-365）
3. **V-norm**: `v = RMSNorm(v)` **重みなし**（`with_scale=False`, HF L348, L366 / vLLM L131 `has_weight=False`）
4. RoPE を q, k に適用（正規化の**後**, HF L368-369 / vLLM L174）
5. **スケーリング 1.0**（`1/sqrt(d)` を掛けない。HF L329 / vLLM L150）
6. GQA: kv ヘッドを `repeat_interleave` で複製（ヘッド h は kv ヘッド `h // n_rep` を使う, HF L263-272）
7. マスク加算 → softmax は fp32（HF L298-302）→ `P·V` → o_proj

### マスクとスライディングウィンドウの「オフセット」
- フル層: キー側パディングのみマスク（`create_bidirectional_mask`, HF L540; `padding_mask_function` は kv_idx のみ参照 HFmask L169-180）。
- スライディング層: **`|q_idx − kv_idx| <= sliding_window`**（両端含む, HFmask `sliding_window_bidirectional_overlay` L142-150）かつパディングマスク。
- オフセットの正体: 因果 Gemma の `sliding_window` は片側幅だが、本モデルでは**半径**（HFcfg L39-43: 参照実装の 1024 幅の半分 = 512）。
  さらに flash-attn/vLLM の窓は「`<= W−1`」なので **+1** して渡している（HF L321-324 `config.sliding_window + 1`、
  vLLM L83 `_WINDOW_OFFSET = 1` コメント「HF masks |q-k| <= W; vLLM ENCODER_ONLY masks <= W-1」, L154）。
  → 自前カーネルでは **`|i−j| <= 512`（1025 トークン幅）** を正とする。`< 512` にすると誤り
  （ランダム重みの変異テストで文ベクトル cos が 0.977 まで劣化し、基準 0.9999 で検出できることを確認済み）。

### RoPE（HF L70-139, L230-260）
- NeoX 形式（前半/後半ペア, `rotate_half`; vLLM L143 `is_neox_style=True`）。`emb = cat(freqs, freqs)`。
- 層種別ごとに別パラメータ: スライディング θ=10,000、フル θ=1,000,000（既定, HFcfg L113-116）。
  inv_freq は**その層種別の head_dim**で計算（HF L94-97: per_layer_config[layer_type] から）。
  → スライディング: d=256、フル: d=512。
- `rope_type` が `"proportional"`（Gemma 4 系の部分回転）の場合: 先頭 `int(partial_rotary_factor·d/2)` 周波数のみ回転、
  残りは周波数 0（HFrope L193-265）。実 config がこれを指定していても参照実装は対応済み。その他の型は未実装でエラー。
- position_ids はパディング込みの `arange(T)`（HF L529-530）。RoPE 内積・窓判定とも相対距離のみに依存するため、
  **左右どちらのパディングでも結果は不変**（テストで確認済み）。

## 5. 正規化（HF L49-67）
- `y = x · (mean(x²) + eps)^(-0.5) · w` を **fp32** で計算し入力 dtype に戻す。
- **重みはそのまま乗算**（Gemma 1-3 の `1 + w` ではない）。vLLM も標準 RMSNorm（vLLM L26, L129-131）。

## 6. エンコーダ層（HF L392-438 / vLLM L279-298）
```
h = x + post_attention_layernorm( Attn( input_layernorm(x) ) )
h = h + post_feedforward_layernorm( MLP( pre_feedforward_layernorm(h) ) )     # MLP = down(gelu_tanh(gate(x)) * up(x))  (HF L167-169)
h = h + post_per_layer_input_norm( per_layer_projection( gelu_tanh(per_layer_input_gate(h)) * ple[:, :, i] ) )   # HF L220-227
h = h * layer_scalar                                                          # HF L437、永続バッファ（チェックポイントに含まれる）
```

## 7. PLE（Per-Layer Embeddings, HF L172-200 / vLLM L301-334）
- **projection-only**: 補助ルックアップテーブルなし。`inputs_embeds`（√H スケール後）から導出。
- `ple = RMSNorm_P( reshape( Linear(H → L·P)(x) · H^-0.5, [B, T, L, P] ) )`、i 層目は `ple[:, :, i, :]`。

## 8. 埋め込み
- `embed_tokens(ids) · sqrt(H)`。スケールは**重み dtype に丸めてから**乗算（HF L150-153, L490-493; vLLM L387-390）。
  bf16 だと √512 = 22.6274 → 22.625。fp32 参照では 22.6274。
  今回は保存された bf16 重みを fp32 に変換し、ST と参照実装の双方を fp32 で検証した
  （`scripts/validate_vs_st.py` L73-81、`results/phase1_validation.json` weights）。

## 9. プーリングと出力（HFdoc L39, L96-99, L134）
- マスク付き平均プーリング（パディング除外）→ **fp32 で L2 正規化**。vLLM の既定プーリングも MEAN（vLLM L979）。
- プロンプトトークンもプーリングに含める（HFdoc L110「prepended to the input (and included in mean pooling)」）。
  実設定は `pooling_mode="mean"`、`include_prompt=true`、後段 Normalize
  （Model `1_Pooling/config.json` L1-5、`modules.json` L14-19）。
- MRL: 768 → 512/256/128 は**先頭を切り出してから再正規化**（HFdoc L134。ST の `truncate_dim` は正規化済み出力を切るだけなので注意）。

## 10. 入力書式（タスク指示プレフィックス）
HFdoc L121-130 の表。Model `config_sentence_transformers.json` L9-29 でも以下の全プレフィックスを確認:

| prompt_name | プレフィックス |
|---|---|
| query / Retrieval-query / Retrieval / Reranking / BitextMining / SearchQuery | `task: search result \| query: ` |
| document / Document / Retrieval-document | `title: none \| text: ` |
| Classification / MultilabelClassification | `task: classification \| query: ` |
| Clustering | `task: clustering \| query: ` |
| CodeRetrieval / InstructionRetrieval | `task: code retrieval \| query: ` |
| FactChecking | `task: fact checking \| query: ` |
| QuestionAnswering | `task: question answering \| query: ` |
| STS / SentenceSimilarity / PairClassification / Summarization | `task: sentence similarity \| query: ` |

- AutoTokenizer 利用時はプレフィックスを文字列連結（HFdoc L84-89）。
- AutoProcessor/ST では**チャットテンプレートの `system` メッセージ**としてプロンプトを渡す（HFdoc L110, L433-447）。
  ST は `modality_config` に `message` がある場合テキストをメッセージ形式に変換し、プロンプトを system メッセージとして前置する
  （ST `base/modules/transformer.py` preprocess L1361-1393、`base/modality.py` L679-696）。
  トークン列は `<bos> [prompt] <text_tokens> <eos>`（HFdoc L400-404）。vLLM も BOS+EOS を付与（vLLM L420-422）。
- **実設定は chat 経路**（Model `sentence_bert_config.json` L20-24）。チャットテンプレートは
  system のテキスト + user のテキストを連結するだけで、turn マーカーや BOS/EOS 自体は挿入しない
  （Model `chat_template.jinja` L1）。Processor がその後トークナイザで BOS/EOS を付与する。
- 特殊トークンの重要な差: Tokenizer の `apply_chat_template(tokenize=True)` は
  `add_special_tokens=False`（HFtok L3123-3131）。Processor 版はテンプレート展開後、BOS で始まる場合のみ
  特殊トークンの追加を無効にする（HFproc L2218-2227, L2250-2255）。ST は Processor 版を呼ぶ
  （ST `base/modules/transformer.py` L1965-1987）。参照実装も展開した文字列をこの条件でトークナイズする。
  初回の BOS/EOS 欠落を修正後、6プロンプト × 63テキストのトークン列は全一致
  （`results/phase1_validation.json` cases.token_mismatches）。
- `max_seq_length` は ST 設定に明示値なし（Model `sentence_bert_config.json` L1-27）。
  実行時の ST は tokenizer の `model_max_length=1000000000000000019884624838656` を返す
  （Model `tokenizer_config.json` L23、ST `base/modules/transformer.py` L1137-1142、
  `results/phase1_checkpoint.json` st_max_seq_length）。モデルの `max_position_embeddings=262144`
  （Model `config.json` L93）とは別の値。参照実装の max_seq_length は None。
  今回は長さ3〜1357のみ検証し、切り詰め・上限近くの挙動は未検証。

## 11. チェックポイント
- テキストバックボーンのキーは `language_model.` 配下（HF `EmbeddingGemma2Model` L649-650、vLLM L1094-1097）。
  **実重みもこのプレフィックス**。ローダは `embed_tokens.weight` で終わるキーからプレフィックスを自動検出し、
  期待キーの過不足を厳密チェックする（`reference/egemma_ref/weights.py`）。
- `vision_tower.* / embed_vision.* / audio_tower.* / embed_audio.*` はテキスト専用時に無視（HF L637-642, vLLM L1117-1129）。
- 実重みのテキスト413テンソルはすべて bf16、tower963テンソルを除外。`layer_scalar` は全24層に存在し、
  デフォルト補完は0（`results/phase1_validation.json` weights、`reference/egemma_ref/weights.py` L122-128）。

## 12. 実装上の注意（フェーズ2/3 向け）
- スケーリング 1.0 のため、QK ロジットの大きさは q_norm/k_norm の学習済み重みに直結する。
  fp16 でロジットを保持するとオーバーフロー/精度落ちの恐れ → **QK 積と softmax は fp32 累積**で実装する。
- スライディング層は窓 1025 幅なので、T≤513 では実質フルアテンションと同じ。
  T≥514 では窓外の組が生じる（最も離れた位置の距離はT−1、HFmask L142-150）。
- パディングクエリ行が全マスクになる場合（短文を長くパディング）でも NaN を出さないよう、`-inf` ではなく有限の最小値で
  マスクする（HF eager と同じ）。

## 13. フェーズ2のネイティブ実装と検証

上記の演算を `native/engine.mm` / `native/kernels.metal` に移植した。
モデル設定と safetensors をC++側で直接読み、Transformer → mean Pooling → Normalize のST構成を確認する。
トークナイズはPythonで第10節と同じProcessor条件を再現する。
未知の数値設定・追加STモジュールを黙って省略せずエラーにする。
詳細なビルド・入出力・設計判断は [`NATIVE.md`](NATIVE.md)。

fp16でも QK積、softmax、RMSNorm、プーリングの和・除算、L2正規化はfp32で計算する。
特にプーリングの和をfp16に丸めると1025トークンの実入力で65504を超えたため、
平均までfp32を維持する（`native/kernels.metal` L169-177、`results/phase2_fp16_diagnosis.txt`）。

フェーズ1のMPS/fp32を正解とする6プロンプト × 63文 × 4次元の比較では、
fp32最小cos≥0.9999、fp16最小cos≥0.999を全ケースで満たした
（[`results/phase2_validation.json`](../results/phase2_validation.json)）。

## フェーズ3の実行経路

数値仕様・トークナイズ・学習済み射影は維持し、C++/Metal側の実行を最適化した。
大きい線形投影はSIMD-group matrixのfp32積和、通常推論は1 command bufferと再利用scratch、
長文は8-query×32-keyのタイルとオンラインsoftmaxでQK/softmax/PVを融合する。
fp16での指数の丸めと演算順が変わるため、各変更後に6×63×4次元を両dtypeで確認する。
CPU/GPU同期を減らしても、GPUのread/write順は各dispatch後のbuffer barrierで保証する。
実測・一次資料・精度・制限は `docs/OPTIMIZATION.md` と `results/phase3_*.json` に記録。
