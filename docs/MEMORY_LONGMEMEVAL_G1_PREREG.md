# G1: 未知の英語履歴で凍結C-final-norwを1回評価する登録

2026-10-13。**生成・採点は未実行。規模・費用と本登録を提示して人間の確認を待つ。** 準備として公式取得・CPUメタデータ抽出のみ実行。LoCoMo本番、長文精度、メモリ監査の数値を置換しない。A128の中断を維持し、256k/軽量rerankへ進まない。

## データと未調整の範囲

[公式LongMemEval](https://github.com/xiaowu0162/LongMemEval/tree/9e0b455f4ef0e2ab8f2e582289761153549043fc)が案内する[cleaned S版](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f)。revision、bytes、SHA256はsampling登録とplanに固定。取得済み277383467 bytesと公式LFS SHA256が一致した。V2、oracle、Mへ代替しない。

HF dataset cardは英語/MITを明記、公式コードは[MIT LICENSE](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/LICENSE)。出典・著作権表示を保持し、本文・正解・生成をGitに入れず再配布しない。公式READMEの背景履歴の由来（ShareGPT/UltraChatなど）も記録する。配布元のライセンス表示の確認であり、全ての元発言の権利を個別に調査したものではない。

評価質問・履歴・正解はパイプライン設計調整に一切使っていない。質問/正解/履歴本文は研究者がまだ閲覧しておらず、抽出はID/type/session ID/dateだけ。将来、取り込み時に機械が履歴を処理するが、正解・has_answer・answer_session_idsは検索/生成/選択へ渡さない。公開ベンチマークなのでQwenの学習データ混入は未確認であり、モデルにとって未見と断定しない。

## 抽出と規模

内容を見る前に8c06e74でseed `longmemeval-g1-v1`、IDのSHA256順位、quotaを登録。メタデータだけで500問から30問を確定。種類ごと6問（multi-session、temporal-reasoning、knowledge-update、single-session、abstention）。single-sessionはuser/assistant/preference各2。末尾`_abs`は棄権層へ。ID一覧・層・順序・メタデータhashをplanに固定し、別seed/差替え/再抽出をしない。

選択30問の延べセッション数1441、ID+date組1441、履歴メタデータ列30。42〜53セッション/問、平均48.033。ID+dateだけで本文が同一かは未確認。完全なfact要求digest一致だけを共有可能とするが、費用の見出しは1441回を仮定し、共有を見込んで安くしない。全500問には23867セッションがある。session IDの重複を一意と誤認した検査だけf79135eで生成前修正、抽出規則は不変（別逸脱文書）。

主全体値は**層別均等抽出30問の平均**。元500問の分布を代表する値と呼ばない。補助の500問構成へ標準化した平均は各層の母集団件数121/127/72/150/30から固定重みを使う。single-session内部は64/56/30を使う。抽出の少なさを結果を見て補充しない。

## 凍結条件とデータadapter

- C-final-norw = `p2,fact/dense/orig/none/rerank-qwen`。rwなし、元選択39ec1ca/追加norw35b1461の関数・prompt・生成schema・順位規則を参照し、原ファイルhashを保護する。
- 比較は**A30/B1536 30/C-final-norw30の計90回答**、Cのrerank30回。Aを後から成功例だけ/短い例だけに絞らない。取り込みfactは最大1441回、新しい埋め込みを1回構築、評価質問埋め込み30件。各生成要求1HTTP試行、受信済み再送なし。
- 同一Qwen3.8-27B-Q4_K_M、SHA256 `c600de0300ae8a0eb3a6c0b8b5561b8b96f16bd2c863c2a66c42de29d391a747`、build `b10964-b29c606e2`。回答temperature0/seed0/max_tokens256/JSON/enable_thinking=false/cache_prompt=true、元am.GENERATIONのまま。
- 先行host-cache-off確認55528a1は既存30問で通常8k/128kと全30回答・payload・raw一致。従って本測定は全Qwen工程で`LLAMA_ARG_CACHE_RAM=0`、active-slot/cache_prompt/checkpointは保持。C/B/fact/rerank/judge窓8192、A窓131072。host保存cacheだけoffで、全キャッシュoffとは呼ばない。先行一致は外部データでの精度保証ではない。
- EmbeddingGemma 300m fp32/768次元/元query/doc prompt/2048 token cut、元native実装・model/runtime fingerprintを再利用。Qwen作業と埋め込みGPU工程を同時に走らせない。索引は質問ごとの履歴内だけ。全候補p2/fact順位を使い、回答に入れるのは原発言だけ。最近1024＋記憶512、B1536は最近1536の完全発言。次が入らなければ打切り、発言の切断・予算変更はしない。
- adapterはrole/content/date/元順序を保持、発言IDは`question_id/D{session位置}:{turn位置}`、textは`role: content`。session IDが再登場しても出現位置を区別する。p2はsession境界を越えない。factは既存mm.messages/parse/max1500を使い、日付と原発言だけから作る。answer/fact/rerankのsystem/template/schemaは原コードのまま。
- 質問文字列は元questionをそのまま渡す。**question_dateは監査metadataに保持するが、凍結回答templateに独立した基準日fieldがないため追加しない。** session dateは既存renderのDate行で渡す。この制約は相対日時問題に不利な可能性がある。日付を追加したpromptへ後から改変しない。従って公式の別promptによるスコアと直接同条件ではない。
- 公式約115kはLlama3換算で、Qwenの入力長保証ではない。生成開始前に全fact/A/Bおよびrerank実入力をQwen tokenizerで検査。Cの実入力はrerank後、回答開始前に全30件検査。入力+予約出力+余裕128が窓を超えたら全体を止めて報告、全文短縮/セッション分割/別窓/256kで救済しない。Qwen tokenizer検査自体も人間確認後に行う。

## 実行順・保存・環境

承認後、元hash・モデル・開始swap/プロセス確認→8k fact入力preflightとfact全件→Qwenを終了した直列embedding工程→8k rerank30→C入力preflight→A128k30→8k B30→8k C30→8k judge90→元131072窓/元loggingへ予定復帰。各answer条件開始fresh PID、質問順はsampling hash順。judge順は匿名回答IDのhash `longmemeval-g1-judge-v1`、モデルに条件名・他条件回答・timingsを見せない。質問の回答は各1回のみ、結果による再生成なし。

1並列、Qwen1つ、他GPU作業なし。開始時のswap量・RSS>=1GiBまたはCPU>=10%のプロセスsnapshotを記録し、ComfyUI/Ollama/Blender/動画作業を閉じる。Claude/Codex等の例外と実測記録を残す。開始swap>=1GiBは既存swap大印、swap量/ins/out増加条件は時間参考値。既知プロセス検査だけで全GPU非稼働が証明されたとは言わない。

毎件fsync journal、25件/工程切替/finally checkpoint。unknown inflight/通信停止/モデル不一致/資源監視失敗なら停止し再送しない。資源限界footprint>48GiB/available推定<8GiB/critical pressureは正常保存後停止。human pauseは持続し明示再開のみ、Qwenはhuman pauseで止めない。取り込みはpreflight開始からfact/embedding完了まで**12時間の予算**。次要求の最大待ち900秒を残して新要求を控え、予算到達時は保存して確認待ち。全30の取り込みが完了しなければ回答・採点へ進まず、完了した短い質問だけで一般化判定をしない。

## 採点を事前固定

主指標は**固定Qwen judgeによる正誤率**。自己採点の偏りが残る。公式GPT-4o採点の公表スコアとは別。公式[評価関数](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py)の種類別prompt（abstention含む）を固定revisionから抽出し、templateとhashを別JSONに保存。MIT著作権・license全文を同JSONに保持。

1 user message、同Qwen/hash/build、temperature0/seed0/max_tokens10/enable_thinking=false/stream=false/cache_prompt=true、JSON強制なし。回答のanswer文字列だけをhypothesisとして渡す。judgeはquestion/gold/hypothesisだけで判定、条件名・検索順位・全履歴・他armの回答は渡さない。

finish_reason=stopかつtrim/casefold後に厳密にyes/noの場合だけ受理。公式コードの`'yes' in response`とは異なる事前固定の厳密parser。invalid/nonstopはlabel0+judge失敗、再送なし。生成側parse/nonstop/context失敗も主0点として全30分母に保持し、成功例だけにしない。failuresとjudge失敗は別に報告。

公式rubricでは、通常事実は必要情報全て/等価表現、preferenceは個人情報を適切に使えば全rubric点を含まなくても可、knowledge-updateは更新値を含めば旧情報併記可、temporalは日数等のoff-by-oneを許容、abstentionは情報不足を認める回答を許容する。**厳密な全意味要素正解率とは異なる操作的尺度**。元am.abstainsの定型棄権率も別記。referenceある棄権を正解にするのではなく、LMEの答えなし層ではjudgeが正しい拒否を評価する。

副指標は元normalizeによる語彙F1・定型棄権率・生成/採点失敗・種類別正誤率、同paired比較、時間・memory。F1は自由記述/rubric回答には不向きな参考値。長い正解を都合よく短縮しない。

## 推論・事前判定

主比較はC−Aの正誤率差、追加C−B。全体と各5層・single3下位type、全条件を表に載せる。個別質問差と履歴metadata fingerprint別差、勝/負/同/nと厳密両側符号検定（ゼロ除外・無補正）を併記。30問を独立した人間30人と呼ばない。

95% percentile paired bootstrap10000回seed0。主はsession ID/date列の同一履歴fingerprintをclusterとして再標本化、cluster内の全対応質問を一緒に保持。今回metadata上30列全て異なるため質問単位と同じ30単位。補助は5層内のpaired質問bootstrap。背景session共有/同一人物の潜在依存を完全に取り除けず、履歴fingerprintは人物IDではない。

- **登録標本で方向の再現を支持**: 主・補助の両95%区間下限が−0.03以上。これを非劣性の操作的基準とし、未知会話全般へ「一般化できた」と無限定に言わない。Aを上回る証拠は両下限>0、実用幅+0.05は別に区間と比較する。
- **登録標本で不成立**: 両95%区間上限が−0.03未満。
- それ以外は**不確実、再現を確認できない**。点差が良い/0を跨ぐだけで非劣性・同等と言わない。±.03の同等性は区間全体が内側の場合だけ別記。
- 全種類への主張は各層全てが同じ非劣性基準を満たす時だけ。6問/層で区間は広く、今回の標本で±.03を示せる保証はない。1問の差は全体で約.0333。

不成立・不確実でも抽出/条件/判定を変えて再挑戦しない。原因分析や他judgeとの比較は未実行の別探索として明示し、人間の別指示を待つ。独立したモデル学習未見・日本語・実際の自然な長期利用・汎用的意味正答率は主張しない。

非劣性だけでは絶対的な高精度を証明しない。双方低得点でも差の基準が成立しうるため、A/Cの正誤率・生成失敗・棄権を必ず併記する。凍結templateのquestion_date制約やAの入力不適合を隠して、Cの有用な一般化を確認したと言わない。

## 費用：実測記録と移植推定を分離

旧LoCoMo fact合計7202.318秒/272セッション=26.479110秒/セッションは**復元した保存実測成分和**。新LMEの実測単価ではない。1441回なら38156.398秒=10.599時間、20%シナリオ12.719時間（確率的p90/上限ではない）。新セッションの長さ・出力・窓適合が違い、起動/preflight/embedding/saveは別。この20%シナリオは12h予算を超えるため、未完了停止の可能性を承認前に説明する。

全500問23867回ならfactだけ175.549時間、20%で210.659時間。未確認の重複で減ることを前提にしない。代案20問は同程度48セッション/問という仮定でfact約7.07h、20%約8.48h。ただし現在登録は30問であり、20問にするなら生成前にquota/ID/人間レビュー件数を変更登録して確認する。実行後の予算理由による部分採点へ替えない。

A30は既存128k cold HTTP約505〜512秒の記録を参考に、500〜800秒/問を移植した**推定**4.17〜6.67h。保存キャッシュ/履歴共有/入力長によって異なる。B/C/rerank/judge/embedding/preflight/起動/保存を追加し、G1全体を**約16〜21h**のシナリオとして提示する。人手確認/待機/実装時間は別。費用JSONに成分範囲を保存する。12h取り込み予算と総所要時間を混同しない。

## 小出し報告とG3

登録commit時、factの25%/50%/75%/完了（件数・時間のみ）、各回答条件完了（未採点の受信/失敗/時間/memory）、全対応回答のjudge25%/50%/75%/完了で短く報告。judge途中は全armがそろった同じquestion集合だけ暫定と明記、抽出を変えない。区間/判定は全30対応が完了した後のみ。10分の無変化確認・同種異常の継続は通知せず、重大異常は新規時のみ通知する。

人間確認一覧は全30問、C/A回答不一致を先、残りID hash順。question/gold/A/B/C回答/judge結果/evidence照合用原文参照をignored privateJSON/HTMLへ。盲検版は各質問でarmを匿名化・順序固定し、元条件名表は別。人間ラベルはLLMラベルへ書戻さず独立監査として保存する。内容はチャットへ自動転載しない。人間30問×3回答×2〜4分を想定し3〜6人時、未実測。

最終は結論3行、主表/種類別/paired差/時間/footprint/RSS/swap/失敗、確認できたこと・できないこと・主張範囲、元hash/commit/tests。公開は集計・計画・hashだけ別ファイル、pushなし。G1後は人間確認を待ち、G2/G3人手実施/A128再開を自動開始しない。
