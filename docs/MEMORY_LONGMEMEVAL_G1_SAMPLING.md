# G1: 内容を開く前の抽出登録

2026-10-13。CPUで取得可否・メタデータ・費用を確認するための登録。GPU生成、埋め込み、Qwen tokenizer呼出、採点、A128再開は行わない。最終計画・見積もりを提示し、人間の確認後に実行する。

- データ: [公式配布のLongMemEval-S cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)、revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`、`longmemeval_s_cleaned.json`、277383467 bytes、SHA256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`。公式HF treeから確認。V2/oracle/M版へ置換しない。
- 公式GitHub revision `9e0b455f4ef0e2ab8f2e582289761153549043fc`。HF cardはEnglish/MIT、公式repo LICENSEもMIT。データ・原文はignored build/eval内だけに保存し再配布しない。ライセンス表示と由来を保持する。
- 内容閲覧前seed `longmemeval-g1-v1`。`sha256(seed+'|'+question_id)`の昇順で抽出。同hashならID順。本文・正解・根拠・得点・履歴長を選択に使わない。
- 30問: multi-session6、temporal-reasoning6、knowledge-update6、single-session6（user2/assistant2/preference2）、abstention6。`question_id`末尾`_abs`は元typeより優先して棄権層へ。各層不足なら停止し代替しない。
- 取得後のCPUコードはID/type/session ID/dateのみをallowlistで読み、選択ID・件数を出す。JSONデコードは機械的にファイルを読むが、question/answer/haystack_sessions/answer_session_idsを参照・表示しない。履歴本文・正解は研究者がまだ閲覧せず、設計調整に使っていない。
- 費用の暫定単価は保存済みfact 7202.318秒/272セッション。セッション延べ数とID+dateの重複数を別記する。ID+date一致だけでは内容一致は未確認なので、再利用可能数を断定しない。実行承認後に完全なfact要求digest一致だけを再利用する。
- 30問を選択した後の費用が12時間以内の取り込み目安を超えるなら、実行せず事実と選択肢を人間へ提示する。費用や内容を見て質問を差し替えない。
- 新結果で凍結C-final-norw・元結果を更新しない。一般化の判定・モデル・adapter・指標は後続の最終事前登録で生成前に固定する。
