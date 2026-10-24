# 外部記憶の検索：最終設定の固定

選択設定：`p2,rw,fact/dense/orig/none/rerank-qwen`。
選択結果のコミット：`16b1d3609c6fb8380e59a4904712ef8a3b5a67f5`。
設定とコード・モデルのhashは `results/eval/memory/final-config.json` に保存した。
選択規則は変更していない。これは同じ10会話の開発用質問での選択であり、本番用は未計測。
「LoCoMoの同じ10会話の、調整に使わなかった質問への性能」はまだ測っていない。
未知の会話・日本語会話への一般化は評価していない。
t1・p2・全候補、会話別の差、会話単位95%区間、符号検定、複数根拠を含む種類別の結果は
[ステージ3](../results/eval/memory/stage3-dev.md)、
[ステージ4](../results/eval/memory/stage4-dev.md)、
[ステージ5](../results/eval/memory/stage5-dev.md)を参照。
別の会話への一度の独立評価には、この設定を再選択せず使う。データと実行手順は別途承認・固定する。
検索全生成・固定の完了後、条件付き実行許可に従って別の開発用回答pilotを開始した。
120問・7条件の計画は [answer-pilot-plan.json](../results/eval/memory/answer-pilot-plan.json)、
検索完了集計は [stage3-5-summary.md](../results/eval/memory/stage3-5-summary.md)を参照。
本番回答計画は、[案](MEMORY_ANSWER_EVAL_DRAFT.md)に定めた時間と区間幅の開発実測後に固定する。
