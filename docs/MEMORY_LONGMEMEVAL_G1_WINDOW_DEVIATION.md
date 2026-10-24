# G1窓超過後の生成前登録逸脱

2026-10-14、人間が原因診断、必要最小窓への拡大、保存済み取り込みでの再開を明示許可。**並べ替え・回答・judgeの出力0件の状態**で登録する。fact取り込み結果だけは以前から保存済み。元5697e93/f80e14a、停止460141fを変更しない。

## 先に決めた規則とCPU診断

規則は診断前のd2e7f4f: `ceil(1.10*(最大入力+120+128))` を満たす16384/32768/65536の最小。余裕を測定後に変更しない。

|入力|最小|中央値|p90|最大|
|---|---:|---:|---:|---:|
|rerank全30候補|5902|9409.5|12084.5|13422|
|C dense順位の参考入力|784|1201.5|1568.7|1590|
|B1536実入力|1089|1394.5|1662.4|1679|
|A実入力|111418|115850|117632.7|118277|

rerank24/30が8192に収まらない。単独で窓超過する候補は0件、最大候補2652。候補数は凍結30のまま、**複数候補の合計**が原因。各質問の初めて超える累積候補番号・候補別unit/原発言ID/token数はwindow-diagnosis.json/.md。

必要値15038、選択窓**16384**。変えるのはrerankの-cだけ。プロンプト/候補数/本文/候補pool/サンプリング/モデル/host-cache-offは不変。C/B回答8192・A131072・judge8192は不変。

Cの各候補先頭30×30参考は最大1674で全て8192に収まる。1024+512予算と空入力template/questionに境界余裕128を加えた構造見積もりは、出力256＋余裕128込み最大2226。これは最終入力の実測でも全順列の証明でもない。**rerank30後、全30の実際のC入力を回答生成前に厳密検査**し、超過なら停止確認待ち。A/Bの保存済み入力は全件検査済み。原文切断/候補削減/救済/再生成はしない。

## 保存と再開

sourceの1441 unique fact journal（各HTTP初回試行）・失敗event3件・prepared・facts・全doc/query vectors・unitsをhashで保護し、取り込み/embeddingを呼ばない新continuationを使用。fact失敗処理は凍結fallbackのまま。新dirにrerank30/answer90/judge90の210登録をfsyncで保存し、旧journal1441には書かない。unknown inflight/通信/監視/資源停止なら自動再送や復旧をしない。

CPU診断のrerank body/orderを保存しhash照合して使う。質問順不変。fresh rerank16384→fresh C8192→fresh B8192→fresh A131072→fresh匿名judge8192→original131072復帰。6planned starts、GPU1並列、各1HTTP試行。元判定3区分・A-eligible/全30運用比較・10000seed0bootstrap/符号検定・judge匿名化不変。

今回人間が許可した**旧G1 overflow停止gateのみ**、実行直前に理由/time/hashを照合してarchive解除する。新しいhuman stopgateは解除しない。G1後は確認待ち、A128/G2/256k/phase3を開始しない。

再開推定4.5〜7.5時間。旧機械時間7時間19分33秒と新機械時間を分け、合算は機械成分和で中断/人間待機を除く。開始時swap・他の重いprocessを再確認し、swap>=1GiBを大印、時間は参考値。

## 運用上の一般化の限界と資源

**元の8k rerank設計は、この長発話データ30問中24問で成り立たなかった。** これは精度・C−A判定とは別の運用上の一般化の限界。16k拡大後の結果を、元の8k設計のまま再現したとは書かない。

各fresh PID開始時のログmodel/KV/compute/output/stateとOS baseline、rerank30/各answer30の50ms footprint/RSSピーク・1秒swapを記録。旧8k rerankは0HTTP、idle baselineだけなので、旧idleと新30問peakを因果的なメモリ差としない。KV確保量は旧8kログと新16kログの実値を並べる。旧8kのrerankHTTP時間は未測定であり速度差を捏造しない。新16kHTTP/prefill/generationと新OSピークを報告する。KV/ログ確保量をRSS/footprintに加算しない。

元immutable/sourceを保護、稼働logのみappendable除外+開始snapshot、journalや原回答は除外しない。公開は別集計/hashのみ、本文はignored build/eval。pushなし、別review作業は触らない。精度の対象は英語固定30問、自己judge偏り・学習混入/独立利用者未確認・小標本の区間幅という限定を保持。
