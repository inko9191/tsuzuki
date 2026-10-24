# G1 結果保存後の報告処理修正の登録逸脱

2026-10-14、人間がd0dfd13確認後に明示許可。これは出力を得た後の報告処理修正であり、生成前の登録ではない。元計画・窓逸脱・原journal・回答・Qwen判定は変更しない。

1. 計測工程のmodel/KV/computeログ必須条件を保持し、original restoreだけ通常verbosityで未記録を許容する。未記録は不明であり0ではない。今回は保存済み復帰ログをCPU確認し、サーバー起動/health/tokenize/回答/judgeを一切呼ばない。
2. 登録5層はstratumのみで集計。single-session内訳はstratumとtypeの両方で限定し、棄権を元種類へ重複させない。型別は傾向参考で、有意性を主張しない。
3. 修正前後の全体指標・A適格集合・C−A/C−B/F1の全比較・区間・符号検定が完全一致することを照合。不一致なら報告確定せず停止。実出力1651unique/attempt1と保護hashを照合。変更する2コードだけ元hashと新hashを別記、元登録hash辞書は変更しない。
4. 30問の質問・正解・A/B/C回答・Qwen判定一覧をprivateへ出力し、C/A判定不一致を先頭。Codex独立採点は条件/Qwen判定を隠した単回読取、公式種類別rubricの補助分析。既知の全体値を含む完全blindではない事後分析で、登録Qwen採点を置換しない。人手採点とも呼ばない。
5. 取り込みは1記憶につき1問の今回構造におけるC総コストとして別枠。30問・英語のみ・回答者と同じQwen採点・16k rerank必要・swap時間参考を明記。新実験なし。元stopgate/workerstatusは証跡として保持し、CPU報告完了を別保存。automation PAUSEDを維持。

修正前hashと検証規則はlongmemeval-g1-finalization-deviation.json。公開は集計/hash/修正コードのみ、本文・レビューはignored build/evalのみ。pushなし。
