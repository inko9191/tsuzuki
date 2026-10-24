# J1：日本語での記憶（答える役は Gemma 4 E2B）

自作の日本語テスト40問。E2B の窓 131072、E2B-AT が窓に入った問 40/40。主判定は Qwen judge。 P の記憶が答えの会話を全部含んだ割合 1.00。

|条件|正答率|棄権率|入力トークン|回答秒|
|---|---:|---:|---:|---:|
|E2B-P|0.550|0.000|3190|0.9|
|E2B-AT|0.275|0.000|47689|20.2|

主：E2B-P − E2B-AT +0.275 [+0.150, +0.425] → 上回る（勝11/負0/同29）
製品の目安（E2B-P ≥ 0.5）：満たす
Qwen での追試：行う

|種類|abstention|knowledge-update|multi-session|single-session-assistant|single-session-preference|single-session-user|temporal-reasoning|
|---|---:|---:|---:|---:|---:|---:|---:|
|E2B-P|1.00|0.50|0.25|1.00|1.00|0.83|0.12|
|E2B-AT|1.00|0.00|0.12|0.75|0.00|0.50|0.00|
