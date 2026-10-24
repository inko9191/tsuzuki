# Files changed for publication

This repository is a single-commit snapshot of the author's private development repository (branch `dsh-plugin`, commit `e4717a8`, 2026-10-24), made with `git archive` and then edited as listed here. 197 files are byte-identical to that commit at the same path. Every other file is listed below with its sha256 before and after, so the hashes recorded in the plan files can still be checked.

## Evaluation records

Changed only to remove personal data or to state the status. Everything else in `docs/MEMORY_*`, `scripts/eval/` and `results/eval/memory/` is unchanged.

| File | What changed | sha256 before | sha256 now |
|---|---|---|---|
| `docs/MEMORY_CODING_C1_PREREG.md` | a note at the top that C1 was stopped without a verdict, and what this repository leaves out. | `623f9880035fad92c177072f0acd7a65549b3c9862e44907f6b17488f7313399` | `a75671ba623fcce2e0b66c0e3cb99d8ac933cf18723e377dae644ca1b2155a10` |
| `docs/MEMORY_LONGMEMEVAL_E1_PREREG.md` | the first sentence of the question named the author's planned product; it now states the premise neutrally (small models on an ordinary PC). `longmemeval-e1-plan.json` records the original hash. | `6f4760e3ac0cb92c89723613558e112925a012649333095abbd7ad936461d65d` | `558de787e06d64ee6d42297d63480c0d652480fbf13311e0aec60f9c7349f16e` |
| `docs/MEMORY_JA_J1_PREREG.md` | the same kind of mention of the author's planned product was removed or made neutral (three phrases); conditions and verdict rules are unchanged. `j1-plan.json` records the original hash. | `505765fe7020826af8ab3404ffb2417e71c77734f380373ed942a269fa045fd1` | `50f69bad00ee0eb619a65c157ca5291160d35e33e3a2d4a235eabdf6930e9746` |
| `results/eval/memory/longmemeval-u1-plan.json` | the local path of the Qwen model file in `server_argv` (it contained the user name) is now `<models>/Qwen3.8-27B-Q4_K_M.gguf`. `longmemeval-u1-report.json` records the original hash as `plan_sha256`. | `c52a6f73cd09de84ce73d65e329653c4e7b75c3e7de425adc6d4f1bf1fa7d963` | `e79f58938ccab7ac709a0ff10450f7d95cddabd22a2616e1c83adb8529fddc25` |
| `results/eval/memory/longmemeval-u2-plan.json` | the same path in `server_argv`, replaced the same way. | `84b91a379234f015ee636968ae5c8d32740c274bc2e5cb21ef23d1d77342c06d` | `a58697a6216acdf963afaad11ff2ccb8311d708c5f96c52b12f5c4ff177dfba2` |

## Not committed at that commit

Taken from the development working copy, unchanged (the scale test report, and the J1 secondary-judge script, which is not among the J1 files whose hashes `j1-plan.json` records).

| File | sha256 |
|---|---|
| `results/eval/memory/scale-report.md` | `dbcc3e9bb7b043afa63ba5360d91aaa39983ad80da0ed21907142d2e6ce909d5` (as written); now `be4159d822ea05186e751c530db450c4817f244b6077e56ed33d6a5dc23567e1` after a note at the top that the listed bugs were fixed later |
| `scripts/eval/j1_claude_judge.py` | `b31f932d98a15dd8fa98a946e4fbdbd64f8058ee23c04802db6bb5cba50196d0` |

Also added: `data/texts_ja_en.jsonl` (60 synthetic sentences used by `scripts/validate_*.py`), unchanged, sha256 `2fef15594d70b5b7ba1208e84e86bf8400e2556b138bfded516b553763b1192f`.

## Renamed to tsuzuki

Code, tests, package files and user documentation: the user-facing names changed from egemma to tsuzuki (see "Names" in the README). Also: `NOTICE` names the project and the Gemma 4 E2B license; `tests/plugins.test.mjs` skips the tests that need a downloaded model or the built engine when they are missing; `integrations/hermes/tests/test_tsuzuki.py` uses a neutral profile name; `docs/NATIVE.md` and `docs/DSH_PLUGIN.md` lost machine-specific paths and got a note on the names; `.gitignore` was trimmed; the development `README.md` was replaced by `README.md` and `README.ja.md`.

| Before | Now | sha256 before | sha256 now |
|---|---|---|---|
| `.gitignore` | `.gitignore` | `b287888b259402bc43987533748d054fe1231940c9a74f20e84c88b8bce15965` | `471c2c0481757ac89e8674268b8ff69087d4514aa13c5e7811e555ccfb18f29a` |
| `NOTICE` | `NOTICE` | `8be5a3d05f729dce1436c4a98e9390537b5fcca875622c96dea766a4f2a4a38e` | `edb434829e590a1dd5561e139f0602e90643461decc7be6ac757b0a50db98f56` |
| `docs/DSH_PLUGIN.md` | `docs/DSH_PLUGIN.md` | `399fbed4ed300d7c2f45d5d80fef28f5d296fd33eb4154e55cd797ffe5880442` | `5a717d9d12e0ec26761dc2715d5eeaae5b49c824d2727c75009be5b3b1c1bfaa` |
| `docs/NATIVE.md` | `docs/NATIVE.md` | `509961b14221866590467e154c2467b8d4242aa0cdd2cc2453173c74fdd9a783` | `018a51f5274cabf84a5142cef1f862cca54b66b1f9df668733797daf1caab384` |
| `dsh-plugins/package.json` | `dsh-plugins/package.json` | `904e40dd09ce6b5d8af73dd2522b6b08cacfbc18798280e7b5677b6f24bf96ff` | `fa0d29b72b1ce1e88499f953792555d8b65ef186b55de3a0722ef0463bacd374` |
| `dsh-plugins/packages/memory-egemma/NOTICE` | `dsh-plugins/packages/tsuzuki/NOTICE` | `8be5a3d05f729dce1436c4a98e9390537b5fcca875622c96dea766a4f2a4a38e` | `edb434829e590a1dd5561e139f0602e90643461decc7be6ac757b0a50db98f56` |
| `dsh-plugins/packages/memory-egemma/README.md` | `dsh-plugins/packages/tsuzuki/README.md` | `4db8ba652daf5c6024dccd78b4a093e930ed800967aad368b5f480fc5ce05218` | `75a87343ec421b4ca3993cfc5002f7cd75391026901e7a77fd103a97d87a90fd` |
| `dsh-plugins/packages/memory-egemma/bin/egemma-memory.mjs` | `dsh-plugins/packages/tsuzuki/bin/tsuzuki-memory.mjs` | `07482980cc56a4a8ac61cf6aa5b78a1bd0576229f1c3c0dec6b450cd92873ecb` | `de262f295c0a6a2135bb763dfae05b48a792a91af387c9a41ea61f932c4bf54c` |
| `dsh-plugins/packages/memory-egemma/cordis.patch.yml` | `dsh-plugins/packages/tsuzuki/cordis.patch.yml` | `65ff230727649ec1f112ffdfdb7d044f3af5e4fffb8fba780dd7e149cec17088` | `5d98a53ed0c9eafe2b791177de413377f5a457e2e024a2b5a2f1ba50a0b09eb1` |
| `dsh-plugins/packages/memory-egemma/package.json` | `dsh-plugins/packages/tsuzuki/package.json` | `3e71fd662f4fcb60e8d1322642f161a0dc4ea72cfa26072d851a3eae310c0852` | `7602b36568f14f6145dd4922df9f89fddd6a3e1e9817d540447f6c4043bc5674` |
| `dsh-plugins/packages/memory-egemma/src/index.ts` | `dsh-plugins/packages/tsuzuki/src/index.ts` | `5edfb9d62e75213b5bb4fdb201d86e3fa54314d68c8796bc09ef1f44132809c6` | `20611795fe27e30b3454cdce088f0eec09866f9cd2739f0166c35e599d925f7d` |
| `dsh-plugins/packages/memory-egemma/src/server.ts` | `dsh-plugins/packages/tsuzuki/src/server.ts` | `a05e80989ff970648a61e9cb309060bce2a214b851db48737f7d803def6f526e` | `8a8882d9b5bd87c67aad538bf6a088afac84fb4b0537855ae892f236596599b2` |
| `dsh-plugins/packages/memory-egemma/test/make_fixtures.py` | `dsh-plugins/packages/tsuzuki/test/make_fixtures.py` | `18bb31583ab27780f0ddae1311e6b3c5c4dfeb1a93f8dee5775a7a977b0e4326` | `7dcde1ec6727951edc8317d5f05f0fc3b7d613fef9f8b671060320f4e14b80e4` |
| `dsh-plugins/packages/memory-egemma/test/server.test.mjs` | `dsh-plugins/packages/tsuzuki/test/server.test.mjs` | `191cebb09dc224a240985ada545c15ee4f2f81682db614e0c96426a2c660d769` | `3869b1d11a5db3929e303d66e6a3205e86dfe9c857453c959a9e986be275689b` |
| `dsh-plugins/packages/text-embeddings-egemma/NOTICE` | `dsh-plugins/packages/text-embeddings-tsuzuki/NOTICE` | `8be5a3d05f729dce1436c4a98e9390537b5fcca875622c96dea766a4f2a4a38e` | `edb434829e590a1dd5561e139f0602e90643461decc7be6ac757b0a50db98f56` |
| `dsh-plugins/packages/text-embeddings-egemma/README.md` | `dsh-plugins/packages/text-embeddings-tsuzuki/README.md` | `cc001489578ef6f678c8df5c8f92f5cd04f23cf1d9b0d4b461dc0bfb6815e637` | `79e847d869a3d652c1c6708c37164d53004aad36817aa059b0552274fa403d75` |
| `dsh-plugins/packages/text-embeddings-egemma/cordis.patch.yml` | `dsh-plugins/packages/text-embeddings-tsuzuki/cordis.patch.yml` | `bee3dd102578389b6f7966e3e6ae09614481dfd6459f8a337f0ef2a50f416e8e` | `081eba7ea83623fb9cd2226c083820129d817439c6a888a9deec7868a1192dc3` |
| `dsh-plugins/packages/text-embeddings-egemma/package.json` | `dsh-plugins/packages/text-embeddings-tsuzuki/package.json` | `52929c7747d454c8bea7239c471d496d79c00686f4891b3bed075ba4f47f97f4` | `5c6b36f75a65b1b2ad773b5e36d02a37b60653f7223a5c8826b6afb628c2f94f` |
| `dsh-plugins/packages/text-embeddings-egemma/src/index.ts` | `dsh-plugins/packages/text-embeddings-tsuzuki/src/index.ts` | `2e3508118fdf74c3d354b8ab28c3d7a26ca822f534eabea020b829a4fd3bc6db` | `039889af097b705b7dc7b88ee619197116067e756d0a3f40b09e47b4c441195a` |
| `dsh-plugins/packages/text-embeddings-openai/NOTICE` | `dsh-plugins/packages/text-embeddings-openai/NOTICE` | `8be5a3d05f729dce1436c4a98e9390537b5fcca875622c96dea766a4f2a4a38e` | `edb434829e590a1dd5561e139f0602e90643461decc7be6ac757b0a50db98f56` |
| `dsh-plugins/packages/text-embeddings/NOTICE` | `dsh-plugins/packages/text-embeddings/NOTICE` | `8be5a3d05f729dce1436c4a98e9390537b5fcca875622c96dea766a4f2a4a38e` | `edb434829e590a1dd5561e139f0602e90643461decc7be6ac757b0a50db98f56` |
| `dsh-plugins/packages/text-embeddings/README.md` | `dsh-plugins/packages/text-embeddings/README.md` | `9a44c787e6cbf7633d7573ea1761df4735e909c14c73d2f08b9a101797e6b781` | `cdda05424fdbb685cabc59ba5658e9907df44ed15e55ad039e1ad8855dced3d9` |
| `dsh-plugins/packages/text-embeddings/src/openai-wire.ts` | `dsh-plugins/packages/text-embeddings/src/openai-wire.ts` | `0b5fb7b30bb34467587019aeb1d9d6c7786244590e52e111094bfc7b2a13530c` | `0aa78b27329c639d697dc9b2833021eccdc24823a37ab7caa2fd83c6694289ca` |
| `dsh-plugins/pnpm-lock.yaml` | `dsh-plugins/pnpm-lock.yaml` | `b376bc1dffe1187e087edd98b405bcf88644ae468e2c28a877338985999c816e` | `ac3155e830fe5f1c292958805293ca932b006d6becd7e977b77f62b2ee4c7f8a` |
| `dsh-plugins/scripts/install-desktop.sh` | `dsh-plugins/scripts/install-desktop.sh` | `21cf8211569dd9efc6a5fb5b3b1b7e4aecd31573d36fb3d3d73c4e9f7565c56c` | `1f78fc82c48186a2d63012a3b62043b38548b405c59558725aa773a02f91591e` |
| `dsh-plugins/tests/dsh-e2e/prime-memory-bench-probe.mjs` | `dsh-plugins/tests/dsh-e2e/prime-memory-bench-probe.mjs` | `52a8f577e9a2111c8ef0268c8834a5bb09ce643cbfc3fdb9a8b300a1979a25f7` | `429979ea2cc9f833fed365aeb8f2849529c628bce3443d4310588f7e14663671` |
| `dsh-plugins/tests/dsh-e2e/run-memory.sh` | `dsh-plugins/tests/dsh-e2e/run-memory.sh` | `2924c50defc214982aa5fbcfb088b3448e9f064cc2aa5f0b98636ce9f8b56a31` | `f923e29d63768c978d21eae96fd24dda1c5cfb3d7edcf7ce042ce214264e09cb` |
| `dsh-plugins/tests/dsh-e2e/run-prime-memory.sh` | `dsh-plugins/tests/dsh-e2e/run-prime-memory.sh` | `8ab571c143d1e532c3fd4feaa95915a85ef32b3ac032fc2e41cc96c96131821c` | `d0b70923b9e52680523fdea003bf6c43e4f5578667f493708f13d3cfb5ab246c` |
| `dsh-plugins/tests/dsh-e2e/run-working-memory.sh` | `dsh-plugins/tests/dsh-e2e/run-working-memory.sh` | `4657751b4e2fa128c0970d52cfd90e2c2394036a10b19356ac13b3198265727b` | `d8d29e1c6df7132ddc4525ed3f1e42e77b61ba996cf234d153f502badda3f153` |
| `dsh-plugins/tests/dsh-e2e/run.sh` | `dsh-plugins/tests/dsh-e2e/run.sh` | `3ed6309173c48a684e6633f43d5690cd62e8676b6295100cb535fb5627bec12d` | `50625501ca45bb05bff0947944c20db198593524d0d16cf4b96f0901a7af9426` |
| `dsh-plugins/tests/plugins.test.mjs` | `dsh-plugins/tests/plugins.test.mjs` | `8f6973944e0ab336b7060b14997c322ce7151276599d73154107a40e24f114fd` | `18439f7f0b0bf454604a6ec04a5aac9bde58b9fe5ef8a9a0684b5b5ba01fd6ce` |
| `integrations/hermes/README.md` | `integrations/hermes/README.md` | `c8e74be18e0300fec72720ddddfdc2056a9d5d36f4547f804889f3ffe7aa4691` | `3f948fd0405c247d847d89aeddb3a4066468c8c7800eb92dcbddf2e01a8ffd55` |
| `integrations/hermes/egemma_memory/__init__.py` | `integrations/hermes/tsuzuki/__init__.py` | `a41252c849a5c87afe33bc6dc86b82e4ed8a94fcf64f96576f1aa21613602af3` | `558ff5bf6c0950e9ef1dac7945bef6df5dc99fb602fa6b7588a71112d19c0a8b` |
| `integrations/hermes/egemma_memory/plugin.yaml` | `integrations/hermes/tsuzuki/plugin.yaml` | `5b36b3eb726f8155deee7b660622c245fe2ae6d3fa779a92e9ce5e3020ae4645` | `652c220a670d1797047a242c23935bf20fd762d160c7bb5fdc3e79802fc5130d` |
| `integrations/hermes/install.py` | `integrations/hermes/install.py` | `60599a7429753da107f6fa1d5ba794499e5cab05379f9761118cb76874592852` | `76595e036b8c561a8d669223d36ff08a531cc3d20179f609e12d27d95f9bc3d4` |
| `integrations/hermes/tests/test_egemma_memory.py` | `integrations/hermes/tests/test_tsuzuki.py` | `b8bcd77c5cb1b0fb6203b6d306188e0d3f18cfb001f78a81b8f1baeb11a0def7` | `af7a77dd1de22f44f18fe6fc0382871edb7b515c89b35ba8908c776881ee2f26` |
| `integrations/hermes/tests/test_install.py` | `integrations/hermes/tests/test_install.py` | `399de064e33e409d891593b4cd1a9e53f6178c39753e0c39914919c151883032` | `4308285e0c76c8aee322f514bf7b5ce009f6eed7472bb28ab7d42251783667fc` |
| `integrations/hermes/tests/test_real_sidecar.py` | `integrations/hermes/tests/test_real_sidecar.py` | `540c198a52c8b7edc420f875c3e6c2aa1ab5b2eb9ef19007b0c7e9a8a3225ff3` | `77de452f660bd117fb4c611cca04d64a00826c6eed3ada1522ed1a3242d9e77c` |
| `server/pyproject.toml` | `server/pyproject.toml` | `345ec5de051791ce2193d310c7bbcb37a1c9cd40a781f3c2f9f9fdcebdb7fda4` | `28d28d54da47b7cec1b6b59513ebbf208fc7f0fafa052810e14723e8ec0a5c5d` |
| `dsh-plugins/packages/memory-egemma/LICENSE` | `dsh-plugins/packages/tsuzuki/LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | same |
| `dsh-plugins/packages/memory-egemma/src/embedding-http.ts` | `dsh-plugins/packages/tsuzuki/src/embedding-http.ts` | `4010a7d9ec254b130a66225ceb27d5d69db00f6ec5b465e367b1457a87ecf572` | same |
| `dsh-plugins/packages/memory-egemma/src/format.ts` | `dsh-plugins/packages/tsuzuki/src/format.ts` | `221f6a8d4ba21574353eb52f2310b88a3e7c6d4d6f1b250ed3666668e8d1b7aa` | same |
| `dsh-plugins/packages/memory-egemma/src/llm.ts` | `dsh-plugins/packages/tsuzuki/src/llm.ts` | `5f457463564b16f089dbdbed43bfaaa844a8c4c0dc4e2a638fa19be1d6861f2f` | same |
| `dsh-plugins/packages/memory-egemma/src/memory.ts` | `dsh-plugins/packages/tsuzuki/src/memory.ts` | `4fe504595bb05f94bd26ecb60afbf8d9f5025f4bd10e32317bfb4d5c0d027d82` | same |
| `dsh-plugins/packages/memory-egemma/src/pipeline.ts` | `dsh-plugins/packages/tsuzuki/src/pipeline.ts` | `400fa03ce30b7f46cfd1d2b59539def4c13ed357b0d76f44e6c6e533cd7f9c1d` | same |
| `dsh-plugins/packages/memory-egemma/src/redact.ts` | `dsh-plugins/packages/tsuzuki/src/redact.ts` | `e79d610733899ba1e08097b9efb8b7a24ba8114612450337fce1f993e87baaa8` | same |
| `dsh-plugins/packages/memory-egemma/src/store.ts` | `dsh-plugins/packages/tsuzuki/src/store.ts` | `c1cc090c2ff8be1c37365e302757a04f33e6a998c0957c287fddc81c3ee1bff9` | same |
| `dsh-plugins/packages/memory-egemma/src/unit-index.ts` | `dsh-plugins/packages/tsuzuki/src/unit-index.ts` | `4e7192270ecb59da9ced2b0af89e35812338bd68bb730ec27ecffa15f5944e4d` | same |
| `dsh-plugins/packages/memory-egemma/test/conformance.test.mjs` | `dsh-plugins/packages/tsuzuki/test/conformance.test.mjs` | `d8c73dffcf511d8f3bf57b1002b0ab19acf169d80d5961fa0c80d5dddf6c3e30` | same |
| `dsh-plugins/packages/memory-egemma/test/embedding-http.test.mjs` | `dsh-plugins/packages/tsuzuki/test/embedding-http.test.mjs` | `285bbd38a23a5bd4c17e967d99ae8c4c907a419c1b4b1112cf6e0c7ace3981c5` | same |
| `dsh-plugins/packages/memory-egemma/test/fixtures/conformance.json` | `dsh-plugins/packages/tsuzuki/test/fixtures/conformance.json` | `07cc9c9c3934d3614d4fee7d7bc36d9a968569dcd5b17f1123612cf3fd66e2dd` | same |
| `dsh-plugins/packages/memory-egemma/test/forget.test.mjs` | `dsh-plugins/packages/tsuzuki/test/forget.test.mjs` | `2b4ae899ef9564136934215c6f368c8d94f043c206c7de116e2abca412c12791` | same |
| `dsh-plugins/packages/memory-egemma/test/memory.test.mjs` | `dsh-plugins/packages/tsuzuki/test/memory.test.mjs` | `b110cf1b027b807e792fcd7eb55a0c241910dc58a4223eab1df98b59905a0a0c` | same |
| `dsh-plugins/packages/memory-egemma/test/plugin.test.mjs` | `dsh-plugins/packages/tsuzuki/test/plugin.test.mjs` | `00c1606ba8a3eecad5f2c6376f50b921e303ea51b81d236a8025acc13f264748` | same |
| `dsh-plugins/packages/memory-egemma/test/redact.test.mjs` | `dsh-plugins/packages/tsuzuki/test/redact.test.mjs` | `2f15ebc431e7cfc7ed1dcaee3cb977996bf089418f4e8d3bb48d81fe41f0fd8d` | same |
| `dsh-plugins/packages/memory-egemma/test/retrieve-equivalence.test.mjs` | `dsh-plugins/packages/tsuzuki/test/retrieve-equivalence.test.mjs` | `2f0c8d8c9ea4cb922f907b4acfb4f3e288fb11c69bb6326aa2496f44ff754578` | same |
| `dsh-plugins/packages/memory-egemma/test/scale.test.mjs` | `dsh-plugins/packages/tsuzuki/test/scale.test.mjs` | `ba2daa02b569171015404d8e27b1e4de734d235dc596dc06c9cd26ad94f3d90c` | same |
| `dsh-plugins/packages/memory-egemma/tsconfig.json` | `dsh-plugins/packages/tsuzuki/tsconfig.json` | `9f3adcaccd0b0473918adc05d945f805d02a560e442ffe6ea98299fa3a4d47a3` | same |
| `dsh-plugins/packages/text-embeddings-egemma/LICENSE` | `dsh-plugins/packages/text-embeddings-tsuzuki/LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | same |
| `dsh-plugins/packages/text-embeddings-egemma/src/supervisor.ts` | `dsh-plugins/packages/text-embeddings-tsuzuki/src/supervisor.ts` | `dbbc8f7cb2859cd55bf3621e13a648b4db06b21345629ea5a6fc45ca856d30f7` | same |
| `dsh-plugins/packages/text-embeddings-egemma/tsconfig.json` | `dsh-plugins/packages/text-embeddings-tsuzuki/tsconfig.json` | `9f3adcaccd0b0473918adc05d945f805d02a560e442ffe6ea98299fa3a4d47a3` | same |

## Changed after publication

Windows fixes, made the same way in the development repository (branch `dsh-plugin`, commit `8520035`), with tests: without `--config`, the sidecar finds its config with `os.homedir()` instead of `HOME`, and `node lib/server.js` starts it on Windows too; the Hermes provider passes the Windows variables Node needs (`SYSTEMROOT`, `USERPROFILE`, `TEMP`, `TMP`) and starts the sidecar where `fcntl` is missing. These four files changed, so their "sha256 now" above includes the fixes; they still differ from the development files only as described under "Renamed to tsuzuki".

| File | sha256 at publication | Development file at `8520035` | its sha256 |
|---|---|---|---|
| `dsh-plugins/packages/tsuzuki/src/server.ts` | `55ae227143a1259f0d84f180ebeda0baced7a9ded418729ac4aaecf1ff493f4a` | `dsh-plugins/packages/memory-egemma/src/server.ts` | `350144105acea8e134467b6deaa0bd19358a34f78eb8cc630259dcb2b8612884` |
| `dsh-plugins/packages/tsuzuki/test/server.test.mjs` | `fbe9b59c35971f33a967000eb8660d3f579a4b97f2f49d8f05711b4a1b1348bd` | `dsh-plugins/packages/memory-egemma/test/server.test.mjs` | `970902b232410e631a8f12a21fa59830aeade771b87060f086cb30e54fa81f51` |
| `integrations/hermes/tsuzuki/__init__.py` | `fb5df681aed532e3e0897d159c1f8873e74dd9531da35f34e77858352a2fc2f2` | `integrations/hermes/egemma_memory/__init__.py` | `12aa80adfd296a68454c3ebc9622296ddf268ef67f7c203aa26703c03e31b2aa` |
| `integrations/hermes/tests/test_tsuzuki.py` | `e7582bc469eb6b11c9278559e4cdbbd45733f204fb3840b8d28e9a6a03b80a72` | `integrations/hermes/tests/test_egemma_memory.py` | `749a6f724108a2473caca85c80106738f5ce1e5f41943b7bcdd3c4b849002191` |

## Left out

- Personal and release-process notes (`docs/release/`), hand-over notes and review requests.
- The stopped coding experiment C1: `scripts/eval/coding_c1/`, `results/eval/coding_c1/`.
- All other results (benchmark logs, dsh end-to-end logs, run records), `data/`, and the engine optimization history docs.
- Untracked files, `build/`, `models/`, `.venv/`, `node_modules/`, `lib/`, `dist/`.
