# README images

- `demo-en.gif`, `demo-ja.gif`: drawn by `make_demo.py` from `transcript-en.json` / `transcript-ja.json`, which `capture.py` recorded: Gemma 4 E2B (`gemma-4-E2B_q4_0-it.gguf`) on llama-server b10964 at temperature 0, once through tsuzuki's proxy (a new chat recalls the first) and once directly (the same question without tsuzuki). The message texts in the images are those outputs, unchanged.
- `results-*-light.png`, `results-*-dark.png`: drawn by `make_chart.py` from the accuracies in the README's result tables (E1, J1, U2-AT, J1-Q; primary judge).

The scripts use macOS's Hiragino Sans font and Pillow / matplotlib.
