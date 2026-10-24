#!/usr/bin/env python3
"""Acceptance test for EmbeddingGemma 300m (v1) in the native engine and egemma-server.

Oracle: sentence-transformers (google/embeddinggemma-300m@57c266a, fp32, MPS): transformers' Gemma 3 is the
reference implementation here, so no separate reference model is written (unlike v2, docs/V1_SUPPORT_ESTIMATE.md).
Criterion as for v2: every vector, minimum cosine >= THRESHOLDS[dtype] (fp32 0.9999), for 6 prompts x 63 texts
(3 longer than the 512-token sliding window) x 768/512/256/128.
Checks:
  tokens      PromptTokenizer (tokenizers only) == ST tokenization for every input plus probes
  engine      in-process engine vs ST (batches of 8, right padding)
  daemon      egemma-server over HTTP (--model-name embeddinggemma-300m, task field) vs ST
  concurrent  16 threads with mixed tasks, dimensions and lengths through the daemon's dynamic batching
  layers      (fp32 only) every layer output for the 3 long texts vs transformers' hidden_states:
              max |diff| / max |reference| <= LAYER_TOLERANCE. Catches what the vectors barely show
              (a sliding-window radius off by one moves the vectors by only ~2e-5 in cosine)
  mutations   each Gemma 3 rule disabled in turn (EGEMMA_TEST_MUTATION) must fail the vector or layer criterion:
              norm_plus_one (x*w instead of x*(1+w)), attn_scale (1 instead of 1/16), window (radius W//2+1 instead
              of W//2), v_norm (normalizing V), dense (skipping the Dense modules)
Output: results/v1_validation.json (+ .txt summary). Exit 1 on failure.
Usage: .venv/bin/python scripts/validate_v1.py [--dtype fp32] [--port 8393] [--skip-mutations]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "build"))

MODEL = ROOT / "models/embeddinggemma-300m"
DIMS = (768, 512, 256, 128)
PROMPT_CASES = ("<none>", "query", "document", "STS", "Classification", "Clustering")
THRESHOLDS = {"fp32": 0.9999, "bf16": 0.999, "fp16": 0.999}
MUTATIONS = ("norm_plus_one", "attn_scale", "window", "v_norm", "dense")
LAYER_TOLERANCE = 1e-3  # fp32: the correct engine stays below 6e-5 at every layer


def load_texts() -> list[str]:
    rows = [json.loads(line) for line in (ROOT / "data/texts_ja_en.jsonl").read_text().splitlines() if line.strip()]
    texts = [r["text"] for r in rows]
    en = " ".join(r["text"] for r in rows if r["lang"] == "en")
    ja = "".join(r["text"] for r in rows if r["lang"] == "ja")
    return texts + [(en + " ") * 2, ja * 3, en + ja]  # as scripts/validate_vs_st.py: past the 512-token window


def truncate(v: np.ndarray, d: int) -> np.ndarray:
    v = v[:, :d].astype(np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def min_cos(a: np.ndarray, b: np.ndarray) -> dict:
    return {d: float((truncate(a, d) * truncate(b, d)).sum(1).min()) for d in DIMS}


def layer_error(engine, tokenizer, hf, texts: list[str]) -> dict:
    """Worst relative error of each layer output vs transformers' hidden_states (batch 1, no prompt)."""
    import tempfile
    import torch
    worst: dict[int, float] = {}
    for text in texts:
        ids = np.asarray([tokenizer.encode([text], None)[0]], np.int32)
        mask = np.ones_like(ids)
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            engine.encode(ids, mask, 768, directory)
            states = hf(input_ids=torch.tensor(ids, dtype=torch.long, device="mps"),
                        attention_mask=torch.tensor(mask, device="mps"), output_hidden_states=True).hidden_states
            # hidden_states[i+1] is the output of layer i, except the last entry (after the final norm).
            for i in range(len(states) - 2):
                got = np.fromfile(f"{directory}/layer{i}.out.f32", np.float32).reshape(ids.shape[1], -1)
                ref = states[i + 1][0].float().cpu().numpy()
                worst[i] = max(worst.get(i, 0.0), float(np.abs(got - ref).max() / np.abs(ref).max()))
    return worst


def engine_vectors(engine, tokenizer, texts: list[str], prompt: str | None) -> np.ndarray:
    out = []
    for start in range(0, len(texts), 8):
        seqs = tokenizer.encode(texts[start:start + 8], prompt)
        width = max(map(len, seqs))
        ids = np.zeros((len(seqs), width), np.int32)
        mask = np.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = s
            mask[i, :len(s)] = 1
        out.append(np.asarray(engine.encode(ids, mask, 768)))
    return np.concatenate(out)


class Daemon:
    def __init__(self, port: int, dtype: str):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "server"))
        env.pop("EGEMMA_TEST_MUTATION", None)
        self.port = port
        self.proc = subprocess.Popen([sys.executable, "-m", "egemma_server", "--port", str(port), "--dtype", dtype,
                                      "--model-dir", str(MODEL), "--model-name", "embeddinggemma-300m"],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        line = self.proc.stdout.readline()
        event = json.loads(line) if line else {"event": "failed", "error": self.proc.stderr.read()}
        if event["event"] != "ready":
            raise RuntimeError(f"daemon failed: {event}")
        self.ready = event

    def embed(self, texts, task=None, dims=None):
        body = {"model": "embeddinggemma-300m", "input": texts}
        if task is not None:
            body["task"] = task
        if dims is not None:
            body["dimensions"] = dims
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/embeddings", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
        return np.asarray([d["embedding"] for d in sorted(out["data"], key=lambda d: d["index"])], np.float32)

    def stats(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/stats") as r:
            return json.loads(r.read())

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=sorted(THRESHOLDS))
    ap.add_argument("--matmul", default="auto")
    ap.add_argument("--port", type=int, default=8393)
    ap.add_argument("--skip-mutations", action="store_true")
    ap.add_argument("--skip-daemon", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out_path = args.out or ROOT / f"results/v1_validation_{args.dtype}.json"
    threshold = THRESHOLDS[args.dtype]

    import _egemma_native as native
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    from egemma_server.tokenizer import PromptTokenizer

    texts = load_texts()
    tokenizer = PromptTokenizer(MODEL)
    st = SentenceTransformer(str(MODEL), device="mps").float()
    report = {"model": "google/embeddinggemma-300m@57c266a", "oracle": f"sentence-transformers {sentence_transformers.__version__} fp32 MPS",
              "dtype": args.dtype, "threshold": threshold, "texts": len(texts), "prompts": list(PROMPT_CASES), "pass": True}

    # -- tokens ----------------------------------------------------------------
    probes = ["", " ", "<bos>", "<bos>hello", "hello<eos>", "  leading", "改行\nあり"]
    mismatches, checked, longest = [], 0, 0
    for case in PROMPT_CASES:
        prompt = None if case == "<none>" else tokenizer.prompt_for(case)
        for t in texts + probes:
            ours = tokenizer.encode([t], prompt)[0]
            feats = st.tokenize([(prompt or "") + t])
            ref = [i for i, m in zip(feats["input_ids"][0].tolist(), feats["attention_mask"][0].tolist()) if m]
            longest = max(longest, len(ours))
            checked += 1
            if ours != ref:
                mismatches.append({"case": case, "text": t[:40]})
    report["tokens"] = {"checked": checked, "mismatches": mismatches, "longest_input_tokens": longest}
    report["pass"] &= not mismatches and longest <= 2048
    print(f"[tokens] {checked} checked, {len(mismatches)} mismatches, longest {longest} tokens", flush=True)

    # -- golden ST vectors ---------------------------------------------------------
    golden = {}
    for case in PROMPT_CASES:
        prompt = None if case == "<none>" else tokenizer.prompt_for(case)
        golden[case] = st.encode([(prompt or "") + t for t in texts], batch_size=8, normalize_embeddings=True)

    # -- engine --------------------------------------------------------------------
    engine = native.Engine(str(MODEL), args.dtype, False, args.matmul)
    info = json.loads(engine.info())
    report["engine_info"] = {k: info.get(k) for k in ("model_family", "dtype", "matmul_backend", "folded_dense_modules", "text_tensors")}
    worst = {}
    for case in PROMPT_CASES:
        prompt = None if case == "<none>" else tokenizer.prompt_for(case)
        worst[case] = min_cos(engine_vectors(engine, tokenizer, texts, prompt), golden[case])
    engine_min = min(v for c in worst.values() for v in c.values())
    report["engine"] = {"min_cos": engine_min, "per_prompt": worst, "pass": engine_min >= threshold}
    report["pass"] &= report["engine"]["pass"]
    print(f"[engine {args.dtype}] min cos {engine_min:.9f} ({'PASS' if engine_min >= threshold else 'FAIL'})", flush=True)
    long_texts = texts[-3:]
    if args.dtype == "fp32":
        layers = layer_error(engine, tokenizer, st[0].model, long_texts)
        report["layers"] = {"max_relative_error": max(layers.values()), "per_layer": layers, "tolerance": LAYER_TOLERANCE,
                            "texts_tokens": [len(tokenizer.encode([t], None)[0]) for t in long_texts],
                            "pass": max(layers.values()) <= LAYER_TOLERANCE}
        report["pass"] &= report["layers"]["pass"]
        print(f"[layers] max relative error {max(layers.values()):.2e} (tolerance {LAYER_TOLERANCE:.0e})", flush=True)
    del engine

    # -- daemon ----------------------------------------------------------------------
    if not args.skip_daemon:
        daemon = Daemon(args.port, args.dtype)
        try:
            dworst = {}
            for case in PROMPT_CASES:
                task = "none" if case == "<none>" else case
                dworst[case] = {}
                for d in DIMS:
                    got = np.concatenate([daemon.embed(texts[s:s + 16], task, d) for s in range(0, len(texts), 16)])
                    dworst[case][d] = float((got.astype(np.float64) * truncate(golden[case], d)).sum(1).min())
            daemon_min = min(v for c in dworst.values() for v in c.values())
            rng = random.Random(0)
            jobs = [(rng.choice(PROMPT_CASES), rng.choice(DIMS), rng.sample(range(len(texts)), rng.randint(1, 6))) for _ in range(64)]

            def run(job):
                case, d, idx = job
                got = daemon.embed([texts[i] for i in idx], "none" if case == "<none>" else case, d)
                return float((got.astype(np.float64) * truncate(golden[case][idx], d)).sum(1).min())
            with ThreadPoolExecutor(16) as pool:
                concurrent = min(pool.map(run, jobs))
            stats = daemon.stats()
            ok = daemon_min >= threshold and concurrent >= threshold and stats.get("errors", 0) == 0
            report["daemon"] = {"min_cos": daemon_min, "per_prompt": dworst, "concurrent_min_cos": concurrent,
                                "requests": stats.get("requests"), "errors": stats.get("errors"),
                                "cold_start": daemon.ready.get("cold_start"), "engine": daemon.ready.get("engine"), "pass": ok}
            report["pass"] &= ok
            print(f"[daemon] min cos {daemon_min:.9f}, concurrent {concurrent:.9f}, errors {stats.get('errors')} "
                  f"({'PASS' if ok else 'FAIL'})", flush=True)
        finally:
            daemon.close()

    # -- mutations: each disabled rule must break the criterion -------------------------
    if not args.skip_mutations:
        report["mutations"] = {}
        for mutation in MUTATIONS:
            os.environ["EGEMMA_TEST_MUTATION"] = mutation
            try:
                engine = native.Engine(str(MODEL), args.dtype, False, args.matmul)
                assert json.loads(engine.info()).get("test_mutation") == mutation
                m = min(min(min_cos(engine_vectors(engine, tokenizer, texts, None if c == "<none>" else tokenizer.prompt_for(c)),
                                    golden[c]).values()) for c in ("<none>", "query"))
                layer = max(layer_error(engine, tokenizer, st[0].model, long_texts).values()) if args.dtype == "fp32" else None
                del engine
            finally:
                os.environ.pop("EGEMMA_TEST_MUTATION", None)
            detected = m < threshold or (layer is not None and layer > LAYER_TOLERANCE)
            report["mutations"][mutation] = {"min_cos": m, "max_layer_relative_error": layer, "detected": detected}
            report["pass"] &= detected
            print(f"[mutation {mutation}] min cos {m:.6f}, layer error {layer if layer is None else f'{layer:.2e}'} "
                  f"({'detected' if detected else 'NOT detected'})", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("overall:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
