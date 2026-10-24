#!/usr/bin/env python3
"""Daemon acceptance test: egemma-server over HTTP vs reference, engine, and Sentence Transformers.

Same criterion as the engine gate (every vector, minimum cosine): fp32 >= 0.9999,
fp16 >= 0.999, for 6 prompts x 63 texts (3 longer than 512 tokens) x 768/512/256/128.
Checks:
  tokens      daemon tokenizer (tokenizers only) == reference and transformers-path IDs,
              for all 378 inputs plus empty/whitespace/BOS probes
  reference   daemon vectors vs phase 1 reference (MPS/fp32, the engine gate's oracle)
  engine      daemon vectors vs the in-process engine (same dtype and matmul mode)
  st          daemon vectors vs SentenceTransformer.encode (fp32, MPS)
  concurrent  16 threads with mixed tasks, dimensions and lengths through dynamic batching
Daemon configurations: fp16 with matmul auto (MPP on M5), fp16 native-only, fp32.
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "native"))
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "scripts"))

from egemma_native import NativePipeline  # noqa: E402
from egemma_ref.model import truncate_mrl  # noqa: E402
from egemma_ref.pipeline import Pipeline  # noqa: E402
from egemma_server.tokenizer import PromptTokenizer  # noqa: E402
from validate_vs_st import DIMS, PROMPT_CASES, load_texts  # noqa: E402

MODEL = ROOT / "models/embeddinggemma-2"
CONFIGS = [("fp16-auto", "fp16", "auto", .999), ("fp16-native", "fp16", "native", .999), ("fp32", "fp32", "auto", .9999)]


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64); b = b.astype(np.float64)
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


class Daemon:
    def __init__(self, port: int, dtype: str, matmul: str):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "server"))
        self.port = port
        self.proc = subprocess.Popen([sys.executable, "-m", "egemma_server", "--port", str(port), "--dtype", dtype,
                                      "--matmul", matmul, "--model-dir", str(MODEL)],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        line = self.proc.stdout.readline()
        event = json.loads(line) if line else {"event": "failed", "error": self.proc.stderr.read()}
        if event["event"] != "ready":
            raise RuntimeError(f"daemon failed: {event}")
        self.ready = event

    def embed(self, texts, task=None, dims=None, model=None):
        body = {"input": texts}
        if task is not None:
            body["task"] = task
        if dims is not None:
            body["dimensions"] = dims
        if model is not None:
            body["model"] = model
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/dsh/server_validation.json")
    ap.add_argument("--port", type=int, default=8391)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    texts = load_texts(ROOT / "data/texts_ja_en.jsonl", True)
    assert len(texts) == 63
    ref = Pipeline(str(MODEL), device="mps", dtype=torch.float32, verbose=False)
    tok = PromptTokenizer(MODEL)
    report = {"oracle": "phase 1 reference MPS/fp32", "texts": len(texts), "pass": True, "configs": {}}

    # -- tokens --------------------------------------------------------------
    hf = NativePipeline(MODEL, "fp32")
    mismatches, checked = [], 0
    probes = ["", " ", "<bos>", "<bos>hello", "hello<eos>", "  leading", "改行\nあり", "<|image|> text"]
    for case in PROMPT_CASES:
        prompt = None if case == "<none>" else ref.prompts[case]
        for t in texts + probes:
            ours = tok.encode([t], prompt)[0]
            if ours != ref.tokenize_one(t, prompt) or ours != hf.tokenize_one(t, prompt):
                mismatches.append({"case": case, "text": t[:40]})
            checked += 1
    del hf
    report["tokens"] = {"checked": checked, "mismatches": mismatches}
    report["pass"] &= not mismatches
    print(f"[tokens] {checked} checked, {len(mismatches)} mismatches", flush=True)

    # -- golden: reference and ST ---------------------------------------------
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(str(MODEL), device="mps", model_kwargs={"dtype": torch.float32},
                             config_kwargs={"vision_config": None, "audio_config": None})
    golden, st_out = {}, {}
    for case in PROMPT_CASES:
        prompt = None if case == "<none>" else ref.prompts[case]
        rows = []
        for start in range(0, len(texts), 8):
            seqs = [ref.tokenize_one(t, prompt) for t in texts[start:start + 8]]
            width = max(map(len, seqs))
            ids = torch.zeros(len(seqs), width, dtype=torch.long); mask = torch.zeros_like(ids)
            for i, s in enumerate(seqs):
                ids[i, :len(s)] = torch.tensor(s); mask[i, :len(s)] = 1
            rows.append(ref.embed_ids(ids, mask).cpu())
        golden[case] = torch.cat(rows)
        st_out[case] = st.encode(texts, prompt=prompt, batch_size=8, convert_to_tensor=True).float().cpu()
    del st
    torch.mps.empty_cache()

    for name, dtype, matmul, threshold in CONFIGS:
        daemon = Daemon(args.port, dtype, matmul)
        try:
            run_config(report, name, daemon, dtype, matmul, threshold, texts, ref, golden, st_out)
        finally:
            daemon.close()
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("overall:", "PASS" if report["pass"] else "FAIL")
    sys.exit(0 if report["pass"] else 1)


def run_config(report, name, daemon, dtype, matmul, threshold, texts, ref, golden, st_out):
    engine = NativePipeline(MODEL, dtype, ROOT / "build")
    engine.engine = __import__("egemma_native").load_extension().Engine(str(MODEL), dtype, False, matmul)
    result = {"threshold": threshold, "engine": daemon.ready["engine"], "cold_start": daemon.ready["cold_start"], "cases": {}}
    worst = {"reference": 1.0, "engine": 1.0, "st": 1.0}
    for case in PROMPT_CASES:
        task = "none" if case == "<none>" else case
        prompt = None if case == "<none>" else ref.prompts[case]
        per_dim = {}
        for dim in DIMS:
            got = np.concatenate([daemon.embed(texts[s:s + 8], task, dim) for s in range(0, len(texts), 8)])
            local = engine.encode(texts, prompt=prompt, batch_size=8, dim=dim)
            c = {"reference": cos(got, truncate_mrl(golden[case], dim).numpy()),
                 "engine": cos(got, local),
                 "st": cos(got, truncate_mrl(st_out[case], dim).numpy())}
            per_dim[dim] = {k: {"mean": float(v.mean()), "min": float(v.min())} for k, v in c.items()}
            for k, v in c.items():
                worst[k] = min(worst[k], float(v.min()))
        result["cases"][case] = per_dim
    # alias form used by clients that only send {model, input}
    alias = daemon.embed(texts[:8], model="embeddinggemma-2:query:256")
    explicit = daemon.embed(texts[:8], "query", 256)
    result["alias_equals_explicit"] = bool(np.array_equal(alias, explicit))
    # concurrency: mixed tasks/dims/lengths batched together
    rng = random.Random(0)
    jobs = [(rng.choice(PROMPT_CASES), rng.choice(DIMS), rng.sample(range(len(texts)), rng.randint(1, 6))) for _ in range(48)]
    def run(job):
        case, dim, idx = job
        got = daemon.embed([texts[i] for i in idx], "none" if case == "<none>" else case, dim)
        return float(cos(got, truncate_mrl(golden[case][idx], dim).numpy()).min())
    with ThreadPoolExecutor(16) as pool:
        concurrent_min = min(pool.map(run, jobs))
    stats = daemon.stats()
    result["concurrent"] = {"requests": len(jobs), "min_cos_vs_reference": concurrent_min,
                            "batches": stats["batches"], "latency_ms": stats["latency_ms"]}
    result["worst"] = worst
    ok = all(v >= threshold for v in worst.values()) and concurrent_min >= threshold and result["alias_equals_explicit"]
    result["pass"] = ok
    report["pass"] &= ok
    report["configs"][name] = result
    print(f"[{name}] {'PASS' if ok else 'FAIL'} worst={worst} concurrent={concurrent_min:.9f} "
          f"batches={stats['batches']['count']} mean_rows={stats['batches']['mean_rows']:.2f}", flush=True)
    del engine


if __name__ == "__main__":
    main()
