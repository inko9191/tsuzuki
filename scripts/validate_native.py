#!/usr/bin/env python3
"""Phase 2 acceptance: every native MRL dimension vs the phase 1 fp32 oracle."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(ROOT / "native"))
from egemma_native import NativePipeline  # noqa: E402
from egemma_ref.hf_compare import compare, format_rows, ref_captures  # noqa: E402
from egemma_ref.model import truncate_mrl  # noqa: E402
from egemma_ref.pipeline import Pipeline  # noqa: E402
from validate_vs_st import DIMS, PROMPT_CASES, load_texts  # noqa: E402


def cosine(a: np.ndarray, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(torch.from_numpy(a), b.float().cpu(), dim=-1)


def debug_layers(engine, ref, ids, mask, build_dir: Path):
    with tempfile.TemporaryDirectory(prefix="native-captures-", dir=build_dir) as directory:
        engine.encode(ids, mask, 768, directory)
        shapes = json.loads((Path(directory) / "captures.json").read_text())
        captures = {
            name: torch.from_numpy(np.fromfile(Path(directory) / (name + ".f32"), dtype=np.float32).reshape(shape))
            for name, shape in shapes.items()
        }
        golden = ref_captures(ref.model, torch.from_numpy(ids).long(), torch.from_numpy(mask).long())
        rows = compare(golden, captures, torch.from_numpy(mask))
        assert len(rows) == 4 + 3 * ref.cfg.num_hidden_layers
        return rows


def interface_checks(native, ref, build_dir: Path, threshold: float) -> dict:
    prompt = ref.prompts["query"]
    texts = ["", " ", "<bos>", "<bos>test", "日本語と English の混在"]
    for p in ("", prompt):
        assert all(native.tokenize_one(t, p) == ref.tokenize_one(t, p) for t in texts)
    seqs = [native.tokenize_one(t, prompt) for t in texts]
    ids, mask = native.batch(seqs)
    right = native.engine.encode(ids, mask)
    left_ids, left_mask = native.batch(seqs, "left")
    left = native.engine.encode(left_ids, left_mask)
    alone = np.concatenate([native.engine.encode(*native.batch([s])) for s in seqs])
    padding_min = min(float(cosine(left, torch.from_numpy(right)).min()), float(cosine(alone, torch.from_numpy(right)).min()))
    assert padding_min >= threshold
    # An all-masked row gives finite zero pooled output; mask holes still use
    # padded positions in RoPE/window distances, matching the oracle.
    hole_mask = mask.copy()
    hole_mask[0] = 0
    hole_mask[1, 1::2] = 0
    hole = native.engine.encode(ids, hole_mask)
    golden = ref.embed_ids(torch.from_numpy(ids).long(), torch.from_numpy(hole_mask).long())
    assert np.isfinite(hole).all() and np.array_equal(hole[0], np.zeros(768, dtype=np.float32))
    mask_min = float(cosine(hole[1:], golden[1:]).min())
    assert mask_min >= threshold
    errors = 0
    bad_cases = [(ids, mask[:, :-1], 768), (ids.astype(np.int64), mask, 768),
                 (np.full_like(ids, -1), mask, 768), (ids, np.full_like(mask, 2), 768),
                 (ids, mask, 769), (ids[:0], mask[:0], 768), (ids, mask, 0)]
    for x, m, d in bad_cases:
        try:
            native.engine.encode(x, m, d)
        except (RuntimeError, ValueError, TypeError):
            errors += 1
    assert errors == len(bad_cases)
    records = [seqs[0], {"input_ids": seqs[1:]}, {"input_ids": ids.tolist(), "attention_mask": hole_mask.tolist()}]
    process = subprocess.run([str(build_dir / "egemma-cli"), "--model-dir", str(native.model_dir),
                              "--dtype", json.loads(native.engine.info())["dtype"], "--dim", "512"],
                             input="".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                             text=True, capture_output=True, check=True)
    outputs = np.array([json.loads(line) for line in process.stdout.splitlines()], dtype=np.float32)
    expected = np.concatenate([native.engine.encode(*native.batch([seqs[0]]), 512),
                               native.engine.encode(*native.batch(seqs[1:]), 512),
                               native.engine.encode(ids, hole_mask, 512)])
    cli_max_abs = float(np.abs(outputs - expected).max())
    assert cli_max_abs == 0
    return {"special_token_probes": 10, "padding_min_cos": padding_min,
            "mask_holes_min_cos": mask_min, "all_masked_row_finite_zero": True,
            "invalid_inputs_rejected": errors, "cli_binding_max_abs": cli_max_abs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--build-dir", type=Path, default=ROOT / "build")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "results/phase2_validation.json")
    args = ap.parse_args()
    assert args.batch_size > 0
    ref = Pipeline(args.model_dir, device="mps", dtype=torch.float32, verbose=False)
    texts = load_texts(ROOT / "data/texts_ja_en.jsonl", True)
    assert len(texts) == 63
    report = {"env": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__},
              "oracle": "phase1 reference, MPS/fp32", "n_texts": len(texts), "batch_size": args.batch_size,
              "dtypes": {}, "pass": True}
    for dtype, threshold in (("fp32", 0.9999), ("fp16", 0.999)):
        native = NativePipeline(args.model_dir, dtype, args.build_dir)
        result = {"threshold": threshold, "engine": json.loads(native.engine.info()), "cases": {}}
        aggregate = {d: [] for d in DIMS}
        for case in PROMPT_CASES:
            prompt = "" if case == "<none>" else ref.prompts[case]
            values = {d: [] for d in DIMS}
            lengths = []
            mismatches = []
            started = time.perf_counter()
            first_failure = None
            for start in range(0, len(texts), args.batch_size):
                chunk = texts[start:start + args.batch_size]
                seqs = [native.tokenize_one(t, prompt) for t in chunk]
                lengths.extend(map(len, seqs))
                mismatches.extend(start + i for i, t in enumerate(chunk) if seqs[i] != ref.tokenize_one(t, prompt))
                ids, mask = native.batch(seqs)
                golden = ref.embed_ids(torch.from_numpy(ids).long(), torch.from_numpy(mask).long())
                for dim in DIMS:
                    out = native.engine.encode(ids, mask, dim)
                    c = cosine(out, truncate_mrl(golden, dim))
                    values[dim].append(c)
                    if float(c.min()) < threshold and first_failure is None:
                        first_failure = debug_layers(native.engine, ref, ids, mask, args.build_dir)
            stats = {}
            for dim in DIMS:
                c = torch.cat(values[dim])
                stats[dim] = {"mean": float(c.mean()), "min": float(c.min())}
                aggregate[dim].append(c)
            ok = not mismatches and all(v["min"] >= threshold for v in stats.values())
            result["cases"][case] = {"prompt": prompt, "cosine": stats, "token_mismatches": mismatches,
                                     "token_length_min": min(lengths), "token_length_max": max(lengths),
                                     "long_text_lengths": lengths[-3:], "seconds": time.perf_counter() - started, "pass": ok}
            assert min(lengths[-3:]) > 512
            if first_failure is not None:
                result["cases"][case]["debug_layers"] = first_failure
                print(format_rows(first_failure), flush=True)
            print(f"[{dtype}/{case}] {'PASS' if ok else 'FAIL'} " + " ".join(
                f"{d}:mean={s['mean']:.9f},min={s['min']:.9f}" for d, s in stats.items()), flush=True)
            report["pass"] &= ok
        result["aggregate"] = {
            d: {"mean": float(torch.cat(v).mean()), "min": float(torch.cat(v).min())} for d, v in aggregate.items()
        }
        result["interface"] = interface_checks(native, ref, args.build_dir, threshold)
        ids, mask = native.batch([native.tokenize_one(t, ref.prompts["query"]) for t in texts[:2]])
        result["debug_layers"] = debug_layers(native.engine, ref, ids, mask, args.build_dir)
        bad = next((r for r in result["debug_layers"] if r["min_cos"] < threshold), None)
        result["first_diverging_point"] = None if bad is None else bad["name"]
        print(f"[{dtype}] interface checks PASS; first diverging capture: {result['first_diverging_point']}", flush=True)
        report["dtypes"][dtype] = result
        del native
        gc.collect()
        torch.mps.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("overall:", "PASS" if report["pass"] else "FAIL", flush=True)
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
