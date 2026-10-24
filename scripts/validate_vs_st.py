#!/usr/bin/env python3
"""Phase 1 acceptance test: reference implementation vs Sentence Transformers on the real checkpoint.

Three separate checks per prompt and MRL dimension (768/512/256/128):
  tokens   : our tokenisation == ST's preprocess() token ids
  forward  : our backbone+pooling on ST's own token ids vs ST.encode()   (isolates the math)
  e2e      : our full pipeline (own tokenisation) vs ST.encode()
Pass criterion (project spec): mean cosine >= 0.9999 for forward and e2e at every dimension.

Usage:
  python scripts/validate_vs_st.py --model-dir models/embeddinggemma-2 [--device mps] [--input-format chat]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))

from egemma_ref.model import truncate_mrl  # noqa: E402
from egemma_ref.pipeline import Pipeline, pick_device  # noqa: E402

DIMS = (768, 512, 256, 128)
PROMPT_CASES = ("<none>", "query", "document", "STS", "Classification", "Clustering")


def load_texts(path: Path, with_long: bool) -> list[str]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    texts = [r["text"] for r in rows]
    if with_long:
        en = " ".join(r["text"] for r in rows if r["lang"] == "en")
        ja = "".join(r["text"] for r in rows if r["lang"] == "ja")
        # several hundred to ~1500 tokens: longer than the 512 sliding-window radius
        texts += [(en + " ") * 2, ja * 3, en + ja]
    return texts


def cos_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    out = {}
    for d in DIMS:
        if d > a.shape[-1]:
            continue
        c = (truncate_mrl(a, d) * truncate_mrl(b, d)).sum(-1)
        out[d] = {"mean": float(c.mean()), "min": float(c.min())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--texts", default=str(ROOT / "data/texts_ja_en.jsonl"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--input-format", default="auto", choices=["auto", "plain", "chat"])
    ap.add_argument("--no-long", action="store_true", help="skip the >512-token texts")
    ap.add_argument("--out", default=str(ROOT / "results/phase1_validation.json"))
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    device = pick_device(args.device)
    texts = load_texts(Path(args.texts), not args.no_long)
    print(f"[validate] {len(texts)} texts, device={device}, torch={torch.__version__}")

    st = SentenceTransformer(
        args.model_dir, device=device,
        model_kwargs={"dtype": torch.float32},
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    print("[validate] Sentence Transformers pipeline:\n" + str(st))
    st_modules = [type(m).__name__ for m in st]

    ref = Pipeline(args.model_dir, device=device, dtype=torch.float32, input_format=args.input_format)
    print("[validate] reference config:\n" + ref.cfg.summary())

    report = {
        "env": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
                "device": device},
        "st_modules": st_modules,
        "ref_post": [n for n, _ in ref.post],
        "input_format": ref.input_format,
        "weights": ref.weight_info,
        "n_texts": len(texts),
        "cases": {},
    }
    all_ok = True
    for case in PROMPT_CASES:
        if case == "<none>":
            prompt = ""
        elif case in st.prompts:
            prompt = st.prompts[case]
        else:
            print(f"[validate] prompt {case!r} not in checkpoint prompts; skipped")
            continue
        t0 = time.time()
        st_emb = st.encode(texts, prompt=prompt, batch_size=args.batch_size, convert_to_tensor=True).float().cpu()

        tok_mismatch, fwd = [], []
        for s in range(0, len(texts), args.batch_size):
            chunk = texts[s : s + args.batch_size]
            feats = st.preprocess(chunk, prompt=prompt or None)
            ids, mask = feats["input_ids"].cpu(), feats["attention_mask"].cpu()
            for j, t in enumerate(chunk):
                st_ids = ids[j][mask[j].bool()].tolist()
                if ref.tokenize_one(t, prompt or None) != st_ids:
                    tok_mismatch.append(s + j)
            fwd.append(ref.embed_ids(ids, mask))
        fwd = torch.cat(fwd)
        e2e = ref.encode(texts, prompt=prompt, batch_size=args.batch_size)

        res = {
            "prompt": prompt,
            "token_mismatches": tok_mismatch,
            "forward": cos_stats(fwd, st_emb),
            "e2e": cos_stats(e2e, st_emb),
            "seconds": round(time.time() - t0, 1),
        }
        ok = all(v["mean"] >= 0.9999 for k in ("forward", "e2e") for v in res[k].values()) and not tok_mismatch
        res["pass"] = ok
        all_ok &= ok
        report["cases"][case] = res
        print(f"\n[{case}] prompt={prompt!r} token mismatches={len(tok_mismatch)} -> {'PASS' if ok else 'FAIL'}")
        for kind in ("forward", "e2e"):
            print("   " + kind.ljust(8) + "  ".join(
                f"{d}: mean={v['mean']:.6f} min={v['min']:.6f}" for d, v in res[kind].items()))

    report["pass"] = all_ok
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[validate] overall: {'PASS' if all_ok else 'FAIL'}  -> {args.out}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
