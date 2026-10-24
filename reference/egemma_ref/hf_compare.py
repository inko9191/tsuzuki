"""Layer-by-layer comparison against the official transformers implementation.

Used only for validation/debugging (the reference forward never imports transformers models).
"""

from __future__ import annotations

import torch

from .model import EmbeddingGemma2Text


def _first(t):
    return t[0] if isinstance(t, (tuple, list)) else t


def hf_captures(hf_text_model, input_ids, attention_mask) -> dict[str, torch.Tensor]:
    """Run the HF EmbeddingGemma2TextModel with forward hooks at the same points as model.py `cap()`."""
    caps: dict[str, torch.Tensor] = {}
    hooks = []

    def reg(module, name):
        hooks.append(module.register_forward_hook(lambda m, i, o: caps.__setitem__(name, _first(o).detach())))

    lm = hf_text_model
    reg(lm.embed_tokens, "embed")
    reg(lm.ple, "ple")
    for i, layer in enumerate(lm.layers):
        reg(layer.self_attn, f"layer{i}.attn")
        reg(layer.mlp, f"layer{i}.mlp")
        reg(layer, f"layer{i}.out")
    reg(lm.norm, "final_norm")
    reg(lm.embedding_projection, "embedding_projection")
    try:
        with torch.no_grad():
            lm(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in hooks:
            h.remove()
    return caps


def ref_captures(ref: EmbeddingGemma2Text, input_ids, attention_mask) -> dict[str, torch.Tensor]:
    caps: dict[str, torch.Tensor] = {}
    ref.forward(input_ids, attention_mask, capture=lambda n, t: caps.__setitem__(n, t.detach()))
    return caps


def compare(ref_caps, hf_caps, attention_mask) -> list[dict]:
    """Per capture point: max abs diff and min per-token cosine over valid (non-pad) tokens."""
    valid = attention_mask.bool().cpu()
    rows = []
    for name, r in ref_caps.items():
        if name not in hf_caps:
            continue
        h = hf_caps[name].to(r.device, r.dtype)
        r_, h_ = r.cpu().float()[valid], h.cpu().float()[valid]  # [N, ...]
        r2, h2 = r_.reshape(r_.shape[0], -1), h_.reshape(h_.shape[0], -1)
        cos = torch.nn.functional.cosine_similarity(r2, h2, dim=-1)
        rows.append({
            "name": name,
            "max_abs": float((r2 - h2).abs().max()),
            "ref_absmax": float(r2.abs().max()),
            "min_cos": float(cos.min()),
            "mean_cos": float(cos.mean()),
        })
    return rows


def format_rows(rows: list[dict]) -> str:
    out = [f"{'point':28s} {'max|diff|':>12s} {'max|ref|':>10s} {'min cos':>12s} {'mean cos':>12s}"]
    for r in rows:
        out.append(f"{r['name']:28s} {r['max_abs']:12.3e} {r['ref_absmax']:10.3g} "
                   f"{r['min_cos']:12.9f} {r['mean_cos']:12.9f}")
    return "\n".join(out)
