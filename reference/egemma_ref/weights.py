"""Load EmbeddingGemma 2 text-backbone tensors directly from safetensors.

The checkpoint holds the full multimodal model; only the text backbone (`language_model.*`) is
used here. The vision/audio towers and their `embed_vision` / `embed_audio` adapters are skipped,
as both HF (modeling_embedding_gemma2.py L637-642) and vLLM (embedding_gemma2.py L1117-1129) do
when the towers are disabled.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .config import TextConfig

_TOWER_MARKERS = ("vision_tower.", "embed_vision.", "audio_tower.", "embed_audio.")


def _shards(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        with open(index) as f:
            return sorted({model_dir / v for v in json.load(f)["weight_map"].values()})
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"no model.safetensors(.index.json) in {model_dir}")


def expected_keys(cfg: TextConfig) -> set[str]:
    keys = {
        "embed_tokens.weight",
        "norm.weight",
        "ple.per_layer_model_projection.weight",
        "ple.per_layer_projection_norm.weight",
        "embedding_projection.weight",
    }
    for i in range(cfg.num_hidden_layers):
        p = f"layers.{i}."
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            keys.add(f"{p}self_attn.{proj}.weight")
            if cfg.attention_bias:
                keys.add(f"{p}self_attn.{proj}.bias")
        keys |= {
            f"{p}self_attn.q_norm.weight",
            f"{p}self_attn.k_norm.weight",  # v_norm has no weight (with_scale=False, L348)
            f"{p}mlp.gate_proj.weight",
            f"{p}mlp.up_proj.weight",
            f"{p}mlp.down_proj.weight",
            f"{p}input_layernorm.weight",
            f"{p}post_attention_layernorm.weight",
            f"{p}pre_feedforward_layernorm.weight",
            f"{p}post_feedforward_layernorm.weight",
            f"{p}layer_scalar",
            f"{p}ple_block.per_layer_input_gate.weight",
            f"{p}ple_block.per_layer_projection.weight",
            f"{p}ple_block.post_per_layer_input_norm.weight",
        }
    return keys


def find_text_prefix(names: list[str]) -> str:
    cands = sorted(
        n[: -len("embed_tokens.weight")]
        for n in names
        if n.endswith("embed_tokens.weight") and not any(m in n for m in _TOWER_MARKERS)
    )
    if len(cands) != 1:
        raise ValueError(f"cannot locate the text backbone; embed_tokens candidates: {cands}")
    return cands[0]


def load_text_weights(
    model_dir: str | Path, cfg: TextConfig, dtype: torch.dtype = torch.float32, verbose: bool = True
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Returns (tensors keyed by backbone-relative name, info dict)."""
    model_dir = Path(model_dir)
    shards = _shards(model_dir)
    all_names: dict[str, Path] = {}
    src_dtypes: dict[str, str] = {}
    for s in shards:
        with safe_open(s, framework="pt") as f:
            for n in f.keys():
                all_names[n] = s
    prefix = find_text_prefix(list(all_names))
    want = expected_keys(cfg)

    out: dict[str, torch.Tensor] = {}
    unexpected = []
    skipped_tower = 0
    for name, shard in all_names.items():
        if not name.startswith(prefix):
            if any(m in name for m in _TOWER_MARKERS):
                skipped_tower += 1
            else:
                unexpected.append(name)
            continue
        rel = name[len(prefix):]
        if rel not in want:
            unexpected.append(name)
            continue
        with safe_open(shard, framework="pt") as f:
            t = f.get_tensor(name)
        src_dtypes[str(t.dtype)] = src_dtypes.get(str(t.dtype), 0) + 1
        out[rel] = t.to(dtype)

    missing = sorted(want - set(out))
    # layer_scalar is a persistent buffer (HF L406 / vLLM L277); HF initialises it to ones when absent
    # (L467-468). Mirror that, loudly. Everything else must be present.
    missing_scalars = [m for m in missing if m.endswith("layer_scalar")]
    missing_other = [m for m in missing if not m.endswith("layer_scalar")]
    if missing_other:
        raise KeyError(f"missing text-backbone tensors (prefix {prefix!r}): {missing_other[:20]} ...")
    for m in missing_scalars:
        out[m] = torch.ones(1, dtype=dtype)
    if unexpected:
        raise KeyError(f"unexpected tensors not consumed by the reference: {unexpected[:20]}")

    info = {
        "prefix": prefix,
        "shards": ",".join(s.name for s in shards),
        "source_dtypes": json.dumps(src_dtypes),
        "skipped_tower_tensors": str(skipped_tower),
        "missing_layer_scalar_defaulted_to_1": str(len(missing_scalars)),
    }
    if verbose:
        print(f"[weights] {len(out)} text tensors, prefix={prefix!r}, src dtypes={src_dtypes}, "
              f"skipped tower tensors={skipped_tower}")
        if missing_scalars:
            print(f"[weights] WARNING: {len(missing_scalars)} layer_scalar buffers absent -> 1.0 (HF default)")
    return out, info
