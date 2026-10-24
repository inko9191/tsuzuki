"""EmbeddingGemma 2 text-config parsing.

Mirrors the derivation in transformers v5.19.0
`models/embedding_gemma2/configuration_embedding_gemma2.py` (EmbeddingGemma2TextConfig, L71-130)
so that a config.json with omitted fields resolves to the same per-layer values HF uses.
Every per-layer quantity the forward pass needs is materialised into a `LayerSpec`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# configuration_embedding_gemma2.py L71-91 (dataclass defaults)
_TEXT_DEFAULTS: dict[str, Any] = {
    "vocab_size": 262_144,
    "hidden_size": 512,
    "intermediate_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "max_position_embeddings": 262_144,
    "rms_norm_eps": 1e-6,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "bos_token_id": 2,
    "rope_parameters": None,
    "attention_bias": False,
    "sliding_window": 512,
    "layer_types": None,
    "hidden_size_per_layer_input": 512,
    "embedding_dim": 768,
}

# configuration_embedding_gemma2.py L113-116
_DEFAULT_ROPE = {
    "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
    "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0},
}

# Keys that carry no numerical meaning for the text forward pass.
_IGNORED_KEYS = {
    "model_type", "architectures", "transformers_version", "dtype", "torch_dtype", "initializer_range",
    "attention_dropout", "use_cache", "tie_word_embeddings", "_name_or_path", "id2label", "label2id",
    "return_dict", "output_hidden_states", "output_attentions", "chunk_size_feed_forward",
    "is_encoder_decoder", "problem_type", "tokenizer_class", "prefix", "task_specific_params",
    "_attn_implementation_autoset",
}

# Per-layer overrides this implementation knows how to apply. Anything else is rejected.
_SUPPORTED_OVERRIDES = {"head_dim", "num_key_value_heads", "intermediate_size"}


@dataclass(frozen=True)
class LayerSpec:
    layer_type: str  # "sliding_attention" | "full_attention"
    head_dim: int
    num_kv_heads: int
    intermediate_size: int
    rope_type: str
    rope_theta: float
    partial_rotary_factor: float
    rope_factor: float
    # Inclusive bidirectional radius: key j visible from query i iff |i - j| <= window.
    window: int | None


@dataclass
class TextConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    hidden_activation: str
    rms_norm_eps: float
    attention_bias: bool
    hidden_size_per_layer_input: int
    embedding_dim: int
    pad_token_id: int | None
    bos_token_id: int | None
    eos_token_id: int | list[int] | None
    layers: list[LayerSpec] = field(default_factory=list)
    unknown_keys: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"vocab={self.vocab_size} hidden={self.hidden_size} layers={self.num_hidden_layers} "
            f"heads={self.num_attention_heads} ple={self.hidden_size_per_layer_input} "
            f"embedding_dim={self.embedding_dim} act={self.hidden_activation} eps={self.rms_norm_eps}"
        ]
        for i, l in enumerate(self.layers):
            lines.append(
                f"  L{i:02d} {l.layer_type:17s} head_dim={l.head_dim} kv={l.num_kv_heads} "
                f"ffn={l.intermediate_size} rope={l.rope_type}(theta={l.rope_theta:g}, "
                f"partial={l.partial_rotary_factor}, factor={l.rope_factor}) window={l.window}"
            )
        if self.unknown_keys:
            lines.append(f"  UNRECOGNISED KEYS (review!): {self.unknown_keys}")
        return "\n".join(lines)


def _text_section(raw: dict[str, Any]) -> dict[str, Any]:
    # Composite EmbeddingGemma2Config nests the backbone under text_config (L166, L179-183).
    if "text_config" in raw and isinstance(raw["text_config"], dict):
        return raw["text_config"]
    return raw


def parse_text_config(raw: dict[str, Any]) -> TextConfig:
    src = dict(_text_section(raw))
    c = {k: src.pop(k, v) for k, v in _TEXT_DEFAULTS.items()}

    # Builder-only kwargs (L94-97, L121-122)
    sliding_window_pattern = src.pop("sliding_window_pattern", 6)
    global_head_dim = src.pop("global_head_dim", 512)
    num_global_kv = src.pop("num_global_key_value_heads", 1)
    per_layer_config = src.pop("per_layer_config", None)

    n = c["num_hidden_layers"]
    layer_types = c["layer_types"]
    if layer_types is None:  # L101-105
        layer_types = [
            "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention" for i in range(n)
        ]
    layer_types = list(layer_types)
    if layer_types[-1] != "full_attention":  # L107-111
        layer_types[-1] = "full_attention"
    if len(layer_types) != n:
        raise ValueError(f"layer_types has {len(layer_types)} entries, expected {n}")

    rope = c["rope_parameters"] or _DEFAULT_ROPE  # L117-118

    if per_layer_config is None:  # L123-128
        per_layer_config = {
            i: {"head_dim": global_head_dim, "num_key_value_heads": num_global_kv}
            for i, t in enumerate(layer_types)
            if t == "full_attention"
        }
    overrides = {int(k): dict(v) for k, v in per_layer_config.items()}
    for idx, ov in overrides.items():
        bad = set(ov) - _SUPPORTED_OVERRIDES
        if bad:
            raise NotImplementedError(
                f"per_layer_config[{idx}] overrides {sorted(bad)}; not implemented in the reference. "
                "See transformers integrations/heterogeneity/configuration_utils.py (_get_layer_config)."
            )

    layers = []
    for i, t in enumerate(layer_types):
        ov = overrides.get(i, {})
        rp = rope[t]
        rope_type = rp.get("rope_type", "default")
        if rope_type not in ("default", "proportional"):
            raise NotImplementedError(
                f"rope_type={rope_type!r} for {t}; only 'default' (modeling_embedding_gemma2.py L102-126) and "
                "'proportional' (modeling_rope_utils.py _compute_proportional_rope_parameters) are implemented."
            )
        layers.append(
            LayerSpec(
                layer_type=t,
                head_dim=ov.get("head_dim", c["head_dim"]),
                num_kv_heads=ov.get("num_key_value_heads", c["num_key_value_heads"]),
                intermediate_size=ov.get("intermediate_size", c["intermediate_size"]),
                rope_type=rope_type,
                rope_theta=float(rp["rope_theta"]),
                partial_rotary_factor=float(rp.get("partial_rotary_factor", 1.0)),
                rope_factor=float(rp.get("factor", 1.0)),
                window=c["sliding_window"] if t == "sliding_attention" else None,
            )
        )

    unknown = sorted(k for k in src if k not in _IGNORED_KEYS)
    return TextConfig(
        vocab_size=c["vocab_size"],
        hidden_size=c["hidden_size"],
        num_hidden_layers=n,
        num_attention_heads=c["num_attention_heads"],
        hidden_activation=c["hidden_activation"],
        rms_norm_eps=float(c["rms_norm_eps"]),
        attention_bias=bool(c["attention_bias"]),
        hidden_size_per_layer_input=c["hidden_size_per_layer_input"],
        embedding_dim=c["embedding_dim"],
        pad_token_id=c["pad_token_id"],
        bos_token_id=c["bos_token_id"],
        eos_token_id=c["eos_token_id"],
        layers=layers,
        unknown_keys=unknown,
    )


def load_text_config(model_dir: str | Path) -> TextConfig:
    with open(Path(model_dir) / "config.json") as f:
        return parse_text_config(json.load(f))
