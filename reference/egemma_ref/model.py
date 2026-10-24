"""From-scratch forward pass of the EmbeddingGemma 2 text backbone (no transformers model classes).

Source of truth: transformers v5.19.0 models/embedding_gemma2/modeling_embedding_gemma2.py
(cited as HF Lxxx) and vLLM @1dfe9fb vllm/model_executor/models/embedding_gemma2.py (vLLM Lxxx).
See docs/ARCHITECTURE.md for the full derivation.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .config import LayerSpec, TextConfig

Capture = Callable[[str, torch.Tensor], None] | None


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    # HF L58-67: computed in fp32, x * pow(mean(x^2) + eps, -0.5) * weight  (plain weight, no "1 +")
    xf = x.float()
    y = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


def activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name in ("gelu_pytorch_tanh", "gelu_tanh"):
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu
    raise NotImplementedError(f"hidden_activation={name!r}")


def inv_freq(spec: LayerSpec) -> torch.Tensor:
    d = spec.head_dim
    if spec.rope_type == "default":
        # HF L120-126
        return 1.0 / (spec.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float) / d))
    # proportional: modeling_rope_utils.py _compute_proportional_rope_parameters
    rope_angles = int(spec.partial_rotary_factor * d // 2)
    rot = 1.0 / (spec.rope_theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float() / d))
    nope = d // 2 - rope_angles
    f = torch.cat([rot, torch.zeros(nope)]) if nope > 0 else rot
    return f / spec.rope_factor


def rope_cos_sin(spec: LayerSpec, positions: torch.Tensor, dtype: torch.dtype):
    # HF L128-139: emb = cat(freqs, freqs); NeoX-style rotate_half (HF L230-260, vLLM L143)
    f = inv_freq(spec).to(positions.device)
    freqs = positions[..., None].float() * f
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class EmbeddingGemma2Text:
    def __init__(self, cfg: TextConfig, weights: dict[str, torch.Tensor], device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype
        self.w = {k: v.to(device=self.device, dtype=dtype) for k, v in weights.items()}
        self.act = activation(cfg.hidden_activation)
        # HF L490-493 / L150-153: sqrt(hidden) materialised in the weight dtype (bf16 -> 22.625).
        self.embed_scale = torch.tensor(cfg.hidden_size**0.5, dtype=torch.float32).to(dtype)

    def _lin(self, x: torch.Tensor, name: str) -> torch.Tensor:
        b = self.w.get(name + ".bias")
        return F.linear(x, self.w[name + ".weight"], b)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                capture: Capture = None) -> torch.Tensor:
        """input_ids/attention_mask: [B, T]. Returns per-token embeddings [B, T, embedding_dim]
        (embedding_projection is applied per token, HF L504-505/L560)."""
        cfg = self.cfg
        cap = capture or (lambda name, t: None)
        input_ids = input_ids.to(self.device)
        B, T = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long)
        key_valid = attention_mask.to(self.device).bool()  # [B, T]

        # --- token embedding, HF L525 ---
        x = F.embedding(input_ids, self.w["embed_tokens.weight"]) * self.embed_scale
        cap("embed", x)

        # --- projection-only PLE, HF L172-200 / vLLM L301-334 ---
        L, P = cfg.num_hidden_layers, cfg.hidden_size_per_layer_input
        ple = F.linear(x, self.w["ple.per_layer_model_projection.weight"]) * (cfg.hidden_size**-0.5)
        ple = rms_norm(ple.reshape(B, T, L, P), self.w["ple.per_layer_projection_norm.weight"], cfg.rms_norm_eps)
        cap("ple", ple)

        # positions: HF L529-530 (arange over the padded length). RoPE is relative, so the result is
        # invariant to left/right padding.
        positions = torch.arange(T, device=self.device)
        rope_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

        # masks, HF L539-542 + masking_utils.py sliding_window_bidirectional_overlay (|q-k| <= W)
        dist = (positions[:, None] - positions[None, :]).abs()  # [T, T]
        neg = torch.finfo(self.dtype).min
        mask_cache: dict[int | None, torch.Tensor] = {}

        def additive_mask(window: int | None) -> torch.Tensor:
            if window not in mask_cache:
                allowed = key_valid[:, None, None, :]  # [B,1,1,T] padding mask on keys only
                if window is not None:
                    allowed = allowed & (dist <= window)[None, None]
                m = torch.zeros(allowed.shape, dtype=self.dtype, device=self.device)
                mask_cache[window] = m.masked_fill(~allowed, neg)
            return mask_cache[window]

        H = cfg.num_attention_heads
        eps = cfg.rms_norm_eps
        for i, spec in enumerate(cfg.layers):
            p = f"layers.{i}."
            key = (spec.rope_type, spec.rope_theta, spec.head_dim, spec.partial_rotary_factor, spec.rope_factor)
            if key not in rope_cache:
                rope_cache[key] = rope_cos_sin(spec, positions, self.dtype)
            cos, sin = rope_cache[key]  # [T, D]

            # ---- attention block, HF L417-427 ----
            residual = x
            h = rms_norm(x, self.w[p + "input_layernorm.weight"], eps)
            D, KV = spec.head_dim, spec.num_kv_heads
            q = self._lin(h, p + "self_attn.q_proj").view(B, T, H, D).transpose(1, 2)
            k = self._lin(h, p + "self_attn.k_proj").view(B, T, KV, D).transpose(1, 2)
            v = self._lin(h, p + "self_attn.v_proj").view(B, T, KV, D).transpose(1, 2)
            q = rms_norm(q, self.w[p + "self_attn.q_norm.weight"], eps)  # HF L364-366
            k = rms_norm(k, self.w[p + "self_attn.k_norm.weight"], eps)
            v = rms_norm(v, None, eps)  # v_norm: no scale (HF L348)
            q = q * cos + rotate_half(q) * sin  # HF L256-259
            k = k * cos + rotate_half(k) * sin
            if KV != H:  # GQA, HF L263-272 (repeat_interleave on the head axis)
                k = k.repeat_interleave(H // KV, dim=1)
                v = v.repeat_interleave(H // KV, dim=1)
            scores = torch.matmul(q, k.transpose(-1, -2))  # scaling = 1.0 (HF L329, vLLM L150)
            scores = scores + additive_mask(spec.window)
            probs = torch.softmax(scores.float(), dim=-1).to(self.dtype)  # HF L302
            a = torch.matmul(probs, v).transpose(1, 2).reshape(B, T, H * D)
            a = self._lin(a, p + "self_attn.o_proj")
            cap(f"layer{i}.attn", a)
            x = residual + rms_norm(a, self.w[p + "post_attention_layernorm.weight"], eps)

            # ---- MLP block, HF L429-433 / L167-169 ----
            residual = x
            h = rms_norm(x, self.w[p + "pre_feedforward_layernorm.weight"], eps)
            h = F.linear(self.act(F.linear(h, self.w[p + "mlp.gate_proj.weight"])) *
                         F.linear(h, self.w[p + "mlp.up_proj.weight"]), self.w[p + "mlp.down_proj.weight"])
            cap(f"layer{i}.mlp", h)
            x = residual + rms_norm(h, self.w[p + "post_feedforward_layernorm.weight"], eps)

            # ---- PLE block, HF L203-227 ----
            g = self.act(F.linear(x, self.w[p + "ple_block.per_layer_input_gate.weight"])) * ple[:, :, i, :]
            g = F.linear(g, self.w[p + "ple_block.per_layer_projection.weight"])
            x = x + rms_norm(g, self.w[p + "ple_block.post_per_layer_input_norm.weight"], eps)

            x = x * self.w[p + "layer_scalar"]  # HF L437
            cap(f"layer{i}.out", x)

        x = rms_norm(x, self.w["norm.weight"], eps)  # HF L559
        cap("final_norm", x)
        out = F.linear(x, self.w["embedding_projection.weight"])  # HF L560 (learned head, 768-d)
        cap("embedding_projection", out)
        return out


def mean_pool(token_embs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # sentence_transformers Pooling(mean) / HF docs embedding_gemma2.md L96-98
    m = attention_mask.to(token_embs.device).unsqueeze(-1).to(token_embs.dtype)
    return (token_embs * m).sum(1) / m.sum(1).clamp(min=1e-9)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1)


def truncate_mrl(emb: torch.Tensor, dim: int) -> torch.Tensor:
    # docs embedding_gemma2.md L134: slice the prefix, then re-normalise
    return l2_normalize(emb[..., :dim])

