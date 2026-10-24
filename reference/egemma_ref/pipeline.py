"""End-to-end text embedding pipeline: tokenizer -> backbone -> ST post-processing modules.

The post-processing chain is read from the checkpoint's Sentence Transformers `modules.json`
instead of being hard-coded, so that every learned layer the checkpoint ships (e.g. Dense heads)
is applied. Unknown module types abort instead of being silently skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .config import load_text_config
from .model import EmbeddingGemma2Text, l2_normalize, mean_pool
from .weights import load_text_weights


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _Dense:
    def __init__(self, d: Path, device, dtype):
        cfg = json.loads((d / "config.json").read_text())
        st = d / "model.safetensors"
        w = load_file(str(st)) if st.exists() else torch.load(d / "pytorch_model.bin", map_location="cpu")
        self.weight = w["linear.weight"].to(device=device, dtype=dtype)
        self.bias = w["linear.bias"].to(device=device, dtype=dtype) if "linear.bias" in w else None
        act = cfg.get("activation_function", "torch.nn.modules.activation.Tanh")
        if act.endswith("Identity"):
            self.act = lambda x: x
        elif act.endswith("Tanh"):
            self.act = torch.tanh
        else:
            raise NotImplementedError(f"Dense activation {act!r} in {d}")
        for k in ("module_input_name", "module_output_name"):
            if cfg.get(k) not in (None, "sentence_embedding"):
                raise NotImplementedError(f"Dense {k}={cfg.get(k)!r} in {d}")
        self.desc = f"Dense({self.weight.shape[1]}->{self.weight.shape[0]}, bias={self.bias is not None}, act={act})"

    def __call__(self, x):
        return self.act(F.linear(x, self.weight, self.bias))


class Pipeline:
    def __init__(self, model_dir: str | Path, device: str = "auto", dtype: torch.dtype = torch.float32,
                 input_format: str = "auto", verbose: bool = True):
        from transformers import AutoTokenizer  # tokenizer only (allowed by the project rules)

        self.model_dir = Path(model_dir)
        self.device = pick_device(device)
        self.cfg = load_text_config(self.model_dir)
        weights, self.weight_info = load_text_weights(self.model_dir, self.cfg, dtype=torch.float32, verbose=verbose)
        self.model = EmbeddingGemma2Text(self.cfg, weights, device=self.device, dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        st_cfg_path = self.model_dir / "config_sentence_transformers.json"
        st_cfg = json.loads(st_cfg_path.read_text()) if st_cfg_path.exists() else {}
        self.prompts: dict[str, str] = st_cfg.get("prompts", {})
        self.default_prompt_name: str | None = st_cfg.get("default_prompt_name")
        sb_path = self.model_dir / "sentence_bert_config.json"
        self.sb_cfg = json.loads(sb_path.read_text()) if sb_path.exists() else {}
        self.max_seq_length = self.sb_cfg.get("max_seq_length")

        if input_format == "auto":
            mc = self.sb_cfg.get("modality_config") or {}
            if "message" in mc:
                input_format = "chat"
            elif mc:
                input_format = "plain"
            else:
                raise ValueError(
                    "sentence_bert_config.json has no modality_config; pass input_format='plain' or 'chat' "
                    "explicitly (scripts/validate_vs_st.py reports which one matches Sentence Transformers)."
                )
        self.input_format = input_format

        self.post = []  # list of (name, callable on [B, E])
        self.pool_desc = None
        modules_path = self.model_dir / "modules.json"
        if not modules_path.exists():
            raise FileNotFoundError("modules.json missing: cannot determine the post-pooling chain")
        for m in json.loads(modules_path.read_text()):
            t, sub = m["type"], self.model_dir / m["path"]
            if t.endswith("Transformer"):
                continue
            if t.endswith("Pooling"):
                pc = json.loads((sub / "config.json").read_text())
                mode = pc.get("pooling_mode") or next(
                    (k[len("pooling_mode_"):] for k, v in pc.items() if k.startswith("pooling_mode_") and v is True),
                    None,
                )
                if mode not in ("mean", "mean_tokens"):
                    raise NotImplementedError(f"pooling mode {mode!r} ({pc})")
                if pc.get("include_prompt", True) is not True:
                    raise NotImplementedError("include_prompt=False is not implemented (see ST pooling.py L125-168)")
                self.pool_desc = f"Pooling(mean, include_prompt=True) cfg={pc}"
            elif t.endswith("Dense"):
                d = _Dense(sub, self.device, dtype)
                self.post.append((d.desc, d))
            elif t.endswith("Normalize"):
                self.post.append(("Normalize", l2_normalize))
            else:
                raise NotImplementedError(f"Sentence Transformers module {t!r} is not implemented")
        if self.pool_desc is None:
            raise ValueError("no Pooling module in modules.json")
        if verbose:
            print(f"[pipeline] device={self.device} input_format={self.input_format} "
                  f"max_seq_length={self.max_seq_length}")
            print(f"[pipeline] {self.pool_desc} -> " + " -> ".join(n for n, _ in self.post))

    # ---------------- tokenisation ----------------
    def resolve_prompt(self, prompt_name: str | None, prompt: str | None) -> str | None:
        # Same precedence as sentence_transformers base/model.py _resolve_prompt (L344-369).
        if prompt is not None:
            return prompt
        if prompt_name is not None:
            return self.prompts[prompt_name]
        if self.default_prompt_name is not None:
            return self.prompts.get(self.default_prompt_name)
        return None

    def tokenize_one(self, text: str, prompt: str | None) -> list[int]:
        if self.input_format == "plain":
            ids = self.tokenizer((prompt or "") + text, add_special_tokens=True)["input_ids"]
        else:
            msgs = []
            if prompt:
                msgs.append({"role": "system", "content": [{"type": "text", "text": prompt}]})
            msgs.append({"role": "user", "content": [{"type": "text", "text": text}]})
            rendered = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            # Mirror ProcessorMixin.apply_chat_template (transformers 5.19.0,
            # processing_utils.py L2218-2227): add special tokens unless the
            # rendered template already starts with BOS. Tokenizer's direct
            # apply_chat_template(tokenize=True) always omits them.
            bos = self.tokenizer.bos_token
            ids = self.tokenizer(rendered, add_special_tokens=not (bos and rendered.startswith(bos)))["input_ids"]
        if self.max_seq_length is not None and len(ids) > self.max_seq_length:
            raise ValueError(f"input of {len(ids)} tokens exceeds max_seq_length={self.max_seq_length}; "
                             "truncation semantics are not implemented in the reference")
        return list(ids)

    def batch(self, seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        T = max(len(s) for s in seqs)
        pad = self.cfg.pad_token_id or 0
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):  # right padding (results are padding-side invariant, see model.py)
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        return ids, mask

    # ---------------- inference ----------------
    def embed_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        tok = self.model.forward(input_ids, attention_mask)
        x = mean_pool(tok, attention_mask)
        for _, fn in self.post:
            x = fn(x)
        return x.float().cpu()

    def encode(self, texts: list[str], prompt_name: str | None = None, prompt: str | None = None,
               batch_size: int = 8) -> torch.Tensor:
        p = self.resolve_prompt(prompt_name, prompt)
        outs = []
        for s in range(0, len(texts), batch_size):
            seqs = [self.tokenize_one(t, p) for t in texts[s: s + batch_size]]
            ids, mask = self.batch(seqs)
            outs.append(self.embed_ids(ids, mask))
        return torch.cat(outs)
