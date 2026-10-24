"""Python tokenizer and pybind11 wrapper; forward inference always runs in Metal."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_extension(build_dir: str | Path = ROOT / "build"):
    directory = str(Path(build_dir).resolve())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module("_egemma_native")


class NativePipeline:
    def __init__(self, model_dir: str | Path, dtype: str = "fp32", build_dir: str | Path = ROOT / "build"):
        from transformers import AutoTokenizer

        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        st = json.loads((self.model_dir / "config_sentence_transformers.json").read_text())
        sb = json.loads((self.model_dir / "sentence_bert_config.json").read_text())
        self.prompts = st.get("prompts", {})
        self.default_prompt_name = st.get("default_prompt_name")
        self.max_seq_length = sb.get("max_seq_length")
        modalities = sb.get("modality_config") or {}
        if not modalities:
            raise ValueError("Cannot determine input format without modality_config")
        self.input_format = "chat" if "message" in modalities else "plain"
        self.engine = load_extension(build_dir).Engine(str(self.model_dir), dtype)

    def tokenize_one(self, text: str, prompt: str | None = None) -> list[int]:
        # Same rule as reference/egemma_ref/pipeline.py L131-149.
        if self.input_format == "chat":
            messages = []
            if prompt:
                messages.append({"role": "system", "content": [{"type": "text", "text": prompt}]})
            messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            bos = self.tokenizer.bos_token
            ids = self.tokenizer(rendered, add_special_tokens=not (bos and rendered.startswith(bos)))["input_ids"]
        else:
            ids = self.tokenizer((prompt or "") + text, add_special_tokens=True)["input_ids"]
        if self.max_seq_length is not None and len(ids) > self.max_seq_length:
            raise ValueError("Input exceeds max_seq_length; reference truncation is unsupported")
        return list(ids)

    def batch(self, sequences: list[list[int]], padding_side: str = "right") -> tuple[np.ndarray, np.ndarray]:
        if not sequences or any(not s for s in sequences):
            raise ValueError("Nonempty batch and token sequences are required")
        if padding_side not in ("left", "right"):
            raise ValueError("padding_side must be left or right")
        width = max(map(len, sequences))
        ids = np.full((len(sequences), width), self.tokenizer.pad_token_id or 0, dtype=np.int32)
        mask = np.zeros_like(ids)
        for row, sequence in enumerate(sequences):
            start = width - len(sequence) if padding_side == "left" else 0
            ids[row, start:start + len(sequence)] = sequence
            mask[row, start:start + len(sequence)] = 1
        return ids, mask

    def encode(self, texts: list[str], prompt: str | None = None, prompt_name: str | None = None,
               batch_size: int = 8, dim: int = 768) -> np.ndarray:
        if not texts or batch_size < 1:
            raise ValueError("Nonempty texts and positive batch_size are required")
        if prompt is None:
            name = prompt_name if prompt_name is not None else self.default_prompt_name
            prompt = self.prompts[name] if name is not None else None
        outputs = []
        for start in range(0, len(texts), batch_size):
            ids, mask = self.batch([self.tokenize_one(t, prompt) for t in texts[start:start + batch_size]])
            outputs.append(self.engine.encode(ids, mask, dim))
        return np.concatenate(outputs)
