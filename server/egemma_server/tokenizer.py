"""Prompt rendering and tokenization without transformers.

The released checkpoint's chat template (``chat_template.jinja``) renders a
system prompt and one user text as plain concatenation ``prompt + text`` with
no turn markers; the Processor then adds BOS/EOS unless the rendered string
already starts with BOS (reference/egemma_ref/pipeline.py L131-149). The
template is pinned by hash so a different template fails loudly instead of
being silently mis-rendered. Equality with the reference tokenization is
tested by scripts/validate_server.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# chat_template.jinja of google/embeddinggemma-2 @ 914f7f89142e33e77833254d9c9b90c3cef7303b
SUPPORTED_CHAT_TEMPLATES = {"4b852efc0b9960283e735363331e6f325b33bc74bdbaa076f595bc4e9b94d85e"}


class PromptTokenizer:
    def __init__(self, model_dir: str | Path):
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        config = json.loads((model_dir / "tokenizer_config.json").read_text())
        self.bos = config["bos_token"]
        self.pad_id = self.tokenizer.token_to_id(config.get("pad_token") or "<pad>")
        st = json.loads((model_dir / "config_sentence_transformers.json").read_text())
        self.prompts: dict[str, str] = st.get("prompts", {})
        self.default_prompt_name: str | None = st.get("default_prompt_name")
        modalities = json.loads((model_dir / "sentence_bert_config.json").read_text()).get("modality_config") or {}
        self.chat = "message" in modalities
        if self.chat:
            template = (model_dir / "chat_template.jinja").read_bytes()
            digest = hashlib.sha256(template).hexdigest()
            if digest not in SUPPORTED_CHAT_TEMPLATES:
                raise ValueError(f"Unsupported chat template (sha256 {digest}); only the released template is implemented")
        # Case-insensitive task lookup; names differing only by case map to the same prompt text.
        self._tasks: dict[str, str] = {}
        for name, text in self.prompts.items():
            previous = self._tasks.setdefault(name.lower(), text)
            if previous != text:
                raise ValueError(f"Prompt names {name!r} differ only by case but have different text")

    def task_names(self) -> list[str]:
        return sorted(self.prompts)

    def prompt_for(self, task: str | None) -> str | None:
        """Prompt text for a task name; ``None``/'none' means no prompt, 'default' the model card default."""
        if task is None or task.lower() == "none":
            return None
        if task.lower() == "default":
            return None if self.default_prompt_name is None else self.prompts[self.default_prompt_name]
        try:
            return self._tasks[task.lower()]
        except KeyError:
            raise KeyError(task) from None

    def encode(self, texts: list[str], prompt: str | None) -> list[list[int]]:
        rendered = [(prompt or "") + text for text in texts]
        if not self.chat:
            return [e.ids for e in self.tokenizer.encode_batch(rendered, add_special_tokens=True)]
        out: list[list[int] | None] = [None] * len(rendered)
        for add_special in (True, False):
            index = [i for i, r in enumerate(rendered) if (not r.startswith(self.bos)) == add_special]
            if index:
                for i, e in zip(index, self.tokenizer.encode_batch([rendered[i] for i in index], add_special_tokens=add_special)):
                    out[i] = e.ids
        return out  # type: ignore[return-value]
