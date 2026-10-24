#!/usr/bin/env python3
"""Install the tsuzuki provider into Hermes profiles (stop Hermes first; restart it afterwards).

    python3 integrations/hermes/install.py [--home DIR] [--allow-remote-chat | --no-chat] <profile> [<profile> ...]

<profile> is ``default`` (the Hermes home itself) or the name of a profile under ``<home>/profiles``; it must already
have a ``config.yaml`` (start Hermes with it once). ``--home`` is the Hermes home (default ``~/.hermes``).

Per profile: back up config.yaml, copy the plugin to <profile>/plugins/tsuzuki, add
``memory.provider: tsuzuki`` (+ the sidecar path, and Hermes' bundled node when there is one), and write
<profile>/tsuzuki/config.json:

- embeddings: this repository's EmbeddingGemma 300m engine (started on demand on port 8299) when it is built here
  (step 2 of the README); otherwise Ollama's ``embeddinggemma`` on 127.0.0.1:11434, and the script says so;
- facts: the profile's own model endpoint (``model.base_url``), but only when it is on this machine (loopback).
  Conversation text is sent there; a remote endpoint needs ``--allow-remote-chat``. ``--no-chat`` turns facts off.

Re-running is safe: existing settings are kept, and an existing config.json is not overwritten. Nothing is changed
in any profile when one of them cannot be installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "integrations" / "hermes" / "tsuzuki"
SERVER_JS = REPO / "dsh-plugins" / "packages" / "tsuzuki" / "lib" / "server.js"
DEFAULT_PORT = 8297
ENGINE_PORT = 8299
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def profile_home(name: str, root: Path) -> Path:
    return root if name == "default" else root / "profiles" / name


def profile_port(data_dir: Path) -> int:  # same rule as tsuzuki.profile_port
    return DEFAULT_PORT + int(hashlib.sha256(str(data_dir.resolve()).encode()).hexdigest(), 16) % 500


# -- reading config.yaml ------------------------------------------------------------------------


def _closing_quote(value: str) -> int:
    quote, k = value[0], 1
    while k < len(value):
        if quote == '"' and value[k] == "\\":
            k += 2
            continue
        if value[k] == quote:
            if quote == "'" and value[k + 1: k + 2] == "'":
                k += 2
                continue
            return k
        k += 1
    return len(value)


def _scalar(value: str) -> Any:
    """A YAML scalar as PyYAML reads the simple cases: quotes removed, a trailing comment dropped."""
    value = value.strip()
    if value[:1] == '"':
        try:
            return json.loads(value[: _closing_quote(value) + 1])
        except ValueError:
            return value.strip('"')
    if value[:1] == "'":
        return value[1:_closing_quote(value)].replace("''", "'")
    value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    return None if value in ("", "~", "null") else value


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Top-level keys and their direct ``key: scalar`` children: enough for ``model`` and ``memory`` when PyYAML is
    not installed (Hermes' own Python has it only inside its environment)."""
    out: dict[str, Any] = {}
    current: Optional[dict[str, Any]] = None
    child_indent: Optional[int] = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, rest = line.strip().partition(":")
        if not sep or key.startswith("-"):
            continue
        value = rest.strip()
        has_value = bool(value) and not value.startswith("#")
        if indent == 0:
            current, child_indent = None, None
            if has_value:
                out[key] = _scalar(value)
            else:
                current = out[key] = {}
        elif current is not None:
            child_indent = child_indent or indent
            if indent == child_indent:
                current[key] = _scalar(value) if has_value else {}
    return out


def read_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return parse_simple_yaml(text)
    loaded = yaml.safe_load(text)
    return loaded if isinstance(loaded, dict) else {}


# -- what this machine has ----------------------------------------------------------------------


def engine_missing(repo: Path) -> list[str]:
    """What this repository's embedding engine still needs (empty: it can serve EmbeddingGemma 300m)."""
    missing = []
    if not (repo / ".venv" / "bin" / "python").exists():
        missing.append(".venv/bin/python")
    if not (repo / "models" / "embeddinggemma-300m" / "model.safetensors").exists():
        missing.append("models/embeddinggemma-300m")
    if not list((repo / "build").glob("_egemma_native*.so")):
        missing.append("build/_egemma_native*.so")
    return missing


def bundled_node(root: Path) -> Optional[Path]:
    node = root / "node" / "bin" / "node"
    return node if node.is_file() and os.access(node, os.X_OK) else None


def is_loopback(url: str) -> bool:
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return False
    return host.lower() in LOOPBACK


# -- one profile ----------------------------------------------------------------------------------


class Plan:
    """A profile that passed the checks, with what will be written."""

    def __init__(self, name: str, home: Path, config: dict[str, Any], text: str, chat_url: str, chat_model: str):
        self.name, self.home, self.config, self.text = name, home, config, text
        self.chat_url, self.chat_model = chat_url, chat_model


def _memory_block(config: dict[str, Any]) -> dict[str, Any]:
    memory = config.get("memory")
    return memory if isinstance(memory, dict) else {}


def plan(name: str, root: Path, *, allow_remote_chat: bool = False, no_chat: bool = False) -> Plan:
    """Checks one profile without changing anything; raises SystemExit with what to do."""
    home = profile_home(name, root)
    config_path = home / "config.yaml"
    if not config_path.is_file():
        hint = "" if root.exists() else f" ({root} does not exist: pass the Hermes home with --home)"
        raise SystemExit(f"{name}: {config_path} not found{hint}. Create the profile in Hermes first "
                         "(start Hermes with it once), then run this again.")
    text = config_path.read_text(encoding="utf-8")
    config = read_config(config_path)
    provider = str(_memory_block(config).get("provider") or "")
    if provider and provider != "tsuzuki":
        raise SystemExit(f"{name}: memory.provider is already {provider!r}; change it by hand if you want tsuzuki")
    if "memory" in config and provider != "tsuzuki" and not re.search(r"^memory:[ \t]*(#.*)?$", text, re.M):
        raise SystemExit(f"{name}: the memory: block of {config_path} is not in block style; add "
                         "'provider: tsuzuki' to it by hand")
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    chat_url = "" if no_chat else str(model.get("base_url") or "").strip()
    chat_model = "" if no_chat else str(model.get("default") or model.get("model") or "").strip()
    if chat_url and not is_loopback(chat_url) and not allow_remote_chat:
        raise SystemExit(
            f"{name}: the profile's model endpoint {chat_url} is not on this machine, and fact extraction would send "
            "your conversations there. Re-run with --no-chat (memory without facts), or with --allow-remote-chat if "
            "you accept that, or point model.base_url at a local server first. Nothing was changed.")
    return Plan(name, home, config, text, chat_url, chat_model)


def install(p: Plan, root: Path, repo: Path = REPO) -> list[str]:
    """Applies a checked plan; returns the lines to print."""
    home, text = p.home, p.text
    config_path = home / "config.yaml"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = config_path.with_name(f"config.yaml.bak-tsuzuki-{stamp}")
    shutil.copy2(config_path, backup)
    out = [f"{p.name}: config backup {backup.name}"]

    target = home / "plugins" / "tsuzuki"
    target.mkdir(parents=True, exist_ok=True)
    for f in ("__init__.py", "plugin.yaml"):
        shutil.copy2(PLUGIN / f, target / f)
    out.append(f"  plugin -> {target}")

    node = bundled_node(root)
    if _memory_block(p.config).get("provider") != "tsuzuki":
        lines = ["  provider: tsuzuki", "  tsuzuki:", f"    server_js: {json.dumps(str(SERVER_JS))}"]
        if node:  # otherwise the provider finds node itself (Hermes' bundled one, then PATH)
            lines.append(f"    node: {json.dumps(str(node))}")
        insert = "\n".join(lines) + "\n"
        if re.search(r"^memory:[ \t]*(#.*)?$", text, re.M):
            text = re.sub(r"^memory:[ \t]*(#.*)?\n", lambda m: m.group(0) + insert, text, count=1, flags=re.M)
        else:
            text = text.rstrip("\n") + "\n\nmemory:\n" + insert
        config_path.write_text(text, encoding="utf-8")
        out.append("  config.yaml: memory.provider = tsuzuki")
    else:
        out.append("  config.yaml: memory.provider is already tsuzuki (kept)")
    if not node and not shutil.which("node"):
        out.append(f"  NOTE: no node found ({root / 'node/bin/node'} or on PATH). Install Node ^22.19 or >=24, or set "
                   "memory.tsuzuki.node in config.yaml.")

    data = home / "tsuzuki"
    data.mkdir(parents=True, exist_ok=True)
    sidecar = data / "config.json"
    if sidecar.exists():
        out.append(f"  sidecar config kept: {sidecar} (delete it and re-run to regenerate)")
        return out
    missing = engine_missing(repo)
    if missing:
        embedding: dict[str, Any] = {"baseUrl": "http://127.0.0.1:11434/v1", "model": "embeddinggemma"}
        out.append(f"  embeddings: Ollama embeddinggemma on 127.0.0.1:11434, because this repository's engine is not "
                   f"built (missing: {', '.join(missing)}). Run 'ollama pull embeddinggemma' (that accepts the Gemma "
                   "Terms of Use), or build the engine (README step 2), delete config.json and run this again.")
    else:
        embedding = {"engine": {"engineRoot": str(repo), "modelDir": str(repo / "models" / "embeddinggemma-300m"),
                                "modelName": "embeddinggemma-300m", "port": ENGINE_PORT}}
        out.append(f"  embeddings: EmbeddingGemma 300m on this repository's engine (started on demand, port {ENGINE_PORT})")
    port = profile_port(data)
    sidecar.write_text(json.dumps({
        "port": port, "dataDir": str(data), "embedding": embedding,
        "chat": {"baseUrl": p.chat_url, "model": p.chat_model, "server": "openai"},
        "recall": {"budgetTokens": 2048},
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out.append(f"  sidecar config: {sidecar} (port {port})")
    if p.chat_url:
        remote = "" if is_loopback(p.chat_url) else " (NOT on this machine: conversation text is sent there)"
        out.append(f"  facts: {p.chat_url}, model {p.chat_model or '(the first from /models)'}{remote}")
    else:
        out.append("  facts: off (no chat endpoint). Memory works on message windows only; set chat.baseUrl in "
                   "config.json to a local server for the tested setting.")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("profiles", nargs="+", help="'default' or a profile name under <home>/profiles")
    ap.add_argument("--home", type=Path, default=Path.home() / ".hermes", help="Hermes home (default ~/.hermes)")
    chat = ap.add_mutually_exclusive_group()
    chat.add_argument("--allow-remote-chat", action="store_true",
                      help="use the profile's model endpoint for facts even when it is not on this machine")
    chat.add_argument("--no-chat", action="store_true", help="no fact extraction (memory on message windows only)")
    args = ap.parse_args(argv)
    if not SERVER_JS.is_file():
        raise SystemExit(f"build the sidecar first (README step 1): {SERVER_JS} is missing")
    root = args.home.expanduser().resolve()
    plans = [plan(name, root, allow_remote_chat=args.allow_remote_chat, no_chat=args.no_chat) for name in args.profiles]
    for p in plans:
        print("\n".join(install(p, root)))
    print("Restart Hermes. A sidecar that is already running reads config.json only when it starts: stop it first "
          "(see integrations/hermes/README.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
