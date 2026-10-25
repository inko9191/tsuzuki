"""Tests for integrations/hermes/install.py on throwaway Hermes homes (stdlib only; nothing under ~/.hermes).

    python3 -I integrations/hermes/tests/test_install.py -v

Any Python 3.10+ works. With PyYAML installed, config.yaml is read with yaml.safe_load; without it (e.g. Hermes'
bundled Python, whose PyYAML lives only in its environment), with the script's small parser. Both are tested.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tsuzuki_install", HERE.parent / "install.py")
install = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install)

HAS_YAML = importlib.util.find_spec("yaml") is not None

CONFIG = """\
# Hermes profile
model:
  default: "qwen3.8-27b"   # quoted, as Hermes writes it
  provider: custom
  base_url: "{url}"
memory:
  memory_enabled: true
"""


class InstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="hermes-home-")).resolve()
        self.addCleanup(shutil.rmtree, self.root, True)
        if not install.SERVER_JS.is_file():
            self.skipTest(f"build the sidecar first: {install.SERVER_JS}")

    def profile(self, name: str, url: str = "http://127.0.0.1:8080/v1", text: str | None = None) -> Path:
        home = install.profile_home(name, self.root)
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(text if text is not None else CONFIG.format(url=url), encoding="utf-8")
        return home

    def run_main(self, *args: str) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(install.main(["--home", str(self.root), *args]), 0)
        return out.getvalue()

    def test_missing_config_yaml_is_a_clear_error(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            install.main(["--home", str(self.root / "nowhere"), "work"])
        self.assertIn("config.yaml not found", str(caught.exception))
        self.assertIn("--home", str(caught.exception))

    def test_quoted_values_and_the_profile_endpoint(self) -> None:
        home = self.profile("work")
        out = self.run_main("work")
        sidecar = json.loads((home / "tsuzuki" / "config.json").read_text())
        self.assertEqual(sidecar["chat"], {"baseUrl": "http://127.0.0.1:8080/v1", "model": "qwen3.8-27b", "server": "openai"})
        self.assertEqual(sidecar["port"], install.profile_port(home / "tsuzuki"))
        self.assertTrue((home / "plugins" / "tsuzuki" / "__init__.py").is_file())
        text = (home / "config.yaml").read_text()
        self.assertIn("memory:\n  provider: tsuzuki\n  tsuzuki:\n    server_js: ", text)
        self.assertIn("  memory_enabled: true", text, "existing settings are kept")
        self.assertIn("facts: http://127.0.0.1:8080/v1", out)
        if HAS_YAML:
            import yaml

            self.assertEqual(yaml.safe_load(text)["memory"]["provider"], "tsuzuki")
        again = self.run_main("work")  # re-running changes nothing that is set
        self.assertIn("already tsuzuki", again)
        self.assertIn("sidecar config kept", again)
        self.assertEqual(text.count("provider: tsuzuki"), 1)

    def test_a_remote_chat_endpoint_needs_consent(self) -> None:
        home = self.profile("cloud", url="https://api.example.com/v1")
        before = (home / "config.yaml").read_text()
        with self.assertRaises(SystemExit) as caught:
            install.main(["--home", str(self.root), "cloud"])
        self.assertIn("not on this machine", str(caught.exception))
        self.assertEqual((home / "config.yaml").read_text(), before, "nothing changed")
        self.assertFalse((home / "tsuzuki").exists())
        self.run_main("--no-chat", "cloud")
        self.assertEqual(json.loads((home / "tsuzuki" / "config.json").read_text())["chat"]["baseUrl"], "")
        other = self.profile("cloud2", url="https://api.example.com/v1")
        out = self.run_main("--allow-remote-chat", "cloud2")
        self.assertEqual(json.loads((other / "tsuzuki" / "config.json").read_text())["chat"]["baseUrl"],
                         "https://api.example.com/v1")
        self.assertIn("NOT on this machine", out)

    def test_one_bad_profile_changes_no_profile(self) -> None:
        good = self.profile("good")
        self.profile("bad", url="https://api.example.com/v1")
        with self.assertRaises(SystemExit):
            install.main(["--home", str(self.root), "good", "bad"])
        self.assertFalse((good / "tsuzuki").exists())
        self.assertNotIn("tsuzuki", (good / "config.yaml").read_text())

    def test_node_is_written_only_when_hermes_has_one(self) -> None:
        home = self.profile("default")
        self.run_main("default")
        self.assertNotIn("node:", (home / "config.yaml").read_text())
        node = self.root / "node" / "bin" / "node"
        node.parent.mkdir(parents=True)
        node.write_text("#!/bin/sh\n")
        node.chmod(0o755)
        other = self.profile("second")
        self.run_main("second")
        self.assertIn(f"    node: {json.dumps(str(node))}", (other / "config.yaml").read_text())

    def test_another_provider_is_not_replaced(self) -> None:
        self.profile("x", text="memory:\n  provider: honcho\n")
        with self.assertRaises(SystemExit) as caught:
            install.main(["--home", str(self.root), "x"])
        self.assertIn("honcho", str(caught.exception))

    def test_an_empty_provider_line_and_four_space_indent_are_handled(self) -> None:
        for name, block in (("empty", "memory:\n  memory_enabled: true\n  provider: ''\n  nudge_interval: 10\n"),
                            ("wide", "memory:\n    memory_enabled: true\n    nudge_interval: 10\n")):
            home = self.profile(name, text="model:\n  base_url: \"http://127.0.0.1:8080/v1\"\n" + block)
            self.run_main(name)
            config = install.read_config_text((home / "config.yaml").read_text())
            self.assertEqual(config["memory"]["provider"], "tsuzuki", name)
            self.assertEqual(config["memory"]["memory_enabled"], True, name)
            self.assertEqual(config["memory"]["nudge_interval"], 10, name)
            self.assertEqual((home / "config.yaml").read_text().count("provider:"), 1, name)
            sidecar = home / "tsuzuki" / "config.json"
            self.assertEqual(sidecar.stat().st_mode & 0o777, 0o600)
            self.assertRegex(json.loads(sidecar.read_text())["token"], r"^[0-9a-f]{48}$")

    def test_llamacpp_embeddings_and_a_local_url_only(self) -> None:
        home = self.profile("lc")
        out = self.run_main("--embedding", "llamacpp", "lc")
        config = json.loads((home / "tsuzuki" / "config.json").read_text())
        self.assertEqual(config["embedding"], {"baseUrl": "http://127.0.0.1:8081/v1", "model": "embeddinggemma"})
        self.assertIn("llama.cpp", out)
        other = self.profile("lc2")
        self.run_main("--embedding", "llamacpp", "--embedding-url", "http://localhost:9000/v1", "lc2")
        self.assertEqual(json.loads((other / "tsuzuki" / "config.json").read_text())["embedding"]["baseUrl"],
                         "http://localhost:9000/v1")
        self.profile("far")
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            install.main(["--home", str(self.root), "--embedding", "llamacpp", "--embedding-url", "http://192.168.1.5:8081/v1", "far"])
        self.assertFalse((install.profile_home("far", self.root) / "tsuzuki").exists())

    def test_embeddings_fall_back_to_ollama_without_the_engine(self) -> None:
        repo = Path(tempfile.mkdtemp(prefix="repo-"))
        self.addCleanup(shutil.rmtree, repo, True)
        self.assertEqual(install.engine_missing(repo), [".venv/bin/python", "models/embeddinggemma-300m",
                                                        "build/_egemma_native*.so"])
        home = self.profile("work")
        lines = install.install(install.plan("work", self.root), self.root, repo=repo)
        sidecar = json.loads((home / "tsuzuki" / "config.json").read_text())
        self.assertEqual(sidecar["embedding"], {"baseUrl": "http://127.0.0.1:11434/v1", "model": "embeddinggemma"})
        self.assertTrue(any("Ollama" in line and ".venv/bin/python" in line for line in lines), lines)
        for path in (".venv/bin/python", "models/embeddinggemma-300m/model.safetensors", "build/_egemma_native.so"):
            (repo / path).parent.mkdir(parents=True, exist_ok=True)
            (repo / path).write_text("")
        self.assertEqual(install.engine_missing(repo), [])
        other = self.profile("engine")
        install.install(install.plan("engine", self.root), self.root, repo=repo)
        engine = json.loads((other / "tsuzuki" / "config.json").read_text())["embedding"]["engine"]
        self.assertEqual(engine["modelName"], "embeddinggemma-300m")
        self.assertEqual(engine["engineRoot"], str(repo))


class SimpleYamlTests(unittest.TestCase):
    """The fallback reader (used when PyYAML is missing) agrees with PyYAML on what install.py reads."""

    SAMPLES = [
        CONFIG.format(url="http://127.0.0.1:8080/v1"),
        "model:\n  default: 'it''s-a-model'\n  base_url: http://localhost:11434/v1  # ollama\nmemory:\n  provider: tsuzuki\n",
        "model:\n    default: \"m \\\"q\\\"\"\n    base_url: ''\nagent:\n  max_turns: 90\n",
        "toolsets:\n  - web\nmodel: plain-string\n",
    ]

    def test_fallback_matches_expected_values(self) -> None:
        first = install.parse_simple_yaml(self.SAMPLES[0])
        self.assertEqual(first["model"]["default"], "qwen3.8-27b")
        self.assertEqual(first["model"]["base_url"], "http://127.0.0.1:8080/v1")
        second = install.parse_simple_yaml(self.SAMPLES[1])
        self.assertEqual(second["model"], {"default": "it's-a-model", "base_url": "http://localhost:11434/v1"})
        self.assertEqual(second["memory"]["provider"], "tsuzuki")
        third = install.parse_simple_yaml(self.SAMPLES[2])
        self.assertEqual(third["model"], {"default": 'm "q"', "base_url": ""})
        self.assertEqual(install.parse_simple_yaml(self.SAMPLES[3])["model"], "plain-string")

    @unittest.skipUnless(HAS_YAML, "PyYAML not importable")
    def test_fallback_agrees_with_pyyaml(self) -> None:
        import yaml

        for text in self.SAMPLES:  # the string fields install.py reads (model.*, memory.provider)
            ours, theirs = install.parse_simple_yaml(text), yaml.safe_load(text)
            for key in ("model", "memory"):
                if isinstance(theirs.get(key), dict):
                    for field, value in theirs[key].items():
                        if isinstance(value, str):
                            self.assertEqual(ours[key][field], value, (text, field))
                elif key in theirs:
                    self.assertEqual(ours[key], theirs[key])


if __name__ == "__main__":
    unittest.main()
