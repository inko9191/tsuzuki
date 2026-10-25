"""Tests for the tsuzuki Hermes memory provider, against a fake sidecar.

Run with Hermes' own Python (stdlib unittest; pytest also works if installed), any version under ~/.hermes/tools:

    "$(ls -d ~/.hermes/tools/python-*/bin/python3 | head -1)" -I integrations/hermes/tests/test_tsuzuki.py -v

The real ``MemoryProvider`` ABC is imported from the Hermes checkout (read-only; set
HERMES_AGENT_ROOT to use another one). When Hermes' committed dependency environment can be
found, its site-packages are added too and the MemoryManager / plugin-loader tests run;
otherwise those are skipped. Nothing is written under ~/.hermes: bytecode writing is off and
HERMES_HOME points at a temporary directory for the whole run.
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True  # never drop __pycache__ into the Hermes checkout

import inspect
import json
import shutil
import signal
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
INTEGRATION_DIR = HERE.parent
PLUGIN_SRC = INTEGRATION_DIR / "tsuzuki"
HERMES_AGENT = Path(os.environ.get("HERMES_AGENT_ROOT") or Path.home() / ".hermes" / "hermes-agent").resolve()


def _bootstrap_hermes() -> Optional[str]:
    """Put the Hermes checkout (and, read-only, its committed site-packages) on sys.path.
    Returns why the full Hermes runtime is unavailable, or None when it is."""
    if not (HERMES_AGENT / "agent" / "memory_provider.py").is_file():
        raise SystemExit(f"Hermes checkout not found at {HERMES_AGENT} (set HERMES_AGENT_ROOT)")
    sys.path.insert(0, str(HERMES_AGENT))
    try:
        import site

        from pm.environments import committed_venv, site_packages  # stdlib-only, reads facts.json

        venv = committed_venv(HERMES_AGENT)
        if venv is None or not site_packages(venv).is_dir():
            return "no committed Hermes dependency environment"
        site.addsitedir(str(site_packages(venv)))
        return None
    except Exception as exc:  # pragma: no cover - depends on the local install
        return f"Hermes dependency environment unavailable: {exc}"


_HERMES_RUNTIME_MISSING = _bootstrap_hermes()

# Everything below runs against a throwaway HERMES_HOME.
_SESSION_HOME = Path(tempfile.mkdtemp(prefix="tsuzuki-hermes-home-")).resolve()
os.environ["HERMES_HOME"] = str(_SESSION_HOME)
for _var in ("TSUZUKI_NODE", "TSUZUKI_SERVER_JS"):
    os.environ.pop(_var, None)

from agent.memory_provider import MemoryProvider  # noqa: E402  (the real ABC)

sys.path.insert(0, str(INTEGRATION_DIR))
import tsuzuki as em  # noqa: E402

_ORIGINAL_CONSTANTS = {name: getattr(em, name) for name in (
    "PREFETCH_TIMEOUT_S", "QUEUED_RECALL_TIMEOUT_S", "STARTUP_TIMEOUT_S", "DRAIN_TIMEOUT_S", "PLUGIN_DIR",
    "RECALL_BREAKER_S", "SEARCH_TIMEOUT_S", "RESTART_INTERVAL_S", "EMBEDDING_CHECK_S")}


# -- fake sidecar -------------------------------------------------------------------------------


class FakeSidecar:
    """Stdlib HTTP server implementing the sidecar contract, recording every request."""

    def __init__(self, port: int = 0) -> None:
        self.requests: list[tuple[str, str, Any]] = []
        self.lock = threading.Lock()
        self.recall_delay = 0.0
        self.search_delay = 0.0
        # (status, body) answered instead of a result, e.g. the sidecar's 503 when its embedding model is down.
        self.recall_error: Optional[tuple[int, dict[str, Any]]] = None
        self.search_error: Optional[tuple[int, dict[str, Any]]] = None
        self.health_extra: dict[str, Any] = {}
        self.forget_reply: tuple[int, dict[str, Any]] = (
            200, {"turns": 2, "units": 3, "facts": 1, "segments": 1, "links": 0, "vacuumed": True})
        sidecar = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _reply(self, obj: Any, status: int = 200) -> None:
                body = json.dumps(obj, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                sidecar._record("GET", self.path, None)
                if self.path == "/health":
                    self._reply({"status": "ok", "version": "fake", "stats": {"messages": 0}, **sidecar.health_extra})
                else:
                    self._reply({"error": "not found"}, 404)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                sidecar._record("POST", self.path, payload)
                if self.path == "/v1/memory/messages":
                    self._reply({"stored": len(payload.get("messages", []))})
                elif self.path in ("/v1/memory/recall", "/v1/memory/search"):
                    if self.path.endswith("recall") and sidecar.recall_delay:
                        time.sleep(sidecar.recall_delay)
                    if self.path.endswith("search") and sidecar.search_delay:
                        time.sleep(sidecar.search_delay)
                    error = sidecar.recall_error if self.path.endswith("recall") else sidecar.search_error
                    if error is not None:
                        self._reply(error[1], error[0])
                        return
                    query = payload.get("query", "")
                    kind = self.path.rsplit("/", 1)[-1]
                    if query == "nothing":
                        self._reply({"text": "", "ids": [], "reranked": False})
                    else:
                        self._reply({"text": f"<memory>{kind}:{query}</memory>", "ids": ["m1", "m2"],
                                     "reranked": kind == "search"})
                elif self.path == "/v1/memory/flush":
                    self._reply({"ok": True})
                elif self.path == "/v1/memory/forget":
                    status, body = sidecar.forget_reply
                    self._reply(body, status)
                else:
                    self._reply({"error": "not found"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05},
                                       daemon=True)
        self.thread.start()

    def _record(self, method: str, path: str, payload: Any) -> None:
        with self.lock:
            self.requests.append((method, path, payload))

    def calls(self, path: str) -> list[Any]:
        with self.lock:
            return [payload for _, p, payload in self.requests if p == path]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


FAKE_SPAWNED_SIDECAR = r'''
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
config = json.load(open(sys.argv[sys.argv.index("--config") + 1]))

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        body = json.dumps({"status": "ok", "version": "fake-spawned", "stats": {}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

server = ThreadingHTTPServer(("127.0.0.1", int(config["port"])), Handler)
print("booting", file=sys.stderr, flush=True)
print(json.dumps({"event": "ready", "port": server.server_address[1]}), flush=True)
server.serve_forever()
'''


FAKE_RECORDING_SIDECAR = r'''
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
path = sys.argv[sys.argv.index("--config") + 1]
config = json.load(open(path))
log = os.path.join(os.path.dirname(path), "posts.jsonl")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def _reply(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        self._reply({"status": "ok", "version": "fake-recording", "stats": {}, "pid": os.getpid()})
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        with open(log, "a") as f:
            f.write(json.dumps({"pid": os.getpid(), "path": self.path, "payload": payload}) + "\n")
        self._reply({"stored": len(payload.get("messages", []))})

server = ThreadingHTTPServer(("127.0.0.1", int(config["port"])), Handler)
print(json.dumps({"event": "ready", "port": server.server_address[1]}), flush=True)
server.serve_forever()
'''


FAKE_SPAWNED_SIDECAR_JS = r'''
const http = require("http");
const fs = require("fs");
const config = JSON.parse(fs.readFileSync(process.argv[process.argv.indexOf("--config") + 1], "utf8"));
const server = http.createServer((req, res) => {
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify({ status: "ok", version: "fake-node", stats: {} }));
});
server.listen(config.port, "127.0.0.1", () => {
  console.log(JSON.stringify({ event: "ready", port: server.address().port }));
});
'''


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class FakeCtx:
    def __init__(self) -> None:
        self.providers: list[Any] = []

    def register_memory_provider(self, provider: Any) -> None:
        self.providers.append(provider)


# -- base -------------------------------------------------------------------------------------


class SidecarTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.sidecar = FakeSidecar()
        self.addCleanup(self.sidecar.close)
        self.home = Path(tempfile.mkdtemp(prefix="tsuzuki-profile-"))
        self.addCleanup(shutil.rmtree, self.home, True)
        self.write_config(self.sidecar.port)
        # Paths that exist, so is_available() is true; nothing is spawned while the fake is up.
        self.server_js = self.home / "server.js"
        self.server_js.write_text("// placeholder\n")
        self.set_env("TSUZUKI_NODE", sys.executable)
        self.set_env("TSUZUKI_SERVER_JS", str(self.server_js))
        self.addCleanup(self._restore_constants)
        self.providers: list[em.TsuzukiProvider] = []

    def _restore_constants(self) -> None:
        for p in self.providers:
            p.shutdown()
        for name, value in _ORIGINAL_CONSTANTS.items():
            setattr(em, name, value)

    def set_env(self, key: str, value: Optional[str]) -> None:
        old = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
        self.addCleanup(lambda: os.environ.__setitem__(key, old) if old is not None else os.environ.pop(key, None))

    def write_config(self, port: int, home: Optional[Path] = None) -> Path:
        data_dir = (home or self.home) / "tsuzuki"
        data_dir.mkdir(parents=True, exist_ok=True)
        config = em.default_sidecar_config(data_dir)
        config["port"] = port
        path = data_dir / "config.json"
        path.write_text(json.dumps(config))
        return path

    def make(self, session_id: str = "s1", **kwargs: Any) -> em.TsuzukiProvider:
        provider = em.TsuzukiProvider()
        kwargs.setdefault("hermes_home", str(self.home))
        kwargs.setdefault("platform", "cli")
        provider.initialize(session_id, **kwargs)
        provider._forget_enabled = True  # memory.tsuzuki.forget: true (off by default; see test_forget_is_off_by_default)
        self.providers.append(provider)
        return provider


# -- tests --------------------------------------------------------------------------------------


class RegistrationTests(unittest.TestCase):
    def test_register_registers_fresh_memory_provider_instances(self) -> None:
        self.assertEqual(MemoryProvider.__module__, "agent.memory_provider")
        self.assertTrue(Path(inspect.getfile(MemoryProvider)).resolve().is_relative_to(HERMES_AGENT))
        ctx = FakeCtx()
        em.register(ctx)
        em.register(ctx)
        self.assertEqual(len(ctx.providers), 2)
        first, second = ctx.providers
        self.assertIsNot(first, second)
        for provider in ctx.providers:
            self.assertIsInstance(provider, MemoryProvider)
            self.assertFalse(inspect.isabstract(type(provider)))
            self.assertEqual(provider.name, "tsuzuki")

    def test_hook_signatures_match_the_abc(self) -> None:
        for hook in ("initialize", "prefetch", "queue_prefetch", "sync_turn", "handle_tool_call",
                     "on_session_end", "on_session_switch", "shutdown", "system_prompt_block",
                     "get_tool_schemas", "recall_status", "is_available", "unavailable_reason"):
            ours = inspect.signature(getattr(em.TsuzukiProvider, hook))
            theirs = inspect.signature(getattr(MemoryProvider, hook))
            self.assertEqual(list(ours.parameters), list(theirs.parameters), hook)

    def test_plugin_yaml_names_the_provider(self) -> None:
        meta = em._read_plugin_yaml(PLUGIN_SRC / "plugin.yaml")
        self.assertEqual(meta["name"], "tsuzuki")
        self.assertEqual(meta["kind"], "exclusive")
        self.assertIn("server_js", meta)


class AvailabilityTests(SidecarTestCase):
    def test_available_with_node_and_server_js_and_no_network(self) -> None:
        def no_network(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("is_available() must not touch the network")

        original = em._http_json
        em._http_json = no_network
        try:
            provider = em.TsuzukiProvider()
            self.assertTrue(provider.is_available())
            self.assertEqual(provider.unavailable_reason(), "")
        finally:
            em._http_json = original
        self.assertEqual(self.sidecar.requests, [])

    def test_unavailable_without_server_js(self) -> None:
        self.set_env("TSUZUKI_SERVER_JS", None)
        em.PLUGIN_DIR = self.home  # no plugin.yaml there
        provider = em.TsuzukiProvider()
        self.assertFalse(provider.is_available())
        self.assertIn("TSUZUKI_SERVER_JS", provider.unavailable_reason())

    def test_unavailable_with_missing_paths(self) -> None:
        self.set_env("TSUZUKI_SERVER_JS", str(self.home / "missing.js"))
        self.set_env("TSUZUKI_NODE", str(self.home / "no-such-node"))
        provider = em.TsuzukiProvider()
        self.assertFalse(provider.is_available())
        reason = provider.unavailable_reason()
        self.assertIn("missing.js", reason)
        self.assertIn("node executable not found", reason)

    def test_server_js_from_plugin_yaml_relative_to_plugin_dir(self) -> None:
        self.set_env("TSUZUKI_SERVER_JS", None)
        plugin_dir = self.home / "plugin"
        (plugin_dir / "dist").mkdir(parents=True)
        (plugin_dir / "dist" / "server.js").write_text("//\n")
        (plugin_dir / "plugin.yaml").write_text(
            'name: tsuzuki\nserver_js: "dist/server.js"  # built sidecar\nnode: ""\n')
        em.PLUGIN_DIR = plugin_dir
        node, server_js, reason = em._resolve_launch()
        self.assertEqual(server_js, str(plugin_dir / "dist" / "server.js"))
        self.assertEqual(node, sys.executable)
        self.assertEqual(reason, "")

    @unittest.skipUnless(__import__("importlib.util").util.find_spec("yaml"), "PyYAML not importable")
    def test_server_js_from_the_initialized_profile_config(self) -> None:
        """A gateway serving several profiles: the active config has no block, the profile's own does."""
        self.set_env("TSUZUKI_SERVER_JS", None)
        em.PLUGIN_DIR = self.home / "no-plugin-yaml"
        profile = self.home / "profiles" / "work"
        profile.mkdir(parents=True)
        script = profile / "server.js"
        script.write_text("//\n")
        (profile / "config.yaml").write_text(
            f"memory:\n  provider: tsuzuki\n  tsuzuki:\n    server_js: {script}\n    node: ''\n")
        self.assertIsNone(em._resolve_launch()[1])
        node, server_js, reason = em._resolve_launch(str(profile))
        self.assertEqual(server_js, str(script))
        self.assertEqual(reason, "")

    def test_default_config_is_created_once(self) -> None:
        data_dir = self.home / "fresh" / "tsuzuki"
        path, config = em._load_or_create_sidecar_config(data_dir)
        self.assertEqual(config["port"], em.profile_port(data_dir))
        self.assertTrue(8297 <= config["port"] < 8797)
        self.assertNotEqual(em.profile_port(data_dir), em.profile_port(self.home / "other" / "tsuzuki"))
        self.assertEqual(config["dataDir"], str(data_dir))
        self.assertEqual(config["embedding"]["model"], "embeddinggemma")
        self.assertEqual(config["embedding"]["queryPrefix"], "task: search result | query: ")
        self.assertEqual(config["embedding"]["documentPrefix"], "title: none | text: ")
        self.assertEqual(config["recall"]["budgetTokens"], 2048)
        path.write_text(json.dumps({"port": 9999}))
        _, again = em._load_or_create_sidecar_config(data_dir)
        self.assertEqual(again, {"port": 9999})  # an existing config is never overwritten


class InitializeTests(SidecarTestCase):
    def test_initialize_uses_running_sidecar(self) -> None:
        provider = self.make(agent_context="primary", user_id="u1", cwd="/tmp")
        self.assertIsNone(provider.spawned_pid)
        self.assertFalse(provider.read_only)
        self.assertEqual(self.sidecar.requests[0][:2], ("GET", "/health"))
        self.assertIn("<memory-context>", provider.system_prompt_block())
        self.assertIn("tsuzuki_search", provider.system_prompt_block())

    def test_sidecar_of_another_profile_is_not_used(self) -> None:
        provider = self.make(agent_context="primary")
        self.sidecar.health_extra = {"dataDir": str(self.home / "someone-else" / "tsuzuki")}
        self.assertFalse(provider._healthy())
        self.sidecar.health_extra = {"dataDir": str(provider._data_dir)}
        self.assertTrue(provider._healthy())

    def test_initialize_spawns_detached_sidecar_when_down(self) -> None:
        port = _free_port()
        self.write_config(port)
        script = self.home / "fake_sidecar.py"
        script.write_text(FAKE_SPAWNED_SIDECAR)
        self.set_env("TSUZUKI_SERVER_JS", str(script))  # node := this Python
        provider = self.make()
        pid = provider.spawned_pid
        self.assertIsNotNone(pid)
        self.addCleanup(self._kill, provider)
        self.assertEqual(os.getpgid(pid), pid)  # own session/process group: survives Hermes' terminal
        self.assertEqual(provider._base_url, f"http://127.0.0.1:{port}")
        self.assertTrue(provider._healthy())
        log = (self.home / "tsuzuki" / "sidecar.log").read_text()
        self.assertIn('"event": "ready"', log)
        self.assertIn("booting", log)  # stderr goes to the same log

        # A second Hermes process finds it running and does not spawn another one.
        second = self.make()
        self.assertIsNone(second.spawned_pid)

        # shutdown() leaves the shared sidecar running.
        provider.shutdown()
        os.kill(pid, 0)
        self.assertTrue(second._healthy())

    @unittest.skipUnless(em._find_node(""), "no bundled ~/.hermes/node/bin/node and no node on PATH")
    def test_spawn_with_default_node(self) -> None:
        self.set_env("TSUZUKI_NODE", None)  # default resolution: Hermes' bundled node first
        port = _free_port()
        self.write_config(port)
        script = self.home / "fake_sidecar.js"
        script.write_text(FAKE_SPAWNED_SIDECAR_JS)
        self.set_env("TSUZUKI_SERVER_JS", str(script))
        provider = self.make()
        self.addCleanup(self._kill, provider)
        self.assertIsNotNone(provider.spawned_pid)
        self.assertEqual(provider._node, em._find_node(""))
        self.assertTrue(provider._healthy())
        self.assertIn('"event":"ready"', (self.home / "tsuzuki" / "sidecar.log").read_text())

    def test_initialize_survives_unstartable_sidecar(self) -> None:
        self.write_config(_free_port())
        script = self.home / "crash.py"
        script.write_text("import sys; sys.exit(3)\n")
        self.set_env("TSUZUKI_SERVER_JS", str(script))
        em.STARTUP_TIMEOUT_S = 5.0
        started = time.monotonic()
        provider = self.make()
        self.assertLess(time.monotonic() - started, 5.0)
        self.addCleanup(self._kill, provider)
        self.assertEqual(provider.prefetch("anything"), "")

    def test_without_fcntl_the_sidecar_is_started_all_the_same(self) -> None:
        """Windows has no fcntl: initialize() starts the sidecar without the lock instead of never starting it."""
        port = _free_port()
        self.write_config(port)
        script = self.home / "fake_sidecar.py"
        script.write_text(FAKE_SPAWNED_SIDECAR)
        self.set_env("TSUZUKI_SERVER_JS", str(script))
        real = sys.modules.get("fcntl")
        sys.modules["fcntl"] = None  # type: ignore[assignment]  # `import fcntl` now fails, as on Windows
        self.addCleanup(lambda: sys.modules.__setitem__("fcntl", real) if real is not None
                        else sys.modules.pop("fcntl", None))
        with em._file_lock(self.home / "unused.lock", 0.1) as locked:
            self.assertTrue(locked)
        provider = self.make()
        self.addCleanup(self._kill, provider)
        self.assertIsNotNone(provider.spawned_pid)
        self.assertEqual(provider._base_url, f"http://127.0.0.1:{port}")
        self.assertTrue(provider._healthy())

    @staticmethod
    def _kill(provider: em.TsuzukiProvider) -> None:
        pid = provider.spawned_pid
        if pid is None:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            pass
        if provider._proc is not None:
            provider._proc.wait(timeout=5)


class ChildEnvTests(unittest.TestCase):
    """The sidecar's environment: what Node needs on each platform, never API keys or proxy settings."""

    def child_env(self, environ: dict[str, str], node: Optional[str] = None) -> dict[str, str]:
        provider = em.TsuzukiProvider()
        provider._hermes_home = "/hermes/profiles/p"
        provider._node = node
        with mock.patch.dict(os.environ, environ, clear=True):
            return provider._child_env()

    def test_windows_variables_pass_and_api_keys_and_proxies_do_not(self) -> None:
        # Upper-case, as os.environ has them on Windows (SystemRoot, Path, Temp, ...).
        needed = {"SYSTEMROOT": r"C:\Windows", "USERPROFILE": r"C:\Users\me",
                  "TEMP": r"C:\Users\me\AppData\Local\Temp", "TMP": r"C:\Users\me\AppData\Local\Temp",
                  "PATH": r"C:\Windows\system32;C:\Program Files\nodejs"}
        not_needed = {"WINDIR": r"C:\Windows", "APPDATA": r"C:\Users\me\AppData\Roaming",
                      "LOCALAPPDATA": r"C:\Users\me\AppData\Local", "HOMEDRIVE": "C:", "HOMEPATH": r"\Users\me",
                      "PATHEXT": ".COM;.EXE;.BAT;.CMD", "COMSPEC": r"C:\Windows\system32\cmd.exe",
                      "PROGRAMDATA": r"C:\ProgramData", "USERNAME": "me"}
        secrets = {"OPENAI_API_KEY": "sk-test", "ANTHROPIC_API_KEY": "sk-ant-test", "HTTPS_PROXY": "http://proxy:3128",
                   "HTTP_PROXY": "http://proxy:3128", "ALL_PROXY": "socks5://proxy:1080", "NO_PROXY": "*"}
        env = self.child_env({**needed, **not_needed, **secrets})
        self.assertEqual(env, {**needed, "HERMES_HOME": "/hermes/profiles/p"})

    def test_posix_environment_is_unchanged(self) -> None:
        kept = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/home/me", "USER": "me", "LOGNAME": "me",
                "LANG": "ja_JP.UTF-8", "LC_ALL": "ja_JP.UTF-8", "LC_CTYPE": "UTF-8", "TMPDIR": "/var/folders/xy/T/",
                "TZ": "Asia/Tokyo", "SHELL": "/bin/zsh", "TSUZUKI_NODE": "/opt/node/bin/node",
                "NODE_OPTIONS": "--max-old-space-size=4096"}
        dropped = {"OPENAI_API_KEY": "sk-test", "HTTPS_PROXY": "http://proxy:3128", "https_proxy": "http://proxy:3128",
                   "no_proxy": "*", "AWS_SECRET_ACCESS_KEY": "secret", "PYTHONPATH": "/elsewhere"}
        node = "/opt/node/bin/node"
        node_dir = str(Path(node).parent)
        env = self.child_env({**kept, **dropped}, node=node)
        self.assertEqual(env, {**kept, "PATH": node_dir + os.pathsep + kept["PATH"], "HERMES_HOME": "/hermes/profiles/p"})
        self.assertEqual(self.child_env({}, node=node)["PATH"], node_dir + os.pathsep + "/usr/bin:/bin")


class PrefetchTests(SidecarTestCase):
    def test_queue_prefetch_result_is_consumed_by_prefetch(self) -> None:
        provider = self.make("s1")
        provider.queue_prefetch("hello", session_id="s1")
        self.assertTrue(_wait_until(lambda: all(p.done.is_set() for p in provider._cache.values())))
        self.assertEqual(len(self.sidecar.calls("/v1/memory/recall")), 1)
        self.assertEqual(provider.prefetch("hello", session_id="s1"), "<memory>recall:hello</memory>")
        self.assertEqual(len(self.sidecar.calls("/v1/memory/recall")), 1)  # served from the cache
        self.assertEqual(self.sidecar.calls("/v1/memory/recall")[0], {"query": "hello", "session": "s1"})
        status = provider.recall_status()
        self.assertIsNotNone(status)
        self.assertEqual(status.count, 2)

    def test_prefetch_without_cache_recalls_synchronously(self) -> None:
        provider = self.make("s1")
        provider.queue_prefetch("previous turn", session_id="s1")
        self.assertEqual(provider.prefetch("new question", session_id="s1"),
                         "<memory>recall:new question</memory>")
        self.assertIn({"query": "new question", "session": "s1"}, self.sidecar.calls("/v1/memory/recall"))
        # session falls back to the one from initialize()
        provider.prefetch("third")
        self.assertIn({"query": "third", "session": "s1"}, self.sidecar.calls("/v1/memory/recall"))

    def test_empty_recall_returns_empty_and_clears_indicator(self) -> None:
        provider = self.make()
        self.assertTrue(provider.prefetch("something"))
        self.assertIsNotNone(provider.recall_status())
        self.assertEqual(provider.prefetch("nothing"), "")
        self.assertIsNone(provider.recall_status())
        self.assertEqual(provider.prefetch("   "), "")

    def test_prefetch_times_out_to_empty(self) -> None:
        em.PREFETCH_TIMEOUT_S = 0.5
        self.sidecar.recall_delay = 2.0
        provider = self.make()
        started = time.monotonic()
        self.assertEqual(provider.prefetch("slow"), "")
        self.assertLess(time.monotonic() - started, 1.5)

    def test_prefetch_waits_for_in_flight_queued_recall_within_budget(self) -> None:
        em.PREFETCH_TIMEOUT_S = 0.5
        self.sidecar.recall_delay = 2.0
        provider = self.make("s1")
        provider.queue_prefetch("slow", session_id="s1")
        started = time.monotonic()
        self.assertEqual(provider.prefetch("slow", session_id="s1"), "")
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(len(self.sidecar.calls("/v1/memory/recall")), 1)  # no duplicate request

    def test_recall_timeout_opens_the_breaker_and_later_turns_do_not_wait(self) -> None:
        """R5: after one timed-out recall, turns skip the synchronous recall; a background success closes it."""
        em.PREFETCH_TIMEOUT_S = 0.4
        self.sidecar.recall_delay = 1.0
        provider = self.make("s1")
        started = time.monotonic()
        with self.assertLogs(em.logger, level="WARNING") as logs:
            self.assertEqual(provider.prefetch("first"), "")
        self.assertLess(time.monotonic() - started, 0.9)
        self.assertTrue(any("skip memory recall for 60s" in line for line in logs.output), logs.output)
        recalls = len(self.sidecar.calls("/v1/memory/recall"))
        for query in ("second", "third"):
            started = time.monotonic()
            self.assertEqual(provider.prefetch(query), "")
            self.assertLess(time.monotonic() - started, 0.1, "no wait while the breaker is open")
        self.assertEqual(len(self.sidecar.calls("/v1/memory/recall")), recalls, "no synchronous recall sent")
        self.sidecar.recall_delay = 0.0
        provider.queue_prefetch("fourth")  # background recalls go on, and their success closes the breaker
        self.assertTrue(_wait_until(lambda: all(p.done.is_set() for p in provider._cache.values())))
        self.assertEqual(provider.prefetch("fifth"), "<memory>recall:fifth</memory>")

    def test_embedding_timeout_in_the_sidecar_warns_once_and_opens_the_breaker(self) -> None:
        """R5, R7: the sidecar's 503 for a hung embedding model is one WARNING, and later turns do not wait."""
        provider = self.make("s1")
        self.sidecar.recall_error = (503, {"error": "the embedding model did not answer the query within 3000 ms",
                                           "cause": "embedding-timeout"})
        with self.assertLogs(em.logger, level="DEBUG") as logs:
            self.assertEqual(provider.prefetch("first"), "")
            self.assertEqual(provider.prefetch("second"), "")
        warnings = [line for line in logs.output if line.startswith("WARNING")]
        self.assertEqual(len(warnings), 1, logs.output)
        self.assertIn("cannot use its embedding model", warnings[0])
        self.assertIn("within 3000 ms", warnings[0])
        self.assertEqual(len(self.sidecar.calls("/v1/memory/recall")), 1)

    def test_unreachable_embedding_model_is_reported_once_at_start(self) -> None:
        """R7: the sidecar runs, but its embedding model does not answer: say so instead of staying silent."""
        reason = "POST http://127.0.0.1:11434/v1/embeddings: fetch failed (ECONNREFUSED)"
        self.sidecar.health_extra = {"embedding": {"ok": False, "lastError": reason}}
        with self.assertLogs(em.logger, level="WARNING") as logs:
            self.make("s1")
            self.assertTrue(_wait_until(lambda: any("11434" in line for line in logs.output), timeout=3))
        self.assertEqual(len([line for line in logs.output if "embedding model" in line]), 1)
        self.sidecar.health_extra = {"embedding": {"ok": True, "lastError": None}}
        with self.assertNoLogs(em.logger, level="WARNING"):
            self.make("s2")
            time.sleep(0.3)

    def test_prefetch_never_raises_when_sidecar_is_down(self) -> None:
        provider = self.make()
        self.sidecar.close()
        self.assertEqual(provider.prefetch("hello"), "")
        provider.queue_prefetch("hello")
        self.assertEqual(provider.prefetch("hello"), "")


class SyncTurnTests(SidecarTestCase):
    def test_sync_turn_posts_both_messages_in_order(self) -> None:
        provider = self.make("s1")
        before = int(time.time() * 1000)
        provider.sync_turn("覚えておいて: キャッシュは 5 分", "了解しました。", session_id="s1",
                           messages=[{"role": "user", "content": "x"}], turn_author={"id": "u"})
        provider.sync_turn("", "assistant only", session_id="s1")
        provider.sync_turn("   ", "", session_id="s1")  # nothing to store
        provider.sync_turn("second session", "ok", session_id="s2")
        provider.shutdown()  # drains the writer queue
        calls = self.sidecar.calls("/v1/memory/messages")
        self.assertEqual(len(calls), 3)
        first = calls[0]
        self.assertEqual(first["session"], "s1")
        self.assertEqual([(m["role"], m["text"]) for m in first["messages"]],
                         [("user", "覚えておいて: キャッシュは 5 分"), ("assistant", "了解しました。")])
        for message in first["messages"]:
            self.assertIsInstance(message["at"], int)
            self.assertGreaterEqual(message["at"], before)
            self.assertNotIn("id", message)
        self.assertEqual(calls[1]["messages"], [{"role": "assistant", "text": "assistant only",
                                                 "at": calls[1]["messages"][0]["at"]}])
        self.assertEqual(calls[2]["session"], "s2")

    def test_sync_turn_is_non_blocking_and_never_raises_when_down(self) -> None:
        provider = self.make()
        self.sidecar.close()
        em.DRAIN_TIMEOUT_S = 2.0
        started = time.monotonic()
        provider.sync_turn("u", "a", session_id="s1")
        self.assertLess(time.monotonic() - started, 0.5)
        provider.shutdown()

    def test_writes_reach_a_sidecar_started_again_after_it_died(self) -> None:
        """R10: the sidecar this process started dies; the next write starts it again and nothing is lost."""
        port = _free_port()
        self.write_config(port)
        script = self.home / "recording_sidecar.py"
        script.write_text(FAKE_RECORDING_SIDECAR)
        self.set_env("TSUZUKI_SERVER_JS", str(script))
        provider = self.make("s1")
        first = provider.spawned_pid
        self.assertIsNotNone(first)
        self.addCleanup(lambda: [self._kill_pid(pid) for pid in {first, provider.spawned_pid} if pid])
        os.kill(first, signal.SIGKILL)
        provider._proc.wait(timeout=5)
        provider.sync_turn("My bike is a red Bianchi.", "Noted.", session_id="s1")
        provider.sync_turn("It needs new brakes.", "OK.", session_id="s1")
        provider.shutdown()
        self.assertNotEqual(provider.spawned_pid, first, "a new sidecar was started")
        posts = [json.loads(line) for line in (self.home / "tsuzuki" / "posts.jsonl").read_text().splitlines()]
        texts = [m["text"] for p in posts if p["path"] == "/v1/memory/messages" for m in p["payload"]["messages"]]
        self.assertEqual(texts, ["My bike is a red Bianchi.", "Noted.", "It needs new brakes.", "OK."])

    def test_writes_are_kept_while_the_sidecar_cannot_start_and_sent_when_it_is_back(self) -> None:
        """R10: bounded resend in order once the sidecar answers again (here: restarted by someone else)."""
        provider = self.make("s1")
        port = self.sidecar.port
        self.sidecar.close()
        self.set_env("TSUZUKI_SERVER_JS", str(self.home / "missing.js"))
        provider._server_js = None  # cannot be started from here
        provider.sync_turn("first while down", "a1", session_id="s1")
        provider.sync_turn("second while down", "a2", session_id="s1")
        self.assertTrue(_wait_until(lambda: len(provider._retry) == 2))
        back = FakeSidecar(port)
        self.addCleanup(back.close)
        provider.sync_turn("after it is back", "a3", session_id="s1")
        provider.shutdown()
        texts = [m["text"] for p in back.calls("/v1/memory/messages") for m in p["messages"]]
        self.assertEqual(texts, ["first while down", "a1", "second while down", "a2", "after it is back", "a3"])
        self.assertEqual(len(provider._retry), 0)

    @staticmethod
    def _kill_pid(pid: int) -> None:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    def test_session_end_and_switch_flush_after_pending_writes(self) -> None:
        provider = self.make("s1")
        provider.sync_turn("u1", "a1", session_id="s1")
        provider.on_session_switch("s2", parent_session_id="s1", reset=False)
        provider.sync_turn("u2", "a2", session_id="s2")
        provider.on_session_end([])
        posts = [(path, payload) for method, path, payload in self.sidecar.requests if method == "POST"]
        self.assertEqual(posts, [
            ("/v1/memory/messages", {"session": "s1", "messages": posts[0][1]["messages"]}),
            ("/v1/memory/flush", {"session": "s1"}),
            ("/v1/memory/messages", {"session": "s2", "messages": posts[2][1]["messages"]}),
            ("/v1/memory/flush", {"session": "s2"}),
        ])


class ForgetSettingTests(SidecarTestCase):
    def test_forget_is_off_by_default(self) -> None:
        provider = em.TsuzukiProvider()
        self.providers.append(provider)
        with mock.patch.object(em, "_profile_settings", lambda home=None: {}):
            provider.initialize("s1", hermes_home=str(self.home), platform="cli")
            self.assertEqual([t["name"] for t in provider.get_tool_schemas()], ["tsuzuki_search"])
            refused = json.loads(provider.handle_tool_call("tsuzuki_forget", {"all": True}))
        self.assertIn("turned off", refused["error"])
        self.assertEqual(self.sidecar.calls("/v1/memory/forget"), [])
        on = em.TsuzukiProvider()
        self.providers.append(on)
        with mock.patch.object(em, "_profile_settings", lambda home=None: {"forget": True}):
            self.assertIn("tsuzuki_forget", [t["name"] for t in on.get_tool_schemas()])

    def test_the_sidecar_token_is_sent(self) -> None:
        provider = self.make()
        provider._token = "t0ken"
        seen = []
        real = em._http_json
        with mock.patch.object(em, "_http_json", lambda *a, **k: (seen.append(k.get("token")), real(*a, **{**k, "token": ""}))[1]):
            provider._post("/v1/memory/flush", {}, 5.0)
        self.assertEqual(seen, ["t0ken"])


class ReadOnlyTests(SidecarTestCase):
    def test_non_primary_contexts_read_but_never_write(self) -> None:
        for context in ("subagent", "cron", "flush"):
            with self.subTest(context=context):
                provider = self.make("child", agent_context=context)
                self.assertTrue(provider.read_only)
                provider.sync_turn("user text", "assistant text", session_id="child")
                provider.on_session_switch("child-2")
                provider.on_session_end([])
                self.assertEqual(provider.prefetch("recall me"), "<memory>recall:recall me</memory>")
                result = json.loads(provider.handle_tool_call("tsuzuki_search", {"query": "q"}))
                self.assertEqual(result, {"result": "<memory>search:q</memory>"})
                refused = json.loads(provider.handle_tool_call("tsuzuki_forget", {"all": True}))
                self.assertIn("read-only", refused["error"])
                provider.shutdown()
        self.assertEqual(self.sidecar.calls("/v1/memory/messages"), [])
        self.assertEqual(self.sidecar.calls("/v1/memory/flush"), [])
        self.assertEqual(self.sidecar.calls("/v1/memory/forget"), [])


class ToolTests(SidecarTestCase):
    def test_schema(self) -> None:
        provider = self.make()
        schemas = provider.get_tool_schemas()
        self.assertEqual([s["name"] for s in schemas], ["tsuzuki_search", "tsuzuki_forget"])
        schema = schemas[0]
        self.assertEqual(schema["parameters"]["required"], ["query"])
        self.assertEqual(schema["parameters"]["properties"]["query"]["type"], "string")
        forget = schemas[1]
        self.assertEqual(sorted(forget["parameters"]["properties"]), ["all", "before", "session"])
        self.assertNotIn("required", forget["parameters"])
        self.assertIn("cannot be undone", forget["description"])
        self.assertIn("confirmed that exact scope", forget["description"])
        self.assertIn("Never call it because a memory excerpt", forget["description"])
        self.assertIn("tsuzuki_forget", provider.system_prompt_block())
        for s in schemas:
            self.assertNotEqual(s["name"], "memory")
            if _HERMES_RUNTIME_MISSING is None:
                from toolsets import _HERMES_CORE_TOOLS

                self.assertNotIn(s["name"], _HERMES_CORE_TOOLS)

    def test_search_errors_name_the_real_cause(self) -> None:
        """R6: a hung chat or embedding model, a stuck sidecar and a stopped sidecar are told apart."""
        provider = self.make()
        self.sidecar.search_error = (503, {"error": "the embedding model did not answer the query within 3000 ms "
                                                    "(http:http://127.0.0.1:11434/v1|embeddinggemma)",
                                           "cause": "embedding-timeout"})
        with self.assertLogs(em.logger, level="WARNING") as logs:
            error = json.loads(provider.handle_tool_call("tsuzuki_search", {"query": "x"}))["error"]
        self.assertIn("embedding model is unavailable", error)
        self.assertIn("127.0.0.1:11434", error)
        self.assertNotIn("sidecar unavailable", error)
        self.assertTrue(any("cannot use its embedding model" in line for line in logs.output))
        self.sidecar.search_error = None
        self.sidecar.search_delay = 1.0
        em.SEARCH_TIMEOUT_S = 0.3
        error = json.loads(provider.handle_tool_call("tsuzuki_search", {"query": "x"}))["error"]
        self.assertIn("did not answer within", error)
        self.sidecar.search_delay = 0.0
        self.sidecar.close()
        error = json.loads(provider.handle_tool_call("tsuzuki_search", {"query": "x"}))["error"]
        self.assertIn("not running", error)
        self.assertIn("sidecar.log", error)

    def test_handle_tool_call(self) -> None:
        provider = self.make()
        result = json.loads(provider.handle_tool_call("tsuzuki_search", {"query": "前回の結論"}))
        self.assertEqual(result, {"result": "<memory>search:前回の結論</memory>"})
        self.assertEqual(self.sidecar.calls("/v1/memory/search"), [{"query": "前回の結論"}])
        self.assertEqual(json.loads(provider.handle_tool_call("tsuzuki_search", {"query": "nothing"})),
                         {"result": "No matching memories."})
        self.assertIn("error", json.loads(provider.handle_tool_call("tsuzuki_search", {"query": " "})))
        self.assertIn("error", json.loads(provider.handle_tool_call("tsuzuki_search", {})))
        self.assertIn("error", json.loads(provider.handle_tool_call("other_tool", {"query": "x"})))
        self.sidecar.close()
        down = json.loads(provider.handle_tool_call("tsuzuki_search", {"query": "x"}))
        self.assertIn("not running", down["error"])

    def test_forget_current_session_after_queued_writes(self) -> None:
        provider = self.make("s1")
        provider.sync_turn("覚えておいて: 鍵は植木鉢の下", "了解しました。", session_id="s1")
        provider.queue_prefetch("鍵", session_id="s1")
        self.assertTrue(_wait_until(lambda: all(p.done.is_set() for p in provider._cache.values())))
        result = json.loads(provider.handle_tool_call("tsuzuki_forget", {"session": "current"}))
        posts = [(path, payload) for method, path, payload in self.sidecar.requests
                 if method == "POST" and path != "/v1/memory/recall"]
        self.assertEqual([p for p, _ in posts], ["/v1/memory/messages", "/v1/memory/forget"],
                         "queued turns reach the sidecar before the forget")
        self.assertEqual(posts[1][1], {"session": "s1"})
        self.assertEqual(result["deleted"]["turns"], 2)
        self.assertTrue(result["result"].startswith(
            "Forgot this conversation: deleted 2 messages, 1 fact and 3 index entries."), result["result"])
        self.assertIn("stored as usual", result["result"])
        self.assertEqual(len(provider._cache), 0, "recalls fetched before the forget are dropped")

    def test_forget_scopes_pass_through_and_ambiguous_calls_send_nothing(self) -> None:
        provider = self.make("s1")
        for args in ({"before": "2026-10-01"}, {"all": True}, {"session": "other-session", "before": None}):
            result = json.loads(provider.handle_tool_call("tsuzuki_forget", args))
            self.assertIn("result", result, args)
        self.assertEqual(self.sidecar.calls("/v1/memory/forget"),
                         [{"before": "2026-10-01"}, {"all": True}, {"session": "other-session"}])
        for bad in ({}, {"session": "current", "all": True}, None, "not json", {"query": "x"}):
            self.assertIn("exactly one", json.loads(provider.handle_tool_call("tsuzuki_forget", bad))["error"])
        self.assertEqual(json.loads(provider.handle_tool_call("tsuzuki_forget", '{"all": true}'))["deleted"]["turns"], 2)
        self.assertEqual(len(self.sidecar.calls("/v1/memory/forget")), 4)
        self.sidecar.forget_reply = (200, {"turns": 0, "units": 0, "facts": 0, "segments": 0, "links": 0, "vacuumed": False})
        nothing = json.loads(provider.handle_tool_call("tsuzuki_forget", {"session": "x"}))["result"]
        self.assertTrue(nothing.startswith("Nothing was stored for session x"), nothing)
        self.assertIn("rewritten without them within about 10 minutes", nothing)
        self.sidecar.forget_reply = (400, {"error": "before must be an ISO 8601 date or date-time"})
        refused = json.loads(provider.handle_tool_call("tsuzuki_forget", {"before": "last week"}))
        self.assertIn("nothing was deleted", refused["error"])
        self.assertIn("ISO 8601", refused["error"])
        self.sidecar.close()
        down = json.loads(provider.handle_tool_call("tsuzuki_forget", {"all": True}))
        self.assertIn("unavailable", down["error"])

    def test_forget_without_a_session(self) -> None:
        provider = self.make("")
        self.assertIn("no current session", json.loads(
            provider.handle_tool_call("tsuzuki_forget", {"session": "current"}))["error"])
        self.assertEqual(self.sidecar.calls("/v1/memory/forget"), [])


@unittest.skipIf(_HERMES_RUNTIME_MISSING, _HERMES_RUNTIME_MISSING or "")
class HermesRuntimeTests(SidecarTestCase):
    """The provider driven by Hermes' own MemoryManager and plugin loader."""

    def setUp(self) -> None:
        super().setUp()
        shutil.rmtree(_SESSION_HOME / "tsuzuki", ignore_errors=True)
        self.write_config(self.sidecar.port, home=_SESSION_HOME)

    def test_memory_manager_round_trip(self) -> None:
        from agent.memory_manager import MemoryManager

        manager = MemoryManager()
        provider = em.TsuzukiProvider()
        provider._forget_enabled = True  # as with memory.tsuzuki.forget: true
        self.providers.append(provider)
        manager.add_provider(provider)
        self.assertTrue(manager.has_tool("tsuzuki_search"))
        manager.initialize_all(session_id="s1", platform="cli", agent_context="primary")
        self.assertEqual(Path(provider._hermes_home).resolve(), _SESSION_HOME)

        self.assertIn("tsuzuki_search", manager.build_system_prompt())
        recalled = manager.prefetch_all("what did we decide about caching?", session_id="s1")
        self.assertEqual(recalled, "<memory>recall:what did we decide about caching?</memory>")
        self.assertIn("recalled 2 memories", manager.describe_recall())

        manager.sync_all("remember: cache TTL is 5 minutes", "Noted.", session_id="s1")
        self.assertTrue(manager.flush_pending(timeout=5))
        self.assertTrue(_wait_until(lambda: len(self.sidecar.calls("/v1/memory/messages")) == 1))
        stored = self.sidecar.calls("/v1/memory/messages")[0]
        self.assertEqual(stored["session"], "s1")
        self.assertEqual([m["role"] for m in stored["messages"]], ["user", "assistant"])

        result = json.loads(manager.handle_tool_call("tsuzuki_search", {"query": "cache TTL"}))
        self.assertEqual(result, {"result": "<memory>search:cache TTL</memory>"})

        self.assertTrue(manager.has_tool("tsuzuki_forget"))
        forgot = json.loads(manager.handle_tool_call("tsuzuki_forget", {"session": "current"}))
        self.assertEqual(forgot["deleted"]["turns"], 2)
        self.assertEqual(self.sidecar.calls("/v1/memory/forget"), [{"session": "s1"}])

        manager.on_session_end([])
        self.assertEqual(self.sidecar.calls("/v1/memory/flush"), [{"session": "s1"}])
        manager.shutdown_all()

    def test_hermes_plugin_loader_discovers_and_loads_the_directory(self) -> None:
        plugins_dir = _SESSION_HOME / "plugins"
        target = plugins_dir / "tsuzuki"
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(PLUGIN_SRC, target, ignore=shutil.ignore_patterns("__pycache__"))
        self.addCleanup(shutil.rmtree, plugins_dir, True)

        import plugins.memory as memory_plugins

        self.assertEqual(memory_plugins.find_provider_dir("tsuzuki").resolve(), target)
        self.assertIn("tsuzuki", memory_plugins.list_memory_provider_names())
        provider = memory_plugins.load_memory_provider("tsuzuki", register_skills=False)
        self.assertIsNotNone(provider)
        self.assertIsInstance(provider, MemoryProvider)
        self.assertEqual(provider.name, "tsuzuki")
        self.assertTrue(provider.is_available())
        again = memory_plugins.load_memory_provider("tsuzuki", register_skills=False)
        self.assertIsNot(provider, again)


def tearDownModule() -> None:
    shutil.rmtree(_SESSION_HOME, ignore_errors=True)


if __name__ == "__main__":
    if _HERMES_RUNTIME_MISSING:
        print(f"note: Hermes runtime tests skipped: {_HERMES_RUNTIME_MISSING}", file=sys.stderr)
    unittest.main()
