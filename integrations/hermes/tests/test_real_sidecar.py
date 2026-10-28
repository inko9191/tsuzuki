"""The Hermes provider against the real sidecar (node lib/server.js) and a fake embedding endpoint.

    "$(ls -d ~/.hermes/tools/python-*/bin/python3 | head -1)" -I integrations/hermes/tests/test_real_sidecar.py -v

Needs the built sidecar (dsh-plugins/packages/tsuzuki/lib/server.js) and Hermes' bundled node.
CPU only; nothing is written under ~/.hermes.
"""
from __future__ import annotations

import json
import os
import re
import signal
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_tsuzuki as base  # noqa: E402  (bootstraps Hermes and imports the plugin as base.em)

em = base.em
REPO = Path(__file__).resolve().parents[3]
SERVER_JS = REPO / "dsh-plugins" / "packages" / "tsuzuki" / "lib" / "server.js"
NODE = Path.home() / ".hermes" / "node" / "bin" / "node"
STOP = {"the", "i", "my", "is", "to", "do", "user", "assistant", "where", "now", "did", "a", "last", "week"}


def vector(text: str) -> list[float]:
    v = [0.0] * 64
    body = re.sub(r"^(task: search result \| query: |title: none \| text: )", "", text)
    for w in re.findall(r"[a-z0-9]+", body.lower()):
        if w in STOP:
            continue
        h = 2166136261
        for c in w:
            h = ((h ^ ord(c)) * 16777619) & 0xFFFFFFFF
        v[h % 64] += 1.0
    return v


class FakeEmbeddings:
    def __init__(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                data = [{"index": i, "embedding": vector(t)} for i, t in enumerate(body["input"])]
                out = json.dumps({"data": data}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()


@unittest.skipUnless(SERVER_JS.is_file() and NODE.is_file(), "built sidecar or Hermes node missing")
class RealSidecarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = Path(tempfile.mkdtemp(prefix="hermes-real-"))
        self.emb = FakeEmbeddings()
        data = self.home / em.DATA_DIR_NAME
        data.mkdir(parents=True)
        port = base._free_port()  # an ephemeral port: never one of the fixed ports other local services use
        (data / "config.json").write_text(json.dumps({
            "port": port, "dataDir": str(data),
            "embedding": {"baseUrl": self.emb.url, "model": "fake", "autoPull": False},
            "recall": {"budgetTokens": 2048}}))
        self.env = {k: os.environ.get(k) for k in ("TSUZUKI_SERVER_JS", "TSUZUKI_NODE")}
        os.environ["TSUZUKI_SERVER_JS"] = str(SERVER_JS)
        os.environ["TSUZUKI_NODE"] = str(NODE)
        self.pids: list[int] = []

    def tearDown(self) -> None:
        for pid in self.pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.emb.close()
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def provider(self, session: str) -> "em.TsuzukiProvider":
        p = em.TsuzukiProvider()
        p.initialize(session, hermes_home=str(self.home), platform="cli", agent_context="primary")
        p._forget_enabled = True  # memory.tsuzuki.forget: true (off by default)
        if p.spawned_pid:
            self.pids.append(p.spawned_pid)
        return p

    def test_turn_captured_then_recalled_from_another_session(self) -> None:
        first = self.provider("s1")
        self.assertIsNotNone(first.spawned_pid, "the real sidecar is started by the provider")
        first.sync_turn("I moved to Kyoto and my cat Tama loves the garden.", "Kyoto sounds lovely!", session_id="s1")
        first.shutdown()  # drains the queued write
        second = self.provider("s2")
        self.assertIsNone(second.spawned_pid, "the running sidecar is reused")
        # Shares words with the stored turn: the toy vectors must clear the sidecar's default recall gate (0.31).
        text = second.prefetch("Did I move to Kyoto with my cat?", session_id="s2")
        self.assertIn("Kyoto", text)
        self.assertIn("Current date:", text)
        second.sync_turn("Did I move to Kyoto with my cat?", "You said Kyoto.", session_id="s2")
        second.shutdown()
        third = self.provider("s3")
        found = json.loads(third.handle_tool_call("tsuzuki_search", {"query": "cat Tama"}))
        self.assertIn("Tama", found["result"])
        health = em._http_json(third._base_url, "GET", "/health", timeout=5)
        self.assertEqual(Path(health["dataDir"]).resolve(), (self.home / em.DATA_DIR_NAME).resolve())
        self.assertEqual(health["stats"]["turns"], 4)
        third.shutdown()

    def test_forget_removes_the_text_from_search_and_from_the_file(self) -> None:
        db_dir = self.home / em.DATA_DIR_NAME

        def on_disk(word: str) -> bool:
            return any(word.encode() in p.read_bytes() for p in db_dir.glob("memory.db*"))

        def search(provider: "em.TsuzukiProvider", query: str) -> str:
            return json.loads(provider.handle_tool_call("tsuzuki_search", {"query": query}))["result"]

        first = self.provider("s1")
        first.sync_turn("My zanzibar locker code is 4417.", "Noted, the zanzibar locker.", session_id="s1")
        first.shutdown()
        second = self.provider("s2")
        second.sync_turn("The quokka locker is in hall C.", "OK, hall C.", session_id="s2")
        self.assertIn("zanzibar", search(second, "zanzibar locker code"))
        self.assertTrue(on_disk("zanzibar"), "sanity: on disk before forgetting")

        forgot = json.loads(second.handle_tool_call("tsuzuki_forget", {"session": "s1"}))
        self.assertEqual(forgot["deleted"]["turns"], 2)
        self.assertTrue(forgot["deleted"]["vacuumed"])
        self.assertNotIn("zanzibar", search(second, "zanzibar locker code"))
        self.assertFalse(on_disk("zanzibar"), "not in memory.db or its WAL")
        self.assertTrue(on_disk("quokka"))

        current = json.loads(second.handle_tool_call("tsuzuki_forget", {"session": "current"}))
        self.assertTrue(current["result"].startswith("Forgot this conversation: deleted 2 messages"), current["result"])
        self.assertFalse(on_disk("quokka"))
        self.assertEqual(search(second, "quokka locker hall"), "No matching memories.")
        second.shutdown()


if __name__ == "__main__":
    unittest.main()
