"""OpenAI-compatible embeddings HTTP service around the Metal engine.

Endpoints: ``POST /v1/embeddings``, ``GET /v1/models``, ``GET /health``,
``GET /stats``. Errors use the OpenAI envelope
``{"error": {"message", "type", "param", "code"}}``.

Task and Matryoshka dimension can be chosen per request with the extension
fields ``task`` (a sentence-transformers prompt name, ``none`` or ``default``),
``prompt`` (raw prompt text) and the standard ``dimensions``; clients that can
only send ``{model, input}`` select them through the model alias
``<model>[:<task>[:<dimensions>]]`` (e.g. ``embeddinggemma-2:query:256``).
Explicit fields override the alias; the alias overrides the server defaults.
"""
from __future__ import annotations

import base64
import hmac
import json
import os
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .batcher import Batcher
from .metrics import Clock, RequestStats, footprint
from .tokenizer import PromptTokenizer

ALLOWED_DIMENSIONS = (768, 512, 256, 128)
MAX_BODY_BYTES = 16 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str, code: str, param: str | None = None,
                 kind: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.code, self.param, self.kind = status, code, param, kind

    def body(self) -> dict:
        return {"error": {"message": str(self), "type": self.kind, "param": self.param, "code": self.code}}


@dataclass
class Settings:
    model_name: str
    default_task: str | None
    default_dimensions: int
    max_tokens: int
    max_inputs: int
    request_timeout: float
    api_key: str | None
    load_wait: float = 15.0  # seconds an embeddings request waits while the engine is still loading


class Service:
    """State shared by all request threads; ``ready`` flips after load and warmup."""

    def __init__(self, settings: Settings, clock: Clock):
        self.settings, self.clock = settings, clock
        self.stats = RequestStats()
        self.state, self.failure = "loading", None
        self.settled = threading.Event()  # set when the state leaves "loading"
        self.tokenizer: PromptTokenizer | None = None
        self.batcher: Batcher | None = None
        self.engine_info: dict = {}

    def set_state(self, state: str, failure: str | None = None) -> None:
        self.state, self.failure = state, failure
        if state != "loading":
            self.settled.set()

    # -- request parsing -------------------------------------------------
    def resolve_model(self, model: object) -> tuple[str | None, int | None]:
        """Parse ``<model>[:<task>[:<dimensions>]]``; returns (task or None if unset, dims or None)."""
        if model is None:
            return None, None
        if not isinstance(model, str):
            raise ApiError(HTTPStatus.BAD_REQUEST, "model must be a string", "invalid_type", "model")
        base, *rest = model.split(":")
        if base != self.settings.model_name or len(rest) > 2:
            raise ApiError(HTTPStatus.NOT_FOUND, f"The model '{model}' does not exist; use "
                           f"'{self.settings.model_name}[:<task>[:<dimensions>]]'", "model_not_found", "model")
        task = rest[0] if rest and rest[0] else None
        dims = None
        if len(rest) == 2:
            if not rest[1].isdigit():
                raise ApiError(HTTPStatus.NOT_FOUND, f"Invalid dimensions in model alias '{model}'", "model_not_found", "model")
            dims = int(rest[1])
        return task, dims

    def parse(self, body: dict) -> tuple[list[str], str | None, str, int, str]:
        raw = body.get("input")
        if isinstance(raw, str):
            texts = [raw]
        elif isinstance(raw, list) and raw and all(isinstance(x, str) for x in raw):
            texts = raw
        elif isinstance(raw, list) and raw and all(isinstance(x, (int, list)) for x in raw):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Token-array inputs are not supported; send text", "unsupported_input", "input")
        else:
            raise ApiError(HTTPStatus.BAD_REQUEST, "input must be a string or a non-empty array of strings", "invalid_type", "input")
        if len(texts) > self.settings.max_inputs:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"At most {self.settings.max_inputs} inputs per request "
                           f"(got {len(texts)})", "too_many_inputs", "input")
        alias_task, alias_dims = self.resolve_model(body.get("model"))
        if "prompt" in body and ("task" in body or "prompt_name" in body):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Give either prompt or task, not both", "conflicting_parameters", "prompt")
        tokenizer = self.tokenizer
        assert tokenizer is not None
        if "prompt" in body:
            if not isinstance(body["prompt"], str):
                raise ApiError(HTTPStatus.BAD_REQUEST, "prompt must be a string", "invalid_type", "prompt")
            prompt, task_label = body["prompt"], "custom"
        else:
            task = body.get("task", body.get("prompt_name", alias_task))
            if task is None:
                task = self.settings.default_task
            if task is not None and not isinstance(task, str):
                raise ApiError(HTTPStatus.BAD_REQUEST, "task must be a string", "invalid_type", "task")
            try:
                prompt = tokenizer.prompt_for(task)
            except KeyError:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"Unknown task '{task}'; valid: none, default, "
                               + ", ".join(tokenizer.task_names()), "unknown_task", "task") from None
            task_label = task or "none"
        dims = body.get("dimensions", alias_dims if alias_dims is not None else self.settings.default_dimensions)
        if not isinstance(dims, int) or isinstance(dims, bool) or dims not in ALLOWED_DIMENSIONS:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"dimensions must be one of {list(ALLOWED_DIMENSIONS)}",
                           "invalid_dimensions", "dimensions")
        encoding = body.get("encoding_format", "float")
        if encoding not in ("float", "base64"):
            raise ApiError(HTTPStatus.BAD_REQUEST, "encoding_format must be float or base64", "invalid_encoding_format", "encoding_format")
        return texts, prompt, task_label, dims, encoding

    def embed(self, body: dict) -> dict:
        # Clients started together with the daemon (e.g. a dsh plugin at boot) would otherwise get 503 for the
        # first seconds and may back off for much longer; hold them until the engine is ready, up to load_wait.
        if self.state == "loading":
            self.settled.wait(self.settings.load_wait)
        if self.state != "ready":
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, f"Server is {self.state}" +
                           (f": {self.failure}" if self.failure else ""), "not_ready", kind="server_error")
        started = time.perf_counter()
        texts, prompt, task, dims, encoding = self.parse(body)
        sequences = self.tokenizer.encode(texts, prompt)
        for index, sequence in enumerate(sequences):
            if len(sequence) > self.settings.max_tokens:
                raise ApiError(HTTPStatus.BAD_REQUEST,
                               f"Input {index} has {len(sequence)} tokens including the task prompt and BOS/EOS; "
                               f"the maximum is {self.settings.max_tokens}. Shorten or split the input.",
                               "context_length_exceeded", "input")
        job = self.batcher.submit(sequences, dims)
        try:
            vectors = job.future.result(timeout=self.settings.request_timeout)
        except FutureTimeout:
            raise ApiError(HTTPStatus.GATEWAY_TIMEOUT, f"Request did not finish within {self.settings.request_timeout}s",
                           "timeout", kind="server_error") from None
        tokens = sum(map(len, sequences))
        self.stats.request(len(texts), tokens, time.perf_counter() - started,
                           (job.started or job.enqueued) - job.enqueued)
        if encoding == "base64":
            data = [base64.b64encode(v.astype("<f4").tobytes()).decode() for v in vectors]
        else:
            data = vectors.tolist()
        model = self.settings.model_name if task == "none" and dims == ALLOWED_DIMENSIONS[0] else \
            f"{self.settings.model_name}:{task}:{dims}"
        return {"object": "list", "model": model,
                "data": [{"object": "embedding", "index": i, "embedding": d} for i, d in enumerate(data)],
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}

    def models(self) -> dict:
        return {"object": "list", "data": [{"id": self.settings.model_name, "object": "model", "created": 0,
                                            "owned_by": "egemma-engine"}]}

    def health(self) -> tuple[HTTPStatus, dict]:
        status = HTTPStatus.OK if self.state == "ready" else HTTPStatus.SERVICE_UNAVAILABLE
        body = {"status": self.state, "model": self.settings.model_name, "pid": os.getpid()}
        if self.failure:
            body["error"] = self.failure
        return status, body

    def snapshot(self) -> dict:
        return {"status": self.state, "model": self.settings.model_name, "pid": os.getpid(),
                "uptime_s": round(time.time() - self.clock.process_start, 3),
                "cold_start": self.clock.report(), "engine": self.engine_info,
                "limits": {"max_tokens": self.settings.max_tokens, "max_inputs": self.settings.max_inputs,
                           "dimensions": list(ALLOWED_DIMENSIONS), "default_dimensions": self.settings.default_dimensions,
                           "default_task": self.settings.default_task or "none"},
                "queue_depth": self.batcher.depth() if self.batcher else 0,
                "memory": {"physical_footprint": footprint()}, **self.stats.snapshot()}


def make_handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        server_version = "egemma-server"
        protocol_version = "HTTP/1.1"
        # TCP_NODELAY: _send writes the headers and the body separately; with Nagle on, the body waits
        # for the client's delayed ACK of the headers (~40 ms per keep-alive request on loopback).
        disable_nagle_algorithm = True

        def log_message(self, *_args):  # request logging is the caller's choice; stats cover volume
            pass

        def _send(self, status: HTTPStatus, body: dict) -> None:
            payload = json.dumps(body, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        # A web page in the user's browser must not use this server: browsers send Origin on cross-site requests, a
        # DNS-rebinding page arrives with a foreign Host, and a no-preflight POST cannot be application/json.
        timeout = 120  # a client that stops sending does not hold a thread forever

        def _refused(self, post: bool) -> Optional[ApiError]:
            if self.headers.get("Origin") is not None:
                return ApiError(HTTPStatus.FORBIDDEN, "browser requests are not accepted", "forbidden")
            host, port = self.server.server_address[:2]
            if str(host) in ("127.0.0.1", "::1", "localhost"):
                allowed = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
                if str(self.headers.get("Host", "")).lower() not in allowed:
                    return ApiError(HTTPStatus.FORBIDDEN, "unexpected Host header", "forbidden")
            if post and not str(self.headers.get("Content-Type", "")).lower().startswith("application/json"):
                return ApiError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json", "invalid_body")
            return None

        def _authorized(self) -> bool:
            key = service.settings.api_key
            if not key:
                return True
            supplied = self.headers.get("Authorization", "")
            return hmac.compare_digest(supplied.encode(), f"Bearer {key}".encode())

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            refused = self._refused(post=False)
            if refused:
                self._send(refused.status, refused.body())
            elif path == "/health":
                self._send(*service.health())
            elif not self._authorized():
                self._send(HTTPStatus.UNAUTHORIZED, ApiError(HTTPStatus.UNAUTHORIZED, "Invalid API key", "invalid_api_key").body())
            elif path == "/v1/models":
                self._send(HTTPStatus.OK, service.models())
            elif path == "/stats":
                self._send(HTTPStatus.OK, service.snapshot())
            else:
                self._send(HTTPStatus.NOT_FOUND, ApiError(HTTPStatus.NOT_FOUND, f"Unknown path {path}", "not_found").body())

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            try:
                refused = self._refused(post=True)
                if refused:
                    raise refused
                if not self._authorized():
                    raise ApiError(HTTPStatus.UNAUTHORIZED, "Invalid API key", "invalid_api_key")
                if path not in ("/v1/embeddings", "/embeddings"):
                    raise ApiError(HTTPStatus.NOT_FOUND, f"Unknown path {path}", "not_found")
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise ApiError(HTTPStatus.BAD_REQUEST, f"Body must be 1..{MAX_BODY_BYTES} bytes of JSON", "invalid_body")
                try:
                    body = json.loads(self.rfile.read(length))
                except json.JSONDecodeError as error:
                    raise ApiError(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {error}", "invalid_json") from None
                if not isinstance(body, dict):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "Body must be a JSON object", "invalid_body")
                self._send(HTTPStatus.OK, service.embed(body))
            except ApiError as error:
                service.stats.error()
                self._send(error.status, error.body())
            except Exception as error:  # engine or encoding failure: report, keep serving
                service.stats.error()
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR,
                           ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(error).__name__}: {error}",
                                    "internal_error", kind="server_error").body())

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    # SO_REUSEADDR only: a restart may bind over TIME_WAIT sockets; a second live
    # listener on the same port still fails with EADDRINUSE (no SO_REUSEPORT).
    allow_reuse_address = True
    # listen() backlog; the stdlib default of 5 resets bursts of concurrent clients.
    request_queue_size = 128


def serve(host: str, port: int, service: Service) -> tuple[Server, threading.Thread]:
    server = Server((host, port), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, name="egemma-http", daemon=True)
    thread.start()
    return server, thread
