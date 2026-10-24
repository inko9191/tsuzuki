"""egemma-server: resident OpenAI-compatible embeddings daemon.

Binds first (``/health`` answers ``loading``), then loads the tokenizer and the
Metal engine, runs one warmup, and reports ``ready``. Embeddings requests that
arrive while loading are held until ready (at most ``--load-wait`` seconds,
then 503); a failed load answers them at once. A single JSON line
``{"event": "ready", ...}`` (or ``{"event": "failed", ...}``) is written to
stdout for supervisors; diagnostics go to stderr. SIGTERM/SIGINT stop
accepting requests and exit after in-flight batches.
"""
from __future__ import annotations

from .metrics import Clock

CLOCK = Clock()  # first statement after the interpreter starts executing this package

import argparse  # noqa: E402
import importlib  # noqa: E402
import ipaddress  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402

from .app import ALLOWED_DIMENSIONS, Service, Settings, serve  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env = os.environ.get
    ap = argparse.ArgumentParser(prog="egemma-server", description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=env("EGEMMA_MODEL_DIR", str(REPO / "models/embeddinggemma-2")))
    ap.add_argument("--build-dir", default=env("EGEMMA_BUILD_DIR", str(REPO / "build")),
                    help="directory containing the built _egemma_native extension")
    ap.add_argument("--host", default=env("EGEMMA_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(env("EGEMMA_PORT", "8299")))
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default=env("EGEMMA_DTYPE", "auto"),
                    help="auto: fp16 for EmbeddingGemma 2, fp32 for EmbeddingGemma 300m (bf16 is its faster, less exact mode)")
    ap.add_argument("--matmul", choices=["auto", "native", "mpp"], default=env("EGEMMA_MATMUL", "auto"))
    ap.add_argument("--model-name", default="embeddinggemma-2")
    ap.add_argument("--default-task", default=None,
                    help="prompt name used when a request names none (default: the model card's default_prompt_name, none)")
    ap.add_argument("--default-dimensions", type=int, choices=ALLOWED_DIMENSIONS, default=768)
    ap.add_argument("--max-tokens", type=int, default=2048, help="per input, including the task prompt and BOS/EOS")
    ap.add_argument("--max-inputs", type=int, default=2048, help="inputs per request")
    ap.add_argument("--max-batch-tokens", type=int, default=32768, help="padded tokens per engine call")
    ap.add_argument("--max-batch-rows", type=int, default=256)
    ap.add_argument("--batch-wait-ms", type=float, default=2.0)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--load-wait", type=float, default=15.0,
                    help="seconds an embeddings request is held while the engine loads (then 503)")
    ap.add_argument("--api-key", default=env("EGEMMA_API_KEY"))
    args = ap.parse_args(argv)
    if args.max_tokens < 2 or args.max_inputs < 1 or args.max_batch_tokens < 1 or args.max_batch_rows < 1:
        ap.error("limits must be positive (max-tokens >= 2)")
    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = args.host == "localhost"
    if not loopback and not args.api_key:
        ap.error("binding a non-loopback address requires --api-key")
    return args


def resolve_dtype(dtype: str, model_dir: str | Path) -> str:
    """'auto' -> fp32 for Gemma 3 checkpoints (EmbeddingGemma 300m: activations overflow fp16, and bf16 moves
    single vectors down to cos 0.983, README), fp16 otherwise (EmbeddingGemma 2, validated in fp16)."""
    if dtype != "auto":
        return dtype
    config = json.loads((Path(model_dir) / "config.json").read_text())
    model_type = (config.get("text_config") or config).get("model_type")
    return "fp32" if model_type == "gemma3_text" else "fp16"


def emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings(args.model_name, args.default_task, args.default_dimensions, args.max_tokens,
                        args.max_inputs, args.request_timeout, args.api_key, args.load_wait)
    service = Service(settings, CLOCK)
    try:
        server, thread = serve(args.host, args.port, service)
    except OSError as error:
        emit({"event": "failed", "stage": "bind", "error": f"{args.host}:{args.port}: {error}"})
        print(f"egemma-server: cannot bind {args.host}:{args.port}: {error}", file=sys.stderr)
        return 2
    CLOCK.mark("bound")
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    try:
        from .batcher import Batcher
        from .tokenizer import PromptTokenizer

        CLOCK.mark("imports")
        tokenizer = PromptTokenizer(args.model_dir)
        if args.default_task is not None:
            tokenizer.prompt_for(args.default_task)  # unknown names fail at startup
        service.tokenizer = tokenizer
        CLOCK.mark("tokenizer")
        build_dir = str(Path(args.build_dir).resolve())
        if build_dir not in sys.path:
            sys.path.insert(0, build_dir)
        native = importlib.import_module("_egemma_native")
        engine = native.Engine(str(args.model_dir), resolve_dtype(args.dtype, args.model_dir), False, args.matmul)
        info = json.loads(engine.info())
        service.engine_info = {k: info.get(k) for k in ("device", "dtype", "matmul_backend", "mpp_status", "mpp_min_rows",
                                                         "metal4_family")}
        CLOCK.mark("engine")
        service.batcher = Batcher(engine, tokenizer.pad_id, service.stats, ALLOWED_DIMENSIONS[0],
                                  args.max_batch_tokens, args.max_batch_rows, args.batch_wait_ms)
        ids = tokenizer.encode(["warmup"], tokenizer.prompt_for("query"))
        service.batcher.submit(ids, ALLOWED_DIMENSIONS[0]).future.result(timeout=120)
        CLOCK.mark("ready")
        service.set_state("ready")
    except Exception as error:  # report the cause on /health and stdout, then exit non-zero
        service.set_state("failed", f"{type(error).__name__}: {error}")
        emit({"event": "failed", "stage": "load", "error": service.failure})
        print(f"egemma-server: startup failed: {service.failure}", file=sys.stderr)
        server.shutdown()
        return 1
    emit({"event": "ready", "host": args.host, "port": server.server_address[1], "pid": os.getpid(),
          "model": args.model_name, "engine": service.engine_info, "cold_start": CLOCK.report()})
    stop.wait()
    service.set_state("stopping")
    server.shutdown()
    service.batcher.stop()
    thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
