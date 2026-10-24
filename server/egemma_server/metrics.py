"""Process facts (start time, physical footprint) and request statistics."""
from __future__ import annotations

import ctypes
import os
import threading
import time
from collections import deque

_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_libproc.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]


def footprint() -> dict[str, int]:
    """Current and lifetime-peak physical footprint (SDK sys/resource.h rusage_info_v4)."""
    buffer = ctypes.create_string_buffer(296)
    if _libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buffer)):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return {"current_bytes": ctypes.c_uint64.from_buffer(buffer, 72).value,
            "lifetime_peak_bytes": ctypes.c_uint64.from_buffer(buffer, 240).value}


def process_start_time() -> float:
    """Wall-clock process start (proc_bsdinfo.pbi_start_tvsec/usec, PROC_PIDTBSDINFO)."""
    buffer = ctypes.create_string_buffer(136)
    if _libproc.proc_pidinfo(os.getpid(), 3, 0, ctypes.byref(buffer), 136) != 136:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    seconds = ctypes.c_uint64.from_buffer(buffer, 120).value
    micros = ctypes.c_uint64.from_buffer(buffer, 128).value
    return seconds + micros / 1e6


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class RequestStats:
    """Thread-safe counters plus a window of recent request latencies."""

    def __init__(self, window: int = 2048):
        self._lock = threading.Lock()
        self._latency = deque(maxlen=window)
        self._queue_wait = deque(maxlen=window)
        self.requests = self.inputs = self.tokens = self.errors = 0
        self.batches = self.batch_rows = self.batch_padded_tokens = 0
        self.engine_seconds = 0.0
        self.peak_metal_allocated_bytes = 0

    def request(self, inputs: int, tokens: int, latency: float, queue_wait: float) -> None:
        with self._lock:
            self.requests += 1
            self.inputs += inputs
            self.tokens += tokens
            self._latency.append(latency)
            self._queue_wait.append(queue_wait)

    def error(self) -> None:
        with self._lock:
            self.errors += 1

    def batch(self, rows: int, padded_tokens: int, seconds: float, metal_peak: int) -> None:
        with self._lock:
            self.batches += 1
            self.batch_rows += rows
            self.batch_padded_tokens += padded_tokens
            self.engine_seconds += seconds
            self.peak_metal_allocated_bytes = max(self.peak_metal_allocated_bytes, metal_peak)

    def snapshot(self) -> dict:
        with self._lock:
            latency = [x * 1e3 for x in self._latency]
            wait = [x * 1e3 for x in self._queue_wait]
            return {
                "requests": self.requests, "inputs": self.inputs, "tokens": self.tokens, "errors": self.errors,
                "latency_ms": {"window": len(latency), "p50": _percentile(latency, .5), "p95": _percentile(latency, .95),
                               "p99": _percentile(latency, .99), "max": max(latency) if latency else None},
                "queue_wait_ms": {"p50": _percentile(wait, .5), "p95": _percentile(wait, .95)},
                "batches": {"count": self.batches,
                            "mean_rows": self.batch_rows / self.batches if self.batches else None,
                            "padded_tokens": self.batch_padded_tokens, "engine_seconds": self.engine_seconds},
                "peak_metal_allocated_bytes": self.peak_metal_allocated_bytes,
            }


class Clock:
    """Cold-start milestones relative to the OS process start."""

    def __init__(self):
        self.process_start = process_start_time()
        self.marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        self.marks[name] = time.time() - self.process_start

    def report(self) -> dict[str, float]:
        return {f"{name}_s": round(value, 4) for name, value in self.marks.items()}
