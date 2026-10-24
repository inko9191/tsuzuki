"""Dynamic batching in front of the single Metal engine.

Requests are queued; one worker thread owns every engine call. After the first
queued job it keeps collecting jobs for at most ``wait_ms`` (or until the
queued rows reach ``max_batch_rows``), sorts all rows by token length and cuts
them into engine calls whose padded size ``rows * longest`` stays within
``max_batch_tokens`` (a single over-budget row still runs alone). Each engine
call returns 768-dimensional L2-normalized vectors; a job asking for fewer
Matryoshka dimensions gets its prefix renormalized, which equals normalizing
the pooled prefix directly because the full normalization is a positive scale.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np

from .metrics import RequestStats


@dataclass
class Job:
    sequences: list[list[int]]
    dimensions: int
    future: Future = field(default_factory=Future)
    enqueued: float = field(default_factory=time.perf_counter)
    started: float | None = None


class Batcher:
    def __init__(self, engine, pad_id: int, stats: RequestStats, full_dimensions: int,
                 max_batch_tokens: int, max_batch_rows: int, wait_ms: float):
        self.engine, self.pad_id, self.stats = engine, pad_id, stats
        self.full_dimensions = full_dimensions
        self.max_batch_tokens, self.max_batch_rows, self.wait = max_batch_tokens, max_batch_rows, wait_ms / 1e3
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="egemma-batcher", daemon=True)
        self._thread.start()

    def submit(self, sequences: list[list[int]], dimensions: int) -> Job:
        job = Job(sequences, dimensions)
        self._queue.put(job)
        return job

    def depth(self) -> int:
        return self._queue.qsize()

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30)

    def _collect(self, first: Job) -> tuple[list[Job], bool]:
        jobs, rows = [first], len(first.sequences)
        deadline = time.perf_counter() + self.wait
        while rows < self.max_batch_rows:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                job = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if job is None:
                return jobs, True
            jobs.append(job)
            rows += len(job.sequences)
        return jobs, False

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return
            jobs, stopping = self._collect(first)
            try:
                self._execute(jobs)
            except BaseException as error:  # every waiting caller must learn the outcome
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(error)
            if stopping:
                return

    def _execute(self, jobs: list[Job]) -> None:
        now = time.perf_counter()
        for job in jobs:
            job.started = now
        rows = [(len(seq), j, i) for j, job in enumerate(jobs) for i, seq in enumerate(job.sequences)]
        rows.sort()
        outputs = [np.empty((len(job.sequences), self.full_dimensions), np.float32) for job in jobs]
        start = 0
        while start < len(rows):
            end = start + 1
            while end < len(rows) and end - start < self.max_batch_rows and (end - start + 1) * rows[end][0] <= self.max_batch_tokens:
                end += 1
            chunk = rows[start:end]
            width = chunk[-1][0]
            ids = np.full((len(chunk), width), self.pad_id, np.int32)
            mask = np.zeros_like(ids)
            for r, (length, j, i) in enumerate(chunk):
                ids[r, :length] = jobs[j].sequences[i]
                mask[r, :length] = 1
            began = time.perf_counter()
            vectors = self.engine.encode(ids, mask, self.full_dimensions)
            metal = json.loads(self.engine.profile()).get("peak_metal_allocated_bytes", 0)
            self.stats.batch(len(chunk), int(ids.size), time.perf_counter() - began, metal)
            for r, (_, j, i) in enumerate(chunk):
                outputs[j][i] = vectors[r]
            start = end
        for job, out in zip(jobs, outputs):
            if job.dimensions < self.full_dimensions:
                out = out[:, :job.dimensions]
                out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), np.float32(1e-12))
            job.future.set_result(np.ascontiguousarray(out, dtype=np.float32))
