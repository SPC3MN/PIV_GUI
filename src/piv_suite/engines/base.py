"""Common engine interface shared by the CPU (openpiv-python) and GPU
(openpiv-python-gpu) backends.

Both backends already expose the same minimal surface in the original four
repos (confirmed identical in practice): `.coords` (x, y arrays),
`.val_locations` (bool array, True = invalid, set after a call),
`.scaling_par`, and being callable as process(frame_a, frame_b) -> (u, v).
`processing.pipeline.process_frames()` is written against this Protocol so
it never needs to branch on which backend built the engine.
"""

from typing import Callable, Optional, Protocol, Tuple

import numpy as np


class EngineCancelled(Exception):
    """Raised by an engine's __call__ (from inside its OWN multi-pass
    loop, between passes) when cancel_check() returns True mid-pair --
    lets pipeline_worker.PipelineWorker abort a single pair without
    waiting for its whole correlation to finish, without resorting to
    killing a thread mid-BLAS/FFT call (unsafe -- see cpu_engine.py /
    pipeline_worker.py for why that's deliberately not done instead).

    Distinct from a plain Exception so pipeline_worker's per-pair
    try/except can tell "cancelled" apart from "this pair genuinely
    failed" -- the former should break the batch loop cleanly (no
    self.error signal, no summary row), the latter should not."""


class PIVEngine(Protocol):
    """Structural interface a PIV engine instance must satisfy. Not meant
    to be subclassed -- CPUPIVProcess and the GPU engine wrapper each
    satisfy this by having the right attributes/methods, per Python's
    structural (duck-typed) Protocol semantics."""

    coords: Tuple[np.ndarray, np.ndarray]
    val_locations: Optional[np.ndarray]
    scaling_par: float

    def __call__(self, frame_a: np.ndarray, frame_b: np.ndarray,
                 cancel_check: Optional[Callable[[], bool]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """cancel_check, if given, is polled by the engine at whatever
        points its own internals make that cheap (CPUPIVProcess: between
        multi-pass iterations; the GPU backend has no such point inside
        piv_gpu itself, see gpu_engine.py) -- raises EngineCancelled the
        moment it returns True. Not every engine has a cheap place to
        check it; engines that don't just ignore the argument."""
        ...
