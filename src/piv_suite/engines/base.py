"""Common engine interface shared by the CPU (openpiv-python) and GPU
(openpiv-python-gpu) backends.

Both backends already expose the same minimal surface in the original four
repos (confirmed identical in practice): `.coords` (x, y arrays),
`.val_locations` (bool array, True = invalid, set after a call),
`.scaling_par`, and being callable as process(frame_a, frame_b) -> (u, v).
`processing.pipeline.process_frames()` is written against this Protocol so
it never needs to branch on which backend built the engine.
"""

from typing import Optional, Protocol, Tuple

import numpy as np


class PIVEngine(Protocol):
    """Structural interface a PIV engine instance must satisfy. Not meant
    to be subclassed -- CPUPIVProcess and the GPU engine wrapper each
    satisfy this by having the right attributes/methods, per Python's
    structural (duck-typed) Protocol semantics."""

    coords: Tuple[np.ndarray, np.ndarray]
    val_locations: Optional[np.ndarray]
    scaling_par: float

    def __call__(self, frame_a: np.ndarray, frame_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...
