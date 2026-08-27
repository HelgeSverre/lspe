"""RMS-scaled additive perturbation kernel."""

from __future__ import annotations

import numpy as np

from .base import KernelResult, as_float32


def rms_scaled_additive(
    activation: np.ndarray, direction: np.ndarray, alpha: float, epsilon: float = 1e-8
) -> KernelResult:
    """Apply an additive direction and restore each vector's RMS magnitude."""

    original = np.asarray(activation)
    if alpha == 0:
        return KernelResult(original.copy(), 0, 0)
    h = as_float32(original)
    r = np.broadcast_to(as_float32(direction), h.shape)
    rms = np.sqrt(np.mean(np.square(h), axis=-1, keepdims=True))
    direction_rms = np.sqrt(np.mean(np.square(r), axis=-1, keepdims=True))
    unit_rms = r / np.maximum(direction_rms, epsilon)
    raw = h + alpha * rms * unit_rms
    raw_rms = np.sqrt(np.mean(np.square(raw), axis=-1, keepdims=True))
    output = raw * rms / np.maximum(raw_rms, epsilon)
    output = np.where(rms > epsilon, output, h)
    return KernelResult(
        output.astype(original.dtype, copy=False),
        int(np.count_nonzero(rms <= epsilon)),
        int(np.count_nonzero(direction_rms <= epsilon)),
    )
