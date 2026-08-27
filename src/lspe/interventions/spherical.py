"""Norm-preserving per-vector spherical residual rotations."""

from __future__ import annotations

import numpy as np

from .base import KernelResult, as_float32


def spherical_rotation(
    activation: np.ndarray, direction: np.ndarray, theta: float, epsilon: float = 1e-8
) -> KernelResult:
    """Rotate vectors in their last dimension while preserving norms to FP tolerance."""

    original = np.asarray(activation)
    if theta == 0:
        return KernelResult(original.copy(), 0, 0)
    h = as_float32(original)
    r = as_float32(direction)
    if h.shape[-1] != r.shape[-1]:
        raise ValueError("Activation and direction hidden widths must match")
    r = np.broadcast_to(r, h.shape)
    norm = np.linalg.vector_norm(h, axis=-1, keepdims=True)
    safe_norm = np.maximum(norm, epsilon)
    u = h / safe_norm
    v_raw = r - np.sum(r * u, axis=-1, keepdims=True) * u
    v_norm = np.linalg.vector_norm(v_raw, axis=-1, keepdims=True)
    # A direction can be collinear with an activation (not merely random noise
    # in a test fixture). Choose a deterministic least-aligned basis vector so
    # the promised rotation and norm invariant still hold.
    degenerate = v_norm <= epsilon
    if np.any(degenerate):
        basis = np.zeros_like(h)
        basis_index = np.argmin(np.abs(u), axis=-1)
        np.put_along_axis(basis, basis_index[..., None], 1.0, axis=-1)
        fallback = basis - np.sum(basis * u, axis=-1, keepdims=True) * u
        v_raw = np.where(degenerate, fallback, v_raw)
        v_norm = np.linalg.vector_norm(v_raw, axis=-1, keepdims=True)
    safe_v_norm = np.maximum(v_norm, epsilon)
    v = v_raw / safe_v_norm
    rotated = norm * (np.cos(theta) * u + np.sin(theta) * v)
    # A zero activation has no well-defined direction, so preserve it exactly.
    rotated = np.where(norm > epsilon, rotated, h)
    return KernelResult(
        rotated.astype(original.dtype, copy=False),
        int(np.count_nonzero(norm <= epsilon)),
        int(np.count_nonzero(v_norm <= epsilon)),
    )
