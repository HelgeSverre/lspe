"""Shared types for norm-aware activation interventions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KernelResult:
    activation: np.ndarray
    near_zero_hidden_count: int
    near_zero_direction_count: int


def as_float32(value: np.ndarray) -> np.ndarray:
    """Perform intervention math in float32 irrespective of model activation dtype."""

    return np.asarray(value, dtype=np.float32)
