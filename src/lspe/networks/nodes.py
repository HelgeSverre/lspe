"""Canonical functional-node identifiers and observed activity containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, order=True)
class HeadNode:
    """One attention head at one decoder layer."""

    layer_index: int
    head_index: int
    shared_kv_index: int | None = None

    def __post_init__(self) -> None:
        if self.layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if self.head_index < 0:
            raise ValueError("head_index must be non-negative")
        if self.shared_kv_index is not None and self.shared_kv_index < 0:
            raise ValueError("shared_kv_index must be non-negative when present")

    @property
    def node_id(self) -> str:
        return f"L{self.layer_index:03d}H{self.head_index:03d}"


@dataclass(frozen=True)
class HeadActivity:
    """Matched residual contribution rows for one functional head node."""

    node: HeadNode
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.array(self.values, copy=True)
        if values.ndim != 2:
            raise ValueError("Head activity must be a samples-by-d_model matrix")
        if values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("Head activity needs at least two samples and one feature")
        if not np.isfinite(values).all():
            raise ValueError("Head activity must contain only finite values")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
