"""Observation-only collection of per-head residual contributions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .nodes import HeadActivity, HeadNode


@dataclass
class InMemoryHeadObserver:
    """Collect matched head rows for integrity tests and bounded mapping runs."""

    last_position_only: bool = False
    attention_bins: int = 16
    _rows: dict[HeadNode, list[np.ndarray]] = field(default_factory=dict)
    _patterns: dict[HeadNode, list[np.ndarray]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.attention_bins < 2:
            raise ValueError("attention_bins must be at least two")

    def record_mlx(self, layer_index: int, contributions: Any) -> None:
        """Copy `[batch, length, heads, d_model]` contributions to host memory."""

        values = np.asarray(contributions, dtype=np.float32)
        if values.ndim != 4:
            raise ValueError("Observed contributions must have four dimensions")
        if self.last_position_only:
            values = values[:, -1:, :, :]
        for head_index in range(values.shape[2]):
            node = HeadNode(layer_index, head_index)
            rows = values[:, :, head_index, :].reshape(-1, values.shape[-1]).copy()
            self._rows.setdefault(node, []).append(rows)

    def record_attention_mlx(self, layer_index: int, patterns: Any) -> None:
        """Bin variable-length attention patterns onto a fixed relative-position grid."""

        values = np.asarray(patterns, dtype=np.float32)
        if values.ndim != 4:
            raise ValueError("Attention patterns must have shape [batch, heads, query, key]")
        if self.last_position_only:
            values = values[:, :, -1:, :]
        key_count = values.shape[-1]
        assignments = np.minimum(
            np.arange(key_count) * self.attention_bins // key_count,
            self.attention_bins - 1,
        )
        binned = np.zeros((*values.shape[:-1], self.attention_bins), dtype=np.float32)
        for bin_index in range(self.attention_bins):
            binned[..., bin_index] = np.sum(values[..., assignments == bin_index], axis=-1)
        for head_index in range(values.shape[1]):
            node = HeadNode(layer_index, head_index)
            rows = binned[:, head_index, :, :].reshape(-1, self.attention_bins).copy()
            self._patterns.setdefault(node, []).append(rows)

    def activities(self) -> list[HeadActivity]:
        """Return immutable activities in canonical node order."""

        return [
            HeadActivity(node, np.concatenate(self._rows[node], axis=0))
            for node in sorted(self._rows)
        ]

    def attention_patterns(self) -> list[HeadActivity]:
        """Return fixed-bin pattern rows in canonical node order."""

        return [
            HeadActivity(node, np.concatenate(self._patterns[node], axis=0))
            for node in sorted(self._patterns)
        ]

    @property
    def observation_count(self) -> int:
        return sum(rows.shape[0] for chunks in self._rows.values() for rows in chunks)


def dense_head_contributions(
    concatenated_heads: np.ndarray, output_weight: np.ndarray, head_count: int
) -> np.ndarray:
    """Reference decomposition whose head sum equals a bias-free dense projection."""

    values = np.asarray(concatenated_heads, dtype=np.float64)
    weight = np.asarray(output_weight, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("Head input must have shape [batch, length, width]")
    if weight.ndim != 2 or weight.shape[1] != values.shape[-1]:
        raise ValueError("Output weight must have shape [output, input width]")
    if head_count <= 0 or values.shape[-1] % head_count:
        raise ValueError("Input width must be divisible by a positive head count")
    if not np.isfinite(values).all() or not np.isfinite(weight).all():
        raise ValueError("Projection inputs must be finite")
    head_width = values.shape[-1] // head_count
    heads = values.reshape(values.shape[0], values.shape[1], head_count, head_width)
    weights = weight.reshape(weight.shape[0], head_count, head_width)
    return np.einsum("blhd,ohd->blho", heads, weights)
