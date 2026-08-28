"""Dependence measures used to construct baseline functional head graphs."""

from __future__ import annotations

import numpy as np

from .nodes import HeadActivity


def linear_cka(first: np.ndarray, second: np.ndarray, epsilon: float = 1e-12) -> float:
    """Return centered linear CKA for two matched sample matrices.

    Rows are observations at identical prompt/token positions. Columns may have
    different widths. Constant matrices have no measurable dependence and are
    rejected instead of silently receiving a zero edge weight.
    """

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    x = _matrix(first, "first")
    y = _matrix(second, "second")
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must contain the same matched samples")
    x = x - np.mean(x, axis=0, keepdims=True)
    y = y - np.mean(y, axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    first_self = np.linalg.norm(x.T @ x, ord="fro")
    second_self = np.linalg.norm(y.T @ y, ord="fro")
    denominator = first_self * second_self
    if denominator <= epsilon:
        raise ValueError("CKA is undefined for constant activity")
    value = float(cross / denominator)
    return float(np.clip(value, 0.0, 1.0))


def pairwise_linear_cka(activities: list[HeadActivity]) -> tuple[list[str], np.ndarray]:
    """Build a deterministic symmetric CKA matrix ordered by canonical node ID."""

    if len(activities) < 2:
        raise ValueError("A functional graph requires at least two head nodes")
    ordered = sorted(activities, key=lambda activity: activity.node)
    sample_counts = {activity.values.shape[0] for activity in ordered}
    if len(sample_counts) != 1:
        raise ValueError("All head activities must use the same matched sample rows")
    identifiers = [activity.node.node_id for activity in ordered]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Functional head node IDs must be unique")
    matrix = np.eye(len(ordered), dtype=np.float64)
    for first in range(len(ordered)):
        for second in range(first + 1, len(ordered)):
            value = linear_cka(ordered[first].values, ordered[second].values)
            matrix[first, second] = value
            matrix[second, first] = value
    return identifiers, matrix


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] < 1:
        raise ValueError(f"{name} must be a two-dimensional sample matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result
