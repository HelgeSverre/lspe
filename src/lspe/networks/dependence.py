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


def mean_cosine_similarity(first: np.ndarray, second: np.ndarray, epsilon: float = 1e-12) -> float:
    """Mean matched-row cosine similarity of residual-contribution matrices."""

    x, y = _matched(first, second)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    valid = denominator > epsilon
    if not np.any(valid):
        raise ValueError("Cosine similarity is undefined for zero-only matched rows")
    return float(np.mean(np.sum(x[valid] * y[valid], axis=1) / denominator[valid]))


def rms_timeseries_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Pearson correlation between matched head-output RMS time series."""

    x, y = _matched(first, second)
    x_rms = np.sqrt(np.mean(np.square(x), axis=1))
    y_rms = np.sqrt(np.mean(np.square(y), axis=1))
    if np.std(x_rms) <= 1e-12 or np.std(y_rms) <= 1e-12:
        raise ValueError("RMS correlation is undefined for a constant time series")
    return float(np.corrcoef(x_rms, y_rms)[0, 1])


def attention_js_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """One minus normalized Jensen–Shannon divergence for matched attention rows."""

    x, y = _matched(first, second)
    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("Attention patterns cannot contain negative probabilities")
    x_total = np.sum(x, axis=1, keepdims=True)
    y_total = np.sum(y, axis=1, keepdims=True)
    if np.any(x_total <= 0) or np.any(y_total <= 0):
        raise ValueError("Attention pattern rows must contain positive mass")
    x = x / x_total
    y = y / y_total
    midpoint = 0.5 * (x + y)
    x_term = np.sum(np.where(x > 0, x * np.log(x / midpoint), 0.0), axis=1)
    y_term = np.sum(np.where(y > 0, y * np.log(y / midpoint), 0.0), axis=1)
    divergence = 0.5 * (x_term + y_term)
    return float(np.clip(1.0 - np.mean(divergence) / np.log(2.0), 0.0, 1.0))


def _matched(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _matrix(first, "first")
    y = _matrix(second, "second")
    if x.shape != y.shape:
        raise ValueError("Dependence inputs must have identical matched shapes")
    return x, y


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] < 1:
        raise ValueError(f"{name} must be a two-dimensional sample matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result
