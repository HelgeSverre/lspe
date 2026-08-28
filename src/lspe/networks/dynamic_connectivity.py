"""Numerical core for dynamic attention-connectivity measurement and flattening."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def standardize_attention_rows(
    scores: np.ndarray, epsilon: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize finite `[heads, keys]` score rows and return their moments."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("Attention scores must have shape [heads >= 2, keys >= 2]")
    if not np.isfinite(values).all():
        raise ValueError("Eligible attention scores must be finite")
    means = np.mean(values, axis=1, keepdims=True)
    scales = np.std(values, axis=1, keepdims=True)
    if np.any(scales < epsilon):
        raise ValueError("Attention score row has near-zero variance")
    return (values - means) / scales, means, scales


def regularized_head_correlation(
    standardized_rows: list[np.ndarray], shrinkage: float | None = None
) -> tuple[np.ndarray, float, int]:
    """Estimate a positive-definite head correlation from variable-length rows."""

    if not standardized_rows:
        raise ValueError("At least one attention observation is required")
    matrices = [np.asarray(row, dtype=np.float64) for row in standardized_rows]
    if any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("All observations must be [heads, keys] with equal head count")
    head_counts = {matrix.shape[0] for matrix in matrices}
    if len(head_counts) != 1:
        raise ValueError("All observations must be [heads, keys] with equal head count")
    if not all(np.isfinite(matrix).all() for matrix in matrices):
        raise ValueError("Correlation observations must be finite")
    samples = np.concatenate([matrix.T for matrix in matrices], axis=0)
    if samples.shape[0] < 2:
        raise ValueError("Head correlation needs at least two key observations")
    empirical = np.corrcoef(samples, rowvar=False)
    if not np.isfinite(empirical).all():
        raise ValueError("Head correlation is non-finite")
    coefficient = _oas_identity_shrinkage(samples) if shrinkage is None else shrinkage
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("Shrinkage must be in [0, 1]")
    regularized = (1.0 - coefficient) * empirical + coefficient * np.eye(empirical.shape[0])
    return regularized, float(coefficient), int(samples.shape[0])


def flattening_transform(correlation: np.ndarray, alpha: float) -> np.ndarray:
    """Return the symmetric fractional whitening transform `C^(-alpha/2)`."""

    matrix = np.asarray(correlation, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Correlation must be square")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("Correlation must be finite and symmetric")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Alpha must be in [0, 1]")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("Correlation must be positive definite")
    transform = (eigenvectors * eigenvalues ** (-alpha / 2.0)) @ eigenvectors.T
    return 0.5 * (transform + transform.T)


def selective_flattening_transform(
    correlation: np.ndarray, alpha: float, eigenvalue_ranks: frozenset[int]
) -> np.ndarray:
    """Whiten only the selected eigenmodes of a correlation matrix."""

    matrix = np.asarray(correlation, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Correlation must be square")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("Correlation must be finite and symmetric")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Alpha must be in [0, 1]")
    invalid = sorted(eigenvalue_ranks - set(range(matrix.shape[0])))
    if invalid:
        raise ValueError(f"Invalid eigenvalue ranks: {invalid}")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("Correlation must be positive definite")
    gains = np.ones_like(eigenvalues)
    selected = np.array(sorted(eigenvalue_ranks), dtype=np.int64)
    if selected.size:
        gains[selected] = eigenvalues[selected] ** (-alpha / 2.0)
    transform = (eigenvectors * gains) @ eigenvectors.T
    return 0.5 * (transform + transform.T)


def random_basis_transform(
    correlation: np.ndarray,
    alpha: float,
    eigenvalue_ranks: frozenset[int],
    seed: int,
) -> np.ndarray:
    """Place a selective transform's eigenvalue gains in a frozen random basis."""

    matrix = np.asarray(correlation, dtype=np.float64)
    selective = selective_flattening_transform(matrix, alpha, eigenvalue_ranks)
    gains = np.linalg.eigvalsh(selective)
    rng = np.random.default_rng(seed)
    basis, triangular = np.linalg.qr(rng.normal(size=matrix.shape))
    signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    basis = basis * signs
    transform = (basis * gains) @ basis.T
    return 0.5 * (transform + transform.T)


@dataclass
class AttentionNoiseController:
    """Add deterministic independent score noise and restore per-head moments."""

    selected_layers: frozenset[int]
    sigma: float
    seed: int
    minimum_keys: int = 8
    maximum_mean_error: float = 0.0
    maximum_scale_error: float = 0.0
    nonfinite_count: int = 0
    zero_variance_count: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sigma < 0.0:
            raise ValueError("Attention-noise sigma must be non-negative")
        self._rng = np.random.default_rng(self.seed)

    def has_effect(self, layer_index: int) -> bool:
        return layer_index in self.selected_layers and self.sigma > 0.0

    def apply_mlx(self, layer_index: int, scores: Any) -> Any:
        if (
            layer_index not in self.selected_layers
            or self.sigma == 0.0
            or scores.shape[-2] != 1
            or scores.shape[-1] < self.minimum_keys
        ):
            return scores
        mx = __import__("mlx.core", fromlist=["array"])
        values = scores.astype(mx.float32)
        means = mx.mean(values, axis=-1, keepdims=True)
        scales = mx.sqrt(mx.mean(mx.square(values - means), axis=-1, keepdims=True))
        standardized = (values - means) / mx.maximum(scales, 1e-6)
        noise = mx.array(self._rng.normal(size=scores.shape), dtype=mx.float32)
        mixed = standardized + self.sigma * noise
        mixed_means = mx.mean(mixed, axis=-1, keepdims=True)
        mixed_scales = mx.sqrt(mx.mean(mx.square(mixed - mixed_means), axis=-1, keepdims=True))
        restored = (mixed - mixed_means) / mx.maximum(mixed_scales, 1e-6) * scales + means
        restored_means = mx.mean(restored, axis=-1, keepdims=True)
        restored_scales = mx.sqrt(
            mx.mean(mx.square(restored - restored_means), axis=-1, keepdims=True)
        )
        mx.eval(restored, restored_means, restored_scales, scales, mixed_scales)
        arrays = [np.asarray(value) for value in (restored, restored_means, restored_scales)]
        if not all(np.isfinite(value).all() for value in arrays):
            self.nonfinite_count += 1
            raise RuntimeError("Non-finite attention-noise output")
        self.zero_variance_count += int(np.sum(np.asarray(mixed_scales) < 1e-6))
        self.maximum_mean_error = max(
            self.maximum_mean_error,
            float(np.max(np.abs(np.asarray(restored_means) - np.asarray(means)))),
        )
        self.maximum_scale_error = max(
            self.maximum_scale_error,
            float(np.max(np.abs(np.asarray(restored_scales) - np.asarray(scales)))),
        )
        return restored.astype(scores.dtype)


def apply_flattening(
    scores: np.ndarray, transform: np.ndarray, epsilon: float = 1e-6
) -> np.ndarray:
    """Mix standardized head rows and restore every row's original moments."""

    standardized, means, scales = standardize_attention_rows(scores, epsilon)
    mixing = np.asarray(transform, dtype=np.float64)
    if mixing.shape != (standardized.shape[0], standardized.shape[0]):
        raise ValueError("Transform shape does not match attention head count")
    if not np.isfinite(mixing).all():
        raise ValueError("Transform must be finite")
    mixed = mixing @ standardized
    mixed_mean = np.mean(mixed, axis=1, keepdims=True)
    mixed_scale = np.std(mixed, axis=1, keepdims=True)
    if np.any(mixed_scale < epsilon):
        raise ValueError("Flattened attention row has near-zero variance")
    restored = (mixed - mixed_mean) / mixed_scale * scales + means
    if not np.isfinite(restored).all():
        raise ValueError("Flattened attention scores are non-finite")
    return restored


def matrix_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Pearson similarity between strict upper triangles."""

    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("Connectivity matrices must have identical square shapes")
    upper = np.triu_indices(x.shape[0], k=1)
    if np.std(x[upper]) <= 1e-12 or np.std(y[upper]) <= 1e-12:
        raise ValueError("Matrix similarity is undefined for constant edges")
    return float(np.corrcoef(x[upper], y[upper])[0, 1])


def mean_absolute_off_diagonal(matrix: np.ndarray) -> float:
    """Mean absolute strict-off-diagonal matrix value."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Matrix must be square")
    return float(np.mean(np.abs(values[np.triu_indices(values.shape[0], k=1)])))


def effective_rank(matrix: np.ndarray) -> float:
    """Entropy effective rank of a positive-semidefinite matrix."""

    eigenvalues = np.clip(np.linalg.eigvalsh(np.asarray(matrix, dtype=np.float64)), 0.0, None)
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        raise ValueError("Effective rank needs positive eigenvalue mass")
    probabilities = eigenvalues[eigenvalues > 0.0] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


@dataclass
class DynamicConnectivityController:
    """Apply frozen per-layer transforms and retain bounded mechanism telemetry."""

    transforms: dict[int, np.ndarray]
    minimum_keys: int = 8
    records: list[dict[str, Any]] = field(default_factory=list)

    def has_effect(self, layer_index: int) -> bool:
        """Return whether the layer's frozen transform differs from identity."""

        transform = self.transforms.get(layer_index)
        return transform is not None and not np.array_equal(transform, np.eye(transform.shape[0]))

    def apply_mlx(self, layer_index: int, scores: Any) -> Any:
        """Apply a transform to cached single-query MLX attention scores."""

        mx = __import__("mlx.core", fromlist=["array"])
        if layer_index not in self.transforms or scores.shape[-2] != 1:
            return scores
        if scores.shape[-1] < self.minimum_keys:
            return scores
        transform = mx.array(self.transforms[layer_index], dtype=mx.float32)
        values = scores.astype(mx.float32)
        means = mx.mean(values, axis=-1, keepdims=True)
        scales = mx.sqrt(mx.mean(mx.square(values - means), axis=-1, keepdims=True))
        standardized = (values - means) / mx.maximum(scales, 1e-6)
        mixed = mx.einsum("ij,bjqk->biqk", transform, standardized)
        mixed_means = mx.mean(mixed, axis=-1, keepdims=True)
        mixed_scales = mx.sqrt(mx.mean(mx.square(mixed - mixed_means), axis=-1, keepdims=True))
        restored = (mixed - mixed_means) / mx.maximum(mixed_scales, 1e-6) * scales + means
        mx.eval(standardized, restored)
        before = np.asarray(standardized[0, :, 0, :], dtype=np.float32)
        after_standardized = np.asarray(
            ((restored - means) / mx.maximum(scales, 1e-6))[0, :, 0, :],
            dtype=np.float32,
        )
        self.records.append({"layer": layer_index, "before": before, "after": after_standardized})
        return restored.astype(scores.dtype)


@dataclass
class DynamicCorrelationObserver:
    """Accumulate per-fold head correlation sufficient statistics on MLX."""

    layer_count: int
    head_count: int
    minimum_keys: int = 8
    current_fold: int = 0
    sums: np.ndarray = field(init=False)
    counts: np.ndarray = field(init=False)
    steps: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.sums = np.zeros(
            (4, self.layer_count, self.head_count, self.head_count), dtype=np.float64
        )
        self.counts = np.zeros((4, self.layer_count), dtype=np.int64)
        self.steps = np.zeros((4, self.layer_count), dtype=np.int64)

    def has_effect(self, layer_index: int) -> bool:
        """Observation never changes attention."""

        return False

    def apply_mlx(self, layer_index: int, scores: Any) -> Any:
        """Accumulate `U U^T` while returning scores untouched."""

        if scores.shape[-2] != 1 or scores.shape[-1] < self.minimum_keys:
            return scores
        mx = __import__("mlx.core", fromlist=["array"])
        values = scores.astype(mx.float32)
        means = mx.mean(values, axis=-1, keepdims=True)
        scales = mx.sqrt(mx.mean(mx.square(values - means), axis=-1, keepdims=True))
        standardized = (values - means) / mx.maximum(scales, 1e-6)
        gram = standardized[0, :, 0, :] @ standardized[0, :, 0, :].T
        mx.eval(gram)
        if not bool(np.isfinite(np.asarray(gram)).all()):
            raise RuntimeError("Non-finite DCF correlation telemetry")
        key_count = int(scores.shape[-1])
        self.sums[self.current_fold, layer_index] += np.asarray(gram, dtype=np.float64)
        self.counts[self.current_fold, layer_index] += key_count
        self.steps[self.current_fold, layer_index] += 1
        return scores

    def transforms(self) -> dict[int, np.ndarray]:
        """Return regularized fold/layer correlation matrices."""

        result = np.empty_like(self.sums)
        for fold in range(4):
            for layer in range(self.layer_count):
                count = int(self.counts[fold, layer])
                if count <= 0:
                    raise RuntimeError(f"No DCF observations for fold {fold}, layer {layer}")
                empirical = self.sums[fold, layer] / count
                coefficient = oas_shrinkage_from_correlation(empirical, count)
                result[fold, layer] = (1.0 - coefficient) * empirical + coefficient * np.eye(
                    self.head_count
                )
        return result

    def restore(self, sums: np.ndarray, counts: np.ndarray, steps: np.ndarray) -> None:
        """Restore a validated resume checkpoint."""

        if sums.shape != self.sums.shape or counts.shape != self.counts.shape:
            raise RuntimeError("DCF checkpoint shape does not match the runtime geometry")
        if steps.shape != self.steps.shape:
            raise RuntimeError("DCF checkpoint step shape does not match the runtime geometry")
        if not np.isfinite(sums).all() or np.any(counts < 0) or np.any(steps < 0):
            raise RuntimeError("DCF checkpoint contains invalid sufficient statistics")
        self.sums[...] = sums
        self.counts[...] = counts
        self.steps[...] = steps


@dataclass
class DynamicMechanismController:
    """Apply DCF while accumulating compact before/after correlation telemetry."""

    transforms: dict[int, np.ndarray]
    minimum_keys: int = 8
    sums_before: dict[int, np.ndarray] = field(default_factory=dict)
    sums_after: dict[int, np.ndarray] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)
    maximum_mean_error: float = 0.0
    maximum_scale_error: float = 0.0
    nonfinite_count: int = 0
    zero_variance_count: int = 0

    def has_effect(self, layer_index: int) -> bool:
        """Return whether this layer has an active transform."""

        return layer_index in self.transforms

    def apply_mlx(self, layer_index: int, scores: Any) -> Any:
        """Transform one cached score row and accumulate mechanism statistics."""

        if layer_index not in self.transforms or scores.shape[-2] != 1:
            return scores
        if scores.shape[-1] < self.minimum_keys:
            return scores
        mx = __import__("mlx.core", fromlist=["array"])
        values = scores.astype(mx.float32)
        means = mx.mean(values, axis=-1, keepdims=True)
        scales = mx.sqrt(mx.mean(mx.square(values - means), axis=-1, keepdims=True))
        transform = mx.array(self.transforms[layer_index], dtype=mx.float32)
        standardized = (values - means) / mx.maximum(scales, 1e-6)
        mixed = mx.einsum("ij,bjqk->biqk", transform, standardized)
        mixed_means = mx.mean(mixed, axis=-1, keepdims=True)
        mixed_scales = mx.sqrt(mx.mean(mx.square(mixed - mixed_means), axis=-1, keepdims=True))
        restored = (mixed - mixed_means) / mx.maximum(mixed_scales, 1e-6) * scales + means
        after = (restored - means) / mx.maximum(scales, 1e-6)
        before_gram = standardized[0, :, 0, :] @ standardized[0, :, 0, :].T
        after_gram = after[0, :, 0, :] @ after[0, :, 0, :].T
        restored_means = mx.mean(restored, axis=-1, keepdims=True)
        restored_scales = mx.sqrt(
            mx.mean(mx.square(restored - restored_means), axis=-1, keepdims=True)
        )
        mx.eval(
            before_gram,
            after_gram,
            restored_means,
            restored_scales,
            scales,
            mixed_scales,
        )
        before_np = np.asarray(before_gram, dtype=np.float64)
        after_np = np.asarray(after_gram, dtype=np.float64)
        if not np.isfinite(before_np).all() or not np.isfinite(after_np).all():
            self.nonfinite_count += 1
            raise RuntimeError("Non-finite DCF mechanism telemetry")
        self.zero_variance_count += int(np.sum(np.asarray(mixed_scales) < 1e-6))
        self.maximum_mean_error = max(
            self.maximum_mean_error,
            float(np.max(np.abs(np.asarray(restored_means) - np.asarray(means)))),
        )
        self.maximum_scale_error = max(
            self.maximum_scale_error,
            float(np.max(np.abs(np.asarray(restored_scales) - np.asarray(scales)))),
        )
        self.sums_before[layer_index] = (
            self.sums_before.get(layer_index, np.zeros_like(before_np)) + before_np
        )
        self.sums_after[layer_index] = (
            self.sums_after.get(layer_index, np.zeros_like(after_np)) + after_np
        )
        self.counts[layer_index] = self.counts.get(layer_index, 0) + int(scores.shape[-1])
        return restored.astype(scores.dtype)

    def correlations(self) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Return empirical before and after correlation matrices by layer."""

        if set(self.sums_before) != set(self.transforms):
            raise RuntimeError("DCF mechanism telemetry is incomplete")
        before = {layer: self.sums_before[layer] / self.counts[layer] for layer in self.transforms}
        after = {layer: self.sums_after[layer] / self.counts[layer] for layer in self.transforms}
        return before, after


def _oas_identity_shrinkage(samples: np.ndarray) -> float:
    """Oracle-approximating shrinkage coefficient for an identity target."""

    count, features = samples.shape
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    covariance = centered.T @ centered / count
    trace = float(np.trace(covariance))
    trace_square = float(np.sum(covariance * covariance))
    numerator = (1.0 - 2.0 / features) * trace_square + trace * trace
    denominator = (count + 1.0 - 2.0 / features) * (trace_square - trace * trace / features)
    if denominator <= 0.0:
        return 1.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def oas_shrinkage_from_correlation(correlation: np.ndarray, count: int) -> float:
    """Compute OAS identity shrinkage from an empirical correlation and sample count."""

    matrix = np.asarray(correlation, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or count < 2:
        raise ValueError("OAS needs a square correlation and at least two observations")
    features = matrix.shape[0]
    trace = float(np.trace(matrix))
    trace_square = float(np.sum(matrix * matrix))
    numerator = (1.0 - 2.0 / features) * trace_square + trace * trace
    denominator = (count + 1.0 - 2.0 / features) * (trace_square - trace * trace / features)
    if denominator <= 0.0:
        return 1.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))
