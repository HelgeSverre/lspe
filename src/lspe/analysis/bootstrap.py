"""Prompt-level cluster bootstrap and paired sign-flip test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    median: float
    ci95: tuple[float, float]
    standardized_effect: float | None
    positive: int
    zero: int
    negative: int
    sign_flip_p_value: float


def paired_bootstrap(
    differences: np.ndarray,
    seed: int,
    samples: int = 10_000,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Resample prompts, never individual generations, for paired inference."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a non-empty 1D array")
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    resampled_means = values[indices].mean(axis=1)
    tail = (1 - confidence_level) / 2
    ci = tuple(float(value) for value in np.quantile(resampled_means, [tail, 1 - tail]))
    std = values.std(ddof=1) if values.size > 1 else 0
    standardized = float(values.mean() / std) if std > 0 else None
    # Exact enumeration is needlessly explosive; a deterministic Monte Carlo test is auditable.
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, values.size))
    null_means = (signs * values).mean(axis=1)
    observed = abs(float(values.mean()))
    p_value = float((1 + np.count_nonzero(np.abs(null_means) >= observed)) / (samples + 1))
    return BootstrapResult(
        estimate=float(values.mean()),
        median=float(np.median(values)),
        ci95=(ci[0], ci[1]),
        standardized_effect=standardized,
        positive=int(np.count_nonzero(values > 0)),
        zero=int(np.count_nonzero(values == 0)),
        negative=int(np.count_nonzero(values < 0)),
        sign_flip_p_value=p_value,
    )
