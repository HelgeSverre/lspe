"""Post-filter sampling entropy and deterministic temperature matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemperatureMatch:
    temperature: float
    target_entropy: float
    achieved_entropy: float
    absolute_mismatch: float


def filtered_probabilities(
    logits: np.ndarray, temperature: float, top_k: int, top_p: float
) -> np.ndarray:
    """Apply the actual sampler's temperature and top-k/top-p filters."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = np.asarray(logits, dtype=np.float64) / temperature
    if top_k > 0 and top_k < values.size:
        keep = np.argpartition(values, -top_k)[-top_k:]
        masked = np.full_like(values, -np.inf)
        masked[keep] = values[keep]
        values = masked
    finite = np.isfinite(values)
    shifted = values[finite] - np.max(values[finite])
    probabilities = np.zeros_like(values)
    probabilities[finite] = np.exp(shifted)
    probabilities /= probabilities.sum()
    if top_p < 1:
        order = np.argsort(probabilities)[::-1]
        cdf = np.cumsum(probabilities[order])
        remove = order[cdf > top_p]
        # Keep the first token whose cumulative mass crosses the threshold.
        if remove.size:
            remove = remove[1:]
            probabilities[remove] = 0
            probabilities /= probabilities.sum()
    return probabilities


def sampling_entropy(logits: np.ndarray, temperature: float, top_k: int, top_p: float) -> float:
    probabilities = filtered_probabilities(logits, temperature, top_k, top_p)
    nonzero = probabilities > 0
    return float(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))


def match_temperature(
    logits_rows: list[np.ndarray],
    target_entropy: float,
    top_k: int,
    top_p: float,
    minimum: float = 0.05,
    maximum: float = 3.0,
    steps: int = 201,
) -> TemperatureMatch:
    """Use a bounded grid to make the frozen calibration easy to audit/replay."""

    if not logits_rows:
        raise ValueError("Temperature matching requires at least one calibration distribution")
    candidates = np.linspace(minimum, maximum, steps)
    means = np.array(
        [
            np.mean([sampling_entropy(row, candidate, top_k, top_p) for row in logits_rows])
            for candidate in candidates
        ]
    )
    index = int(np.argmin(np.abs(means - target_entropy)))
    achieved = float(means[index])
    return TemperatureMatch(
        temperature=float(candidates[index]),
        target_entropy=target_entropy,
        achieved_entropy=achieved,
        absolute_mismatch=abs(achieved - target_entropy),
    )
