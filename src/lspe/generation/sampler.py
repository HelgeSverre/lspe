"""Deterministic local sampler with auditable post-filter telemetry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..calibration.entropy import filtered_probabilities


@dataclass(frozen=True)
class SampledToken:
    token_id: int
    log_probability: float
    entropy: float
    top1_probability: float
    top1_top2_margin: float
    top_token_ids: tuple[int, ...]
    top_log_probabilities: tuple[float, ...]


def sample_token(
    logits: np.ndarray,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    store_top_logprobs: int,
) -> SampledToken:
    """Sample exactly once from the distribution whose metrics are stored."""

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise FloatingPointError("Sampler received non-finite logits")
    probabilities = filtered_probabilities(values, temperature, top_k, top_p)
    rng = np.random.default_rng(seed)
    token_id = int(rng.choice(probabilities.size, p=probabilities))
    nonzero = probabilities > 0
    entropy = float(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))
    order = np.argsort(probabilities)[::-1]
    size = min(store_top_logprobs, probabilities.size)
    selected = order[:size]
    top1 = float(probabilities[order[0]])
    top2 = float(probabilities[order[1]]) if probabilities.size > 1 else 0.0
    return SampledToken(
        token_id=token_id,
        log_probability=float(np.log(probabilities[token_id])),
        entropy=entropy,
        top1_probability=top1,
        top1_top2_margin=top1 - top2,
        top_token_ids=tuple(int(value) for value in selected),
        top_log_probabilities=tuple(float(np.log(probabilities[value])) for value in selected),
    )
