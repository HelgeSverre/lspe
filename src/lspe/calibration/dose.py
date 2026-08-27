"""Teacher-forced next-token distribution comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistributionDivergence:
    kl_altered_baseline: float
    js: float
    top1_agreement: bool
    top_k_overlap: float


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / weights.sum()


def distribution_divergence(
    altered_logits: np.ndarray, baseline_logits: np.ndarray, top_k: int = 10
) -> DistributionDivergence:
    """Compute full-softmax metrics from logits evaluated under the same prefix."""

    altered = softmax(altered_logits)
    baseline = softmax(baseline_logits)
    epsilon = np.finfo(np.float64).tiny
    kl = float(np.sum(altered * (np.log(altered + epsilon) - np.log(baseline + epsilon))))
    midpoint = (altered + baseline) / 2
    js = float(
        0.5 * np.sum(altered * (np.log(altered + epsilon) - np.log(midpoint + epsilon)))
        + 0.5 * np.sum(baseline * (np.log(baseline + epsilon) - np.log(midpoint + epsilon)))
    )
    count = min(top_k, altered.size)
    altered_top = set(np.argpartition(altered, -count)[-count:])
    baseline_top = set(np.argpartition(baseline, -count)[-count:])
    return DistributionDivergence(
        kl_altered_baseline=kl,
        js=js,
        top1_agreement=int(np.argmax(altered)) == int(np.argmax(baseline)),
        top_k_overlap=len(altered_top & baseline_top) / count,
    )
