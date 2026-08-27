"""Multiple-comparison procedures used only for secondary outcome families."""

from __future__ import annotations

import numpy as np


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return familywise Holm-adjusted p-values in original order."""

    values = np.asarray(p_values, dtype=float)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        [(values[index] * (len(values) - rank)) for rank, index in enumerate(order)]
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1)
    return adjusted.tolist()
