"""Primary valid semantic diversity calculation."""

from __future__ import annotations

import numpy as np


def valid_semantic_diversity(valid: list[bool], embeddings: np.ndarray) -> float:
    """Mean gated pairwise cosine distance, with invalid pairs contributing zero."""

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(valid):
        raise ValueError("One 2D embedding row is required for each validity result")
    count = len(valid)
    if count < 2:
        raise ValueError("VSD needs at least two generations")
    norms = np.linalg.vector_norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embeddings must be non-zero")
    normalized = values / norms
    total = 0.0
    for first in range(count):
        for second in range(first + 1, count):
            if valid[first] and valid[second]:
                total += 1 - float(np.dot(normalized[first], normalized[second]))
    return 2 * total / (count * (count - 1))
