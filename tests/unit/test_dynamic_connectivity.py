import numpy as np
import pytest

from lspe.networks.dynamic_connectivity import (
    apply_flattening,
    effective_rank,
    flattening_transform,
    matrix_similarity,
    mean_absolute_off_diagonal,
    regularized_head_correlation,
    standardize_attention_rows,
)


def test_identity_flattening_is_exact_sham_and_preserves_moments() -> None:
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(8, 23))
    correlation = np.eye(8)
    sham = apply_flattening(scores, flattening_transform(correlation, 0.0))
    assert np.allclose(sham, scores, atol=1e-12)
    mixed = apply_flattening(scores, flattening_transform(correlation + 0.1, 0.5))
    assert np.allclose(np.mean(mixed, axis=1), np.mean(scores, axis=1), atol=1e-12)
    assert np.allclose(np.std(mixed, axis=1), np.std(scores, axis=1), atol=1e-12)


def test_fractional_whitening_reduces_synthetic_head_correlation() -> None:
    rng = np.random.default_rng(19)
    common = rng.normal(size=400)
    scores = np.stack([common + 0.3 * rng.normal(size=400) for _ in range(8)])
    standardized, _, _ = standardize_attention_rows(scores)
    correlation, _, count = regularized_head_correlation([standardized], shrinkage=0.01)
    transform = flattening_transform(correlation, 1.0)
    flattened = apply_flattening(scores, transform)
    before = np.corrcoef(scores)
    after = np.corrcoef(flattened)
    assert count == 400
    assert mean_absolute_off_diagonal(after) < mean_absolute_off_diagonal(before)
    assert effective_rank(after) > effective_rank(before)


def test_matrix_similarity_uses_edges_not_diagonal() -> None:
    first = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.8], [0.4, 0.8, 1.0]])
    assert matrix_similarity(first, first) == pytest.approx(1.0)


def test_nonfinite_and_nonpositive_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite"):
        standardize_attention_rows(np.array([[0.0, np.nan], [1.0, 2.0]]))
    with pytest.raises(ValueError, match="positive definite"):
        flattening_transform(np.array([[1.0, 2.0], [2.0, 1.0]]), 0.5)
