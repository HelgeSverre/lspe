import numpy as np
import pytest

from lspe.networks.dynamic_connectivity import (
    AttentionNoiseController,
    apply_flattening,
    effective_rank,
    flattening_transform,
    matrix_similarity,
    mean_absolute_off_diagonal,
    random_basis_transform,
    regularized_head_correlation,
    selective_flattening_transform,
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


def test_selective_flattening_changes_only_selected_eigenmode() -> None:
    correlation = np.array([[1.0, 0.5], [0.5, 1.0]])
    transform = selective_flattening_transform(correlation, 1.0, frozenset({0}))
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    in_basis = eigenvectors.T @ transform @ eigenvectors
    assert in_basis[0, 0] == pytest.approx(eigenvalues[0] ** -0.5)
    assert in_basis[1, 1] == pytest.approx(1.0)
    assert abs(in_basis[0, 1]) < 1e-12


def test_selective_flattening_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="Invalid eigenvalue ranks"):
        selective_flattening_transform(np.eye(3), 0.4, frozenset({3}))


def test_random_basis_preserves_selective_transform_spectrum() -> None:
    correlation = np.array([[1.0, 0.4], [0.4, 1.0]])
    selective = selective_flattening_transform(correlation, 0.5, frozenset({0}))
    random = random_basis_transform(correlation, 0.5, frozenset({0}), seed=12)
    assert np.allclose(np.linalg.eigvalsh(random), np.linalg.eigvalsh(selective))
    assert np.allclose(random, random.T)


def test_random_basis_control_allows_super_whitening() -> None:
    transform = random_basis_transform(
        np.array([[1.0, 0.4], [0.4, 1.0]]), 2.0, frozenset({0}), seed=12
    )
    assert np.max(np.linalg.eigvalsh(transform)) > 1.0


def test_attention_noise_rejects_negative_sigma() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        AttentionNoiseController(frozenset({1}), -0.1, 3)


def test_matrix_similarity_uses_edges_not_diagonal() -> None:
    first = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.8], [0.4, 0.8, 1.0]])
    assert matrix_similarity(first, first) == pytest.approx(1.0)


def test_nonfinite_and_nonpositive_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite"):
        standardize_attention_rows(np.array([[0.0, np.nan], [1.0, 2.0]]))
    with pytest.raises(ValueError, match="positive definite"):
        flattening_transform(np.array([[1.0, 2.0], [2.0, 1.0]]), 0.5)
