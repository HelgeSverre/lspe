import numpy as np
import pytest

from lspe.networks import (
    HeadActivity,
    HeadNode,
    adjusted_rand_index,
    attention_js_similarity,
    degree_preserving_null_modularities,
    density_threshold,
    linear_cka,
    mean_cosine_similarity,
    pairwise_linear_cka,
    rms_timeseries_correlation,
    spectral_communities,
    weighted_modularity,
)


def test_head_node_has_stable_identifier_and_validates_indices() -> None:
    assert HeadNode(2, 7, 1).node_id == "L002H007"
    with pytest.raises(ValueError, match="layer_index"):
        HeadNode(-1, 0)


def test_head_activity_owns_an_immutable_snapshot() -> None:
    source = np.arange(6.0).reshape(3, 2)
    activity = HeadActivity(HeadNode(0, 0), source)
    source[0, 0] = 99.0
    assert activity.values[0, 0] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        activity.values[0, 0] = 1.0


def test_linear_cka_is_invariant_to_orthogonal_feature_rotation() -> None:
    first = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert linear_cka(first, first @ rotation) == pytest.approx(1.0)


def test_linear_cka_rejects_unmatched_and_constant_activity() -> None:
    with pytest.raises(ValueError, match="same matched samples"):
        linear_cka(np.ones((3, 2)), np.ones((4, 2)))
    with pytest.raises(ValueError, match="constant activity"):
        linear_cka(np.ones((3, 2)), np.arange(6).reshape(3, 2))


def test_pairwise_cka_uses_canonical_node_order() -> None:
    values = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    identifiers, graph = pairwise_linear_cka(
        [
            HeadActivity(HeadNode(2, 0), values),
            HeadActivity(HeadNode(1, 3), values.copy()),
        ]
    )
    assert identifiers == ["L001H003", "L002H000"]
    assert np.allclose(graph, np.ones((2, 2)))


def test_secondary_dependence_measures_have_expected_extremes() -> None:
    first = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    assert mean_cosine_similarity(first, first) == pytest.approx(1.0)
    assert rms_timeseries_correlation(first, first) == pytest.approx(1.0)
    patterns = np.array([[0.8, 0.2], [0.3, 0.7]])
    assert attention_js_similarity(patterns, patterns) == pytest.approx(1.0)


def test_weighted_modularity_detects_two_separated_communities() -> None:
    adjacency = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    assert weighted_modularity(adjacency, [0, 0, 1, 1]) == pytest.approx(0.5)
    assert weighted_modularity(adjacency, [0, 1, 0, 1]) == pytest.approx(-0.5)


def test_weighted_modularity_fails_closed_on_invalid_graphs() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        weighted_modularity(np.array([[0.0, 1.0], [0.0, 0.0]]), [0, 1])
    with pytest.raises(ValueError, match="no edges"):
        weighted_modularity(np.zeros((2, 2)), [0, 1])


def test_spectral_communities_and_null_are_deterministic() -> None:
    graph = np.array(
        [
            [0.0, 0.9, 0.8, 0.1, 0.1, 0.0],
            [0.9, 0.0, 0.7, 0.0, 0.1, 0.1],
            [0.8, 0.7, 0.0, 0.1, 0.0, 0.1],
            [0.1, 0.0, 0.1, 0.0, 0.8, 0.9],
            [0.1, 0.1, 0.0, 0.8, 0.0, 0.7],
            [0.0, 0.1, 0.1, 0.9, 0.7, 0.0],
        ]
    )
    thresholded = density_threshold(graph, 0.5)
    labels = spectral_communities(thresholded, 2, seed=7)
    assert adjusted_rand_index(labels, [0, 0, 0, 1, 1, 1]) == pytest.approx(1.0)
    first = degree_preserving_null_modularities(thresholded, labels, samples=3, seed=9)
    second = degree_preserving_null_modularities(thresholded, labels, samples=3, seed=9)
    assert np.array_equal(first, second)
