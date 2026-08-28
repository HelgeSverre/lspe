import numpy as np
import pytest

from lspe.networks import InMemoryHeadObserver, dense_head_contributions


def test_dense_head_contributions_sum_to_dense_projection() -> None:
    values = np.arange(24.0).reshape(2, 3, 4)
    weight = np.arange(20.0).reshape(5, 4) / 10
    contributions = dense_head_contributions(values, weight, head_count=2)
    expected = values @ weight.T
    assert contributions.shape == (2, 3, 2, 5)
    assert np.allclose(np.sum(contributions, axis=2), expected)


def test_in_memory_observer_builds_canonical_activity_rows() -> None:
    observer = InMemoryHeadObserver()
    observer.record_mlx(2, np.arange(24.0).reshape(1, 3, 2, 4))
    activities = observer.activities()
    assert [activity.node.node_id for activity in activities] == ["L002H000", "L002H001"]
    assert all(activity.values.shape == (3, 4) for activity in activities)
    assert observer.observation_count == 6


def test_in_memory_observer_can_retain_only_last_position() -> None:
    observer = InMemoryHeadObserver(last_position_only=True)
    observer.record_mlx(2, np.arange(24.0).reshape(1, 3, 2, 4))
    observer.record_mlx(2, np.arange(24.0).reshape(1, 3, 2, 4))
    activities = observer.activities()
    assert all(activity.values.shape == (2, 4) for activity in activities)
    assert observer.observation_count == 4


def test_attention_patterns_are_binned_and_normalized() -> None:
    observer = InMemoryHeadObserver(last_position_only=True, attention_bins=2)
    patterns = np.array(
        [
            [[[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]],
            [[[0.2, 0.3, 0.1, 0.4], [0.4, 0.1, 0.2, 0.3]]],
        ]
    )
    observer.record_attention_mlx(1, patterns)
    activity = observer.attention_patterns()[0]
    assert activity.values.shape == (2, 2)
    assert np.allclose(activity.values, [[0.3, 0.7], [0.5, 0.5]])


def test_dense_head_contributions_rejects_invalid_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        dense_head_contributions(np.ones((1, 1, 3)), np.ones((2, 3)), head_count=2)
