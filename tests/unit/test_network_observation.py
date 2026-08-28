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


def test_dense_head_contributions_rejects_invalid_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        dense_head_contributions(np.ones((1, 1, 3)), np.ones((2, 3)), head_count=2)
