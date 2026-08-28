from lspe.networks.consensus_runner import _select_candidate


def test_corrected_heldout_values_cannot_change_selection() -> None:
    candidates = [
        {
            "density": 0.075,
            "community_count": 3,
            "eligible_nodes": 100,
            "tuning_ari": 0.8,
            "heldout_ari": 0.1,
            "heldout_coverage": 0.95,
        },
        {
            "density": 0.1,
            "community_count": 4,
            "eligible_nodes": 100,
            "tuning_ari": 0.7,
            "heldout_ari": 0.99,
            "heldout_coverage": 1.0,
        },
    ]
    assert _select_candidate(candidates)["density"] == 0.075
