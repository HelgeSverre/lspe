from lspe.networks.mapping_sensitivity import select_nested_candidate


def test_nested_selection_does_not_use_heldout_score() -> None:
    selected = select_nested_candidate(
        [
            {
                "density": 0.1,
                "community_count": 4,
                "eligible_nodes": 100,
                "tuning_ari": 0.8,
                "heldout_ari": 0.1,
            },
            {
                "density": 0.2,
                "community_count": 3,
                "eligible_nodes": 100,
                "tuning_ari": 0.7,
                "heldout_ari": 0.99,
            },
        ]
    )
    assert selected["density"] == 0.1
