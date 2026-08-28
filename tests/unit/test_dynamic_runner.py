from lspe.networks.dynamic_runner import select_dynamic_window


def test_dynamic_window_selection_never_uses_heldout_values() -> None:
    candidates = [
        {
            "start_layer": 0,
            "stop_layer_exclusive": 8,
            "median_tuning_similarity": 0.8,
            "median_tuning_synchrony": 0.03,
            "heldout_similarity": 0.1,
        },
        {
            "start_layer": 1,
            "stop_layer_exclusive": 9,
            "median_tuning_similarity": 0.7,
            "median_tuning_synchrony": 0.04,
            "heldout_similarity": 0.99,
        },
    ]
    assert select_dynamic_window(candidates)["start_layer"] == 0
