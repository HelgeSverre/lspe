from lspe.hashing import content_id


def test_content_id_changes_for_every_scientific_input() -> None:
    base = {
        "prompt_hash": "p",
        "model_revision": "r",
        "condition": "coherent",
        "layer_indices": [1, 2],
        "dose": 0.1,
        "direction_seed": 4,
        "sampling_seed": 5,
    }
    baseline_id = content_id(base)
    for key, replacement in {
        "prompt_hash": "different",
        "model_revision": "r2",
        "condition": "white",
        "layer_indices": [3],
        "dose": 0.2,
        "direction_seed": 6,
        "sampling_seed": 7,
    }.items():
        changed = {**base, key: replacement}
        assert content_id(changed) != baseline_id
