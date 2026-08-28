from collections import Counter

from lspe.networks.mapping_data import (
    build_network_map_dataset,
    load_network_map_dataset,
    network_map_hash,
)


def test_network_map_corpus_is_balanced_paired_and_stable(tmp_path) -> None:
    path = tmp_path / "network-map.jsonl"
    assert build_network_map_dataset(path) == 200
    rows = load_network_map_dataset(path)
    assert Counter(row.category for row in rows) == {
        "constrained": 34,
        "factual": 34,
        "narrative": 33,
        "analogical": 33,
        "code": 33,
        "control": 33,
    }
    assert len({row.pair_id for row in rows if row.pair_kind == "paraphrase"}) == 50
    assert len({row.pair_id for row in rows if row.pair_kind == "unrelated"}) == 25
    first_hash = network_map_hash(path)
    build_network_map_dataset(path, force=True)
    assert network_map_hash(path) == first_hash
