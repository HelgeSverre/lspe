import json
from collections import Counter

import pytest

from lspe.networks.mapping_data_v2 import (
    audit_v2_mapping_leakage,
    build_network_map_v2_dataset,
    load_network_map_v2_dataset,
)


def test_v2_mapping_corpus_is_fresh_balanced_and_paired(tmp_path) -> None:
    path = tmp_path / "network-map-v2.jsonl"
    assert build_network_map_v2_dataset(path) == 240
    rows = load_network_map_v2_dataset(path)
    assert set(Counter(row.category for row in rows).values()) == {40}
    assert len({row.pair_id for row in rows if row.pair_kind == "paraphrase"}) == 60
    assert len({row.pair_id for row in rows if row.pair_kind == "unrelated"}) == 30


def test_v2_mapping_leakage_audit_rejects_an_earlier_prompt(tmp_path) -> None:
    path = tmp_path / "network-map-v2.jsonl"
    build_network_map_v2_dataset(path)
    row = load_network_map_v2_dataset(path)[0]
    earlier = tmp_path / "earlier.jsonl"
    earlier.write_text(json.dumps({"prompt_id": "old", "prompt": row.prompt}) + "\n")
    with pytest.raises(ValueError, match="leakage"):
        audit_v2_mapping_leakage(path, [earlier])
