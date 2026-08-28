from lspe.networks.mapping_data import build_network_map_dataset
from lspe.networks.phase2_data import (
    SPLIT_COUNTS,
    audit_phase2_leakage,
    build_phase2_datasets,
    load_phase2_dataset,
    phase2_data_hashes,
)


def test_all_phase2_splits_are_disjoint_frozen_and_audited(tmp_path) -> None:
    build_network_map_dataset(tmp_path / "network_map.jsonl")
    assert build_phase2_datasets(tmp_path) == SPLIT_COUNTS
    for split, count in SPLIT_COUNTS.items():
        assert len(load_phase2_dataset(tmp_path / f"{split}.jsonl", split)) == count
    assert set(phase2_data_hashes(tmp_path)) == {"network_map", *SPLIT_COUNTS}
    audit = audit_phase2_leakage(tmp_path)
    assert audit["passed"] is True
    assert audit["prompt_count"] == 440
