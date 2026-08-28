import json
from collections import Counter

import pytest

from lspe.networks.dynamic_data import (
    audit_dynamic_map_leakage,
    build_dynamic_map_dataset,
    load_dynamic_map_dataset,
)


def test_dynamic_map_is_balanced_across_categories_and_folds(tmp_path) -> None:
    path = tmp_path / "dynamic-map.jsonl"
    assert build_dynamic_map_dataset(path) == 96
    rows = load_dynamic_map_dataset(path)
    assert set(Counter((row.category, row.fold) for row in rows).values()) == {4}
    for category in {row.category for row in rows}:
        category_rows = [row for row in rows if row.category == category]
        assert all(
            len({row.prompt for row in category_rows if row.fold == fold}) == 4
            for fold in range(4)
        )


def test_dynamic_map_refuses_overwrite(tmp_path) -> None:
    path = tmp_path / "dynamic-map.jsonl"
    build_dynamic_map_dataset(path)
    with pytest.raises(FileExistsError):
        build_dynamic_map_dataset(path)


def test_dynamic_map_leakage_rejects_an_earlier_prompt(tmp_path) -> None:
    path = tmp_path / "dynamic-map.jsonl"
    build_dynamic_map_dataset(path)
    row = load_dynamic_map_dataset(path)[0]
    earlier = tmp_path / "earlier.jsonl"
    earlier.write_text(json.dumps({"prompt_id": "old", "prompt": row.prompt}) + "\n")
    with pytest.raises(ValueError, match="leakage"):
        audit_dynamic_map_leakage(path, [earlier])
