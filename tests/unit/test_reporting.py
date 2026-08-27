import json
from pathlib import Path

import pytest

from lspe.reporting import build_report


def test_report_builder_writes_all_required_formats(tmp_path: Path) -> None:
    report = {
        "schema_version": 1,
        "run_id": "run-1",
        "status": "NOT_SUPPORTED",
        "primary": {
            "metric": "valid_semantic_diversity",
            "contrast": "coherent-temp_match",
            "estimate": 0.0,
            "ci95": [0.0, 0.0],
            "p_value": 1.0,
            "n_prompts": 1,
        },
        "validity": {},
        "competence": {},
        "degeneration": {},
        "replication": {},
        "integrity": {},
        "artifact_root_hash": None,
    }
    build_report(tmp_path, report)
    assert all((tmp_path / name).is_file() for name in ("report.json", "report.md", "report.html"))


def test_report_builder_rejects_invalid_status(tmp_path: Path) -> None:
    report = {
        "schema_version": 1,
        "run_id": "run-1",
        "status": "UNKNOWN",
        "primary": {
            "metric": "valid_semantic_diversity",
            "contrast": "coherent-temp_match",
            "estimate": None,
            "ci95": [None, None],
            "p_value": None,
            "n_prompts": 0,
        },
        "validity": {},
        "competence": {},
        "degeneration": {},
        "replication": {},
        "integrity": {},
        "artifact_root_hash": None,
    }
    with pytest.raises(ValueError):
        build_report(tmp_path, report)


def test_published_schemas_are_valid_json() -> None:
    root = Path("schemas")
    names = {
        "config.schema.json",
        "prompt.schema.json",
        "generation.schema.json",
        "manifest.schema.json",
        "report.schema.json",
    }
    assert {path.name for path in root.glob("*.schema.json")} >= names
    for name in names:
        assert json.loads((root / name).read_text(encoding="utf-8"))["$schema"]
