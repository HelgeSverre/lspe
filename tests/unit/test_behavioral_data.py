import json

from lspe.networks.behavioral_data import build_behavioral_datasets
from lspe.tasks.validators import validate_response


def test_behavioral_data_is_balanced_and_unique(tmp_path) -> None:
    counts = build_behavioral_datasets(tmp_path)
    assert counts == {"calibration": 12, "pilot": 24, "confirm": 48}
    pilot = [json.loads(line) for line in (tmp_path / "pilot.jsonl").read_text().splitlines()]
    assert {row["category"] for row in pilot} == {
        "open_association",
        "analogical",
        "narrative",
        "constrained",
        "factual",
        "code",
    }
    assert all(
        sum(row["category"] == category for row in pilot) == 4
        for category in {row["category"] for row in pilot}
    )
    assert json.loads((tmp_path / "leakage-audit.json").read_text())["passed"]


def test_bounded_prose_validator_enforces_word_range() -> None:
    expected = {"minimum_words": 3, "maximum_words": 5}
    assert validate_response("bounded_prose", "one two three", expected).valid
    assert not validate_response("bounded_prose", "one two", expected).valid
