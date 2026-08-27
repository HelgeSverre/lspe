import numpy as np

from lspe.metrics.deterministic import valid_semantic_diversity
from lspe.tasks.validators import validate_response


def test_invalid_outputs_contribute_zero_to_vsd_pairs() -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert valid_semantic_diversity([True, False, True], embeddings) == 2 / 3


def test_divergent_words_validator_requires_exact_json_array() -> None:
    words = [
        "apple",
        "bridge",
        "cloud",
        "drum",
        "engine",
        "forest",
        "galaxy",
        "hammer",
        "island",
        "jacket",
    ]
    text = str(words).replace("'", '"')
    assert validate_response("divergent_words", text).valid
    assert not validate_response("divergent_words", "apple, bridge").valid


def test_json_validator_enforces_preregistered_schema() -> None:
    assert validate_response("json", '{"answer": 7}', {"answer": 7}).valid
    assert not validate_response("json", '{"answer": 8}', {"answer": 7}).valid


def test_python_function_validator_runs_preregistered_pure_function_cases() -> None:
    expected = {
        "function_name": "add_one",
        "cases": [{"args": [1], "expected": 2}, {"args": [-2], "expected": -1}],
    }
    assert validate_response(
        "python_function", "def add_one(x):\n    return x + 1\n", expected
    ).valid
    rejected = validate_response(
        "python_function", "import os\ndef add_one(x):\n return x\n", expected
    )
    assert rejected.failure_code == "CODE_IMPORT_FORBIDDEN"
