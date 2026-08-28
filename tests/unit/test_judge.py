from lspe.judge import _pairs, _parse_assessment, _reparse_records


def test_judge_parser_requires_exact_rubric_schema() -> None:
    text = (
        '{"A":{"novelty":1,"usefulness":2,"coherence":3,"constraint_adherence":4,'
        '"plausibility":5},"B":{"novelty":5,"usefulness":4,"coherence":3,'
        '"constraint_adherence":2,"plausibility":1}}'
    )
    assessment, error = _parse_assessment(text)
    assert error is None
    assert assessment is not None
    assert assessment["A"]["novelty"] == 1
    assert _parse_assessment('{"A": {}}')[1] == "INVALID_TOP_LEVEL_SCHEMA"


def test_judge_parser_accepts_qwen_trailing_chat_terminator() -> None:
    text = (
        '{"A":{"novelty":1,"usefulness":2,"coherence":3,"constraint_adherence":4,'
        '"plausibility":5},"B":{"novelty":5,"usefulness":4,"coherence":3,'
        '"constraint_adherence":2,"plausibility":1}}<|im_end|>'
    )
    assessment, error = _parse_assessment(text)
    assert error is None
    assert assessment is not None
    assert assessment["B"]["plausibility"] == 1


def test_reparse_records_only_updates_parser_derived_fields() -> None:
    response = (
        '{"A":{"novelty":1,"usefulness":2,"coherence":3,"constraint_adherence":4,'
        '"plausibility":5},"B":{"novelty":5,"usefulness":4,"coherence":3,'
        '"constraint_adherence":2,"plausibility":1}}<|im_end|>'
    )
    original = {
        "pair_index": 3,
        "response": response,
        "ratings": None,
        "parse_failure": "INVALID_JSON",
    }
    repaired = _reparse_records([original])
    assert original["ratings"] is None
    assert repaired[0]["pair_index"] == 3
    assert repaired[0]["parse_failure"] is None
    assert repaired[0]["ratings"]["A"]["coherence"] == 3


def test_judge_pairs_are_paired_and_condition_agnostic_to_the_prompt() -> None:
    rows = [
        {"generation_id": "b", "prompt_id": "p", "generation_index": 0, "condition": "baseline"},
        {"generation_id": "c", "prompt_id": "p", "generation_index": 0, "condition": "coherent"},
        {"generation_id": "w", "prompt_id": "p", "generation_index": 0, "condition": "white"},
    ]
    pairs = _pairs(rows, master_seed=3)
    assert {
        tuple(sorted((left["generation_id"], right["generation_id"]))) for left, right in pairs
    } == {
        ("b", "c"),
        ("c", "w"),
    }
