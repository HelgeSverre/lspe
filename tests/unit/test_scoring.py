import json
from pathlib import Path

import numpy as np

from lspe.scoring import score_run


def test_score_run_preserves_prompt_metadata_per_aggregate(
    tmp_path: Path, monkeypatch
) -> None:
    prompts = [
        {
            "schema_version": 1,
            "prompt_id": "creative",
            "split": "pilot",
            "task_type": "divergent_words",
            "system_variant": "neutral",
            "prompt": "Return two unrelated nouns as a JSON array.",
            "response_schema": "divergent_words.v1",
            "validator": "divergent_words",
            "expected": None,
            "tags": ["pilot", "creativity"],
        },
        {
            "schema_version": 1,
            "prompt_id": "control",
            "split": "controls",
            "task_type": "arithmetic",
            "system_variant": "neutral",
            "prompt": "Return only the integer result of 2 * 3.",
            "response_schema": "text.v1",
            "validator": "exact_answer",
            "expected": "6",
            "tags": ["pilot", "control"],
        },
    ]
    (tmp_path / "prompts.snapshot.jsonl").write_text(
        "".join(json.dumps(prompt) + "\n" for prompt in prompts), encoding="utf-8"
    )
    generations = [
        {
            "generation_id": f"g{index}",
            "prompt_id": prompt_id,
            "condition": "coherent",
            "output_text": output,
            "output_token_ids": [1, 2],
        }
        for index, (prompt_id, output) in enumerate(
            (("creative", '["harbor", "forest"]'), ("control", "6")), start=1
        )
    ]
    (tmp_path / "generations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in generations), encoding="utf-8"
    )
    monkeypatch.setattr(
        "lspe.scoring._embed_texts",
        lambda texts, _model, _revision: np.ones((len(texts), 2), dtype=np.float64),
    )

    score_run(tmp_path, "fixture")

    effects = {
        row["prompt_id"]: row
        for row in (
            json.loads(line)
            for line in (tmp_path / "prompt-effects.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    assert effects["creative"]["split"] == "pilot"
    assert effects["creative"]["tags"] == ["pilot", "creativity"]
    assert effects["control"]["split"] == "controls"
    assert effects["control"]["tags"] == ["pilot", "control"]
