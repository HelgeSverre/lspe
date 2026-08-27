import json
from pathlib import Path

from lspe.human_review import export_human_review


def test_human_review_export_blinds_condition_and_persists_key(tmp_path: Path) -> None:
    prompts = [
        {
            "prompt_id": "p1",
            "task_type": "arithmetic",
            "prompt": "Return 2.",
        }
    ]
    generations = [
        {
            "generation_id": "sha256:one",
            "prompt_id": "p1",
            "condition": "baseline",
            "output_text": "2",
            "stop_reason": "EOS",
            "validator": {"valid": True},
        },
        {
            "generation_id": "sha256:two",
            "prompt_id": "p1",
            "condition": "coherent",
            "output_text": "3",
            "stop_reason": "EOS",
            "validator": {"valid": False},
        },
    ]
    (tmp_path / "prompts.snapshot.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in prompts), encoding="utf-8"
    )
    (tmp_path / "generations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in generations), encoding="utf-8"
    )
    summary = export_human_review(tmp_path, master_seed=3, sample=1)
    review = json.loads((tmp_path / "human-review" / "review.json").read_text())
    key = json.loads((tmp_path / "human-review" / "unblinding-key.json").read_text())
    assert summary == {"sampled": 2, "available": 2}
    assert "condition" not in review["items"][0]
    assert {row["condition"] for row in key["items"]} == {"baseline", "coherent"}
    assert (tmp_path / "human-review" / "index.html").is_file()
