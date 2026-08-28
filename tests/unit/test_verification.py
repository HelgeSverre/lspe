from pathlib import Path

from lspe.verification import (
    _stratified_replay_sample,
    _verify_stored_sham_pairs,
    verify_artifact_checksums,
    verify_replay,
    verify_run_projection,
    write_artifact_checksums,
)


def test_checksums_detect_mutation(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    artifact = run / "manifest.json"
    artifact.write_text('{"run_id":"one"}', encoding="utf-8")
    write_artifact_checksums(run)
    assert verify_artifact_checksums(run).passed
    artifact.write_text('{"run_id":"two"}', encoding="utf-8")
    verified = verify_artifact_checksums(run)
    assert not verified.passed
    assert "Checksum mismatch: manifest.json" in verified.reasons


def test_verification_rejects_incomplete_generation_projection(tmp_path: Path) -> None:
    run = tmp_path / "run"
    journal = run / "journal"
    journal.mkdir(parents=True)
    (run / "manifest.json").write_text('{"run_id":"one"}', encoding="utf-8")
    (run / "generation-plan.jsonl").write_text('{"generation_id":"sha256:one"}\n', encoding="utf-8")
    (run / "generations.jsonl").write_text("", encoding="utf-8")
    write_artifact_checksums(run)
    verified = verify_artifact_checksums(run)
    assert not verified.passed
    assert "Expected 1 generation rows but observed 0" in verified.reasons
    projection = verify_run_projection(run)
    assert not projection.passed
    assert "Expected 1 generation rows but observed 0" in projection.reasons


def test_replay_sampling_is_condition_stratified_and_checks_sham() -> None:
    plan = [
        {"generation_id": f"sha256:{condition}", "condition": condition}
        for condition in ("baseline", "sham", "coherent", "white", "temp_match")
    ]
    selected = _stratified_replay_sample(plan, sample=1, master_seed=3)
    assert {row["condition"] for row in selected} == {
        "baseline",
        "sham",
        "coherent",
        "white",
        "temp_match",
    }
    raw = {
        "baseline": {
            "prompt_id": "p",
            "generation_index": 0,
            "condition": "baseline",
            "output_token_ids": [1],
        },
        "sham": {
            "prompt_id": "p",
            "generation_index": 0,
            "condition": "sham",
            "output_token_ids": [2],
        },
    }
    paired_plan = [
        {"prompt_id": "p", "generation_index": 0, "condition": "baseline"},
    ]
    assert _verify_stored_sham_pairs(raw, paired_plan) == [
        "Stored baseline/sham outputs differ: p:0"
    ]
    assert _verify_stored_sham_pairs(raw, paired_plan, require_sham=False) == []


def test_replay_refuses_missing_run_artifacts(tmp_path: Path) -> None:
    result = verify_replay(tmp_path, sample=1)
    assert not result.passed
    assert result.reasons[0].startswith("Cannot prepare replay:")
