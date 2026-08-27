from pathlib import Path

from lspe.run_store import RunStore


def test_resume_is_idempotent(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.initialize({"run_id": "test"})
    inputs = {"prompt": "p", "seed": 1, "condition": "baseline"}
    record = {"text": "answer"}

    first = store.commit_generation(inputs, record)
    second = store.commit_generation(inputs, record)

    assert first.committed is True
    assert second.committed is False
    assert store.completed_ids() == {first.generation_id}
    assert len(store.generations_path.read_text(encoding="utf-8").splitlines()) == 1
