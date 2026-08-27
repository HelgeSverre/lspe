from pathlib import Path

from lspe.config import load_config
from lspe.locking import create_experiment_lock, load_experiment_lock, write_experiment_lock


def test_experiment_lock_embeds_immutable_resolved_configuration(tmp_path: Path) -> None:
    config_path = Path("configs/pilot.gemma4-e4b.yaml")
    config = load_config(config_path)
    lock = create_experiment_lock(
        config,
        config_path,
        "pilot-run",
        "immutablecommit123",
        [1],
        ["DecoderLayer"],
        0.01,
        0.1,
        0.8,
    )
    path = tmp_path / "experiment.lock.yaml"
    write_experiment_lock(lock, path)
    restored = load_experiment_lock(path)
    assert restored.model["revision"] == "immutablecommit123"
    assert restored.resolved_config["schema_version"] == 1
