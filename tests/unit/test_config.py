from pathlib import Path

import pytest

from lspe.config import load_config


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config = Path("configs/smoke.gemma4-e4b.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(config + "\nunknown_key: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_config(path)


def test_config_loads_strict_reference_profile() -> None:
    config = load_config(Path("configs/smoke.gemma4-e4b.yaml"))
    assert config.experiment.phase == "smoke"
    assert config.execution.batch_size == 1
