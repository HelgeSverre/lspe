from pathlib import Path

import pytest

from lspe.config import ModelConfig
from lspe.fetch import fetch_model


def test_fetch_local_model_requires_immutable_revision(tmp_path: Path) -> None:
    (tmp_path / "weights.safetensors").write_bytes(b"weights")
    model = ModelConfig(adapter="mlx_qwen3", repo_id="owner/model", local_path=tmp_path)
    with pytest.raises(RuntimeError, match="immutable source commit"):
        fetch_model(model)


def test_fetch_local_model_hashes_weight_files(tmp_path: Path) -> None:
    (tmp_path / "weights.safetensors").write_bytes(b"weights")
    model = ModelConfig(
        adapter="mlx_qwen3", repo_id="owner/model", local_path=tmp_path, revision="abc123def"
    )
    result = fetch_model(model)
    assert result.revision == "abc123def"
    assert set(result.weight_files) == {"weights.safetensors"}
