from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lspe.config import load_config
from lspe.execution import ExperimentRunner
from lspe.models.base import ArchitectureInfo, LayerInfo


class FakeAdapter:
    def load(self, spec: Any) -> None:
        return None

    def unload(self) -> None:
        return None

    def format_prompt(self, messages: list[dict[str, str]]) -> list[int]:
        return [1, 2, 3]

    def architecture(self) -> ArchitectureInfo:
        return ArchitectureInfo(
            decoder_layer_path="layers",
            hidden_width=4,
            vocabulary_size=3,
            layers=(LayerInfo(0, 0.0, "fake", "unknown"),),
            final_norm_path="norm",
            output_head_path="head",
            cache_type="fake",
            cache_count=1,
            has_per_layer_inputs=False,
        )

    def wrap_layers(self, controller: Any) -> None:
        self.controller = controller

    def unwrap_layers(self) -> None:
        self.controller = None

    def make_cache(self) -> dict[str, str]:
        return {}

    def forward(self, token_ids: list[int], cache: Any = None) -> Any:
        return type("Forward", (), {"logits": np.array([[[0.0, 1.0, 2.0]]])})()

    def decode(self, token_ids: list[int]) -> str:
        return "[]"

    def eos_token_ids(self) -> set[int]:
        return {2}


def test_runner_creates_complete_append_only_paired_matrix(
    monkeypatch: Any, tmp_path: Path
) -> None:
    config = load_config("configs/smoke.gemma4-e4b.yaml")
    config = config.model_copy(
        update={
            "experiment": config.experiment.model_copy(update={"output_root": tmp_path}),
            "sampling": config.sampling.model_copy(update={"max_new_tokens": 1}),
        }
    )
    monkeypatch.setattr("lspe.execution.create_adapter", lambda _: FakeAdapter())
    runner = ExperimentRunner(config, Path("configs/smoke.gemma4-e4b.yaml"), "commit123", [0], 0.1)
    summary = runner.run()
    assert summary.failures == 0
    assert summary.committed_generations == summary.expected_generations
    assert summary.expected_generations == 80
    assert (
        len((summary.run_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()) == 80
    )
    assert (summary.run_dir / "generation-plan.jsonl").is_file()
    assert (summary.run_dir / "prompt-renders.jsonl").is_file()
    assert (summary.run_dir / "environment.txt").is_file()
    assert (summary.run_dir / "packages.lock.json").is_file()


def test_runner_refuses_resume_into_incompatible_manifest(
    monkeypatch: Any, tmp_path: Path
) -> None:
    config = load_config("configs/smoke.gemma4-e4b.yaml")
    config = config.model_copy(
        update={
            "experiment": config.experiment.model_copy(update={"output_root": tmp_path}),
            "sampling": config.sampling.model_copy(update={"max_new_tokens": 1}),
        }
    )
    monkeypatch.setattr("lspe.execution.create_adapter", lambda _: FakeAdapter())
    ExperimentRunner(config, Path("configs/smoke.gemma4-e4b.yaml"), "commit123", [0], 0.1).run()

    incompatible = ExperimentRunner(
        config, Path("configs/smoke.gemma4-e4b.yaml"), "commit123", [0], 0.2
    )
    with pytest.raises(RuntimeError, match="Existing run manifest is incompatible"):
        incompatible.run(resume=True)
