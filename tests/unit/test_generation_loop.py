from typing import Any

import numpy as np

from lspe.config import load_config
from lspe.generation.loop import GenerationLoop


class FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def make_cache(self) -> dict[str, str]:
        return {}

    def forward(self, token_ids: list[int], cache: Any = None) -> Any:
        self.calls.append(token_ids)
        return type("Forward", (), {"logits": np.array([[[0.0, 1.0, 2.0]]])})()

    def decode(self, token_ids: list[int]) -> str:
        return "".join(str(token) for token in token_ids)

    def eos_token_ids(self) -> set[int]:
        return {2}


def test_decode_only_does_not_modify_early_prefill() -> None:
    config = load_config("configs/smoke.gemma4-e4b.yaml")
    adapter = FakeAdapter()
    result = GenerationLoop(adapter, config.sampling, 4).generate([8, 9, 10], "p", 0, "baseline")
    assert adapter.calls[0] == [8, 9]
    assert adapter.calls[1] == [10]
    assert result.stop_reason == "EOS"
    assert result.output_token_ids == ()
    assert result.text == ""
