import numpy as np
import pytest

from lspe.preflight import baseline_generation_sanity, compare_logits, intervention_liveness


def test_layer_wrapper_zero_dose_logits() -> None:
    comparison = compare_logits(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]))
    assert comparison.passed


def test_nonfinite_intervention_is_fatal() -> None:
    with pytest.raises(FloatingPointError):
        compare_logits(np.array([[1.0]]), np.array([[np.nan]]))


def test_liveness_accepts_changed_logits_without_argmax_change() -> None:
    class Adapter:
        def make_cache(self) -> object:
            return object()

        def forward(self, tokens: list[int], cache: object | None = None) -> object:
            logits = np.array([[[2.0, 1.0]]])
            if getattr(self, "wrapped", False):
                logits = np.array([[[2.1, 1.0]]])
            return type("Result", (), {"logits": logits})()

        def wrap_layers(self, controller: object) -> None:
            self.wrapped = True

        def unwrap_layers(self) -> None:
            self.wrapped = False

    result = intervention_liveness(
        Adapter(), [1, 2], master_seed=3, run_id="run", selected_layers=[0], dose=0.01
    )
    assert result.passed
    assert result.greedy_equal


def test_baseline_generation_sanity_rejects_control_token_leakage() -> None:
    class Adapter:
        def make_cache(self) -> object:
            return object()

        def forward(self, _tokens: list[int], cache: object | None = None) -> object:
            return type("Result", (), {"logits": np.array([[[0.0, 0.0, 0.0, 1.0]]])})()

        def eos_token_ids(self) -> set[int]:
            return {1}

        def decode(self, token_ids: list[int]) -> str:
            return "".join("<turn|>" if token == 3 else "x" for token in token_ids)

    result = baseline_generation_sanity(Adapter(), [[9]], [("4",)], max_new_tokens=2)
    assert not result["passed"]
    assert result["outputs"][0]["control_token_leak"]
