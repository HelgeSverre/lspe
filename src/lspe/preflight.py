"""Fail-closed model integrity checks used before pilot/confirmatory execution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .interventions.controller import InterventionController
from .models.base import ArchitectureInfo


@dataclass(frozen=True)
class LogitComparison:
    maximum_absolute_error: float
    mean_absolute_error: float
    greedy_equal: bool
    passed: bool


@dataclass(frozen=True)
class ZeroDoseSuite:
    """Aggregate zero-dose evidence over no-cache and cached decode paths."""

    prompt_count: int
    cached_decode_steps: int
    no_cache: LogitComparison
    cached_decode: LogitComparison

    @property
    def passed(self) -> bool:
        return self.no_cache.passed and self.cached_decode.passed


def compare_logits(
    baseline: np.ndarray, instrumented: np.ndarray, tolerance: float = 1e-6
) -> LogitComparison:
    """Check zero-dose/cache identity without weakening numerical tolerances silently."""

    first = np.asarray(baseline, dtype=np.float64)
    second = np.asarray(instrumented, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(
            f"Cannot compare logits with different shapes: {first.shape} vs {second.shape}"
        )
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise FloatingPointError("Model-integrity comparison received non-finite logits")
    absolute = np.abs(first - second)
    maximum = float(np.max(absolute))
    mean = float(np.mean(absolute))
    greedy_equal = bool(np.array_equal(np.argmax(first, axis=-1), np.argmax(second, axis=-1)))
    return LogitComparison(maximum, mean, greedy_equal, maximum <= tolerance and greedy_equal)


def cache_equivalence(
    adapter: Any, token_ids: list[int], tolerance: float = 1e-6
) -> LogitComparison:
    """Compare full forward with semantically equivalent cached token decoding."""

    if len(token_ids) < 2:
        raise ValueError("Cache equivalence requires at least two tokens")
    full = adapter.forward(token_ids).logits
    cache = adapter.make_cache()
    adapter.forward(token_ids[:-1], cache=cache)
    cached = adapter.forward(token_ids[-1:], cache=cache).logits
    full_last = np.asarray(full).reshape(-1, np.asarray(full).shape[-1])[-1]
    cached_last = np.asarray(cached).reshape(-1, np.asarray(cached).shape[-1])[-1]
    return compare_logits(full_last, cached_last, tolerance)


def write_architecture(architecture: ArchitectureInfo, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(architecture), indent=2, sort_keys=True), encoding="utf-8")


def baseline_logit_sanity(adapter: Any, prompt_token_ids: list[int]) -> dict[str, Any]:
    """Confirm the adapter can produce finite logits before intervention is enabled."""

    result = adapter.forward(prompt_token_ids)
    logits = np.asarray(result.logits)
    if not np.isfinite(logits).all():
        raise FloatingPointError("Baseline inference produced non-finite logits")
    flattened = logits.reshape(-1, logits.shape[-1])[-1]
    return {
        "vocabulary_size": int(flattened.size),
        "greedy_token_id": int(np.argmax(flattened)),
        "finite": True,
    }


def baseline_generation_sanity(
    adapter: Any,
    prompt_token_ids: list[list[int]],
    expected_fragments: list[tuple[str, ...]],
    *,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Exercise the real text path and reject obvious instruction-model failures."""

    if len(prompt_token_ids) != len(expected_fragments) or not prompt_token_ids:
        raise ValueError("Baseline generation prompts and expectations must align")
    if max_new_tokens < 1:
        raise ValueError("Baseline generation requires a positive token budget")
    control_markers = ("<turn", "<|turn", "<|channel", "<channel|>", "<eos>", "<bos>")
    outputs: list[dict[str, Any]] = []
    for tokens, expected in zip(prompt_token_ids, expected_fragments, strict=True):
        output_ids, stop_reason = _greedy_generate(adapter, tokens, max_new_tokens)
        text = adapter.decode(output_ids)
        lowered = text.lower()
        control_token_leak = any(marker in lowered for marker in control_markers)
        repeated = _has_repeated_4gram(output_ids)
        outputs.append(
            {
                "token_count": len(output_ids),
                "stop_reason": stop_reason,
                "control_token_leak": control_token_leak,
                "repeated_4gram": repeated,
                "expected_answer_observed": all(
                    fragment.lower() in lowered for fragment in expected
                ),
            }
        )
    passed = (
        not any(output["control_token_leak"] for output in outputs)
        and not any(output["repeated_4gram"] for output in outputs)
        and any(output["expected_answer_observed"] for output in outputs)
    )
    return {"prompt_count": len(outputs), "outputs": outputs, "passed": passed}


def zero_dose_identity(
    adapter: Any,
    prompt_token_ids: list[int],
    *,
    master_seed: int,
    run_id: str,
    selected_layers: list[int],
    tolerance: float = 1e-6,
) -> LogitComparison:
    """Wrap layers at zero dose and require baseline logits/tokens to remain identical."""

    baseline = _decode_last_logits(adapter, prompt_token_ids)
    controller = InterventionController(
        master_seed=master_seed,
        run_id=run_id,
        prompt_id="zero-dose-integrity",
        generation_index=0,
        condition_id="sham",
        selected_layers=frozenset(selected_layers),
        dose=0.0,
        mode="zero",
        decode_start_token=max(0, len(prompt_token_ids) - 1),
    )
    adapter.wrap_layers(controller)
    try:
        sham = _decode_last_logits(adapter, prompt_token_ids)
    finally:
        adapter.unwrap_layers()
    return compare_logits(baseline, sham, tolerance)


def zero_dose_suite(
    adapter: Any,
    prompt_token_ids: list[list[int]],
    *,
    master_seed: int,
    run_id: str,
    selected_layers: list[int],
    cached_decode_steps: int = 100,
    tolerance: float = 1e-6,
) -> ZeroDoseSuite:
    """Verify zero-dose identity for five+ prompts and a fixed cached decode budget.

    The baseline continuation is greedy solely to keep both paths on the same
    prefix.  The sampled experimental loop is not used here, so this is an
    integrity test rather than a behavioural measurement.
    """

    if len(prompt_token_ids) < 5:
        raise ValueError("Zero-dose preflight requires at least five prompts")
    if cached_decode_steps < 100:
        raise ValueError("Zero-dose preflight requires at least 100 cached decode steps")
    if any(len(tokens) < 2 for tokens in prompt_token_ids):
        raise ValueError("Zero-dose preflight prompts must contain at least two tokens")

    baseline_no_cache = [adapter.forward(tokens).logits for tokens in prompt_token_ids]
    controller = _zero_controller(
        master_seed, run_id, "zero-dose-no-cache", selected_layers, decode_start_token=0
    )
    adapter.wrap_layers(controller)
    try:
        sham_no_cache = [adapter.forward(tokens).logits for tokens in prompt_token_ids]
    finally:
        adapter.unwrap_layers()
    no_cache = _aggregate_comparisons(
        [
            compare_logits(left, right, tolerance)
            for left, right in zip(baseline_no_cache, sham_no_cache, strict=True)
        ]
    )

    allocations = _decode_allocations(len(prompt_token_ids), cached_decode_steps)
    baseline_cached = [
        _greedy_decode_logits(adapter, tokens, steps)
        for tokens, steps in zip(prompt_token_ids, allocations, strict=True)
    ]
    controller = _zero_controller(
        master_seed, run_id, "zero-dose-cached", selected_layers, decode_start_token=0
    )
    adapter.wrap_layers(controller)
    try:
        sham_cached = [
            _greedy_decode_logits(adapter, tokens, steps)
            for tokens, steps in zip(prompt_token_ids, allocations, strict=True)
        ]
    finally:
        adapter.unwrap_layers()
    cached = _aggregate_comparisons(
        [
            compare_logits(left, right, tolerance)
            for baseline, sham in zip(baseline_cached, sham_cached, strict=True)
            for left, right in zip(baseline, sham, strict=True)
        ]
    )
    return ZeroDoseSuite(len(prompt_token_ids), cached_decode_steps, no_cache, cached)


def intervention_liveness(
    adapter: Any,
    prompt_token_ids: list[int],
    *,
    master_seed: int,
    run_id: str,
    selected_layers: list[int],
    dose: float,
) -> LogitComparison:
    """Require a non-zero intervention to affect finite output logits."""

    baseline = _decode_last_logits(adapter, prompt_token_ids)
    controller = InterventionController(
        master_seed=master_seed,
        run_id=run_id,
        prompt_id="liveness-integrity",
        generation_index=0,
        condition_id="coherent",
        selected_layers=frozenset(selected_layers),
        dose=dose,
        mode="coherent_per_layer",
        decode_start_token=max(0, len(prompt_token_ids) - 1),
    )
    adapter.wrap_layers(controller)
    try:
        altered = _decode_last_logits(adapter, prompt_token_ids)
    finally:
        adapter.unwrap_layers()
    comparison = compare_logits(baseline, altered, tolerance=0.0)
    if comparison.maximum_absolute_error == 0:
        raise RuntimeError("Non-zero intervention did not change logits")
    # Liveness requires finite, changed logits. A changed argmax is neither
    # necessary nor desirable at the small diagnostic dose used here.
    return LogitComparison(
        comparison.maximum_absolute_error,
        comparison.mean_absolute_error,
        comparison.greedy_equal,
        True,
    )


def _decode_last_logits(adapter: Any, token_ids: list[int]) -> np.ndarray:
    """Use the same prefill/final-token schedule as decode-only intervention."""

    if len(token_ids) < 2:
        return adapter.forward(token_ids).logits
    cache = adapter.make_cache()
    adapter.forward(token_ids[:-1], cache=cache)
    return adapter.forward(token_ids[-1:], cache=cache).logits


def _greedy_generate(
    adapter: Any, token_ids: list[int], max_new_tokens: int
) -> tuple[list[int], str]:
    if not token_ids:
        raise ValueError("Baseline generation requires non-empty prompt tokens")
    cache = adapter.make_cache()
    if len(token_ids) > 1:
        adapter.forward(token_ids[:-1], cache=cache)
    next_input = [token_ids[-1]]
    output_ids: list[int] = []
    for _ in range(max_new_tokens):
        logits = np.asarray(adapter.forward(next_input, cache=cache).logits)
        if logits.ndim < 1 or not np.isfinite(logits).all():
            raise FloatingPointError("Baseline generation produced non-finite logits")
        token_id = int(np.argmax(logits.reshape(-1, logits.shape[-1])[-1]))
        if token_id in adapter.eos_token_ids():
            return output_ids, "EOS"
        output_ids.append(token_id)
        next_input = [token_id]
    return output_ids, "MAX_TOKENS"


def _has_repeated_4gram(token_ids: list[int]) -> bool:
    if len(token_ids) < 8:
        return False
    grams = [tuple(token_ids[index : index + 4]) for index in range(len(token_ids) - 3)]
    return len(set(grams)) != len(grams)


def _zero_controller(
    master_seed: int,
    run_id: str,
    prompt_id: str,
    selected_layers: list[int],
    decode_start_token: int,
) -> InterventionController:
    return InterventionController(
        master_seed=master_seed,
        run_id=run_id,
        prompt_id=prompt_id,
        generation_index=0,
        condition_id="sham",
        selected_layers=frozenset(selected_layers),
        dose=0.0,
        mode="zero",
        decode_start_token=decode_start_token,
    )


def _decode_allocations(prompt_count: int, total_steps: int) -> list[int]:
    base, remainder = divmod(total_steps, prompt_count)
    return [base + int(index < remainder) for index in range(prompt_count)]


def _greedy_decode_logits(adapter: Any, token_ids: list[int], steps: int) -> list[np.ndarray]:
    cache = adapter.make_cache()
    adapter.forward(token_ids[:-1], cache=cache)
    next_input = [token_ids[-1]]
    values: list[np.ndarray] = []
    for _ in range(steps):
        logits = adapter.forward(next_input, cache=cache).logits
        flattened = np.asarray(logits).reshape(-1, np.asarray(logits).shape[-1])[-1]
        values.append(flattened)
        next_input = [int(np.argmax(flattened))]
    return values


def _aggregate_comparisons(comparisons: list[LogitComparison]) -> LogitComparison:
    if not comparisons:
        raise ValueError("Expected at least one numerical comparison")
    return LogitComparison(
        maximum_absolute_error=max(item.maximum_absolute_error for item in comparisons),
        mean_absolute_error=float(np.mean([item.mean_absolute_error for item in comparisons])),
        greedy_equal=all(item.greedy_equal for item in comparisons),
        passed=all(item.passed for item in comparisons),
    )
