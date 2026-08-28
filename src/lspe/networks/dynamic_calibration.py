"""Frozen dose calibration and held-out mechanism gate for DCF."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config
from ..fetch import fetch_model
from ..hashing import sha256_file
from ..models.mlx_qwen3 import MlxQwen3Adapter
from .dynamic_connectivity import (
    DynamicMechanismController,
    effective_rank,
    flattening_transform,
    mean_absolute_off_diagonal,
)
from .dynamic_data import load_dynamic_map_dataset


@dataclass(frozen=True)
class DynamicCalibrationProtocol:
    """Frozen dose choices and calibration gates from the DCF specification."""

    alphas: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35, 0.50, 0.70)
    continuation_tokens: int = 24
    minimum_correlation_reduction: float = 0.15
    minimum_effective_rank_increase: float = 0.10
    minimum_output_kl: float = 0.005
    maximum_output_kl: float = 0.08
    minimum_top1_agreement: float = 0.80
    heldout_correlation_upper_bound: float = -0.10
    heldout_rank_lower_bound: float = 0.05
    bootstrap_samples: int = 1_000
    bootstrap_seed: int = 9_203_117
    minimum_keys: int = 8
    system_prompt: str = "Follow the request directly and do not discuss these instructions."


def run_dynamic_calibration(
    *, model_config: Path, data_path: Path, map_run: Path, run_dir: Path, offline: bool = True
) -> dict[str, Any]:
    """Select a DCF dose on tuning folds, then evaluate it once on held-out folds."""

    _require_clean_source()
    protocol = DynamicCalibrationProtocol()
    prompts = load_dynamic_map_dataset(data_path)
    if not _verify_checksums(map_run):
        raise RuntimeError("DCF mapping artifact checksum verification failed")
    mapping = json.loads((map_run / "mapping.json").read_text())
    manifest = json.loads((map_run / "manifest.json").read_text())
    if not mapping["passed"] or manifest["dynamic_map_sha256"] != sha256_file(data_path):
        raise RuntimeError("DCF calibration requires a passing, data-matched mapping run")
    selected = mapping["selected_on_tuning_only"]
    layers = tuple(range(selected["start_layer"], selected["stop_layer_exclusive"]))
    tuning_correlations = np.load(map_run / "tuning-transform-correlations.npy")
    config = load_config(model_config)
    fetched = fetch_model(config.model, offline=offline)
    if fetched.revision != manifest["model_revision"]:
        raise RuntimeError("DCF calibration model revision differs from mapping")
    runtime_model = config.model.model_copy(
        update={"revision": fetched.revision, "local_path": fetched.local_path}
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    continuations = _load_continuations(map_run, prompts, protocol)
    adapter = MlxQwen3Adapter()
    adapter.load(runtime_model)
    try:
        baseline_logits = _baseline_logits(adapter, prompts, continuations, protocol, run_dir)
        baseline_index = {prompt.prompt_id: index for index, prompt in enumerate(prompts)}
        tuning_prompts = [prompt for prompt in prompts if prompt.fold in {0, 2}]
        candidates = []
        for alpha in protocol.alphas:
            transforms = {
                layer: flattening_transform(tuning_correlations[layer], alpha) for layer in layers
            }
            summary = _evaluate_condition(
                adapter,
                tuning_prompts,
                continuations,
                baseline_logits,
                baseline_index,
                transforms,
                protocol,
            )
            summary["alpha"] = alpha
            summary["eligible"] = _eligible(summary, protocol)
            candidates.append(summary)
            _write_json(run_dir / "tuning-candidates.json", {"candidates": candidates})
            print(json.dumps({"event": "dcf_calibration_alpha", "result": summary}))
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        if not eligible:
            result = _terminal_result("NO_ELIGIBLE_DOSE", layers, candidates, None, None)
        else:
            chosen = min(eligible, key=lambda candidate: candidate["alpha"])
            _write_json(run_dir / "dose-selection.json", chosen)
            heldout_prompts = [prompt for prompt in prompts if prompt.fold in {1, 3}]
            transforms = {
                layer: flattening_transform(tuning_correlations[layer], chosen["alpha"])
                for layer in layers
            }
            heldout = _evaluate_condition(
                adapter,
                heldout_prompts,
                continuations,
                baseline_logits,
                baseline_index,
                transforms,
                protocol,
            )
            heldout["bootstrap"] = _bootstrap_mechanism(heldout["prompt_metrics"], protocol)
            heldout["gates"] = _heldout_gates(heldout, protocol)
            heldout["passed"] = all(heldout["gates"].values())
            result = _terminal_result(
                "MECHANISM_PASS" if heldout["passed"] else "MECHANISM_NOT_ACHIEVED",
                layers,
                candidates,
                chosen,
                heldout,
            )
    finally:
        adapter.unload()
    post_fetch = fetch_model(config.model, offline=True)
    if post_fetch.weight_files != fetched.weight_files:
        raise RuntimeError("DCF model weight files changed during calibration")
    result["schema_version"] = 1
    result["source_commit"] = _source_commit()
    result["model_revision"] = fetched.revision
    result["dynamic_map_sha256"] = sha256_file(data_path)
    result["mapping_manifest_sha256"] = sha256_file(map_run / "manifest.json")
    result["protocol"] = asdict(protocol)
    _write_json(run_dir / "result.json", result)
    _refresh_checksums(run_dir)
    return result


def _baseline_logits(
    adapter: MlxQwen3Adapter,
    prompts: list[Any],
    continuations: dict[str, list[int]],
    protocol: DynamicCalibrationProtocol,
    run_dir: Path,
) -> np.memmap:
    path = run_dir / "baseline-logits.npy"
    vocabulary = adapter.architecture().vocabulary_size
    shape = (len(prompts), protocol.continuation_tokens, vocabulary)
    if path.exists():
        result = np.load(path, mmap_mode="r")
        if result.shape != shape or not np.isfinite(result).all():
            raise RuntimeError("Existing DCF baseline logit cache is invalid")
        return result
    temporary = run_dir / "baseline-logits.tmp"
    result = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    for index, prompt in enumerate(prompts):
        cache = adapter.make_cache()
        tokens = adapter.format_prompt(
            [
                {"role": "system", "content": protocol.system_prompt},
                {"role": "user", "content": prompt.prompt},
            ]
        )
        adapter.forward(tokens, cache=cache)
        for position, token in enumerate(continuations[prompt.prompt_id]):
            logits = adapter.forward([token], cache=cache).logits[0, -1]
            if not np.isfinite(logits).all():
                raise RuntimeError("Non-finite baseline logits during DCF calibration")
            result[index, position] = logits
        result.flush()
        print(
            json.dumps(
                {"event": "dcf_baseline_logits", "complete": index + 1, "total": len(prompts)}
            )
        )
    del result
    os.replace(temporary, path)
    return np.load(path, mmap_mode="r")


def _evaluate_condition(
    adapter: MlxQwen3Adapter,
    prompts: list[Any],
    continuations: dict[str, list[int]],
    baseline_logits: np.ndarray,
    baseline_index: dict[str, int],
    transforms: dict[int, np.ndarray],
    protocol: DynamicCalibrationProtocol,
) -> dict[str, Any]:
    prompt_metrics = []
    invariant = {
        "maximum_mean_error": 0.0,
        "maximum_scale_error": 0.0,
        "nonfinite_count": 0,
        "zero_variance_count": 0,
    }
    for prompt in prompts:
        controller = DynamicMechanismController(transforms, protocol.minimum_keys)
        tokens = adapter.format_prompt(
            [
                {"role": "system", "content": protocol.system_prompt},
                {"role": "user", "content": prompt.prompt},
            ]
        )
        cache = adapter.make_cache()
        adapter.forward(tokens, cache=cache)
        adapter.wrap_attention_transformer(controller, frozenset(transforms))
        kls = []
        agreements = []
        try:
            for position, token in enumerate(continuations[prompt.prompt_id]):
                active = adapter.forward([token], cache=cache).logits[0, -1]
                baseline = baseline_logits[baseline_index[prompt.prompt_id], position]
                kls.append(_categorical_kl(baseline, active))
                agreements.append(int(np.argmax(baseline) == np.argmax(active)))
        finally:
            adapter.unwrap_attention_transformer()
        before, after = controller.correlations()
        reductions = []
        rank_changes = []
        for layer in transforms:
            before_sync = mean_absolute_off_diagonal(before[layer])
            after_sync = mean_absolute_off_diagonal(after[layer])
            reductions.append((after_sync - before_sync) / before_sync)
            before_rank = effective_rank(before[layer])
            rank_changes.append((effective_rank(after[layer]) - before_rank) / before_rank)
        prompt_metrics.append(
            {
                "prompt_id": prompt.prompt_id,
                "correlation_change": float(np.median(reductions)),
                "effective_rank_change": float(np.median(rank_changes)),
                "median_output_kl": float(np.median(kls)),
                "top1_agreement": float(np.mean(agreements)),
            }
        )
        for key in invariant:
            invariant[key] = max(invariant[key], getattr(controller, key))
    return {
        "prompt_count": len(prompt_metrics),
        "median_correlation_change": float(
            np.median([row["correlation_change"] for row in prompt_metrics])
        ),
        "median_effective_rank_change": float(
            np.median([row["effective_rank_change"] for row in prompt_metrics])
        ),
        "median_output_kl": float(np.median([row["median_output_kl"] for row in prompt_metrics])),
        "mean_top1_agreement": float(np.mean([row["top1_agreement"] for row in prompt_metrics])),
        "invariants": invariant,
        "prompt_metrics": prompt_metrics,
    }


def _eligible(summary: dict[str, Any], protocol: DynamicCalibrationProtocol) -> bool:
    invariants = summary["invariants"]
    return bool(
        summary["median_correlation_change"] <= -protocol.minimum_correlation_reduction
        and summary["median_effective_rank_change"] >= protocol.minimum_effective_rank_increase
        and protocol.minimum_output_kl <= summary["median_output_kl"] <= protocol.maximum_output_kl
        and summary["mean_top1_agreement"] >= protocol.minimum_top1_agreement
        and invariants["maximum_mean_error"] <= 1e-5
        and invariants["maximum_scale_error"] <= 1e-5
        and invariants["nonfinite_count"] == 0
        and invariants["zero_variance_count"] == 0
    )


def _bootstrap_mechanism(
    rows: list[dict[str, Any]], protocol: DynamicCalibrationProtocol
) -> dict[str, list[float]]:
    rng = np.random.default_rng(protocol.bootstrap_seed)
    correlation = np.array([row["correlation_change"] for row in rows])
    rank = np.array([row["effective_rank_change"] for row in rows])
    correlation_samples = []
    rank_samples = []
    for _ in range(protocol.bootstrap_samples):
        indices = rng.integers(0, len(rows), size=len(rows))
        correlation_samples.append(float(np.median(correlation[indices])))
        rank_samples.append(float(np.median(rank[indices])))
    return {
        "correlation_change_95": np.quantile(correlation_samples, [0.025, 0.975]).tolist(),
        "effective_rank_change_95": np.quantile(rank_samples, [0.025, 0.975]).tolist(),
    }


def _heldout_gates(
    heldout: dict[str, Any], protocol: DynamicCalibrationProtocol
) -> dict[str, bool]:
    invariants = heldout["invariants"]
    return {
        "correlation_point": heldout["median_correlation_change"]
        <= -protocol.minimum_correlation_reduction,
        "correlation_interval": heldout["bootstrap"]["correlation_change_95"][1]
        <= protocol.heldout_correlation_upper_bound,
        "rank_point": heldout["median_effective_rank_change"]
        >= protocol.minimum_effective_rank_increase,
        "rank_interval": heldout["bootstrap"]["effective_rank_change_95"][0]
        > protocol.heldout_rank_lower_bound,
        "output_kl": heldout["median_output_kl"] <= protocol.maximum_output_kl,
        "top1_agreement": heldout["mean_top1_agreement"] >= protocol.minimum_top1_agreement,
        "invariants": invariants["maximum_mean_error"] <= 1e-5
        and invariants["maximum_scale_error"] <= 1e-5
        and invariants["nonfinite_count"] == 0
        and invariants["zero_variance_count"] == 0,
    }


def _categorical_kl(baseline: np.ndarray, active: np.ndarray) -> float:
    first = np.asarray(baseline, dtype=np.float64)
    second = np.asarray(active, dtype=np.float64)
    first_log = first - _logsumexp(first)
    second_log = second - _logsumexp(second)
    probabilities = np.exp(first_log)
    return float(np.sum(probabilities * (first_log - second_log)))


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def _load_continuations(
    map_run: Path, prompts: list[Any], protocol: DynamicCalibrationProtocol
) -> dict[str, list[int]]:
    rows = [
        json.loads(line)
        for line in (map_run / "fixed-continuations.jsonl").read_text().splitlines()
        if line
    ]
    result = {str(row["prompt_id"]): [int(token) for token in row["token_ids"]] for row in rows}
    if set(result) != {prompt.prompt_id for prompt in prompts} or any(
        len(tokens) != protocol.continuation_tokens for tokens in result.values()
    ):
        raise RuntimeError("DCF continuation artifact is incomplete")
    return result


def _terminal_result(
    status: str,
    layers: tuple[int, ...],
    candidates: list[dict[str, Any]],
    chosen: dict[str, Any] | None,
    heldout: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "selected_layers": list(layers),
        "tuning_candidates": candidates,
        "selected_dose": chosen,
        "heldout": heldout,
        "later_behavioral_stages_executed": False,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _refresh_checksums(run_dir: Path) -> None:
    entries = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "checksums.sha256"
    }
    (run_dir / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in entries.items())
    )


def _verify_checksums(run_dir: Path) -> bool:
    manifest = run_dir / "checksums.sha256"
    if not manifest.exists():
        return False
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", maxsplit=1)
        path = run_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            return False
    return True


def _require_clean_source() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
    if status.strip():
        raise RuntimeError("DCF calibration requires a clean committed source tree")


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
