"""Paired behavioral experiment for the frozen SCCF mechanism."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..calibration.entropy import match_temperature, sampling_entropy
from ..config import SamplingConfig, load_config
from ..fetch import fetch_model
from ..generation.loop import GenerationLoop
from ..hashing import sha256_file
from ..metrics.degeneration import degeneration_metrics
from ..models.mlx_qwen3 import MlxQwen3Adapter
from ..tasks.validators import validate_response
from .dynamic_calibration import _categorical_kl, _verify_checksums
from .dynamic_connectivity import (
    AttentionNoiseController,
    DynamicConnectivityController,
    DynamicMechanismController,
    random_basis_transform,
)
from .selective_connectivity import build_selective_transforms

CONDITIONS = ("baseline", "sham", "sccf", "random_basis", "attn_noise", "temp_match")
PROTECTED = ("constrained", "factual", "code")
TARGETS = ("open_association", "analogical")


@dataclass(frozen=True)
class BehavioralProtocol:
    """Frozen SCBE control matching, sampling, and decision gates."""

    master_seed: int = 84_120_771
    selected_alpha: float = 0.50
    calibration_tokens: int = 16
    generations_per_prompt: int = 3
    max_new_tokens: int = 128
    base_temperature: float = 0.80
    top_k: int = 64
    top_p: float = 1.0
    random_alphas: tuple[float, ...] = (0.80, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00)
    noise_sigmas: tuple[float, ...] = (0.32, 0.40, 0.50, 0.65, 0.80, 1.00, 1.30)
    control_relative_kl_tolerance: float = 0.25
    temperature_entropy_tolerance: float = 0.02
    pilot_validity_margin: float = 0.15
    confirm_validity_margin: float = 0.10
    pilot_degeneration_margin: float = 0.05
    confirm_degeneration_margin: float = 0.02
    bootstrap_samples: int = 10_000
    minimum_keys: int = 8
    system_prompt: str = "Follow the request exactly. Return only the requested answer."


def run_behavioral_experiment(
    *,
    model_config: Path,
    data_root: Path,
    map_run: Path,
    sccf_run: Path,
    run_dir: Path,
    offline: bool = True,
    protocol: BehavioralProtocol | None = None,
    protocol_document: Path = Path("SCBE_CONTROL_MATCH_AMENDMENT.md"),
) -> dict[str, Any]:
    """Calibrate controls, run pilot, and run confirmation only if its gate passes."""

    _require_clean_source()
    protocol = protocol or BehavioralProtocol()
    datasets = {
        split: _load_rows(data_root / f"{split}.jsonl", split)
        for split in ("calibration", "pilot", "confirm")
    }
    _validate_inputs(data_root, map_run, sccf_run, datasets)
    map_manifest = json.loads((map_run / "manifest.json").read_text())
    sccf_result = json.loads((sccf_run / "result.json").read_text())
    correlations = np.load(map_run / "tuning-transform-correlations.npy")
    chosen = sccf_result["selected_candidate"]
    modes = chosen["modes"]
    if chosen["mask_size"] != 16 or chosen["alpha"] != protocol.selected_alpha:
        raise RuntimeError("SCBE parent selection differs from the frozen protocol")
    config = load_config(model_config)
    fetched = fetch_model(config.model, offline=offline)
    if (
        fetched.revision != map_manifest["model_revision"]
        or fetched.revision != sccf_result["model_revision"]
    ):
        raise RuntimeError("SCBE model revision differs from its parent artifacts")
    runtime_model = config.model.model_copy(
        update={"revision": fetched.revision, "local_path": fetched.local_path}
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _lock_or_validate(
        run_dir, data_root, map_run, sccf_run, protocol, protocol_document, fetched.revision
    )
    adapter = MlxQwen3Adapter()
    adapter.load(runtime_model)
    try:
        calibration = _calibrate_controls(
            adapter, datasets["calibration"], correlations, modes, protocol, run_dir
        )
        if not calibration["passed"]:
            result = _terminal("CONTROL_MATCH_FAILED", calibration, None, None)
        else:
            pilot_rows = _run_split(
                adapter,
                datasets["pilot"],
                "pilot",
                correlations,
                modes,
                calibration,
                protocol,
                run_dir,
            )
            pilot = summarize_split(pilot_rows, datasets["pilot"], protocol, "pilot")
            _write_json(run_dir / "pilot-summary.json", pilot)
            if not pilot["passed"]:
                result = _terminal("PILOT_GATE_FAILED", calibration, pilot, None)
            else:
                confirm_rows = _run_split(
                    adapter,
                    datasets["confirm"],
                    "confirm",
                    correlations,
                    modes,
                    calibration,
                    protocol,
                    run_dir,
                )
                confirm = summarize_split(confirm_rows, datasets["confirm"], protocol, "confirm")
                _write_json(run_dir / "confirm-summary.json", confirm)
                result = _terminal("GENERATION_COMPLETE", calibration, pilot, confirm)
    finally:
        adapter.unload()
    if fetch_model(config.model, offline=True).weight_files != fetched.weight_files:
        raise RuntimeError("SCBE model weight files changed during execution")
    result.update(
        {
            "schema_version": 1,
            "source_commit": _source_commit(),
            "model_revision": fetched.revision,
            "protocol": asdict(protocol),
            "protocol_document_sha256": sha256_file(protocol_document),
        }
    )
    _write_json(run_dir / "result.json", result)
    _refresh_checksums(run_dir)
    return result


def _calibrate_controls(
    adapter: MlxQwen3Adapter,
    prompts: list[dict[str, Any]],
    correlations: np.ndarray,
    modes: list[dict[str, Any]],
    protocol: BehavioralProtocol,
    run_dir: Path,
) -> dict[str, Any]:
    path = run_dir / "control-calibration.json"
    if path.exists():
        return json.loads(path.read_text())
    continuations, baseline = _baseline_calibration(adapter, prompts, protocol)
    sccf_transforms = build_selective_transforms(correlations, modes, protocol.selected_alpha)
    sccf = _evaluate_calibration_condition(
        adapter, prompts, continuations, baseline, protocol, transforms=sccf_transforms
    )
    target_kl = sccf["median_output_kl"]
    random_curve = []
    for alpha in protocol.random_alphas:
        transforms = _random_transforms(correlations, modes, alpha, protocol.master_seed)
        row = _evaluate_calibration_condition(
            adapter, prompts, continuations, baseline, protocol, transforms=transforms
        )
        row["value"] = alpha
        random_curve.append(row)
    noise_curve = []
    layers = frozenset(int(mode["layer"]) for mode in modes)
    for sigma in protocol.noise_sigmas:
        row = _evaluate_calibration_condition(
            adapter,
            prompts,
            continuations,
            baseline,
            protocol,
            noise_sigma=sigma,
            noise_layers=layers,
        )
        row["value"] = sigma
        noise_curve.append(row)
    random_choice = _closest_kl(random_curve, target_kl)
    noise_choice = _closest_kl(noise_curve, target_kl)
    target_entropy = sccf["mean_sampling_entropy"]
    match = match_temperature(
        [row for prompt_rows in baseline for row in prompt_rows],
        target_entropy,
        protocol.top_k,
        protocol.top_p,
    )
    sham = _evaluate_calibration_condition(
        adapter,
        prompts,
        continuations,
        baseline,
        protocol,
        transforms={layer: np.eye(correlations.shape[-1]) for layer in layers},
        sham=True,
    )
    gates = {
        "active_sccf": target_kl > 0.0,
        "random_kl_match": _relative_error(random_choice["median_output_kl"], target_kl)
        <= protocol.control_relative_kl_tolerance,
        "noise_kl_match": _relative_error(noise_choice["median_output_kl"], target_kl)
        <= protocol.control_relative_kl_tolerance,
        "temperature_entropy_match": match.absolute_mismatch
        <= protocol.temperature_entropy_tolerance,
        "sham_logit_equivalence": sham["maximum_absolute_logit_error"] <= 1e-5
        and sham["top1_agreement"] == 1.0,
        "invariants": all(row["invariants_passed"] for row in [sccf, random_choice, noise_choice]),
    }
    result = {
        "sccf": sccf,
        "target_output_kl": target_kl,
        "random_curve": random_curve,
        "random_choice": random_choice,
        "noise_curve": noise_curve,
        "noise_choice": noise_choice,
        "temperature_match": asdict(match),
        "sham": sham,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _write_json(path, result)
    return result


def _baseline_calibration(
    adapter: MlxQwen3Adapter, prompts: list[dict[str, Any]], protocol: BehavioralProtocol
) -> tuple[dict[str, list[int]], list[list[np.ndarray]]]:
    continuations: dict[str, list[int]] = {}
    all_logits: list[list[np.ndarray]] = []
    for prompt in prompts:
        tokens = _format(adapter, prompt, protocol)
        cache = adapter.make_cache()
        logits = adapter.forward(tokens, cache=cache).logits[0, -1]
        generated = []
        rows = []
        for _ in range(protocol.calibration_tokens):
            token = int(np.argmax(logits))
            generated.append(token)
            logits = adapter.forward([token], cache=cache).logits[0, -1]
            rows.append(np.asarray(logits, dtype=np.float32))
        continuations[prompt["prompt_id"]] = generated
        all_logits.append(rows)
    return continuations, all_logits


def _evaluate_calibration_condition(
    adapter: MlxQwen3Adapter,
    prompts: list[dict[str, Any]],
    continuations: dict[str, list[int]],
    baseline: list[list[np.ndarray]],
    protocol: BehavioralProtocol,
    *,
    transforms: dict[int, np.ndarray] | None = None,
    noise_sigma: float | None = None,
    noise_layers: frozenset[int] = frozenset(),
    sham: bool = False,
) -> dict[str, Any]:
    kls = []
    entropies = []
    top1 = []
    maximum_error = 0.0
    invariant_rows = []
    for prompt_index, prompt in enumerate(prompts):
        tokens = _format(adapter, prompt, protocol)
        cache = adapter.make_cache()
        adapter.forward(tokens, cache=cache)
        if noise_sigma is not None:
            controller: Any = AttentionNoiseController(
                noise_layers,
                noise_sigma,
                _seed(protocol.master_seed, "calibration-noise", prompt["prompt_id"]),
            )
            selected = noise_layers
        elif sham:
            controller = DynamicConnectivityController(transforms or {}, protocol.minimum_keys)
            selected = frozenset(transforms or {})
        else:
            controller = DynamicMechanismController(transforms or {}, protocol.minimum_keys)
            selected = frozenset(transforms or {})
        adapter.wrap_attention_transformer(controller, selected)
        try:
            for position, token in enumerate(continuations[prompt["prompt_id"]]):
                active = adapter.forward([token], cache=cache).logits[0, -1]
                ordinary = baseline[prompt_index][position]
                kls.append(_categorical_kl(ordinary, active))
                entropies.append(
                    sampling_entropy(
                        active, protocol.base_temperature, protocol.top_k, protocol.top_p
                    )
                )
                top1.append(int(np.argmax(ordinary) == np.argmax(active)))
                maximum_error = max(maximum_error, float(np.max(np.abs(ordinary - active))))
        finally:
            adapter.unwrap_attention_transformer()
        invariant_rows.append(_controller_invariants(controller))
    invariants_passed = all(row["passed"] for row in invariant_rows)
    return {
        "median_output_kl": float(np.median(kls)),
        "mean_output_kl": float(np.mean(kls)),
        "mean_sampling_entropy": float(np.mean(entropies)),
        "top1_agreement": float(np.mean(top1)),
        "maximum_absolute_logit_error": maximum_error,
        "invariants_passed": invariants_passed,
    }


def _run_split(
    adapter: MlxQwen3Adapter,
    prompts: list[dict[str, Any]],
    split: str,
    correlations: np.ndarray,
    modes: list[dict[str, Any]],
    calibration: dict[str, Any],
    protocol: BehavioralProtocol,
    run_dir: Path,
) -> list[dict[str, Any]]:
    path = run_dir / f"{split}-generations.jsonl"
    rows = _load_jsonl_if_exists(path)
    completed = {
        (str(row["prompt_id"]), int(row["generation_index"]), str(row["condition"])) for row in rows
    }
    sccf = build_selective_transforms(correlations, modes, protocol.selected_alpha)
    random = _random_transforms(
        correlations,
        modes,
        float(calibration["random_choice"]["value"]),
        protocol.master_seed,
    )
    layers = frozenset(int(mode["layer"]) for mode in modes)
    for prompt in prompts:
        for generation_index in range(protocol.generations_per_prompt):
            order = sorted(
                CONDITIONS,
                key=lambda condition: _seed(
                    protocol.master_seed,
                    "condition-order",
                    split,
                    prompt["prompt_id"],
                    generation_index,
                    condition,
                ),
            )
            for condition in order:
                identity = (prompt["prompt_id"], generation_index, condition)
                if identity in completed:
                    continue
                controller: Any | None = None
                selected = frozenset()
                sampling = _sampling(protocol)
                if condition == "sham":
                    controller = DynamicConnectivityController(
                        {layer: np.eye(correlations.shape[-1]) for layer in layers},
                        protocol.minimum_keys,
                    )
                    selected = layers
                elif condition == "sccf":
                    controller = DynamicMechanismController(sccf, protocol.minimum_keys)
                    selected = frozenset(sccf)
                elif condition == "random_basis":
                    controller = DynamicMechanismController(random, protocol.minimum_keys)
                    selected = frozenset(random)
                elif condition == "attn_noise":
                    controller = AttentionNoiseController(
                        layers,
                        float(calibration["noise_choice"]["value"]),
                        _seed(
                            protocol.master_seed,
                            "attention-noise",
                            prompt["prompt_id"],
                            generation_index,
                        ),
                        protocol.minimum_keys,
                    )
                    selected = layers
                elif condition == "temp_match":
                    sampling = sampling.model_copy(
                        update={
                            "temperature": float(calibration["temperature_match"]["temperature"])
                        }
                    )
                prompt_tokens = _format(adapter, prompt, protocol)
                if controller is not None:
                    adapter.wrap_attention_transformer(controller, selected)
                try:
                    generated = GenerationLoop(adapter, sampling, protocol.master_seed).generate(
                        prompt_tokens,
                        prompt["prompt_id"],
                        generation_index,
                        condition,
                        tuple(sorted(selected)),
                        intervention_active=controller is not None and condition != "sham",
                        intervention_dose=(
                            protocol.selected_alpha
                            if condition == "sccf"
                            else float(calibration["random_choice"]["value"])
                            if condition == "random_basis"
                            else float(calibration["noise_choice"]["value"])
                            if condition == "attn_noise"
                            else 0.0
                        ),
                    )
                finally:
                    if controller is not None:
                        adapter.unwrap_attention_transformer()
                validation = validate_response(
                    str(prompt["validator"]), generated.text, prompt.get("expected")
                )
                degeneration = degeneration_metrics(generated.output_token_ids)
                row = {
                    "schema_version": 1,
                    "generation_id": f"{prompt['prompt_id']}:{generation_index}:{condition}",
                    "prompt_id": prompt["prompt_id"],
                    "split": split,
                    "category": prompt["category"],
                    "condition": condition,
                    "generation_index": generation_index,
                    "output_token_ids": list(generated.output_token_ids),
                    "output_text": generated.text,
                    "stop_reason": generated.stop_reason,
                    "valid": validation.valid,
                    "failure_code": validation.failure_code,
                    "degeneration": degeneration,
                    "mean_sampling_entropy": float(
                        np.mean([metric.entropy for metric in generated.token_metrics])
                    ),
                    "controller_invariants": _controller_invariants(controller),
                    "token_metrics": [asdict(metric) for metric in generated.token_metrics],
                }
                _append_jsonl(path, row)
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "event": "scbe_generation",
                            "split": split,
                            "complete": len(rows),
                            "total": len(prompts)
                            * protocol.generations_per_prompt
                            * len(CONDITIONS),
                            "generation_id": row["generation_id"],
                            "valid": row["valid"],
                        }
                    )
                )
    return rows


def summarize_split(
    rows: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    protocol: BehavioralProtocol,
    split: str,
) -> dict[str, Any]:
    """Calculate completeness, paired-sham, validity, and degeneration gates."""

    expected = len(prompts) * protocol.generations_per_prompt * len(CONDITIONS)
    by_cell: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault((str(row["prompt_id"]), int(row["generation_index"])), {})[
            str(row["condition"])
        ] = row
    sham_equal = all(
        conditions.get("baseline", {}).get("output_text")
        == conditions.get("sham", {}).get("output_text")
        for conditions in by_cell.values()
    )
    category_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for category in sorted({str(prompt["category"]) for prompt in prompts}):
        category_metrics[category] = {}
        for condition in CONDITIONS:
            subset = [
                row for row in rows if row["category"] == category and row["condition"] == condition
            ]
            category_metrics[category][condition] = {
                "validity": float(np.mean([row["valid"] for row in subset])),
                "degeneration": float(
                    np.mean(
                        [
                            row["degeneration"]["repeated_4gram_ratio"] > 0
                            or row["degeneration"]["max_identical_run"] >= 8
                            for row in subset
                        ]
                    )
                ),
                "mean_entropy": float(np.mean([row["mean_sampling_entropy"] for row in subset])),
            }
    validity_margin = (
        protocol.pilot_validity_margin if split == "pilot" else protocol.confirm_validity_margin
    )
    degeneration_margin = (
        protocol.pilot_degeneration_margin
        if split == "pilot"
        else protocol.confirm_degeneration_margin
    )
    gates: dict[str, bool] = {
        "complete": len(rows) == expected
        and len(by_cell) == len(prompts) * protocol.generations_per_prompt,
        "sham_text_identity": sham_equal,
        "controller_invariants": all(row["controller_invariants"]["passed"] for row in rows),
    }
    for category in PROTECTED:
        metrics = category_metrics[category]
        gates[f"{category}_validity"] = (
            metrics["sccf"]["validity"] + validity_margin >= metrics["baseline"]["validity"]
        )
    for category, metrics in category_metrics.items():
        gates[f"{category}_degeneration"] = (
            metrics["sccf"]["degeneration"]
            <= metrics["baseline"]["degeneration"] + degeneration_margin
        )
    return {
        "split": split,
        "generation_count": len(rows),
        "expected_generation_count": expected,
        "category_metrics": category_metrics,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _random_transforms(
    correlations: np.ndarray,
    modes: list[dict[str, Any]],
    alpha: float,
    master_seed: int,
) -> dict[int, np.ndarray]:
    by_layer: dict[int, set[int]] = {}
    for mode in modes:
        by_layer.setdefault(int(mode["layer"]), set()).add(int(mode["eigenvalue_rank"]))
    return {
        layer: random_basis_transform(
            correlations[layer],
            alpha,
            frozenset(ranks),
            _seed(master_seed, "random-basis", layer),
        )
        for layer, ranks in sorted(by_layer.items())
    }


def _closest_kl(rows: list[dict[str, Any]], target: float) -> dict[str, Any]:
    if target <= 0.0:
        raise RuntimeError("Cannot match controls to a non-positive SCCF KL")
    return min(
        rows,
        key=lambda row: (
            abs(np.log(max(row["median_output_kl"], 1e-15) / target)),
            row["value"],
        ),
    )


def _relative_error(value: float, target: float) -> float:
    return abs(value - target) / target if target > 0.0 else float("inf")


def _sampling(protocol: BehavioralProtocol) -> SamplingConfig:
    return SamplingConfig(
        temperature=protocol.base_temperature,
        top_k=protocol.top_k,
        top_p=protocol.top_p,
        repetition_penalty=1.0,
        max_new_tokens=protocol.max_new_tokens,
        stop_on_eos=True,
        store_top_logprobs=16,
    )


def _format(
    adapter: MlxQwen3Adapter, prompt: dict[str, Any], protocol: BehavioralProtocol
) -> list[int]:
    return adapter.format_prompt(
        [
            {"role": "system", "content": protocol.system_prompt},
            {"role": "user", "content": str(prompt["prompt"])},
        ]
    )


def _controller_invariants(controller: Any | None) -> dict[str, Any]:
    if controller is None or isinstance(controller, DynamicConnectivityController):
        return {
            "maximum_mean_error": 0.0,
            "maximum_scale_error": 0.0,
            "nonfinite_count": 0,
            "zero_variance_count": 0,
            "passed": True,
        }
    result = {
        "maximum_mean_error": float(controller.maximum_mean_error),
        "maximum_scale_error": float(controller.maximum_scale_error),
        "nonfinite_count": int(controller.nonfinite_count),
        "zero_variance_count": int(controller.zero_variance_count),
    }
    result["passed"] = bool(
        result["maximum_mean_error"] <= 1e-5
        and result["maximum_scale_error"] <= 1e-5
        and result["nonfinite_count"] == 0
        and result["zero_variance_count"] == 0
    )
    return result


def _validate_inputs(
    data_root: Path,
    map_run: Path,
    sccf_run: Path,
    datasets: dict[str, list[dict[str, Any]]],
) -> None:
    if not _verify_checksums(map_run) or not _verify_checksums(sccf_run):
        raise RuntimeError("SCBE parent artifact checksum verification failed")
    mapping = json.loads((map_run / "mapping.json").read_text())
    sccf = json.loads((sccf_run / "result.json").read_text())
    audit = json.loads((data_root / "leakage-audit.json").read_text())
    if not mapping["passed"] or sccf["status"] != "MECHANISM_PASS":
        raise RuntimeError("SCBE requires passing DCF mapping and SCCF mechanism artifacts")
    if not audit["passed"] or audit["row_count"] != sum(map(len, datasets.values())):
        raise RuntimeError("SCBE fresh-data leakage audit failed")
    expected = {"calibration": 12, "pilot": 24, "confirm": 48}
    actual = {split: len(rows) for split, rows in datasets.items()}
    if actual != expected:
        raise RuntimeError(f"SCBE dataset sizes differ from protocol: {actual}")


def _lock_or_validate(
    run_dir: Path,
    data_root: Path,
    map_run: Path,
    sccf_run: Path,
    protocol: BehavioralProtocol,
    protocol_document: Path,
    model_revision: str,
) -> None:
    expected = {
        "schema_version": 1,
        "source_commit": _source_commit(),
        "model_revision": model_revision,
        "protocol": asdict(protocol),
        "protocol_document_sha256": sha256_file(protocol_document),
        "mapping_result_sha256": sha256_file(map_run / "mapping.json"),
        "sccf_result_sha256": sha256_file(sccf_run / "result.json"),
        "data": {
            name: sha256_file(data_root / name)
            for name in (
                "calibration.jsonl",
                "pilot.jsonl",
                "confirm.jsonl",
                "leakage-audit.json",
            )
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
        },
    }
    path = run_dir / "run-lock.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise RuntimeError("SCBE resume provenance differs from its run lock")
        return
    if any(run_dir.iterdir()):
        raise RuntimeError("SCBE output directory is non-empty without a run lock")
    _write_json(path, expected)


def _load_rows(path: Path, expected_split: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    required = {
        "schema_version",
        "prompt_id",
        "split",
        "category",
        "prompt",
        "validator",
        "expected",
    }
    if not rows or any(set(row) != required or row["split"] != expected_split for row in rows):
        raise RuntimeError(f"Malformed SCBE dataset: {path}")
    identifiers = [str(row["prompt_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"Duplicate SCBE prompt ID: {path}")
    return rows


def _terminal(
    status: str,
    calibration: dict[str, Any],
    pilot: dict[str, Any] | None,
    confirm: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "control_calibration": calibration,
        "pilot": pilot,
        "confirm": confirm,
        "analysis_executed": False,
        "judge_executed": False,
    }


def _seed(master_seed: int, domain: str, *components: Any) -> int:
    payload = json.dumps(
        [master_seed, domain, components], sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _refresh_checksums(run_dir: Path) -> None:
    entries = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "checksums.sha256"
    }
    (run_dir / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in entries.items())
    )


def _require_clean_source() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
    if status.strip():
        raise RuntimeError("SCBE requires a clean committed source tree")


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
