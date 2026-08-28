"""Frozen selective causal connectivity-flattening experiment."""

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
from .dynamic_calibration import (
    DynamicCalibrationProtocol,
    _bootstrap_mechanism,
    _categorical_kl,
    _evaluate_condition,
    _load_continuations,
    _verify_checksums,
)
from .dynamic_connectivity import DynamicMechanismController, selective_flattening_transform
from .dynamic_data import load_dynamic_map_dataset


@dataclass(frozen=True)
class SelectiveConnectivityProtocol:
    """Frozen SCCF screen, candidate grid, and gates."""

    screen_alpha: float = 0.42
    screen_tokens: int = 12
    screen_prompts_per_category: int = 4
    target_categories: tuple[str, ...] = ("analogical", "open_association")
    protection_categories: tuple[str, ...] = ("constrained", "factual", "code")
    mask_sizes: tuple[int, ...] = (8, 16, 32, 64)
    alphas: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60)
    continuation_tokens: int = 24
    minimum_correlation_reduction: float = 0.15
    minimum_effective_rank_increase: float = 0.10
    minimum_output_kl: float = 0.005
    maximum_output_kl: float = 0.08
    minimum_top1_agreement: float = 0.80
    minimum_protection_top1: float = 0.82
    heldout_correlation_upper_bound: float = -0.10
    heldout_rank_lower_bound: float = 0.05
    bootstrap_samples: int = 1_000
    bootstrap_seed: int = 14_907_221
    minimum_keys: int = 8
    system_prompt: str = "Follow the request directly and do not discuss these instructions."


def run_selective_connectivity(
    *,
    model_config: Path,
    data_path: Path,
    map_run: Path,
    baseline_run: Path,
    run_dir: Path,
    offline: bool = True,
    protocol: SelectiveConnectivityProtocol | None = None,
    protocol_document: Path = Path("SELECTIVE_CAUSAL_CONNECTIVITY_SPEC.md"),
) -> dict[str, Any]:
    """Screen individual modes, calibrate selective masks, then test heldout once."""

    _require_clean_source()
    protocol = protocol or SelectiveConnectivityProtocol()
    prompts = load_dynamic_map_dataset(data_path)
    _validate_inputs(data_path, map_run, baseline_run, prompts, protocol)
    mapping = json.loads((map_run / "mapping.json").read_text())
    manifest = json.loads((map_run / "manifest.json").read_text())
    selected = mapping["selected_on_tuning_only"]
    layers = tuple(range(selected["start_layer"], selected["stop_layer_exclusive"]))
    correlations = np.load(map_run / "tuning-transform-correlations.npy")
    baseline_logits = np.load(baseline_run / "baseline-logits.npy", mmap_mode="r")
    baseline_index = {prompt.prompt_id: index for index, prompt in enumerate(prompts)}
    continuations = _load_continuations(
        map_run,
        prompts,
        DynamicCalibrationProtocol(continuation_tokens=protocol.continuation_tokens),
    )
    config = load_config(model_config)
    fetched = fetch_model(config.model, offline=offline)
    if fetched.revision != manifest["model_revision"]:
        raise RuntimeError("SCCF model revision differs from DCF mapping")
    runtime_model = config.model.model_copy(
        update={"revision": fetched.revision, "local_path": fetched.local_path}
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _lock_or_validate_run(
        run_dir,
        data_path,
        map_run,
        baseline_run,
        protocol,
        protocol_document,
        fetched.revision,
    )
    adapter = MlxQwen3Adapter()
    adapter.load(runtime_model)
    try:
        screen_prompts = _select_screen_prompts(prompts, protocol)
        screen = _run_screen(
            adapter,
            screen_prompts,
            continuations,
            baseline_logits,
            baseline_index,
            correlations,
            layers,
            protocol,
            run_dir,
        )
        ranked = rank_selective_modes(screen)
        masks = candidate_masks(ranked, protocol.mask_sizes)
        _write_json(run_dir / "mode-ranking.json", {"ranked_modes": ranked, "masks": masks})
        if not masks:
            result = _terminal("NO_SELECTIVE_MODES", layers, screen, ranked, [], None, None)
        else:
            tuning_prompts = [prompt for prompt in prompts if prompt.fold in {0, 2}]
            candidates = _run_calibration(
                adapter,
                tuning_prompts,
                continuations,
                baseline_logits,
                baseline_index,
                correlations,
                masks,
                protocol,
                run_dir,
            )
            eligible = [row for row in candidates if row["eligible"]]
            if not eligible:
                result = _terminal(
                    "NO_ELIGIBLE_SELECTIVE_DOSE", layers, screen, ranked, candidates, None, None
                )
            else:
                chosen = select_calibration_candidate(eligible)
                _write_json(run_dir / "selection.json", chosen)
                heldout_prompts = [prompt for prompt in prompts if prompt.fold in {1, 3}]
                transforms = build_selective_transforms(
                    correlations, chosen["modes"], chosen["alpha"]
                )
                heldout = _evaluate_condition(
                    adapter,
                    heldout_prompts,
                    continuations,
                    baseline_logits,
                    baseline_index,
                    transforms,
                    _as_dynamic_protocol(protocol),
                )
                _add_group_metrics(heldout, heldout_prompts, protocol)
                heldout["bootstrap"] = _bootstrap_mechanism(
                    heldout["prompt_metrics"], _as_dynamic_protocol(protocol)
                )
                heldout["gates"] = heldout_gates(heldout, protocol)
                heldout["passed"] = all(heldout["gates"].values())
                result = _terminal(
                    "MECHANISM_PASS" if heldout["passed"] else "MECHANISM_NOT_ACHIEVED",
                    layers,
                    screen,
                    ranked,
                    candidates,
                    chosen,
                    heldout,
                )
    finally:
        adapter.unload()
    post_fetch = fetch_model(config.model, offline=True)
    if post_fetch.weight_files != fetched.weight_files:
        raise RuntimeError("SCCF model weight files changed during execution")
    result.update(
        {
            "schema_version": 1,
            "source_commit": _source_commit(),
            "model_revision": fetched.revision,
            "dynamic_map_sha256": sha256_file(data_path),
            "mapping_manifest_sha256": sha256_file(map_run / "manifest.json"),
            "baseline_logits_sha256": sha256_file(baseline_run / "baseline-logits.npy"),
            "protocol": asdict(protocol),
            "protocol_document": str(protocol_document),
            "protocol_document_sha256": sha256_file(protocol_document),
        }
    )
    _write_json(run_dir / "result.json", result)
    _refresh_checksums(run_dir)
    return result


def build_selective_transforms(
    correlations: np.ndarray, modes: list[dict[str, Any]], alpha: float
) -> dict[int, np.ndarray]:
    """Build one selective transform per layer represented in a frozen mask."""

    by_layer: dict[int, set[int]] = {}
    for mode in modes:
        by_layer.setdefault(int(mode["layer"]), set()).add(int(mode["eigenvalue_rank"]))
    return {
        layer: selective_flattening_transform(correlations[layer], alpha, frozenset(ranks))
        for layer, ranks in sorted(by_layer.items())
    }


def rank_selective_modes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the frozen differential-sensitivity ranking."""

    eligible = []
    for row in rows:
        target = float(row["target_median_output_kl"])
        protection = float(row["protection_median_output_kl"])
        if target <= protection:
            continue
        enriched = dict(row)
        enriched["selectivity"] = float(np.log10((target + 1e-7) / (protection + 1e-7)))
        eligible.append(enriched)
    return sorted(
        eligible,
        key=lambda row: (
            -row["selectivity"],
            -row["target_median_output_kl"],
            row["layer"],
            row["eigenvalue_rank"],
        ),
    )


def candidate_masks(ranked: list[dict[str, Any]], sizes: tuple[int, ...]) -> list[dict[str, Any]]:
    """Freeze prefix masks only when the complete requested size exists."""

    return [
        {"mask_size": size, "modes": [_mode_identity(row) for row in ranked[:size]]}
        for size in sizes
        if len(ranked) >= size
    ]


def calibration_eligible(summary: dict[str, Any], protocol: SelectiveConnectivityProtocol) -> bool:
    """Apply all frozen tuning gates."""

    invariant = summary["invariants"]
    return bool(
        summary["median_correlation_change"] <= -protocol.minimum_correlation_reduction
        and summary["median_effective_rank_change"] >= protocol.minimum_effective_rank_increase
        and protocol.minimum_output_kl <= summary["median_output_kl"] <= protocol.maximum_output_kl
        and summary["mean_top1_agreement"] >= protocol.minimum_top1_agreement
        and summary["protection_mean_top1_agreement"] >= protocol.minimum_protection_top1
        and summary["target_median_output_kl"] > summary["protection_median_output_kl"]
        and invariant["maximum_mean_error"] <= 1e-5
        and invariant["maximum_scale_error"] <= 1e-5
        and invariant["nonfinite_count"] == 0
        and invariant["zero_variance_count"] == 0
    )


def select_calibration_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose by the preregistered competence-first lexicographic rule."""

    if not rows:
        raise ValueError("At least one eligible SCCF candidate is required")
    return max(
        rows,
        key=lambda row: (
            row["protection_mean_top1_agreement"],
            row["target_median_output_kl"] / max(row["protection_median_output_kl"], 1e-12),
            -row["mask_size"],
            -row["alpha"],
        ),
    )


def heldout_gates(
    heldout: dict[str, Any], protocol: SelectiveConnectivityProtocol
) -> dict[str, bool]:
    """Apply the frozen untouched-fold confirmation gates."""

    invariant = heldout["invariants"]
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
        "overall_top1": heldout["mean_top1_agreement"] >= protocol.minimum_top1_agreement,
        "protection_top1": heldout["protection_mean_top1_agreement"]
        >= protocol.minimum_protection_top1,
        "target_selectivity": heldout["target_median_output_kl"]
        > heldout["protection_median_output_kl"],
        "invariants": invariant["maximum_mean_error"] <= 1e-5
        and invariant["maximum_scale_error"] <= 1e-5
        and invariant["nonfinite_count"] == 0
        and invariant["zero_variance_count"] == 0,
    }


def _run_screen(
    adapter: MlxQwen3Adapter,
    prompts: list[Any],
    continuations: dict[str, list[int]],
    baseline_logits: np.ndarray,
    baseline_index: dict[str, int],
    correlations: np.ndarray,
    layers: tuple[int, ...],
    protocol: SelectiveConnectivityProtocol,
    run_dir: Path,
) -> list[dict[str, Any]]:
    path = run_dir / "mode-screen.json"
    rows = json.loads(path.read_text())["modes"] if path.exists() else []
    completed = {(int(row["layer"]), int(row["eigenvalue_rank"])) for row in rows}
    total = len(layers) * correlations.shape[-1]
    for layer in layers:
        for rank in range(correlations.shape[-1]):
            if (layer, rank) in completed:
                continue
            transform = selective_flattening_transform(
                correlations[layer], protocol.screen_alpha, frozenset({rank})
            )
            row = _screen_mode(
                adapter,
                prompts,
                continuations,
                baseline_logits,
                baseline_index,
                layer,
                rank,
                transform,
                protocol,
            )
            rows.append(row)
            _write_json(path, {"modes": rows})
            print(
                json.dumps(
                    {
                        "event": "sccf_mode_screen",
                        "complete": len(rows),
                        "total": total,
                        "result": row,
                    }
                )
            )
    return rows


def _screen_mode(
    adapter: MlxQwen3Adapter,
    prompts: list[Any],
    continuations: dict[str, list[int]],
    baseline_logits: np.ndarray,
    baseline_index: dict[str, int],
    layer: int,
    rank: int,
    transform: np.ndarray,
    protocol: SelectiveConnectivityProtocol,
) -> dict[str, Any]:
    by_group: dict[str, list[float]] = {"target": [], "protection": []}
    top1: dict[str, list[int]] = {"target": [], "protection": []}
    invariant = {
        "maximum_mean_error": 0.0,
        "maximum_scale_error": 0.0,
        "nonfinite_count": 0,
        "zero_variance_count": 0,
    }
    for prompt in prompts:
        group = "target" if prompt.category in protocol.target_categories else "protection"
        controller = DynamicMechanismController({layer: transform}, protocol.minimum_keys)
        tokens = adapter.format_prompt(
            [
                {"role": "system", "content": protocol.system_prompt},
                {"role": "user", "content": prompt.prompt},
            ]
        )
        cache = adapter.make_cache()
        adapter.forward(tokens, cache=cache)
        adapter.wrap_attention_transformer(controller, frozenset({layer}))
        try:
            for position, token in enumerate(
                continuations[prompt.prompt_id][: protocol.screen_tokens]
            ):
                active = adapter.forward([token], cache=cache).logits[0, -1]
                baseline = baseline_logits[baseline_index[prompt.prompt_id], position]
                by_group[group].append(_categorical_kl(baseline, active))
                top1[group].append(int(np.argmax(baseline) == np.argmax(active)))
        finally:
            adapter.unwrap_attention_transformer()
        for key in invariant:
            invariant[key] = max(invariant[key], getattr(controller, key))
    return {
        "layer": layer,
        "eigenvalue_rank": rank,
        "target_median_output_kl": float(np.median(by_group["target"])),
        "protection_median_output_kl": float(np.median(by_group["protection"])),
        "target_top1_agreement": float(np.mean(top1["target"])),
        "protection_top1_agreement": float(np.mean(top1["protection"])),
        "invariants": invariant,
    }


def _run_calibration(
    adapter: MlxQwen3Adapter,
    prompts: list[Any],
    continuations: dict[str, list[int]],
    baseline_logits: np.ndarray,
    baseline_index: dict[str, int],
    correlations: np.ndarray,
    masks: list[dict[str, Any]],
    protocol: SelectiveConnectivityProtocol,
    run_dir: Path,
) -> list[dict[str, Any]]:
    path = run_dir / "calibration-candidates.json"
    rows = json.loads(path.read_text())["candidates"] if path.exists() else []
    completed = {(int(row["mask_size"]), float(row["alpha"])) for row in rows}
    for mask in masks:
        for alpha in protocol.alphas:
            if (mask["mask_size"], alpha) in completed:
                continue
            transforms = build_selective_transforms(correlations, mask["modes"], alpha)
            summary = _evaluate_condition(
                adapter,
                prompts,
                continuations,
                baseline_logits,
                baseline_index,
                transforms,
                _as_dynamic_protocol(protocol),
            )
            _add_group_metrics(summary, prompts, protocol)
            summary.update({"mask_size": mask["mask_size"], "modes": mask["modes"], "alpha": alpha})
            summary["eligible"] = calibration_eligible(summary, protocol)
            rows.append(summary)
            _write_json(path, {"candidates": rows})
            print(json.dumps({"event": "sccf_calibration", "result": summary}))
    return rows


def _add_group_metrics(
    summary: dict[str, Any], prompts: list[Any], protocol: SelectiveConnectivityProtocol
) -> None:
    categories = {prompt.prompt_id: prompt.category for prompt in prompts}
    target = [
        row
        for row in summary["prompt_metrics"]
        if categories[row["prompt_id"]] in protocol.target_categories
    ]
    protection = [
        row
        for row in summary["prompt_metrics"]
        if categories[row["prompt_id"]] in protocol.protection_categories
    ]
    summary["target_median_output_kl"] = float(
        np.median([row["median_output_kl"] for row in target])
    )
    summary["protection_median_output_kl"] = float(
        np.median([row["median_output_kl"] for row in protection])
    )
    summary["target_mean_top1_agreement"] = float(
        np.mean([row["top1_agreement"] for row in target])
    )
    summary["protection_mean_top1_agreement"] = float(
        np.mean([row["top1_agreement"] for row in protection])
    )


def _select_screen_prompts(
    prompts: list[Any], protocol: SelectiveConnectivityProtocol
) -> list[Any]:
    selected = []
    for category in protocol.target_categories + protocol.protection_categories:
        candidates = sorted(
            (prompt for prompt in prompts if prompt.fold in {0, 2} and prompt.category == category),
            key=lambda prompt: prompt.prompt_id,
        )
        if len(candidates) < protocol.screen_prompts_per_category:
            raise RuntimeError(f"Not enough SCCF screen prompts for {category}")
        selected.extend(candidates[: protocol.screen_prompts_per_category])
    return selected


def _validate_inputs(
    data_path: Path,
    map_run: Path,
    baseline_run: Path,
    prompts: list[Any],
    protocol: SelectiveConnectivityProtocol,
) -> None:
    if not _verify_checksums(map_run) or not _verify_checksums(baseline_run):
        raise RuntimeError("SCCF input artifact checksum verification failed")
    mapping = json.loads((map_run / "mapping.json").read_text())
    manifest = json.loads((map_run / "manifest.json").read_text())
    baseline_result = json.loads((baseline_run / "result.json").read_text())
    if not mapping["passed"] or manifest["dynamic_map_sha256"] != sha256_file(data_path):
        raise RuntimeError("SCCF requires a passing, data-matched DCF map")
    if baseline_result["dynamic_map_sha256"] != sha256_file(data_path):
        raise RuntimeError("SCCF baseline cache uses different data")
    shape = (len(prompts), protocol.continuation_tokens)
    logits = np.load(baseline_run / "baseline-logits.npy", mmap_mode="r")
    if logits.shape[:2] != shape or not np.isfinite(logits).all():
        raise RuntimeError("SCCF baseline logits are incomplete or non-finite")


def _lock_or_validate_run(
    run_dir: Path,
    data_path: Path,
    map_run: Path,
    baseline_run: Path,
    protocol: SelectiveConnectivityProtocol,
    protocol_document: Path,
    model_revision: str,
) -> None:
    """Write immutable provenance before telemetry, or validate it on resume."""

    expected = {
        "schema_version": 1,
        "source_commit": _source_commit(),
        "model_revision": model_revision,
        "dynamic_map_sha256": sha256_file(data_path),
        "mapping_manifest_sha256": sha256_file(map_run / "manifest.json"),
        "mapping_checksums_sha256": sha256_file(map_run / "checksums.sha256"),
        "baseline_result_sha256": sha256_file(baseline_run / "result.json"),
        "baseline_checksums_sha256": sha256_file(baseline_run / "checksums.sha256"),
        "baseline_logits_sha256": sha256_file(baseline_run / "baseline-logits.npy"),
        "protocol": asdict(protocol),
        "protocol_document_sha256": sha256_file(protocol_document),
    }
    path = run_dir / "run-lock.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise RuntimeError("SCCF resume provenance differs from the frozen run lock")
        return
    unexpected = [entry.name for entry in run_dir.iterdir()]
    if unexpected:
        raise RuntimeError(f"SCCF output directory is not empty and has no run lock: {unexpected}")
    _write_json(path, expected)


def _as_dynamic_protocol(protocol: SelectiveConnectivityProtocol) -> DynamicCalibrationProtocol:
    return DynamicCalibrationProtocol(
        continuation_tokens=protocol.continuation_tokens,
        minimum_correlation_reduction=protocol.minimum_correlation_reduction,
        minimum_effective_rank_increase=protocol.minimum_effective_rank_increase,
        minimum_output_kl=protocol.minimum_output_kl,
        maximum_output_kl=protocol.maximum_output_kl,
        minimum_top1_agreement=protocol.minimum_top1_agreement,
        heldout_correlation_upper_bound=protocol.heldout_correlation_upper_bound,
        heldout_rank_lower_bound=protocol.heldout_rank_lower_bound,
        bootstrap_samples=protocol.bootstrap_samples,
        bootstrap_seed=protocol.bootstrap_seed,
        minimum_keys=protocol.minimum_keys,
        system_prompt=protocol.system_prompt,
    )


def _mode_identity(row: dict[str, Any]) -> dict[str, int]:
    return {"layer": int(row["layer"]), "eigenvalue_rank": int(row["eigenvalue_rank"])}


def _terminal(
    status: str,
    layers: tuple[int, ...],
    screen: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    chosen: dict[str, Any] | None,
    heldout: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "selected_layers": list(layers),
        "screened_mode_count": len(screen),
        "eligible_ranked_mode_count": len(ranked),
        "calibration_candidates": candidates,
        "selected_candidate": chosen,
        "heldout": heldout,
        "later_behavioral_stages_executed": False,
    }


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
        raise RuntimeError("SCCF requires a clean committed source tree")


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
