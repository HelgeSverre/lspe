"""Resumable mapping-feasibility runner for Dynamic Connectivity Flattening."""

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
    DynamicCorrelationObserver,
    matrix_similarity,
    mean_absolute_off_diagonal,
)
from .dynamic_data import load_dynamic_map_dataset


@dataclass(frozen=True)
class DynamicMappingProtocol:
    """Frozen DCF mapping choices."""

    continuation_tokens: int = 24
    selected_layers: tuple[int, ...] = tuple(range(36))
    window_width: int = 8
    minimum_keys: int = 8
    minimum_observations: int = 1_000
    minimum_stable_layers: int = 24
    minimum_layer_similarity: float = 0.70
    minimum_window_median_similarity: float = 0.75
    minimum_window_layer_similarity: float = 0.60
    minimum_window_synchrony: float = 0.02
    system_prompt: str = "Follow the request directly and do not discuss these instructions."


def run_dynamic_mapping(
    *, model_config: Path, data_path: Path, run_dir: Path, offline: bool = True
) -> dict[str, Any]:
    """Collect frozen greedy trajectories and evaluate DCF mapping feasibility."""

    _require_clean_source()
    protocol = DynamicMappingProtocol()
    prompts = load_dynamic_map_dataset(data_path)
    config = load_config(model_config)
    fetched = fetch_model(config.model, offline=offline)
    runtime_model = config.model.model_copy(
        update={"revision": fetched.revision, "local_path": fetched.local_path}
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter = MlxQwen3Adapter()
    adapter.load(runtime_model)
    try:
        geometry = adapter.attention_geometry()
        if geometry["layers"] != 36 or geometry["attention_heads"] != 32:
            raise RuntimeError(f"DCF frozen geometry mismatch: {geometry}")
        sham = _sham_equivalence(adapter, prompts[0], protocol, geometry)
        if not sham["passed"]:
            raise RuntimeError(f"DCF sham equivalence failed: {sham}")
        observer = DynamicCorrelationObserver(
            geometry["layers"], geometry["attention_heads"], protocol.minimum_keys
        )
        completed, continuation_rows = _restore_checkpoint(run_dir, observer, data_path)
        _validate_resume_rows(continuation_rows, prompts, protocol)
        adapter.wrap_attention_transformer(observer, frozenset(protocol.selected_layers))
        _collect_trajectories(
            adapter,
            observer,
            prompts,
            completed,
            continuation_rows,
            protocol,
            run_dir,
            data_path,
        )
        adapter.unwrap_attention_transformer()
    finally:
        adapter.unload()
    post_fetch = fetch_model(config.model, offline=True)
    if post_fetch.weight_files != fetched.weight_files:
        raise RuntimeError("DCF model weight files changed during mapping")
    result = _evaluate_mapping(observer, protocol, run_dir, sham)
    manifest = {
        "schema_version": 1,
        "stage": "dynamic_connectivity_mapping",
        "source_commit": _source_commit(),
        "model_repo_id": fetched.repo_id,
        "model_revision": fetched.revision,
        "model_weight_files": fetched.weight_files,
        "dynamic_map_sha256": sha256_file(data_path),
        "protocol": asdict(protocol),
        "geometry": geometry,
        "result": result,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_checksums(run_dir)
    return manifest


def _collect_trajectories(
    adapter: MlxQwen3Adapter,
    observer: DynamicCorrelationObserver,
    prompts: list[Any],
    completed: set[str],
    continuation_rows: list[dict[str, Any]],
    protocol: DynamicMappingProtocol,
    run_dir: Path,
    data_path: Path,
) -> None:
    for index, prompt in enumerate(prompts, start=1):
        if prompt.prompt_id in completed:
            continue
        observer.current_fold = prompt.fold
        tokens = adapter.format_prompt(
            [
                {"role": "system", "content": protocol.system_prompt},
                {"role": "user", "content": prompt.prompt},
            ]
        )
        cache = adapter.make_cache()
        logits = adapter.forward(tokens, cache=cache).logits[0, -1]
        generated: list[int] = []
        for _ in range(protocol.continuation_tokens):
            token = int(np.argmax(logits))
            generated.append(token)
            logits = adapter.forward([token], cache=cache).logits[0, -1]
        row = {
            "prompt_id": prompt.prompt_id,
            "category": prompt.category,
            "fold": prompt.fold,
            "token_ids": generated,
            "text": adapter.decode(generated),
        }
        continuation_rows.append(row)
        _write_checkpoint(run_dir, observer, data_path, continuation_rows)
        _write_continuations(run_dir, continuation_rows)
        print(
            json.dumps(
                {
                    "event": "dcf_mapping_prompt",
                    "complete": index,
                    "total": len(prompts),
                    "prompt_id": prompt.prompt_id,
                }
            )
        )


def _evaluate_mapping(
    observer: DynamicCorrelationObserver,
    protocol: DynamicMappingProtocol,
    run_dir: Path,
    sham: dict[str, Any],
) -> dict[str, Any]:
    fold_matrices = observer.transforms()
    tuning_similarity = np.array(
        [matrix_similarity(fold_matrices[0, layer], fold_matrices[2, layer]) for layer in range(36)]
    )
    heldout_similarity = np.array(
        [matrix_similarity(fold_matrices[1, layer], fold_matrices[3, layer]) for layer in range(36)]
    )
    tuning_average = 0.5 * (fold_matrices[0] + fold_matrices[2])
    tuning_synchrony = np.array(
        [mean_absolute_off_diagonal(tuning_average[layer]) for layer in range(36)]
    )
    candidates = []
    for start in range(36 - protocol.window_width + 1):
        stop = start + protocol.window_width
        candidates.append(
            {
                "start_layer": start,
                "stop_layer_exclusive": stop,
                "median_tuning_similarity": float(np.median(tuning_similarity[start:stop])),
                "minimum_tuning_similarity": float(np.min(tuning_similarity[start:stop])),
                "median_tuning_synchrony": float(np.median(tuning_synchrony[start:stop])),
            }
        )
    selected = select_dynamic_window(candidates)
    start, stop = selected["start_layer"], selected["stop_layer_exclusive"]
    gates = {
        "sham_equivalence": sham["passed"],
        "minimum_observations": bool(np.all(observer.counts >= protocol.minimum_observations)),
        "stable_tuning_layers": int(np.sum(tuning_similarity >= protocol.minimum_layer_similarity))
        >= protocol.minimum_stable_layers,
        "stable_heldout_layers": int(
            np.sum(heldout_similarity >= protocol.minimum_layer_similarity)
        )
        >= protocol.minimum_stable_layers,
        "window_tuning_median": selected["median_tuning_similarity"]
        >= protocol.minimum_window_median_similarity,
        "window_tuning_minimum": selected["minimum_tuning_similarity"]
        >= protocol.minimum_window_layer_similarity,
        "window_heldout_median": float(np.median(heldout_similarity[start:stop]))
        >= protocol.minimum_window_median_similarity,
        "window_heldout_minimum": float(np.min(heldout_similarity[start:stop]))
        >= protocol.minimum_window_layer_similarity,
        "meaningful_synchrony": selected["median_tuning_synchrony"]
        >= protocol.minimum_window_synchrony,
    }
    np.save(run_dir / "fold-correlations.npy", fold_matrices)
    np.save(run_dir / "tuning-transform-correlations.npy", tuning_average)
    result = {
        "status": "MAPPING_PASS" if all(gates.values()) else "MECHANISM_NOT_ACHIEVED",
        "selected_on_tuning_only": selected,
        "selected_heldout_median_similarity": float(
            np.median(heldout_similarity[start:stop])
        ),
        "selected_heldout_minimum_similarity": float(np.min(heldout_similarity[start:stop])),
        "tuning_stable_layer_count": int(
            np.sum(tuning_similarity >= protocol.minimum_layer_similarity)
        ),
        "heldout_stable_layer_count": int(
            np.sum(heldout_similarity >= protocol.minimum_layer_similarity)
        ),
        "tuning_similarity_by_layer": tuning_similarity.tolist(),
        "heldout_similarity_by_layer": heldout_similarity.tolist(),
        "tuning_synchrony_by_layer": tuning_synchrony.tolist(),
        "observation_counts": observer.counts.tolist(),
        "decode_steps": observer.steps.tolist(),
        "sham_equivalence": sham,
        "gates": gates,
        "passed": all(gates.values()),
        "candidates": candidates,
    }
    (run_dir / "mapping.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _sham_equivalence(
    adapter: MlxQwen3Adapter,
    prompt: Any,
    protocol: DynamicMappingProtocol,
    geometry: dict[str, int],
) -> dict[str, Any]:
    tokens = adapter.format_prompt(
        [
            {"role": "system", "content": protocol.system_prompt},
            {"role": "user", "content": prompt.prompt},
        ]
    )

    def decode_logits() -> np.ndarray:
        cache = adapter.make_cache()
        prefill = adapter.forward(tokens, cache=cache).logits[0, -1]
        token = int(np.argmax(prefill))
        return adapter.forward([token], cache=cache).logits[0, -1]

    baseline = decode_logits()
    observer = DynamicCorrelationObserver(
        geometry["layers"], geometry["attention_heads"], protocol.minimum_keys
    )
    adapter.wrap_attention_transformer(observer, frozenset(protocol.selected_layers))
    try:
        wrapped = decode_logits()
    finally:
        adapter.unwrap_attention_transformer()
    maximum = float(np.max(np.abs(baseline - wrapped)))
    return {
        "maximum_absolute_error": maximum,
        "top1_equal": int(np.argmax(baseline)) == int(np.argmax(wrapped)),
        "tolerance": 1e-5,
        "passed": maximum <= 1e-5 and int(np.argmax(baseline)) == int(np.argmax(wrapped)),
    }


def select_dynamic_window(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select only from tuning statistics using the frozen lexicographic rule."""

    if not candidates:
        raise ValueError("At least one DCF layer-window candidate is required")
    return max(
        candidates,
        key=lambda row: (
            row["median_tuning_similarity"],
            row["median_tuning_synchrony"],
            -row["start_layer"],
        ),
    )


def _validate_resume_rows(
    rows: list[dict[str, Any]], prompts: list[Any], protocol: DynamicMappingProtocol
) -> None:
    expected = {prompt.prompt_id: prompt for prompt in prompts}
    for row in rows:
        prompt_id = str(row.get("prompt_id", ""))
        if prompt_id not in expected:
            raise RuntimeError(f"Unknown DCF resume prompt: {prompt_id}")
        prompt = expected[prompt_id]
        if row.get("category") != prompt.category or row.get("fold") != prompt.fold:
            raise RuntimeError(f"DCF resume metadata mismatch for {prompt_id}")
        token_ids = row.get("token_ids")
        if not isinstance(token_ids, list) or len(token_ids) != protocol.continuation_tokens:
            raise RuntimeError(f"DCF resume token count mismatch for {prompt_id}")
        if not all(isinstance(token, int) and token >= 0 for token in token_ids):
            raise RuntimeError(f"DCF resume contains invalid token IDs for {prompt_id}")


def _restore_checkpoint(
    run_dir: Path, observer: DynamicCorrelationObserver, data_path: Path
) -> tuple[set[str], list[dict[str, Any]]]:
    continuation_path = run_dir / "fixed-continuations.jsonl"
    checkpoint_path = run_dir / "mapping-checkpoint.npz"
    rows: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        checkpoint = np.load(checkpoint_path)
        if str(checkpoint["data_sha256"].item()) != sha256_file(data_path):
            raise RuntimeError("DCF checkpoint data hash mismatch")
        observer.restore(checkpoint["sums"], checkpoint["counts"], checkpoint["steps"])
        rows = json.loads(str(checkpoint["continuation_rows_json"].item()))
    elif continuation_path.exists():
        raise RuntimeError("DCF continuation artifact exists without its checkpoint")
    identifiers = [str(row["prompt_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("DCF resume contains duplicate prompt rows")
    if continuation_path.exists():
        artifact_rows = [
            json.loads(line) for line in continuation_path.read_text().splitlines() if line
        ]
        if artifact_rows != rows:
            _write_continuations(run_dir, rows)
    return set(identifiers), rows


def _write_checkpoint(
    run_dir: Path,
    observer: DynamicCorrelationObserver,
    data_path: Path,
    continuation_rows: list[dict[str, Any]],
) -> None:
    destination = run_dir / "mapping-checkpoint.npz"
    temporary = run_dir / "mapping-checkpoint.tmp"
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            sums=observer.sums,
            counts=observer.counts,
            steps=observer.steps,
            data_sha256=np.array(sha256_file(data_path)),
            continuation_rows_json=np.array(json.dumps(continuation_rows, sort_keys=True)),
        )
    os.replace(temporary, destination)


def _write_continuations(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    destination = run_dir / "fixed-continuations.jsonl"
    temporary = run_dir / "fixed-continuations.tmp"
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


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
        raise RuntimeError("DCF mapping requires a clean committed source tree")


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
