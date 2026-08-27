"""Immutable confirmatory-lock construction and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .config import LspeConfig
from .hashing import canonical_json, sha256_bytes, sha256_file


@dataclass(frozen=True)
class ExperimentLock:
    schema_version: int
    experiment_id: str
    created_at: str
    pilot_run_id: str
    config_hash: str
    data_hashes: dict[str, str]
    model: dict[str, Any]
    intervention: dict[str, Any]
    sampling: dict[str, Any]
    statistics: dict[str, Any]
    resolved_config: dict[str, Any]

    def dump(self) -> dict[str, Any]:
        return asdict(self)


def create_experiment_lock(
    config: LspeConfig,
    config_path: Path,
    pilot_run_id: str,
    model_revision: str,
    selected_layers: list[int],
    selected_layer_types: list[str],
    raw_dose: float,
    achieved_kl: float,
    matched_temperature: float,
    white_raw_dose: float | None = None,
    white_achieved_kl: float | None = None,
) -> ExperimentLock:
    """Freeze every scientific input required by a confirmatory run."""

    if config.experiment.phase not in {"pilot", "confirm"}:
        raise ValueError("Only pilot/confirm profiles may create a confirmatory experiment lock")
    if not model_revision:
        raise ValueError("A model revision must be resolved to an immutable commit before freezing")
    if len(selected_layers) != len(selected_layer_types) or not selected_layers:
        raise ValueError(
            "Selected layer indices and discovered types must be non-empty and aligned"
        )
    data_paths = config.data.model_dump(mode="python")
    hashes = {key: sha256_file(Path(path)) for key, path in data_paths.items()}
    model = config.model.model_dump(mode="json")
    model["revision"] = model_revision
    intervention = {
        "kernel": config.intervention.kernel,
        "direction_mode": config.intervention.direction_mode,
        "selected_layers": selected_layers,
        "selected_layer_types": selected_layer_types,
        "raw_dose": raw_dose,
        "achieved_target_kl_nats": achieved_kl,
        "white_raw_dose": white_raw_dose if white_raw_dose is not None else raw_dose,
        "white_achieved_target_kl_nats": (
            white_achieved_kl if white_achieved_kl is not None else achieved_kl
        ),
        "timing": config.intervention.timing,
    }
    payload = {
        "pilot_run_id": pilot_run_id,
        "config_hash": sha256_file(config_path),
        "data_hashes": hashes,
        "model": model,
        "intervention": intervention,
        "sampling": {
            **config.sampling.model_dump(mode="json"),
            "matched_temperature": matched_temperature,
        },
        "statistics": config.statistics.model_dump(mode="json"),
        "resolved_config": config.model_dump(mode="json"),
    }
    experiment_id = f"lspe-{sha256_bytes(canonical_json(payload))[:20]}"
    return ExperimentLock(
        schema_version=1,
        experiment_id=experiment_id,
        created_at=datetime.now(UTC).isoformat(),
        **payload,
    )


def write_experiment_lock(lock: ExperimentLock, path: Path) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = yaml.safe_dump(lock.dump(), sort_keys=True).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to modify immutable experiment lock: {path}") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def load_experiment_lock(path: Path) -> ExperimentLock:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment lock must be a YAML mapping")
    return ExperimentLock(**raw)
