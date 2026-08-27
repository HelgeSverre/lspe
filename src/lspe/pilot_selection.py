"""Preregistered pilot eligibility and deterministic candidate selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import LspeConfig


@dataclass(frozen=True)
class PilotCandidate:
    candidate_id: str
    selected_layers: tuple[int, ...]
    target_kl: float
    raw_dose: float
    achieved_median_kl: float
    white_raw_dose: float
    white_achieved_median_kl: float
    validity_baseline: float
    validity_coherent: float
    degeneration_baseline: float
    degeneration_coherent: float
    vsd_coherent_temp: float
    vsd_coherent_white: float
    utility: float
    eligible: bool
    eligibility_reasons: tuple[str, ...]


@dataclass(frozen=True)
class PilotSelection:
    status: str
    selected: PilotCandidate
    candidates: tuple[PilotCandidate, ...]


def select_pilot_candidate(
    run_dir: Path,
    config: LspeConfig,
    calibration: dict[str, Any],
    *,
    candidate_id: str = "single_best_pilot_layer",
    target_kl: float | None = None,
    persist: bool = True,
) -> PilotSelection:
    """Apply the fixed rule and retain a null/degradation choice when needed."""

    effects = _read_jsonl(run_dir / "prompt-effects.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = calibration["summary"]
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in effects:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    required = {"baseline", "coherent", "white", "temp_match"}
    missing = sorted(required - set(by_condition))
    if missing:
        raise ValueError(
            "Pilot is missing conditions required for selection: " + ", ".join(missing)
        )

    averages = {condition: _averages(rows) for condition, rows in by_condition.items()}
    target = target_kl if target_kl is not None else config.intervention.target_kl_nats
    achieved = float(summary["achieved_median_kl"])
    white_achieved = float(summary.get("white_achieved_median_kl", achieved))
    reasons: list[str] = []
    if not target * 0.8 <= achieved <= target * 1.2:
        reasons.append("KL_OUTSIDE_TARGET_BAND")
    if abs(white_achieved - achieved) > max(target * 0.2, 0.005):
        reasons.append("WHITE_KL_MATCH_FAILED")
    if averages["coherent"]["validity"] < averages["baseline"]["validity"] - 0.05:
        reasons.append("VALIDITY_NONINFERIORITY_FAILED")
    if averages["coherent"]["degeneration"] > averages["baseline"]["degeneration"] + 0.02:
        reasons.append("DEGENERATION_MARGIN_FAILED")
    utility = (
        averages["coherent"]["vsd"]
        - averages["temp_match"]["vsd"]
        + 0.25 * (averages["coherent"]["vsd"] - averages["white"]["vsd"])
        - 0.50 * max(0.0, averages["baseline"]["validity"] - averages["coherent"]["validity"])
    )
    candidate = PilotCandidate(
        candidate_id=candidate_id,
        selected_layers=tuple(int(value) for value in manifest["selected_layers"]),
        target_kl=target,
        raw_dose=float(summary["raw_dose"]),
        achieved_median_kl=achieved,
        white_raw_dose=float(summary.get("white_raw_dose", summary["raw_dose"])),
        white_achieved_median_kl=white_achieved,
        validity_baseline=averages["baseline"]["validity"],
        validity_coherent=averages["coherent"]["validity"],
        degeneration_baseline=averages["baseline"]["degeneration"],
        degeneration_coherent=averages["coherent"]["degeneration"],
        vsd_coherent_temp=averages["coherent"]["vsd"] - averages["temp_match"]["vsd"],
        vsd_coherent_white=averages["coherent"]["vsd"] - averages["white"]["vsd"],
        utility=utility,
        eligible=not reasons,
        eligibility_reasons=tuple(reasons),
    )
    result = PilotSelection(
        status="ELIGIBLE" if candidate.eligible else "NO_ELIGIBLE_INTERVENTION",
        selected=candidate,
        candidates=(candidate,),
    )
    if persist:
        (run_dir / "pilot-selection.json").write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def select_pilot_matrix(candidates: list[PilotCandidate]) -> PilotSelection:
    """Choose an eligible candidate by utility and the preregistered tie breaks."""

    if not candidates:
        raise ValueError("Pilot selection requires at least one candidate")
    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        lowest = min(candidates, key=lambda candidate: (candidate.raw_dose, candidate.candidate_id))
        return PilotSelection("NO_ELIGIBLE_INTERVENTION", lowest, tuple(candidates))
    selected = min(
        eligible,
        key=lambda candidate: (
            -candidate.utility,
            candidate.raw_dose,
            len(candidate.selected_layers),
            sum(candidate.selected_layers) / len(candidate.selected_layers),
            candidate.candidate_id,
        ),
    )
    return PilotSelection("ELIGIBLE", selected, tuple(candidates))


def _averages(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = [float(row["vsd"]) for row in rows if row.get("vsd") is not None]
    if not values:
        raise ValueError("Pilot condition has no valid-semantic-diversity estimates")
    return {
        "vsd": float(np.mean(values)),
        "validity": float(np.mean([float(row["validity_rate"]) for row in rows])),
        "degeneration": float(np.mean([float(row.get("degeneration_rate", 0.0)) for row in rows])),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required pilot artifact: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
