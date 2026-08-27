"""Paired prompt-level analysis from deterministic score artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..rng import derive_seed
from .bootstrap import paired_bootstrap
from .status import StatusInputs, classify_status
from .tests import holm_adjust


def calculate_analysis(run_dir: Path, master_seed: int, bootstrap_samples: int) -> dict[str, Any]:
    """Compute the deterministic analysis without mutating the run directory."""

    rows = _read_jsonl(run_dir / "prompt-effects.jsonl")
    by_prompt: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_id"], {})[row["condition"]] = row
    differences: list[float] = []
    coherent_validities: list[float] = []
    temp_validities: list[float] = []
    baseline_validities: list[float] = []
    coherent_degenerations: list[float] = []
    baseline_degenerations: list[float] = []
    white_validities: list[float] = []
    white_degenerations: list[float] = []
    coherent_white_vsd: list[float] = []
    coherent_white_validity: list[float] = []
    coherent_white_degeneration: list[float] = []
    control_coherent_validities: list[float] = []
    control_white_validities: list[float] = []
    for conditions in by_prompt.values():
        coherent = conditions.get("coherent")
        temp_match = conditions.get("temp_match")
        baseline = conditions.get("baseline")
        white = conditions.get("white")
        if (
            coherent is None
            or temp_match is None
            or coherent["vsd"] is None
            or temp_match["vsd"] is None
        ):
            continue
        differences.append(float(coherent["vsd"]) - float(temp_match["vsd"]))
        coherent_validities.append(float(coherent["validity_rate"]))
        temp_validities.append(float(temp_match["validity_rate"]))
        if baseline is not None:
            baseline_validities.append(float(baseline["validity_rate"]))
            baseline_degenerations.append(float(baseline.get("degeneration_rate", 0.0)))
            coherent_degenerations.append(float(coherent.get("degeneration_rate", 0.0)))
        if white is not None:
            white_validities.append(float(white["validity_rate"]))
            white_degenerations.append(float(white.get("degeneration_rate", 0.0)))
            coherent_white_validity.append(
                float(coherent["validity_rate"]) - float(white["validity_rate"])
            )
            coherent_white_degeneration.append(
                float(coherent.get("degeneration_rate", 0.0))
                - float(white.get("degeneration_rate", 0.0))
            )
            if coherent["vsd"] is not None and white["vsd"] is not None:
                coherent_white_vsd.append(float(coherent["vsd"]) - float(white["vsd"]))
            if _is_control(coherent):
                control_coherent_validities.append(float(coherent["validity_rate"]))
                control_white_validities.append(float(white["validity_rate"]))
    if not differences:
        raise ValueError("No complete coherent/temp_match prompt pairs are available")
    statistics = paired_bootstrap(
        np.array(differences),
        derive_seed(master_seed, "bootstrap", run_dir.name),
        samples=bootstrap_samples,
    )
    baseline_reference = baseline_validities or temp_validities
    validity_retained = np.mean(coherent_validities) + 0.05 >= np.mean(baseline_reference)
    degeneration_retained = not coherent_degenerations or (
        np.mean(coherent_degenerations) <= np.mean(baseline_degenerations) + 0.02
    )
    competence_coherent = control_coherent_validities or coherent_validities
    competence_white = control_white_validities or white_validities
    competence_difference = (
        float(np.mean(competence_coherent) - np.mean(competence_white))
        if competence_coherent and competence_white
        else None
    )
    h2_vsd_difference = float(np.mean(coherent_white_vsd)) if coherent_white_vsd else None
    secondary = _secondary_family(
        {
            "h2_coherent_minus_white_vsd": coherent_white_vsd,
            "h2_coherent_minus_white_validity": coherent_white_validity,
            "h2_coherent_minus_white_degeneration": coherent_white_degeneration,
            "h2_coherent_minus_white_competence": (
                [
                    coherent - white
                    for coherent, white in zip(
                        control_coherent_validities, control_white_validities, strict=True
                    )
                ]
                if control_coherent_validities and control_white_validities
                else coherent_white_validity
            ),
        },
        master_seed,
        run_dir.name,
        bootstrap_samples,
    )
    status = classify_status(
        StatusInputs(
            integrity_ok=True,
            h1_estimate=statistics.estimate,
            h1_ci95=statistics.ci95,
            validity_retained=bool(validity_retained),
            degeneration_retained=bool(degeneration_retained),
            coherent_beats_white_vsd=h2_vsd_difference is not None and h2_vsd_difference > 0,
            coherent_competence_not_worse_than_white=competence_difference is not None
            and competence_difference >= -0.05,
            replication_positive=None,
        )
    )
    analysis = {
        "schema_version": 1,
        "status": status,
        "primary": {
            "metric": "valid_semantic_diversity",
            "contrast": "coherent-temp_match",
            "estimate": statistics.estimate,
            "median": statistics.median,
            "ci95": list(statistics.ci95),
            "p_value": statistics.sign_flip_p_value,
            "n_prompts": len(differences),
            "positive": statistics.positive,
            "zero": statistics.zero,
            "negative": statistics.negative,
            "standardized_effect": statistics.standardized_effect,
        },
        "validity": {
            "coherent_mean": float(np.mean(coherent_validities)),
            "temp_match_mean": float(np.mean(temp_validities)),
            "baseline_mean": float(np.mean(baseline_reference)),
            "coherent_minus_baseline": float(
                np.mean(coherent_validities) - np.mean(baseline_reference)
            ),
            "coherent_minus_white": float(np.mean(coherent_validities) - np.mean(white_validities))
            if white_validities
            else None,
        },
        "competence": {
            "metric": "deterministic_control_validity",
            "coherent_mean": float(np.mean(competence_coherent)),
            "white_mean": float(np.mean(competence_white)) if competence_white else None,
            "coherent_minus_white": competence_difference,
            "n_control_prompts": len(control_coherent_validities),
        },
        "degeneration": {
            "coherent_mean": float(np.mean(coherent_degenerations))
            if coherent_degenerations
            else None,
            "baseline_mean": float(np.mean(baseline_degenerations))
            if baseline_degenerations
            else None,
            "white_mean": float(np.mean(white_degenerations)) if white_degenerations else None,
            "coherent_minus_baseline": float(
                np.mean(coherent_degenerations) - np.mean(baseline_degenerations)
            )
            if coherent_degenerations and baseline_degenerations
            else None,
            "coherent_minus_white": float(
                np.mean(coherent_degenerations) - np.mean(white_degenerations)
            )
            if coherent_degenerations and white_degenerations
            else None,
        },
        "secondary": {
            "h2_coherent_minus_white_vsd": h2_vsd_difference,
            "h2_coherent_minus_white_validity": float(
                np.mean(coherent_validities) - np.mean(white_validities)
            )
            if white_validities
            else None,
            "family": secondary,
        },
        "integrity": {"complete_paired_prompts": len(differences)},
    }
    return analysis


def analyze_run(run_dir: Path, master_seed: int, bootstrap_samples: int) -> dict[str, Any]:
    """Compute and persist preregistered prompt-clustered statistics."""

    analysis = calculate_analysis(run_dir, master_seed, bootstrap_samples)
    (run_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return analysis


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _is_control(row: dict[str, Any]) -> bool:
    return row.get("split") == "controls" or "control" in row.get("tags", [])


def _secondary_family(
    values: dict[str, list[float]], master_seed: int, run_id: str, bootstrap_samples: int
) -> dict[str, dict[str, Any]]:
    """Compute paired secondary outcomes and Holm-adjust their sign-flip p-values."""

    results: dict[str, dict[str, Any]] = {}
    raw_p_values: list[float] = []
    names: list[str] = []
    for name, rows in values.items():
        if not rows:
            results[name] = {
                "estimate": None,
                "ci95": [None, None],
                "p_value": None,
                "n_prompts": 0,
            }
            continue
        outcome = paired_bootstrap(
            np.asarray(rows, dtype=np.float64),
            derive_seed(master_seed, "bootstrap", run_id, "secondary", name),
            samples=bootstrap_samples,
        )
        results[name] = {
            "estimate": outcome.estimate,
            "ci95": list(outcome.ci95),
            "p_value": outcome.sign_flip_p_value,
            "p_value_holm": None,
            "n_prompts": len(rows),
        }
        names.append(name)
        raw_p_values.append(outcome.sign_flip_p_value)
    for name, adjusted in zip(names, holm_adjust(raw_p_values), strict=True):
        results[name]["p_value_holm"] = adjusted
    return results
