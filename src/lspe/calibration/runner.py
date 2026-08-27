"""Teacher-forced calibration against full next-token distributions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

import numpy as np

from ..columnar import write_parquet
from ..config import LspeConfig
from ..interventions.controller import InterventionController
from ..models.factory import create_adapter
from ..tasks.loader import load_prompts
from .dose import distribution_divergence
from .entropy import match_temperature, sampling_entropy

TEACHER_FORCED_CONTINUATION_TOKENS = 16


@dataclass(frozen=True)
class CalibrationPoint:
    direction_mode: str
    raw_dose: float
    direction_seed_index: int
    prompt_id: str
    continuation_position: int
    kl_nats: float
    js: float
    top1_agreement: bool
    top_k_overlap: float


@dataclass(frozen=True)
class CalibrationSummary:
    run_dir: Path
    selected_layers: tuple[int, ...]
    raw_dose: float
    achieved_median_kl: float
    white_raw_dose: float
    white_achieved_median_kl: float
    points: int
    teacher_forced_tokens: int
    matched_temperature: float
    target_sampling_entropy: float
    achieved_sampling_entropy: float
    entropy_absolute_mismatch: float


def choose_sentinel_layers(total_layers: int, requested: list[int] | str) -> list[int]:
    if isinstance(requested, list):
        if not requested:
            raise ValueError("selected_layers must not be empty")
        return requested
    if total_layers < 1:
        raise ValueError("Model has no decoder layers")
    return [round((total_layers - 1) * 0.5)]


def calibrate(
    config: LspeConfig,
    model_revision: str,
    output_root: Path | None = None,
    direction_seeds: int = 3,
    forced_raw_dose: float | None = None,
) -> CalibrationSummary:
    """Calibrate on teacher-forced cached continuations and match sampling entropy."""

    if direction_seeds < 1:
        raise ValueError("direction_seeds must be positive")
    run_dir = (output_root or config.experiment.output_root) / (
        f"calibration-{config.experiment.name}-{model_revision[:12]}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter = create_adapter(config.model)
    points: list[CalibrationPoint] = []
    try:
        adapter.load(config.model)
        architecture = adapter.architecture()
        layers = choose_sentinel_layers(
            len(architecture.layers), config.intervention.selected_layers
        )
        prompts = load_prompts(config.data.calibration, "calibration")
        for prompt in prompts:
            messages = [
                {"role": "system", "content": config.prompting.system},
                {"role": "user", "content": prompt.prompt},
            ]
            token_ids = adapter.format_prompt(messages)
            continuation = _greedy_continuation(
                adapter, token_ids, TEACHER_FORCED_CONTINUATION_TOKENS
            )
            baseline_rows = _teacher_forced_logits(adapter, token_ids, continuation)
            for direction_mode, condition_id in (
                ("coherent_per_layer", "coherent"),
                ("white_per_token", "white"),
            ):
                for raw_dose in config.intervention.raw_dose_grid:
                    for seed_index in range(direction_seeds):
                        controller = InterventionController(
                            master_seed=config.experiment.master_seed,
                            run_id=_calibration_direction_namespace(config, layers),
                            prompt_id=prompt.prompt_id,
                            generation_index=seed_index,
                            condition_id=condition_id,
                            selected_layers=frozenset(layers),
                            dose=raw_dose,
                            mode=direction_mode,
                            kernel=config.intervention.kernel,
                            decode_start_token=len(token_ids) - 1,
                        )
                        altered_rows = _teacher_forced_logits(
                            adapter, token_ids, continuation, controller
                        )
                        for position, (altered, baseline) in enumerate(
                            zip(altered_rows, baseline_rows, strict=True)
                        ):
                            divergence = distribution_divergence(altered, baseline)
                            points.append(
                                CalibrationPoint(
                                    direction_mode=direction_mode,
                                    raw_dose=raw_dose,
                                    direction_seed_index=seed_index,
                                    prompt_id=prompt.prompt_id,
                                    continuation_position=position,
                                    kl_nats=divergence.kl_altered_baseline,
                                    js=divergence.js,
                                    top1_agreement=divergence.top1_agreement,
                                    top_k_overlap=divergence.top_k_overlap,
                                )
                            )
    finally:
        adapter.unload()
    if not points:
        raise RuntimeError("Calibration produced no points")
    medians = {
        mode: {
            dose: median(
                point.kl_nats
                for point in points
                if point.direction_mode == mode and point.raw_dose == dose
            )
            for dose in config.intervention.raw_dose_grid
        }
        for mode in ("coherent_per_layer", "white_per_token")
    }
    coherent_medians = medians["coherent_per_layer"]
    white_medians = medians["white_per_token"]
    if forced_raw_dose is not None:
        if forced_raw_dose not in coherent_medians:
            raise ValueError("forced_raw_dose must be present in intervention.raw_dose_grid")
        selected_dose = forced_raw_dose
    else:
        selected_dose = min(
            coherent_medians,
            key=lambda dose: (
                abs(coherent_medians[dose] - config.intervention.target_kl_nats),
                dose,
            ),
        )
    # A pilot curve serves multiple preregistered target cells.  Its default
    # target is only a descriptive summary: each candidate below is matched
    # and gated independently in ``derive_calibration_from_curve``.  Applying
    # that gate here can discard a completed curve before lower or higher
    # pilot targets have a chance to be evaluated.
    is_pilot_curve = config.experiment.phase == "pilot" and bool(
        config.intervention.pilot_target_kl_bands
    )
    if is_pilot_curve:
        selected_white_dose = min(
            white_medians,
            key=lambda dose: (
                abs(white_medians[dose] - coherent_medians[selected_dose]),
                dose,
            ),
        )
    else:
        selected_white_dose = select_matching_white_dose(
            white_medians,
            coherent_medians[selected_dose],
            config.intervention.target_kl_nats,
        )
    entropy_adapter = create_adapter(config.model)
    try:
        entropy_adapter.load(config.model)
        baseline_entropy_rows, coherent_entropy_rows = _entropy_calibration_rows(
            entropy_adapter,
            config,
            layers,
            selected_dose,
            TEACHER_FORCED_CONTINUATION_TOKENS,
        )
    finally:
        entropy_adapter.unload()
    target_entropy = float(
        np.mean(
            [
                sampling_entropy(
                    logits,
                    config.sampling.temperature,
                    config.sampling.top_k,
                    config.sampling.top_p,
                )
                for logits in coherent_entropy_rows
            ]
        )
    )
    entropy_match = match_temperature(
        baseline_entropy_rows,
        target_entropy,
        config.sampling.top_k,
        config.sampling.top_p,
    )
    summary = CalibrationSummary(
        run_dir=run_dir,
        selected_layers=tuple(layers),
        raw_dose=selected_dose,
        achieved_median_kl=float(coherent_medians[selected_dose]),
        white_raw_dose=selected_white_dose,
        white_achieved_median_kl=float(white_medians[selected_white_dose]),
        points=len(points),
        teacher_forced_tokens=TEACHER_FORCED_CONTINUATION_TOKENS,
        matched_temperature=entropy_match.temperature,
        target_sampling_entropy=entropy_match.target_entropy,
        achieved_sampling_entropy=entropy_match.achieved_entropy,
        entropy_absolute_mismatch=entropy_match.absolute_mismatch,
    )
    (run_dir / "calibration.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "model_revision": model_revision,
                "summary": {**asdict(summary), "run_dir": str(summary.run_dir)},
                "points": [asdict(point) for point in points],
                "median_kl_by_raw_dose": medians,
                "entropy_match": asdict(entropy_match),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_parquet(run_dir / "calibration.parquet", [asdict(point) for point in points])
    return summary


def derive_calibration_from_curve(
    config: LspeConfig,
    model_revision: str,
    curve_path: Path,
    target_kl: float,
) -> CalibrationSummary:
    """Reuse one group-level teacher-forced curve for a preregistered dose cell."""

    source = json.loads(curve_path.read_text(encoding="utf-8"))
    if source.get("model_revision") != model_revision:
        raise ValueError("Calibration curve was produced by a different model revision")
    medians = source["median_kl_by_raw_dose"]
    coherent = {float(dose): float(value) for dose, value in medians["coherent_per_layer"].items()}
    white = {float(dose): float(value) for dose, value in medians["white_per_token"].items()}
    selected_raw_dose = select_target_dose(coherent, target_kl)
    source_summary = source["summary"]
    layers = [int(value) for value in source_summary["selected_layers"]]
    selected_white_dose = select_matching_white_dose(
        white,
        coherent[selected_raw_dose],
        target_kl,
    )
    entropy_adapter = create_adapter(config.model)
    try:
        entropy_adapter.load(config.model)
        baseline_rows, coherent_rows = _entropy_calibration_rows(
            entropy_adapter,
            config,
            layers,
            selected_raw_dose,
            TEACHER_FORCED_CONTINUATION_TOKENS,
        )
    finally:
        entropy_adapter.unload()
    target_entropy = float(
        np.mean(
            [
                sampling_entropy(
                    logits,
                    config.sampling.temperature,
                    config.sampling.top_k,
                    config.sampling.top_p,
                )
                for logits in coherent_rows
            ]
        )
    )
    entropy_match = match_temperature(
        baseline_rows,
        target_entropy,
        config.sampling.top_k,
        config.sampling.top_p,
    )
    run_dir = config.experiment.output_root / (
        f"calibration-{config.experiment.name}-{model_revision[:12]}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = CalibrationSummary(
        run_dir=run_dir,
        selected_layers=tuple(layers),
        raw_dose=selected_raw_dose,
        achieved_median_kl=coherent[selected_raw_dose],
        white_raw_dose=selected_white_dose,
        white_achieved_median_kl=white[selected_white_dose],
        points=0,
        teacher_forced_tokens=TEACHER_FORCED_CONTINUATION_TOKENS,
        matched_temperature=entropy_match.temperature,
        target_sampling_entropy=target_entropy,
        achieved_sampling_entropy=entropy_match.achieved_entropy,
        entropy_absolute_mismatch=entropy_match.absolute_mismatch,
    )
    (run_dir / "calibration.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "model_revision": model_revision,
                "curve_source": str(curve_path.resolve()),
                "summary": {**asdict(summary), "run_dir": str(summary.run_dir)},
                "median_kl_by_raw_dose": medians,
                "entropy_match": asdict(entropy_match),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def select_target_dose(coherent_medians: dict[float, float], target_kl: float) -> float:
    """Select the closest in-band dose, failing closed when the curve is too coarse."""

    in_band = [
        dose
        for dose, value in coherent_medians.items()
        if target_kl * 0.8 <= value <= target_kl * 1.2
    ]
    if not in_band:
        closest = min(
            coherent_medians,
            key=lambda dose: (abs(coherent_medians[dose] - target_kl), dose),
        )
        achieved = coherent_medians[closest]
        raise ValueError(
            "Calibration grid cannot resolve target KL band: "
            f"target={target_kl:.6g}, closest_raw_dose={closest:.6g}, "
            f"achieved={achieved:.6g}. Refine raw_dose_grid before generating a pilot candidate."
        )
    selected = min(
        in_band,
        key=lambda dose: (abs(coherent_medians[dose] - target_kl), dose),
    )
    return selected


def select_matching_white_dose(
    white_medians: dict[float, float], coherent_kl: float, target_kl: float
) -> float:
    """Choose a white-noise dose only when its KL match is protocol-valid."""

    selected = min(white_medians, key=lambda dose: (abs(white_medians[dose] - coherent_kl), dose))
    achieved = white_medians[selected]
    tolerance = max(target_kl * 0.2, 0.005)
    if abs(achieved - coherent_kl) > tolerance:
        raise ValueError(
            "Calibration grid cannot match white-noise KL: "
            f"target={target_kl:.6g}, coherent={coherent_kl:.6g}, "
            f"closest_white_raw_dose={selected:.6g}, white_achieved={achieved:.6g}. "
            "Refine raw_dose_grid before generating a pilot candidate."
        )
    return selected


def _last_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits)
    if values.ndim < 1 or not np.isfinite(values).all():
        raise FloatingPointError("Calibration forward returned non-finite logits")
    return values.reshape(-1, values.shape[-1])[-1]


def _greedy_continuation(adapter: object, prompt_ids: list[int], tokens: int) -> list[int]:
    cache = adapter.make_cache()
    adapter.forward(prompt_ids[:-1], cache=cache)
    next_input = [prompt_ids[-1]]
    continuation: list[int] = []
    for _ in range(tokens):
        logits = _last_logits(adapter.forward(next_input, cache=cache).logits)
        token = int(np.argmax(logits))
        continuation.append(token)
        next_input = [token]
    return continuation


def _teacher_forced_logits(
    adapter: object,
    prompt_ids: list[int],
    continuation: list[int],
    controller: InterventionController | None = None,
) -> list[np.ndarray]:
    """Return distributions under exactly the same prompt+continuation prefixes."""

    cache = adapter.make_cache()
    adapter.forward(prompt_ids[:-1], cache=cache)
    if controller is not None:
        adapter.wrap_layers(controller)
    try:
        next_inputs = [prompt_ids[-1], *continuation[:-1]]
        return [_last_logits(adapter.forward([token], cache=cache).logits) for token in next_inputs]
    finally:
        if controller is not None:
            adapter.unwrap_layers()


def _entropy_calibration_rows(
    adapter: object,
    config: LspeConfig,
    layers: list[int],
    raw_dose: float,
    continuation_tokens: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    baseline_rows: list[np.ndarray] = []
    coherent_rows: list[np.ndarray] = []
    for prompt in load_prompts(config.data.calibration, "calibration"):
        prompt_ids = adapter.format_prompt(
            [
                {"role": "system", "content": config.prompting.system},
                {"role": "user", "content": prompt.prompt},
            ]
        )
        continuation = _greedy_continuation(adapter, prompt_ids, continuation_tokens)
        baseline_rows.extend(_teacher_forced_logits(adapter, prompt_ids, continuation))
        controller = InterventionController(
            master_seed=config.experiment.master_seed,
            run_id=f"entropy-{_calibration_direction_namespace(config, layers)}",
            prompt_id=prompt.prompt_id,
            generation_index=0,
            condition_id="coherent",
            selected_layers=frozenset(layers),
            dose=raw_dose,
            mode="coherent_per_layer",
            kernel=config.intervention.kernel,
            decode_start_token=len(prompt_ids) - 1,
        )
        coherent_rows.extend(_teacher_forced_logits(adapter, prompt_ids, continuation, controller))
    return baseline_rows, coherent_rows


def _calibration_direction_namespace(config: LspeConfig, layers: list[int]) -> str:
    """Keep teacher-forced direction draws stable when refining a dose grid."""

    layer_key = ",".join(str(layer) for layer in layers)
    return f"calibration:{config.model.repo_id}:layers={layer_key}"
