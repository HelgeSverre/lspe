"""Command-line contract for LSPE."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from .analysis.runner import analyze_run
from .calibration.runner import calibrate as run_calibration
from .calibration.runner import derive_calibration_from_curve
from .config import LspeConfig, load_config
from .doctor import inspect_environment, report_dict
from .execution import ExperimentRunner
from .fetch import fetch_model
from .human_review import export_human_review
from .judge import judge_run, reparse_judge_run
from .locking import create_experiment_lock, load_experiment_lock, write_experiment_lock
from .models.factory import create_adapter
from .networks.mapping_runner import MappingProtocol, run_functional_mapping
from .networks.mapping_sensitivity import run_nested_mapping_sensitivity
from .pilot_selection import select_pilot_candidate, select_pilot_matrix
from .preflight import (
    baseline_generation_sanity,
    baseline_logit_sanity,
    cache_equivalence,
    intervention_liveness,
    write_architecture,
    zero_dose_identity,
    zero_dose_suite,
)
from .reporting import build_report
from .reporting.combined import build_combined_report
from .scoring import score_run
from .tasks.default_data import build_default_datasets
from .tasks.loader import load_prompts
from .verification import (
    verify_full,
    verify_replay,
    verify_run_projection,
    verify_scientific_artifacts,
    write_artifact_checksums,
)

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Auditable local LSPE harness")
ConfigOption = Annotated[Path, typer.Option("--config", exists=True, readable=True)]
RunOption = Annotated[Path, typer.Option("--run", exists=True, file_okay=False)]


def _load_config_or_exit(path: Path) -> LspeConfig:
    try:
        return load_config(path)
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error


def _event(**fields: Any) -> None:
    typer.echo(json.dumps(fields, sort_keys=True))


def _dry_run(config: LspeConfig) -> None:
    paths = config.data.model_dump(mode="python")
    counts = {split: len(load_prompts(Path(path), split)) for split, path in paths.items()}
    prompt_count = sum(counts.values())
    _event(
        event="dry_run",
        phase=config.experiment.phase,
        prompts_by_split=counts,
        conditions=config.conditions,
        generations_per_prompt=config.execution.generations_per_prompt,
        expected_generation_count=prompt_count
        * len(config.conditions)
        * config.execution.generations_per_prompt,
    )


def _unavailable(command: str, **extra: Any) -> None:
    _event(
        event="command_blocked", command=command, reason="MODEL_PIPELINE_NOT_YET_VALIDATED", **extra
    )
    raise typer.Exit(code=3)


@app.command()
def doctor(config: ConfigOption) -> None:
    """Inspect the local environment without loading any model."""

    report = inspect_environment(_load_config_or_exit(config), Path.cwd())
    typer.echo(json.dumps(report_dict(report), indent=2, sort_keys=True))
    if not report.ok:
        raise typer.Exit(code=1)


@app.command("build-data")
def build_data(
    output: Annotated[Path, typer.Option("--output")] = Path("data"),
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Build the versioned offline prompt corpus at its protocol split sizes."""

    try:
        counts = build_default_datasets(output, force=force)
    except OSError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    _event(event="datasets_built", output=str(output), counts=counts)


@app.command("map-networks")
def map_networks(
    config: ConfigOption,
    data: Annotated[Path, typer.Option("--data", exists=True, readable=True)] = Path(
        "data/phase2/network_map.jsonl"
    ),
    output: Annotated[Path, typer.Option("--output")] = Path("runs/fnde-network-map-qwen3"),
    offline: Annotated[bool, typer.Option("--offline")] = True,
) -> None:
    """Run the frozen observation-only Qwen functional mapping stage."""

    result = run_functional_mapping(
        model_config=config,
        data_path=data,
        run_dir=output,
        protocol=MappingProtocol(),
        offline=offline,
    )
    _event(event="functional_mapping_complete", run=str(output), result=result["result"])


@app.command("map-sensitivity")
def map_sensitivity(run: RunOption) -> None:
    """Run nested mapping-only sensitivity without behavioral outcomes."""

    result = run_nested_mapping_sensitivity(run)
    _event(event="mapping_sensitivity_complete", run=str(run), result=result)


@app.command()
def fetch(
    config: ConfigOption,
    offline: Annotated[bool, typer.Option("--offline")] = False,
) -> None:
    """Fetch and hash only the configured model revision."""

    loaded = _load_config_or_exit(config)
    try:
        result = fetch_model(loaded.model, offline=offline)
    except RuntimeError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    _event(
        event="model_fetched",
        repo_id=result.repo_id,
        revision=result.revision,
        local_path=str(result.local_path),
        weight_files=result.weight_files,
    )


@app.command()
def preflight(
    config: ConfigOption,
    offline: Annotated[bool, typer.Option("--offline")] = False,
    fail_fast: Annotated[bool, typer.Option("--fail-fast")] = False,
) -> None:
    """Load the configured subject and execute initial integrity checks."""

    loaded = _load_config_or_exit(config)
    if offline:
        _event(
            event="offline_preflight",
            note="Runtime must already have a cached model and dependencies.",
        )
    fetched = fetch_model(loaded.model, offline=offline)
    runtime_config = loaded.model_copy(
        update={
            "model": loaded.model.model_copy(
                update={"revision": fetched.revision, "local_path": fetched.local_path}
            )
        }
    )
    adapter = create_adapter(runtime_config.model)
    try:
        adapter.load(runtime_config.model)
        architecture = adapter.architecture()
        output = (
            loaded.experiment.output_root
            / f"preflight-{loaded.experiment.name}"
            / "architecture.json"
        )
        write_architecture(architecture, output)
        questions = [
            "Return only the integer result of 2 + 2.",
            "Return valid JSON with one key named answer and value 7.",
            "Write exactly one sentence containing the word fjord.",
            "Name the capital city of Norway in one word.",
            "State whether water freezes at zero Celsius. Answer yes or no.",
        ]
        prompt_ids = [
            adapter.format_prompt(
                [
                    {"role": "system", "content": loaded.prompting.system},
                    {"role": "user", "content": question},
                ]
            )
            for question in questions
        ]
        token_ids = prompt_ids[0]
        sanity = baseline_logit_sanity(adapter, token_ids)
        generation_sanity = baseline_generation_sanity(
            adapter,
            prompt_ids,
            [("4",), ("answer", "7"), ("fjord",), ("oslo",), ("yes",)],
        )
        cache = _aggregate_cache_preflight(
            adapter, prompt_ids, loaded.integrity.cache_logit_tolerance
        )
        selected_layers = [len(architecture.layers) // 2]
        sham = zero_dose_identity(
            adapter,
            token_ids,
            master_seed=loaded.experiment.master_seed,
            run_id=loaded.experiment.name,
            selected_layers=selected_layers,
            tolerance=loaded.integrity.zero_dose_logit_tolerance,
        )
        zero_suite = zero_dose_suite(
            adapter,
            prompt_ids,
            master_seed=loaded.experiment.master_seed,
            run_id=loaded.experiment.name,
            selected_layers=selected_layers,
            tolerance=loaded.integrity.zero_dose_logit_tolerance,
        )
        liveness = intervention_liveness(
            adapter,
            token_ids,
            master_seed=loaded.experiment.master_seed,
            run_id=loaded.experiment.name,
            selected_layers=selected_layers,
            dose=min(dose for dose in loaded.intervention.raw_dose_grid if dose > 0),
        )
        if (
            not generation_sanity["passed"]
            or not cache.passed
            or not sham.passed
            or not zero_suite.passed
            or not liveness.passed
        ):
            raise RuntimeError(
                "Baseline generation, cache, zero-dose identity, or intervention liveness "
                "preflight failed"
            )
        result = {
            "schema_version": 1,
            "event": "preflight_passed",
            "model_repo": runtime_config.model.repo_id,
            "model_revision": fetched.revision,
            "architecture_path": str(output),
            "baseline": sanity,
            "baseline_generation": generation_sanity,
            "cache_equivalence": cache.__dict__,
            "zero_dose_identity": sham.__dict__,
            "zero_dose_suite": {
                "prompt_count": zero_suite.prompt_count,
                "cached_decode_steps": zero_suite.cached_decode_steps,
                "no_cache": zero_suite.no_cache.__dict__,
                "cached_decode": zero_suite.cached_decode.__dict__,
                "passed": zero_suite.passed,
            },
            "intervention_liveness": liveness.__dict__,
            "cache_logit_tolerance": loaded.integrity.cache_logit_tolerance,
            "cache_tolerance_reason": loaded.integrity.cache_tolerance_reason,
        }
        (output.parent / "preflight.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _event(
            **result,
        )
    except Exception as error:
        _event(
            event="preflight_failed",
            error_type=type(error).__name__,
            error=str(error),
            fail_fast=fail_fast,
        )
        raise typer.Exit(code=1) from error
    finally:
        adapter.unload()


def _aggregate_cache_preflight(adapter: Any, prompts: list[list[int]], tolerance: float) -> Any:
    comparisons = [cache_equivalence(adapter, tokens, tolerance) for tokens in prompts]
    maximum = max(item.maximum_absolute_error for item in comparisons)
    mean = sum(item.mean_absolute_error for item in comparisons) / len(comparisons)
    greedy_equal = all(item.greedy_equal for item in comparisons)
    passed = all(item.passed for item in comparisons)
    return type(comparisons[0])(maximum, mean, greedy_equal, passed)


def _require_preflight(config: LspeConfig, model_revision: str) -> None:
    """Refuse model-backed experimental phases without persisted integrity evidence."""

    path = config.experiment.output_root / f"preflight-{config.experiment.name}" / "preflight.json"
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RuntimeError(
            f"Missing mandatory preflight evidence at {path}; "
            "run `lspe preflight --config ...` first"
        ) from error
    if evidence.get("event") != "preflight_passed":
        raise RuntimeError("Persisted preflight evidence does not record a pass")
    if evidence.get("model_repo") != config.model.repo_id:
        raise RuntimeError("Persisted preflight evidence is for a different model repository")
    if evidence.get("model_revision") != model_revision:
        raise RuntimeError("Persisted preflight evidence is for a different model revision")
    if not evidence.get("cache_equivalence", {}).get("passed"):
        raise RuntimeError("Persisted preflight cache-equivalence check did not pass")
    if not evidence.get("baseline_generation", {}).get("passed"):
        raise RuntimeError("Persisted preflight baseline generation check did not pass")
    if not evidence.get("zero_dose_suite", {}).get("passed"):
        raise RuntimeError("Persisted preflight zero-dose suite did not pass")


@app.command()
def calibrate(
    config: ConfigOption,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    offline: Annotated[bool, typer.Option("--offline")] = False,
    fail_fast: Annotated[bool, typer.Option("--fail-fast")] = False,
) -> None:
    """Calibrate raw intervention doses against teacher-forced KL."""

    loaded = _load_config_or_exit(config)
    if dry_run:
        _dry_run(loaded)
        return
    try:
        fetched = fetch_model(loaded.model, offline=offline)
        runtime_config = loaded.model_copy(
            update={
                "model": loaded.model.model_copy(
                    update={"revision": fetched.revision, "local_path": fetched.local_path}
                )
            }
        )
        _require_preflight(runtime_config, fetched.revision)
        summary = run_calibration(runtime_config, fetched.revision)
    except Exception as error:
        _event(event="calibration_failed", error_type=type(error).__name__, error=str(error))
        raise typer.Exit(code=1) from error
    _event(
        event="calibration_complete",
        run_dir=str(summary.run_dir),
        selected_layers=summary.selected_layers,
        raw_dose=summary.raw_dose,
        achieved_median_kl=summary.achieved_median_kl,
        white_raw_dose=summary.white_raw_dose,
        white_achieved_median_kl=summary.white_achieved_median_kl,
        matched_temperature=summary.matched_temperature,
        points=summary.points,
        resume=resume,
        fail_fast=fail_fast,
    )


@app.command()
def pilot(
    config: ConfigOption,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    offline: Annotated[bool, typer.Option("--offline")] = False,
    fail_fast: Annotated[bool, typer.Option("--fail-fast")] = False,
) -> None:
    """Run the preregistered pilot matrix."""

    loaded = _load_config_or_exit(config)
    if dry_run:
        _dry_run(loaded)
        return
    try:
        fetched = fetch_model(loaded.model, offline=offline)
        runtime_config = loaded.model_copy(
            update={
                "model": loaded.model.model_copy(
                    update={"revision": fetched.revision, "local_path": fetched.local_path}
                )
            }
        )
        _require_preflight(runtime_config, fetched.revision)
        groups = runtime_config.intervention.pilot_candidate_groups
        targets = runtime_config.intervention.pilot_target_kl_bands
        if runtime_config.experiment.phase == "pilot" and (not groups or not targets):
            raise RuntimeError("Pilot requires preregistered candidate groups and dose bands")
        if not groups:
            groups = [
                type(
                    "ImplicitGroup",
                    (),
                    {
                        "candidate_id": "configured",
                        "layers": runtime_config.intervention.selected_layers,
                    },
                )()
            ]
        if not targets:
            targets = [runtime_config.intervention.target_kl_nats]
        summaries = []
        for group in groups:
            if not isinstance(group.layers, list):
                raise RuntimeError("Pilot candidate groups must resolve to explicit layer indices")
            group_config = runtime_config.model_copy(
                update={
                    "experiment": runtime_config.experiment.model_copy(
                        update={"name": f"{runtime_config.experiment.name}-{group.candidate_id}"}
                    ),
                    "intervention": runtime_config.intervention.model_copy(
                        update={"selected_layers": group.layers}
                    ),
                }
            )
            curve_path = group_config.experiment.output_root / (
                f"calibration-{group_config.experiment.name}-{fetched.revision[:12]}"
            ) / "calibration.json"
            if resume and _calibration_curve_matches(group_config, fetched.revision, curve_path):
                _event(event="pilot_calibration_reused", calibration_path=str(curve_path))
            else:
                group_curve = run_calibration(group_config, fetched.revision)
                curve_path = group_curve.run_dir / "calibration.json"
            for target in targets:
                candidate_config = group_config.model_copy(
                    update={
                        "experiment": group_config.experiment.model_copy(
                            update={"name": f"{group_config.experiment.name}-t{target}"}
                        )
                    }
                )
                calibration = derive_calibration_from_curve(
                    candidate_config, fetched.revision, curve_path, target
                )
                runner = ExperimentRunner(
                    candidate_config,
                    config,
                    fetched.revision,
                    list(calibration.selected_layers),
                    calibration.raw_dose,
                    calibration.matched_temperature,
                    calibration.white_raw_dose,
                )
                summaries.append(runner.run(resume=resume, fail_fast=fail_fast))
    except Exception as error:
        _event(event="pilot_failed", error_type=type(error).__name__, error=str(error))
        raise typer.Exit(code=1) from error
    _event(
        event="pilot_complete",
        candidates=[
            {
                "run_id": summary.run_id,
                "run_dir": str(summary.run_dir),
                "expected_generations": summary.expected_generations,
                "committed_generations": summary.committed_generations,
                "failures": summary.failures,
            }
            for summary in summaries
        ],
    )


@app.command()
def freeze(
    config: ConfigOption,
    pilot_run: Annotated[Path, typer.Option("--pilot-run", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("experiment.lock.yaml"),
) -> None:
    """Freeze a selected pilot candidate into an immutable experiment lock."""

    loaded = _load_config_or_exit(config)
    try:
        manifest = json.loads((pilot_run / "manifest.json").read_text(encoding="utf-8"))
        selection = json.loads((pilot_run / "pilot-selection.json").read_text(encoding="utf-8"))
        if selection["status"] != "ELIGIBLE":
            raise RuntimeError(
                "Pilot selection found no eligible intervention; refusing a confirmatory lock"
            )
        architecture = json.loads((pilot_run / "architecture.json").read_text(encoding="utf-8"))
        candidate_config = load_config(pilot_run / "resolved-config.yaml")
        calibration_path = (
            candidate_config.experiment.output_root
            / f"calibration-{candidate_config.experiment.name}-{manifest['model_revision'][:12]}"
            / "calibration.json"
        )
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        selected_layers = list(selection["selected"]["selected_layers"])
        if selected_layers != list(manifest["selected_layers"]):
            raise RuntimeError("Pilot selection layers do not match the selected pilot manifest")
        if float(selection["selected"]["raw_dose"]) != float(calibration["summary"]["raw_dose"]):
            raise RuntimeError("Pilot selection dose does not match the selected calibration")
        candidate_runs = selection.get("candidate_runs")
        if candidate_runs is not None:
            selected_source = candidate_runs.get(selection["selected"]["candidate_id"])
            if not isinstance(selected_source, dict):
                raise RuntimeError("Pilot selection is missing the selected candidate provenance")
            if selected_source.get("run_id") != manifest["run_id"]:
                raise RuntimeError("Pilot selection points to a different pilot run")
            if Path(str(selected_source.get("run_dir"))).resolve() != pilot_run.resolve():
                raise RuntimeError("Pilot selection path does not match --pilot-run")
        selected_types = [architecture["layers"][index]["layer_type"] for index in selected_layers]
        frozen_config = loaded.model_copy(
            update={
                "intervention": loaded.intervention.model_copy(
                    update={
                        "selected_layers": selected_layers,
                        "target_kl_nats": float(selection["selected"]["target_kl"]),
                        "pilot_candidate_groups": [],
                        "pilot_target_kl_bands": [],
                    }
                )
            }
        )
        lock = create_experiment_lock(
            frozen_config,
            config,
            manifest["run_id"],
            manifest["model_revision"],
            selected_layers,
            selected_types,
            float(calibration["summary"]["raw_dose"]),
            float(calibration["summary"]["achieved_median_kl"]),
            float(calibration["summary"]["matched_temperature"]),
            float(calibration["summary"].get("white_raw_dose", calibration["summary"]["raw_dose"])),
            float(
                calibration["summary"].get(
                    "white_achieved_median_kl", calibration["summary"]["achieved_median_kl"]
                )
            ),
        )
        write_experiment_lock(lock, output)
    except Exception as error:
        _event(event="freeze_failed", error_type=type(error).__name__, error=str(error))
        raise typer.Exit(code=1) from error
    _event(event="freeze_complete", lock=str(output), experiment_id=lock.experiment_id)


@app.command("select-pilot")
def select_pilot(
    config: ConfigOption,
    pilot_runs: Annotated[list[Path], typer.Option("--pilot-run", exists=True, file_okay=False)],
) -> None:
    """Select from the complete preregistered pilot candidate matrix."""

    loaded = _load_config_or_exit(config)
    try:
        expected = _expected_pilot_candidates(loaded)
        if len(pilot_runs) != len(expected):
            raise RuntimeError(
                f"Pilot selection requires all {len(expected)} preregistered candidates; "
                f"received {len(pilot_runs)}"
            )
        candidates = []
        candidate_runs: dict[str, dict[str, str]] = {}
        seen: set[str] = set()
        model_revision: str | None = None
        for pilot_run in pilot_runs:
            projection = verify_run_projection(pilot_run)
            if not projection.passed:
                raise RuntimeError(
                    f"Pilot run is incomplete or inconsistent: {pilot_run}: "
                    + "; ".join(projection.reasons)
                )
            manifest = json.loads((pilot_run / "manifest.json").read_text(encoding="utf-8"))
            candidate_config = load_config(pilot_run / "resolved-config.yaml")
            candidate_id = candidate_config.experiment.name
            specification = expected.get(candidate_id)
            if specification is None:
                raise RuntimeError(f"Pilot run is not a preregistered candidate: {candidate_id}")
            if candidate_id in seen:
                raise RuntimeError(f"Duplicate pilot candidate provided: {candidate_id}")
            if candidate_config.intervention.selected_layers != specification["layers"]:
                raise RuntimeError(
                    f"Pilot candidate layers differ from preregistration: {candidate_id}"
                )
            _require_complete_pilot_scores(
                pilot_run, candidate_config.execution.generations_per_prompt
            )
            revision = str(manifest["model_revision"])
            if model_revision is not None and revision != model_revision:
                raise RuntimeError("All pilot candidates must use the identical model revision")
            model_revision = revision
            calibration_path = (
                candidate_config.experiment.output_root
                / f"calibration-{candidate_config.experiment.name}-{revision[:12]}"
                / "calibration.json"
            )
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            candidate_selection = select_pilot_candidate(
                pilot_run,
                loaded,
                calibration,
                candidate_id=candidate_id,
                target_kl=float(specification["target_kl"]),
                persist=False,
            )
            candidates.append(candidate_selection.selected)
            candidate_runs[candidate_id] = {
                "run_id": str(manifest["run_id"]),
                "run_dir": str(pilot_run.resolve()),
                "calibration_path": str(calibration_path.resolve()),
            }
            seen.add(candidate_id)
        missing = sorted(set(expected) - seen)
        if missing:
            raise RuntimeError("Missing preregistered pilot candidates: " + ", ".join(missing))
        selection = select_pilot_matrix(candidates)
        payload = {
            "schema_version": 1,
            **asdict(selection),
            "candidate_runs": candidate_runs,
        }
        if model_revision is None:
            raise RuntimeError("Pilot selection received no model revision")
        matrix_path = loaded.experiment.output_root / (
            f"pilot-selection-{loaded.experiment.name}-{model_revision[:12]}.json"
        )
        _write_immutable_json(matrix_path, payload)
        selected_run = Path(candidate_runs[selection.selected.candidate_id]["run_dir"])
        _write_immutable_json(selected_run / "pilot-selection.json", payload)
    except Exception as error:
        _event(event="pilot_selection_failed", error_type=type(error).__name__, error=str(error))
        raise typer.Exit(code=1) from error
    _event(
        event="pilot_selection_complete",
        status=selection.status,
        selected=asdict(selection.selected),
        selection_path=str(matrix_path),
        selected_run=str(selected_run),
    )


def _calibration_curve_matches(
    config: LspeConfig, model_revision: str, curve_path: Path
) -> bool:
    """Return whether a persisted curve is safe to reuse for ``pilot --resume``."""

    try:
        curve = json.loads(curve_path.read_text(encoding="utf-8"))
        medians = curve["median_kl_by_raw_dose"]
        configured_doses = {float(value) for value in config.intervention.raw_dose_grid}
        coherent_doses = {float(value) for value in medians["coherent_per_layer"]}
        white_doses = {float(value) for value in medians["white_per_token"]}
        selected_layers = [int(value) for value in curve["summary"]["selected_layers"]]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        curve.get("model_revision") == model_revision
        and selected_layers == config.intervention.selected_layers
        and coherent_doses == configured_doses
        and white_doses == configured_doses
    )


def _expected_pilot_candidates(config: LspeConfig) -> dict[str, dict[str, Any]]:
    if config.experiment.phase != "pilot":
        raise RuntimeError("Pilot selection requires a pilot configuration")
    groups = config.intervention.pilot_candidate_groups
    targets = config.intervention.pilot_target_kl_bands
    if not groups or not targets:
        raise RuntimeError("Pilot configuration has no preregistered candidate matrix")
    result: dict[str, dict[str, Any]] = {}
    for group in groups:
        for target in targets:
            candidate_id = f"{config.experiment.name}-{group.candidate_id}-t{target}"
            if candidate_id in result:
                raise RuntimeError(f"Duplicate preregistered pilot candidate: {candidate_id}")
            result[candidate_id] = {"layers": group.layers, "target_kl": target}
    return result


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"Refusing to overwrite existing selection artifact: {path}")
        return
    path.write_text(encoded, encoding="utf-8")


def _require_complete_pilot_scores(run_dir: Path, generations_per_prompt: int) -> None:
    """Reject stale or partial scoring projections before scientific selection."""

    try:
        plan = [
            json.loads(line)
            for line in (run_dir / "generation-plan.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        effects = [
            json.loads(line)
            for line in (run_dir / "prompt-effects.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
    except OSError as error:
        raise RuntimeError(f"Missing required pilot scoring artifact: {error.filename}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Malformed pilot scoring artifact: {error}") from error
    expected = {(str(row["prompt_id"]), str(row["condition"])) for row in plan}
    actual = {(str(row["prompt_id"]), str(row["condition"])) for row in effects}
    if len(actual) != len(effects):
        raise RuntimeError(
            "Pilot prompt-effects projection contains duplicate prompt-condition rows"
        )
    if actual != expected:
        raise RuntimeError(
            "Pilot prompt-effects projection does not cover exactly the completed generation plan"
        )
    incorrect_counts = [
        f"{row.get('prompt_id')}:{row.get('condition')}"
        for row in effects
        if int(row.get("n_generations", -1)) != generations_per_prompt
    ]
    if incorrect_counts:
        raise RuntimeError(
            "Pilot prompt-effects generation counts are incomplete: " + ", ".join(incorrect_counts)
        )


@app.command(name="run")
def run_experiment(
    lock: Annotated[Path, typer.Option("--lock", exists=True, readable=True)],
    resume: Annotated[bool, typer.Option("--resume")] = False,
    offline: Annotated[bool, typer.Option("--offline")] = False,
    fail_fast: Annotated[bool, typer.Option("--fail-fast")] = False,
) -> None:
    """Run the immutable confirmatory generation matrix."""

    try:
        experiment_lock = load_experiment_lock(lock)
        loaded = LspeConfig.model_validate(experiment_lock.resolved_config)
        loaded = loaded.model_copy(
            update={
                "model": loaded.model.model_copy(
                    update={"revision": experiment_lock.model["revision"]}
                )
            }
        )
        _require_preflight(loaded, str(experiment_lock.model["revision"]))
        runner = ExperimentRunner(
            loaded,
            lock,
            str(experiment_lock.model["revision"]),
            list(experiment_lock.intervention["selected_layers"]),
            float(experiment_lock.intervention["raw_dose"]),
            float(experiment_lock.sampling["matched_temperature"]),
            float(
                experiment_lock.intervention.get(
                    "white_raw_dose", experiment_lock.intervention["raw_dose"]
                )
            ),
        )
        summary = runner.run(resume=resume, fail_fast=fail_fast)
    except Exception as error:
        _event(
            event="run_failed", error_type=type(error).__name__, error=str(error), offline=offline
        )
        raise typer.Exit(code=1) from error
    _event(
        event="run_complete",
        run_id=summary.run_id,
        run_dir=str(summary.run_dir),
        expected_generations=summary.expected_generations,
        committed_generations=summary.committed_generations,
        failures=summary.failures,
    )


@app.command()
def score(run: RunOption) -> None:
    """Rebuild deterministic and embedding scores from raw generations."""

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    config = load_config(run / "resolved-config.yaml")
    try:
        summary = score_run(run, config.scoring.embedding_model, config.scoring.embedding_revision)
        review = (
            export_human_review(run, config.experiment.master_seed)
            if config.scoring.human_review_export
            else None
        )
    except Exception as error:
        _event(event="score_failed", error_type=type(error).__name__, error=str(error))
        raise typer.Exit(code=1) from error
    _event(
        event="score_complete", run_id=manifest["run_id"], **summary.__dict__, human_review=review
    )


@app.command()
def analyze(run: RunOption) -> None:
    """Compute preregistered prompt-clustered statistics."""

    config = load_config(run / "resolved-config.yaml")
    try:
        analysis = analyze_run(
            run, config.experiment.master_seed, config.statistics.bootstrap_samples
        )
    except Exception as error:
        _event(event="analysis_failed", error_type=type(error).__name__, error=str(error))
        raise typer.Exit(code=1) from error
    _event(event="analysis_complete", status=analysis["status"], primary=analysis["primary"])


@app.command()
def judge(
    run: RunOption,
    offline: Annotated[bool, typer.Option("--offline")] = False,
    reparse: Annotated[
        bool,
        typer.Option(
            "--reparse",
            help="Reparse existing blinded judge responses without regeneration.",
        ),
    ] = False,
) -> None:
    """Run blinded Qwen pairwise secondary judging after subject-model execution."""

    config = load_config(run / "resolved-config.yaml")
    if not config.scoring.judge_enabled:
        _event(event="judge_skipped", reason="JUDGE_DISABLED")
        return
    try:
        summary = (
            reparse_judge_run(run, config.scoring.local_judge_model, offline)
            if reparse
            else judge_run(
                run,
                config.scoring.local_judge_model,
                config.experiment.master_seed,
                offline,
            )
        )
    except Exception as error:
        _event(event="judge_failed", error_type=type(error).__name__, error=str(error))
        raise typer.Exit(code=1) from error
    _event(event="judge_complete", reparse=reparse, **summary.__dict__)


@app.command()
def report(run: RunOption) -> None:
    """Build Markdown, HTML, and machine-readable reports."""

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    analysis = json.loads((run / "analysis.json").read_text(encoding="utf-8"))
    config = load_config(run / "resolved-config.yaml")
    report_data = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "status": analysis["status"],
        "primary": analysis["primary"],
        "validity": analysis.get("validity", {}),
        "competence": analysis.get("competence", {}),
        "degeneration": analysis.get("degeneration", {}),
        "replication": analysis.get("replication", {}),
        "integrity": analysis.get("integrity", {}),
        "secondary": analysis.get("secondary", {}),
        "execution": {
            "expected_generations": manifest.get("prompt_count", 0)
            * len(config.conditions)
            * config.execution.generations_per_prompt,
            "observed_generations": len(
                [
                    line
                    for line in (run / "generations.jsonl").read_text(encoding="utf-8").splitlines()
                    if line
                ]
            ),
        },
        "limitations": [
            (
                "This report concerns a local model intervention and makes no claim "
                "about consciousness, "
                "intoxication, psychedelics, or biological equivalence."
            ),
            (
                "Replication evidence is required before treating a positive primary estimate as "
                "supported."
            ),
        ],
        "artifact_root_hash": None,
    }
    try:
        build_report(run, report_data)
        root = write_artifact_checksums(run)
    except Exception as error:
        _event(event="report_failed", error_type=type(error).__name__, error=str(error))
        raise typer.Exit(code=1) from error
    _event(
        event="report_complete",
        run_id=manifest["run_id"],
        status=analysis["status"],
        root_digest=root,
    )


@app.command("combine")
def combine_reports(
    primary_run: Annotated[Path, typer.Option("--primary-run", exists=True, file_okay=False)],
    replication_run: Annotated[
        Path, typer.Option("--replication-run", exists=True, file_okay=False)
    ],
    output: Annotated[Path, typer.Option("--output")] = Path("runs/combined-report"),
) -> None:
    """Build a non-pooled, cross-model final report from verified run reports."""

    for source in (primary_run, replication_run):
        verification = verify_scientific_artifacts(source)
        if not verification.passed:
            raise typer.BadParameter(
                "Source run is not scientifically verified: "
                f"{source}: {'; '.join(verification.reasons)}"
            )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise typer.BadParameter(
            f"Refusing to overwrite non-empty combined-report directory: {output}"
        )
    try:
        combined = build_combined_report(output, primary_run, replication_run)
    except Exception as error:
        _event(event="combine_failed", error_type=type(error).__name__, error=str(error))
        raise typer.Exit(code=1) from error
    _event(
        event="combine_complete",
        output=str(output),
        status=combined["conclusion"]["status"],
        primary_run=combined["primary"]["run_id"],
        replication_run=combined["architecture_replication"]["run_id"],
    )


@app.command()
def verify(
    run: RunOption,
    level: Annotated[str, typer.Option("--level")] = "artifact",
    sample: Annotated[int, typer.Option("--sample")] = 20,
) -> None:
    """Verify artifacts or replay a deterministic stratified sample."""

    if level not in {"artifact", "replay", "full"}:
        typer.echo("--level must be artifact, replay, or full", err=True)
        raise typer.Exit(code=2)
    if level == "artifact":
        result = verify_scientific_artifacts(run)
        _event(
            event="artifact_verification",
            passed=result.passed,
            root_digest=result.root_digest,
            reasons=result.reasons,
        )
        if not result.passed:
            raise typer.Exit(code=1)
        return
    result = verify_replay(run, sample) if level == "replay" else verify_full(run)
    _event(
        event=f"{level}_verification",
        passed=result.passed,
        root_digest=result.root_digest,
        reasons=result.reasons,
        sample=sample if level == "replay" else None,
    )
    if not result.passed:
        raise typer.Exit(code=1)
