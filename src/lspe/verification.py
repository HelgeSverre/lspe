"""Artifact checksum verification that never loads the subject model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analysis.runner import calculate_analysis
from .artifact_schema import validate_report
from .config import load_config
from .execution import ExperimentRunner
from .generation.plan import GenerationPlanItem
from .hashing import root_digest, sha256_file
from .metrics.degeneration import degeneration_metrics
from .models.factory import create_adapter
from .rng import derive_seed
from .tasks.loader import PromptRecord
from .tasks.validators import validate_response

CHECKSUM_FILE = "checksums.sha256"
EPHEMERAL_TOP_LEVEL = frozenset({"logs"})


@dataclass(frozen=True)
class ArtifactVerification:
    passed: bool
    root_digest: str | None
    reasons: tuple[str, ...]


def artifact_checksums(run_dir: Path) -> dict[str, str]:
    """Hash every immutable file, excluding only logs and the checksum inventory itself."""

    records: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if relative.name == CHECKSUM_FILE or relative.parts[0] in EPHEMERAL_TOP_LEVEL:
            continue
        records[relative.as_posix()] = sha256_file(path)
    return records


def write_artifact_checksums(run_dir: Path) -> str:
    records = artifact_checksums(run_dir)
    digest = root_digest(records)
    lines = [f"{checksum}  {relative}" for relative, checksum in sorted(records.items())]
    lines.append(f"ROOT  {digest}")
    (run_dir / CHECKSUM_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return digest


def verify_artifact_checksums(run_dir: Path) -> ArtifactVerification:
    inventory = run_dir / CHECKSUM_FILE
    if not inventory.is_file():
        return ArtifactVerification(False, None, (f"Missing {CHECKSUM_FILE}",))
    expected: dict[str, str] = {}
    expected_root: str | None = None
    reasons: list[str] = []
    for line in inventory.read_text(encoding="utf-8").splitlines():
        try:
            checksum, relative = line.split("  ", 1)
        except ValueError:
            reasons.append(f"Malformed checksum line: {line!r}")
            continue
        if checksum == "ROOT":
            expected_root = relative
        else:
            expected[relative] = checksum
    actual = artifact_checksums(run_dir)
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        changed = sorted(
            path for path in set(actual) & set(expected) if actual[path] != expected[path]
        )
        reasons.extend(f"Missing immutable artifact: {path}" for path in missing)
        reasons.extend(f"Unexpected immutable artifact: {path}" for path in unexpected)
        reasons.extend(f"Checksum mismatch: {path}" for path in changed)
    computed_root = root_digest(actual)
    if expected_root != computed_root:
        reasons.append("Root artifact digest mismatch")
    reasons.extend(_verify_run_projection(run_dir))
    return ArtifactVerification(not reasons, computed_root, tuple(reasons))


def verify_scientific_artifacts(run_dir: Path) -> ArtifactVerification:
    """Verify checksums plus independent deterministic score/analysis rebuilds."""

    checksum = verify_artifact_checksums(run_dir)
    reasons = list(checksum.reasons)
    if (run_dir / "scores.jsonl").is_file():
        reasons.extend(_verify_deterministic_scores(run_dir))
    if (run_dir / "analysis.json").is_file():
        reasons.extend(_verify_analysis_and_report(run_dir))
    return ArtifactVerification(not reasons, checksum.root_digest, tuple(reasons))


def verify_run_projection(run_dir: Path) -> ArtifactVerification:
    """Verify that a run has one durable, planned generation for every cell.

    Unlike checksum verification this is usable before reporting has created a
    checksum inventory.  Selection uses it to reject incomplete pilot runs.
    """

    required = ("generation-plan.jsonl", "generations.jsonl", "journal")
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        return ArtifactVerification(
            False,
            None,
            tuple(f"Missing required run artifact: {name}" for name in missing),
        )
    reasons = _verify_run_projection(run_dir)
    return ArtifactVerification(not reasons, None, tuple(reasons))


def verify_replay(run_dir: Path, sample: int) -> ArtifactVerification:
    """Replay a deterministic, condition-stratified subset against raw records."""

    if sample < 1:
        return ArtifactVerification(False, None, ("Replay sample must be positive",))
    try:
        config = load_config(run_dir / "resolved-config.yaml")
        manifest = _read_json(run_dir / "manifest.json")
        plan = _read_jsonl(run_dir / "generation-plan.jsonl")
        raw_rows = _read_jsonl(run_dir / "generations.jsonl")
        prompts = [
            PromptRecord.model_validate(row)
            for row in _read_jsonl(run_dir / "prompts.snapshot.jsonl")
        ]
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return ArtifactVerification(False, None, (f"Cannot prepare replay: {error}",))
    projection = verify_run_projection(run_dir)
    if not projection.passed:
        return projection
    raw_by_id = {str(row["generation_id"]): row for row in raw_rows}
    selected = _stratified_replay_sample(plan, sample, config.experiment.master_seed)
    reasons = _verify_stored_sham_pairs(
        raw_by_id, selected, require_sham="sham" in config.conditions
    )
    adapter = create_adapter(config.model)
    runner = ExperimentRunner(
        config,
        run_dir / "resolved-config.yaml",
        str(manifest["model_revision"]),
        [int(value) for value in manifest["selected_layers"]],
        float(manifest["raw_dose"]),
        float(manifest["matched_temperature"])
        if manifest.get("matched_temperature") is not None
        else None,
        float(manifest.get("white_raw_dose", manifest["raw_dose"])),
    )
    try:
        adapter.load(config.model)
        for plan_row in selected:
            item = GenerationPlanItem(**plan_row)
            expected = raw_by_id.get(item.generation_id)
            if expected is None:
                reasons.append(f"Missing raw generation for replay cell: {item.generation_id}")
                continue
            actual, _ = runner._execute_item(adapter, item, prompts)
            if actual["rendered_token_ids"] != expected.get("rendered_token_ids"):
                reasons.append(f"Rendered prompt mismatch: {item.generation_id}")
            if actual["output_token_ids"] != expected.get("output_token_ids"):
                reasons.append(f"Sampled output mismatch: {item.generation_id}")
            expected_fingerprints = expected.get("direction_fingerprints", [])
            if expected_fingerprints and actual["direction_fingerprints"] != expected_fingerprints:
                reasons.append(f"Intervention direction fingerprint mismatch: {item.generation_id}")
    except Exception as error:
        reasons.append(f"Replay execution failed: {type(error).__name__}: {error}")
    finally:
        adapter.unload()
    return ArtifactVerification(not reasons, None, tuple(reasons))


def verify_full(run_dir: Path) -> ArtifactVerification:
    """Rerun the complete locked matrix into an isolated verification root."""

    try:
        config = load_config(run_dir / "resolved-config.yaml")
        manifest = _read_json(run_dir / "manifest.json")
        original_rows = _read_jsonl(run_dir / "generations.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return ArtifactVerification(False, None, (f"Cannot prepare full replay: {error}",))
    projection = verify_run_projection(run_dir)
    if not projection.passed:
        return projection
    verification_root = run_dir.parent / "verification-full"
    replay_config = config.model_copy(
        update={
            "experiment": config.experiment.model_copy(update={"output_root": verification_root})
        }
    )
    runner = ExperimentRunner(
        replay_config,
        run_dir / "resolved-config.yaml",
        str(manifest["model_revision"]),
        [int(value) for value in manifest["selected_layers"]],
        float(manifest["raw_dose"]),
        float(manifest["matched_temperature"])
        if manifest.get("matched_temperature") is not None
        else None,
        float(manifest.get("white_raw_dose", manifest["raw_dose"])),
    )
    replay_dir = verification_root / str(manifest["run_id"])
    try:
        runner.run(resume=replay_dir.exists(), fail_fast=True)
        replay_projection = verify_run_projection(replay_dir)
        if not replay_projection.passed:
            return replay_projection
        replay_rows = _read_jsonl(replay_dir / "generations.jsonl")
    except Exception as error:
        return ArtifactVerification(False, None, (f"Full replay execution failed: {error}",))
    original_by_id = {str(row["generation_id"]): row for row in original_rows}
    reasons: list[str] = []
    for row in replay_rows:
        original = original_by_id.get(str(row["generation_id"]))
        if original is None:
            reasons.append(f"Full replay produced an unplanned generation: {row['generation_id']}")
            continue
        if row.get("output_token_ids") != original.get("output_token_ids"):
            reasons.append(f"Full replay output mismatch: {row['generation_id']}")
    if len(replay_rows) != len(original_rows):
        reasons.append("Full replay generation count differs from the original run")
    return ArtifactVerification(not reasons, None, tuple(reasons))


def _verify_run_projection(run_dir: Path) -> list[str]:
    """Verify the durable journal, JSONL projection, and complete plan agree."""

    plan_path = run_dir / "generation-plan.jsonl"
    rows_path = run_dir / "generations.jsonl"
    journal = run_dir / "journal"
    if not plan_path.exists() or not rows_path.exists() or not journal.exists():
        return []
    try:
        planned = [
            json.loads(line)["generation_id"] for line in plan_path.read_text().splitlines() if line
        ]
        rows = [
            json.loads(line)["generation_id"] for line in rows_path.read_text().splitlines() if line
        ]
        journal_ids = {f"sha256:{path.stem}" for path in journal.glob("*.json")}
    except (OSError, json.JSONDecodeError, KeyError) as error:
        return [f"Malformed run projection: {error}"]
    reasons: list[str] = []
    if len(rows) != len(set(rows)):
        reasons.append("Duplicate generation IDs in generations.jsonl")
    if set(rows) != journal_ids:
        reasons.append("Generation projection and journal IDs differ")
    if set(rows) - set(planned):
        reasons.append("Generation projection contains unplanned IDs")
    if len(rows) != len(planned):
        reasons.append(f"Expected {len(planned)} generation rows but observed {len(rows)}")
    return reasons


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _stratified_replay_sample(
    plan: list[dict[str, Any]], sample: int, master_seed: int
) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in plan:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for condition, rows in sorted(by_condition.items()):
        selected.append(
            min(
                rows,
                key=lambda row: derive_seed(
                    master_seed, "bootstrap", "replay", condition, row["generation_id"]
                ),
            )
        )
    remaining = [row for row in plan if row not in selected]
    remaining.sort(
        key=lambda row: derive_seed(
            master_seed, "bootstrap", "replay", "remaining", row["generation_id"]
        )
    )
    return selected + remaining[: max(0, sample - len(selected))]


def _verify_stored_sham_pairs(
    raw_by_id: dict[str, dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    require_sham: bool = True,
) -> list[str]:
    """Check stored sham identity only for protocols that actually include sham."""

    if not require_sham:
        return []
    by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in raw_by_id.values():
        key = (str(row["prompt_id"]), int(row["generation_index"]))
        by_pair.setdefault(key, {})[str(row["condition"])] = row
    selected_pairs = {(str(row["prompt_id"]), int(row["generation_index"])) for row in selected}
    reasons: list[str] = []
    for pair in selected_pairs:
        pair_rows = by_pair[pair]
        baseline = pair_rows.get("baseline")
        sham = pair_rows.get("sham")
        if baseline is None or sham is None:
            reasons.append(f"Replay sample lacks baseline/sham pair: {pair[0]}:{pair[1]}")
        elif baseline.get("output_token_ids") != sham.get("output_token_ids"):
            reasons.append(f"Stored baseline/sham outputs differ: {pair[0]}:{pair[1]}")
    return reasons


def _verify_deterministic_scores(run_dir: Path) -> list[str]:
    try:
        prompts = {
            record.prompt_id: record
            for record in (
                PromptRecord.model_validate(row)
                for row in _read_jsonl(run_dir / "prompts.snapshot.jsonl")
            )
        }
        generations = _read_jsonl(run_dir / "generations.jsonl")
        scores = _read_jsonl(run_dir / "scores.jsonl")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [f"Cannot rebuild deterministic scores: {error}"]
    score_by_id = {str(row.get("generation_id")): row for row in scores}
    generation_ids = {str(row.get("generation_id")) for row in generations}
    reasons: list[str] = []
    if len(score_by_id) != len(scores):
        reasons.append("Duplicate generation IDs in scores.jsonl")
    if set(score_by_id) != generation_ids:
        reasons.append("scores.jsonl does not cover exactly the raw generations")
    for generation in generations:
        generation_id = str(generation["generation_id"])
        score = score_by_id.get(generation_id)
        prompt = prompts.get(str(generation["prompt_id"]))
        if score is None or prompt is None:
            continue
        validation = validate_response(
            prompt.validator, str(generation.get("output_text", "")), prompt.expected
        )
        expected = {
            "valid": validation.valid,
            "failure_code": validation.failure_code,
            "response_length": len(str(generation.get("output_text", ""))),
            **degeneration_metrics(generation.get("output_token_ids", [])),
        }
        for field, value in expected.items():
            if score.get(field) != value:
                reasons.append(f"Deterministic score mismatch: {generation_id}:{field}")
    return reasons


def _verify_analysis_and_report(run_dir: Path) -> list[str]:
    try:
        config = load_config(run_dir / "resolved-config.yaml")
        expected_analysis = _read_json(run_dir / "analysis.json")
        rebuilt = calculate_analysis(
            run_dir, config.experiment.master_seed, config.statistics.bootstrap_samples
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [f"Cannot rebuild analysis: {error}"]
    reasons: list[str] = []
    if expected_analysis != rebuilt:
        reasons.append("analysis.json differs from the deterministic rebuild")
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        return reasons
    try:
        report = validate_report(_read_json(report_path))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [f"Report schema validation failed: {error}", *reasons]
    if report["status"] != rebuilt["status"]:
        reasons.append("report.json status differs from analysis.json")
    if report["primary"] != rebuilt["primary"]:
        reasons.append("report.json primary outcome differs from analysis.json")
    return reasons
