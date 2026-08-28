"""Model-backed paired execution with durable, content-addressed records."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from .columnar import write_jsonl_parquet
from .config import LspeConfig
from .generation.loop import GenerationLoop
from .generation.plan import GenerationPlanItem, build_generation_plan
from .hashing import canonical_json, sha256_bytes, sha256_file
from .interventions.controller import InterventionController
from .memory_guard import MemoryGuard
from .models.factory import create_adapter
from .preflight import write_architecture
from .run_store import RunStore
from .tasks.loader import PromptRecord, load_prompts
from .tasks.validators import validate_response


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    run_dir: Path
    expected_generations: int
    committed_generations: int
    failures: int


def load_phase_prompts(config: LspeConfig, phase: str) -> list[PromptRecord]:
    """Load the phase split and controls; each record retains its original split label."""

    source = {
        "smoke": config.data.pilot,
        "pilot": config.data.pilot,
        "confirm": config.data.confirm,
        "replicate": config.data.confirm,
    }[phase]
    expected = "pilot" if phase == "smoke" else "confirm" if phase == "replicate" else phase
    prompts = _tagged_for_phase(load_prompts(source, expected), phase) + _tagged_for_phase(
        load_prompts(config.data.controls, "controls"), phase
    )
    identifiers = [prompt.prompt_id for prompt in prompts]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Phase prompt set contains duplicate IDs")
    return prompts


def _tagged_for_phase(prompts: list[PromptRecord], phase: str) -> list[PromptRecord]:
    return [prompt for prompt in prompts if phase in prompt.tags]


def _load_prompt_snapshot(path: Path) -> list[PromptRecord]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        PromptRecord.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


class ExperimentRunner:
    """Executes all paired conditions without reseeding failures or mutating weights."""

    def __init__(
        self,
        config: LspeConfig,
        config_path: Path,
        model_revision: str,
        selected_layers: list[int],
        raw_dose: float,
        matched_temperature: float | None = None,
        white_raw_dose: float | None = None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.model_revision = model_revision
        self.selected_layers = selected_layers
        self.raw_dose = raw_dose
        self.matched_temperature = matched_temperature
        self.white_raw_dose = white_raw_dose if white_raw_dose is not None else raw_dose

    def run(self, resume: bool = False, fail_fast: bool = False) -> RunSummary:
        prompts = load_phase_prompts(self.config, self.config.experiment.phase)
        run_id = self._run_id()
        run_dir = self.config.experiment.output_root / run_id
        store = RunStore(run_dir)
        if not run_dir.exists():
            store.initialize(self._manifest(run_id, prompts))
            self._write_static_artifacts(run_dir, prompts)
        elif not resume:
            raise FileExistsError(
                f"Run already exists: {run_dir}; use --resume to reuse complete rows"
            )
        else:
            self._validate_resume_artifacts(run_dir, run_id, prompts)
        plan = build_generation_plan(
            self.config,
            prompts,
            self.model_revision,
            self.selected_layers,
            self.raw_dose,
            self.matched_temperature,
            self.white_raw_dose,
        )
        (run_dir / "generation-plan.jsonl").write_text(
            "".join(json.dumps(item.record(), sort_keys=True) + "\n" for item in plan),
            encoding="utf-8",
        )
        completed = store.completed_ids() if resume else set()
        adapter = create_adapter(self.config.model)
        committed = 0
        failures = 0
        token_lines: list[str] = []
        memory_guard = MemoryGuard(
            self.config.hardware.memory_soft_limit_fraction,
            self.config.hardware.memory_hard_limit_fraction,
        )
        try:
            adapter.load(self.config.model)
            write_architecture(adapter.architecture(), run_dir / "architecture.json")
            self._write_prompt_renders(adapter, prompts, run_dir)
            for item in plan:
                if item.generation_id in completed:
                    continue
                memory_guard.enforce()
                try:
                    record, token_rows = self._execute_item(adapter, item, prompts)
                    result = store.commit_generation(item.scientific_inputs, record)
                    if result.committed:
                        committed += 1
                        token_lines.extend(
                            json.dumps(row, sort_keys=True) + "\n" for row in token_rows
                        )
                except Exception as error:
                    failures += 1
                    failure_record = {
                        "prompt_id": item.prompt_id,
                        "condition": item.condition,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "failure_code": "RUNTIME_FAILURE",
                        "output_token_ids": [],
                        "output_text": "",
                        "stop_reason": "RUNTIME_FAILURE",
                    }
                    store.commit_generation(item.scientific_inputs, failure_record)
                    if fail_fast:
                        raise
        finally:
            adapter.unload()
        if token_lines:
            with (run_dir / "token-metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.writelines(token_lines)
        write_jsonl_parquet(run_dir / "token-metrics.jsonl", run_dir / "token-metrics.parquet")
        return RunSummary(run_id, run_dir, len(plan), committed, failures)

    def _execute_item(
        self, adapter: Any, item: GenerationPlanItem, prompts: list[PromptRecord]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        prompt = next(record for record in prompts if record.prompt_id == item.prompt_id)
        messages = [
            {"role": "system", "content": self.config.prompting.system},
            {"role": "user", "content": prompt.prompt},
        ]
        prompt_ids = adapter.format_prompt(messages)
        controller = self._controller_for(item, len(prompt_ids))
        if controller is not None:
            adapter.wrap_layers(controller)
        sampling = self.config.sampling
        if item.condition == "temp_match" and self.matched_temperature is not None:
            sampling = sampling.model_copy(update={"temperature": self.matched_temperature})
        started = perf_counter()
        try:
            generation = GenerationLoop(
                adapter, sampling, self.config.experiment.master_seed
            ).generate(
                prompt_ids,
                prompt.prompt_id,
                item.generation_index,
                item.condition,
                tuple(self.selected_layers),
                intervention_active=controller is not None and controller.dose != 0,
                intervention_dose=controller.dose if controller is not None else 0.0,
            )
        finally:
            adapter.unwrap_layers()
        validation = validate_response(prompt.validator, generation.text, prompt.expected)
        record = {
            "prompt_id": prompt.prompt_id,
            "prompt_hash": prompt.content_hash,
            "rendered_token_ids": prompt_ids,
            "rendered_token_hash": sha256_bytes(canonical_json(prompt_ids)),
            "model_repo": self.config.model.repo_id,
            "model_revision": self.model_revision,
            "condition": item.condition,
            "generation_index": item.generation_index,
            "layer_indices": self.selected_layers,
            "kernel": self.config.intervention.kernel,
            "dose": controller.dose if controller is not None else 0.0,
            "direction_mode": controller.mode if controller is not None else "none",
            "sampling_seed": item.sampling_seed,
            "sampling": sampling.model_dump(mode="json"),
            "output_token_ids": list(generation.output_token_ids),
            "output_text": generation.text,
            "stop_reason": generation.stop_reason,
            "validator": asdict(validation),
            "runtime_duration_seconds": perf_counter() - started,
            "failure_code": None,
            "intervention_telemetry": [
                asdict(row) for row in (controller.telemetry if controller else [])
            ],
            "direction_fingerprints": controller.direction_fingerprints() if controller else [],
        }
        token_rows = [
            {"generation_id": item.generation_id, **metric.record()}
            for metric in generation.token_metrics
        ]
        return record, token_rows

    def _controller_for(
        self, item: GenerationPlanItem, decode_start_token: int
    ) -> InterventionController | None:
        if item.condition == "baseline" or item.condition == "temp_match":
            return None
        mode = {
            "sham": "zero",
            "coherent": "coherent_per_layer",
            "white": "white_per_token",
        }[item.condition]
        dose = self.raw_dose if item.condition == "coherent" else self.white_raw_dose
        return InterventionController(
            master_seed=self.config.experiment.master_seed,
            run_id=self._run_id(),
            prompt_id=item.prompt_id,
            generation_index=item.generation_index,
            condition_id=item.condition,
            selected_layers=frozenset(self.selected_layers),
            dose=dose,
            mode=mode,
            kernel=self.config.intervention.kernel,
            decode_start_token=decode_start_token - 1,
        )

    def _run_id(self) -> str:
        return f"{self.config.experiment.name}-{self.model_revision[:12]}"

    def _manifest(self, run_id: str, prompts: list[PromptRecord]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "config_hash": sha256_file(self.config_path),
            "model_repo": self.config.model.repo_id,
            "model_revision": self.model_revision,
            "prompt_count": len(prompts),
            "selected_layers": self.selected_layers,
            "raw_dose": self.raw_dose,
            "white_raw_dose": self.white_raw_dose,
            "matched_temperature": self.matched_temperature,
        }

    def _write_static_artifacts(self, run_dir: Path, prompts: list[PromptRecord]) -> None:
        (run_dir / "resolved-config.yaml").write_text(
            yaml.safe_dump(self.config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
        )
        (run_dir / "prompts.snapshot.jsonl").write_text(
            "".join(prompt.model_dump_json() + "\n" for prompt in prompts), encoding="utf-8"
        )
        environment = {
            "platform": platform.platform(),
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        (run_dir / "environment.txt").write_text(
            "".join(f"{key}={value}\n" for key, value in sorted(environment.items())),
            encoding="utf-8",
        )
        packages = {
            distribution.metadata["Name"]: distribution.version
            for distribution in distributions()
            if distribution.metadata.get("Name")
        }
        (run_dir / "packages.lock.json").write_text(
            json.dumps(packages, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        model_files = self._model_files()
        (run_dir / "model-files.json").write_text(
            json.dumps(model_files, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _validate_resume_artifacts(
        self, run_dir: Path, run_id: str, prompts: list[PromptRecord]
    ) -> None:
        """Refuse resume unless the immutable scientific inputs are identical.

        Run IDs are intentionally human-readable and therefore can collide
        across an earlier attempt with changed calibration or a new lock.  A
        content-ID check alone is not sufficient: it would permit mixing two
        scientific matrices under one manifest and stale prompt snapshot.
        """

        path = run_dir / "manifest.json"
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Cannot validate existing run manifest: {path}") from error
        expected = self._manifest(run_id, prompts)
        fields = (
            "run_id",
            "config_hash",
            "model_repo",
            "model_revision",
            "prompt_count",
            "selected_layers",
            "raw_dose",
            "white_raw_dose",
            "matched_temperature",
        )
        mismatched = [field for field in fields if actual.get(field) != expected[field]]
        if mismatched:
            raise RuntimeError(
                "Existing run manifest is incompatible with --resume: "
                + ", ".join(mismatched)
            )
        try:
            snapshot = _load_prompt_snapshot(run_dir / "prompts.snapshot.jsonl")
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise RuntimeError("Cannot validate existing prompt snapshot") from error
        expected_by_id = {prompt.prompt_id: prompt for prompt in prompts}
        snapshot_by_id = {prompt.prompt_id: prompt for prompt in snapshot}
        if snapshot_by_id != expected_by_id:
            raise RuntimeError("Existing prompt snapshot is incompatible with --resume")

    def _write_prompt_renders(
        self, adapter: Any, prompts: list[PromptRecord], run_dir: Path
    ) -> None:
        path = run_dir / "prompt-renders.jsonl"
        if path.exists():
            return
        records = []
        for prompt in prompts:
            token_ids = adapter.format_prompt(
                [
                    {"role": "system", "content": self.config.prompting.system},
                    {"role": "user", "content": prompt.prompt},
                ]
            )
            records.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "prompt_hash": prompt.content_hash,
                    "token_ids": token_ids,
                    "token_hash": sha256_bytes(canonical_json(token_ids)),
                }
            )
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )

    def _model_files(self) -> dict[str, str]:
        local = self.config.model.local_path
        if local is None or not local.is_dir():
            return {}
        return {
            path.relative_to(local).as_posix(): sha256_file(path)
            for path in sorted(local.rglob("*"))
            if path.is_file() and path.suffix in {".safetensors", ".npz", ".gguf"}
        }
