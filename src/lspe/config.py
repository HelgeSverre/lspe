"""Strict, immutable experiment configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    """Shared policy: unknown configuration keys are always an error."""

    # YAML necessarily represents paths as strings; field-level constraints
    # enforce scientific inputs while `extra="forbid"` rejects undeclared keys.
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentConfig(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    phase: Literal["smoke", "pilot", "confirm", "replicate"]
    master_seed: int = Field(ge=0)
    output_root: Path = Path("runs")


class HardwareConfig(StrictModel):
    expected_platform: str = "darwin-arm64"
    minimum_memory_gb: float = Field(ge=1)
    memory_soft_limit_fraction: float = Field(gt=0, le=1)
    memory_hard_limit_fraction: float = Field(gt=0, le=1)

    @field_validator("memory_hard_limit_fraction")
    @classmethod
    def hard_limit_exceeds_soft(cls, value: float, info: object) -> float:
        data = getattr(info, "data", {})
        soft = data.get("memory_soft_limit_fraction")
        if soft is not None and value <= soft:
            raise ValueError("memory_hard_limit_fraction must exceed memory_soft_limit_fraction")
        return value


class IntegrityConfig(StrictModel):
    """Numerical bounds for mandatory preflight checks.

    The 1e-6 default is intentionally strict.  A quantized/BF16 backend may
    only use a larger cache bound when its configuration records why; this
    keeps a backend-specific numerical limitation visible in every lock and
    report rather than silently weakening the scientific gate.
    """

    cache_logit_tolerance: float = Field(default=1e-6, ge=0, le=10)
    zero_dose_logit_tolerance: float = Field(default=1e-6, ge=0, le=1e-3)
    cache_tolerance_reason: str | None = None

    @model_validator(mode="after")
    def requires_reason_for_relaxed_cache_bound(self) -> IntegrityConfig:
        if self.cache_logit_tolerance > 1e-6 and not self.cache_tolerance_reason:
            raise ValueError(
                "cache_tolerance_reason is required when cache_logit_tolerance exceeds 1e-6"
            )
        return self


class ModelConfig(StrictModel):
    adapter: Literal["mlx_gemma4", "mlx_qwen3"]
    repo_id: str = Field(min_length=3)
    revision: str | None = None
    local_path: Path | None = None
    trust_remote_code: bool = False
    text_only: bool = True
    thinking: bool = False
    speculative_decoding: bool = False
    kv_cache_quantization: bool = False


class PromptingConfig(StrictModel):
    system: str = Field(min_length=1)
    use_model_chat_template: bool = True
    max_prompt_tokens: int = Field(ge=1, le=2048)


class SamplingConfig(StrictModel):
    temperature: float = Field(gt=0, le=5)
    top_k: int = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    repetition_penalty: float = Field(ge=1, le=3)
    max_new_tokens: int = Field(ge=1, le=192)
    stop_on_eos: bool = True
    store_top_logprobs: int = Field(ge=1, le=256)


class PilotLayerGroup(StrictModel):
    """A preregistered layer group eligible for pilot selection."""

    candidate_id: str = Field(min_length=1)
    layers: list[int] = Field(min_length=1)

    @field_validator("layers")
    @classmethod
    def layers_are_unique_and_nonnegative(cls, value: list[int]) -> list[int]:
        if any(layer < 0 for layer in value) or len(set(value)) != len(value):
            raise ValueError("candidate layer indices must be unique non-negative integers")
        return value


class InterventionConfig(StrictModel):
    site: Literal["post_decoder_layer"] = "post_decoder_layer"
    kernel: Literal["spherical_rotation", "rms_scaled_additive"]
    timing: Literal["decode_only", "prefill_and_decode"] = "decode_only"
    direction_mode: Literal["coherent_per_layer", "coherent_shared", "white_per_token", "zero"] = (
        "coherent_per_layer"
    )
    selected_layers: list[int] | Literal["auto"] = "auto"
    target_kl_nats: float = Field(ge=0)
    raw_dose_grid: list[float] = Field(min_length=1)
    preserve_norm: bool = True
    group_scale: Literal["inverse_sqrt_count"] = "inverse_sqrt_count"
    pilot_candidate_groups: list[PilotLayerGroup] = Field(default_factory=list)
    pilot_target_kl_bands: list[float] = Field(default_factory=list)

    @field_validator("raw_dose_grid")
    @classmethod
    def dose_grid_is_nonnegative_and_unique(cls, value: list[float]) -> list[float]:
        if any(dose < 0 for dose in value):
            raise ValueError("raw_dose_grid values must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("raw_dose_grid values must be unique")
        return value

    @field_validator("pilot_target_kl_bands")
    @classmethod
    def pilot_targets_are_positive_and_unique(cls, value: list[float]) -> list[float]:
        if any(target <= 0 for target in value) or len(set(value)) != len(value):
            raise ValueError("pilot_target_kl_bands must contain unique positive values")
        return value


class DataConfig(StrictModel):
    calibration: Path
    pilot: Path
    confirm: Path
    controls: Path


class ExecutionConfig(StrictModel):
    generations_per_prompt: int = Field(ge=1)
    batch_size: int = Field(ge=1, le=1)
    randomized_condition_order: bool = True
    save_every: int = Field(ge=1)
    flush_every: int = Field(ge=1)


class ScoringConfig(StrictModel):
    embedding_model: str
    embedding_revision: str | None = None
    local_judge_model: str
    judge_enabled: bool = True
    human_review_export: bool = True


class StatisticsConfig(StrictModel):
    bootstrap_samples: int = Field(ge=1)
    confidence_level: float = Field(gt=0, lt=1)
    familywise_method: Literal["holm"] = "holm"
    validity_noninferiority_margin_pp: float = Field(ge=0, le=100)
    degeneration_margin_pp: float = Field(ge=0, le=100)


CoreCondition = Literal["baseline", "sham", "coherent", "white", "temp_match"]


class LspeConfig(StrictModel):
    schema_version: Literal[1]
    experiment: ExperimentConfig
    hardware: HardwareConfig
    integrity: IntegrityConfig = IntegrityConfig()
    model: ModelConfig
    prompting: PromptingConfig
    sampling: SamplingConfig
    intervention: InterventionConfig
    conditions: Annotated[list[CoreCondition], Field(min_length=1)]
    data: DataConfig
    execution: ExecutionConfig
    scoring: ScoringConfig
    statistics: StatisticsConfig

    @field_validator("conditions")
    @classmethod
    def conditions_are_unique(cls, value: list[CoreCondition]) -> list[CoreCondition]:
        if len(set(value)) != len(value):
            raise ValueError("conditions must not contain duplicates")
        return value


def load_config(path: Path | str) -> LspeConfig:
    """Load and validate a YAML experiment file without accepting unknown keys."""

    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Unable to read configuration {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration {path} must contain a YAML mapping")
    try:
        return LspeConfig.model_validate(raw)
    except ValidationError as error:
        raise ValueError(f"Invalid configuration {path}: {error}") from error
