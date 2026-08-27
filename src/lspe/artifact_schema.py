"""Strict machine-readable artifact contracts used by report generation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PrimaryOutcome(_ArtifactModel):
    metric: str
    contrast: str
    estimate: float | None
    ci95: tuple[float | None, float | None]
    p_value: float | None = Field(default=None, ge=0, le=1)
    n_prompts: int = Field(ge=0)
    median: float | None = None
    standardized_effect: float | None = None
    positive: int = Field(default=0, ge=0)
    zero: int = Field(default=0, ge=0)
    negative: int = Field(default=0, ge=0)


class ReportArtifact(_ArtifactModel):
    schema_version: Literal[1]
    run_id: str = Field(min_length=1)
    status: Literal["SUPPORTED", "PROMISING", "NOT_SUPPORTED", "DEGENERATIVE", "INVALID_RUN"]
    primary: PrimaryOutcome
    validity: dict[str, Any]
    competence: dict[str, Any]
    degeneration: dict[str, Any]
    replication: dict[str, Any]
    integrity: dict[str, Any]
    artifact_root_hash: str | None
    secondary: dict[str, Any] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


def validate_report(value: dict[str, Any]) -> dict[str, Any]:
    """Validate report data and return a JSON-mode normalized representation."""

    return ReportArtifact.model_validate(value).model_dump(mode="json")
