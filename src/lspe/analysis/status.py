"""Mechanical implementation of the specification's result taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Status = Literal["SUPPORTED", "PROMISING", "NOT_SUPPORTED", "DEGENERATIVE", "INVALID_RUN"]


@dataclass(frozen=True)
class StatusInputs:
    integrity_ok: bool
    h1_estimate: float | None
    h1_ci95: tuple[float, float] | None
    validity_retained: bool
    degeneration_retained: bool
    coherent_beats_white_vsd: bool
    coherent_competence_not_worse_than_white: bool
    replication_positive: bool | None
    diversity_due_to_failure: bool = False


def classify_status(inputs: StatusInputs) -> Status:
    if not inputs.integrity_ok:
        return "INVALID_RUN"
    if inputs.diversity_due_to_failure or not inputs.degeneration_retained:
        return "DEGENERATIVE"
    if inputs.h1_estimate is None or inputs.h1_ci95 is None or inputs.h1_estimate <= 0:
        return "NOT_SUPPORTED"
    if (
        inputs.h1_ci95[0] > 0
        and inputs.validity_retained
        and inputs.coherent_beats_white_vsd
        and inputs.coherent_competence_not_worse_than_white
        and inputs.replication_positive is True
    ):
        return "SUPPORTED"
    if inputs.validity_retained:
        return "PROMISING"
    return "NOT_SUPPORTED"
