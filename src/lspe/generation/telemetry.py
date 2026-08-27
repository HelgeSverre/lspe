"""Serializable token-level records for every generated token."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TokenTelemetry:
    token_index: int
    token_id: int
    token_fragment: str
    selected_token_log_probability: float
    entropy: float
    top1_probability: float
    top1_top2_margin: float
    top_token_ids: tuple[int, ...]
    top_log_probabilities: tuple[float, ...]
    intervention_active: bool
    intervention_dose: float
    selected_layers: tuple[int, ...]
    finite: bool

    def record(self) -> dict[str, Any]:
        return asdict(self)
