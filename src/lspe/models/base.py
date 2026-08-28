"""Backend-neutral contracts for subject-model adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class LayerInfo:
    index: int
    normalized_depth: float
    layer_type: str
    attention_kind: str
    kv_sharing_role: str | None = None


@dataclass(frozen=True)
class ArchitectureInfo:
    decoder_layer_path: str
    hidden_width: int
    vocabulary_size: int
    layers: tuple[LayerInfo, ...]
    final_norm_path: str
    output_head_path: str
    cache_type: str
    cache_count: int
    has_per_layer_inputs: bool


@dataclass(frozen=True)
class ForwardResult:
    logits: np.ndarray
    hidden_summaries: dict[int, np.ndarray]
    cache: Any


class ModelAdapter(Protocol):
    """The core never imports MLX/MLX-VLM classes directly."""

    def load(self, spec: Any) -> None: ...

    def unload(self) -> None: ...

    def format_prompt(self, messages: Sequence[dict[str, str]]) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def eos_token_ids(self) -> set[int]: ...

    def architecture(self) -> ArchitectureInfo: ...

    def wrap_layers(self, controller: Any) -> None: ...

    def unwrap_layers(self) -> None: ...

    def wrap_attention_observer(
        self, observer: Any, selected_layers: frozenset[int]
    ) -> None: ...

    def unwrap_attention_observer(self) -> None: ...

    def make_cache(self) -> Any: ...

    def forward(self, token_ids: Sequence[int], cache: Any | None = None) -> ForwardResult: ...
