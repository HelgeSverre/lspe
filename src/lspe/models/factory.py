"""Single adapter selection point used by every model-backed command."""

from __future__ import annotations

from typing import Any

from ..config import ModelConfig
from .mlx_gemma4 import MlxGemma4Adapter
from .mlx_qwen3 import MlxQwen3Adapter


def create_adapter(model: ModelConfig) -> Any:
    match model.adapter:
        case "mlx_gemma4":
            return MlxGemma4Adapter()
        case "mlx_qwen3":
            return MlxQwen3Adapter()
        case _:
            raise ValueError(f"Unknown configured model adapter: {model.adapter}")
