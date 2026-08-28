"""Optional MLX runtime helpers with actionable failure messages."""

from __future__ import annotations

import importlib
from typing import Any


class RuntimeUnavailableError(RuntimeError):
    """Raised when a requested adapter's pinned runtime is not installed."""


def import_module(name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise RuntimeUnavailableError(
            f"{name} is unavailable. Install the matching locked MLX runtime profile "
            f"({install_hint}) "
            "and rerun `lspe preflight`; do not substitute an unreviewed backend."
        ) from error


def resolve_attribute(root: Any, candidates: tuple[str, ...]) -> tuple[str, Any]:
    """Discover a version-sensitive nested runtime path rather than hardcode it."""

    for dotted in candidates:
        value = root
        try:
            for part in dotted.split("."):
                value = getattr(value, part)
        except AttributeError:
            continue
        return dotted, value
    rendered = ", ".join(candidates)
    raise RuntimeUnavailableError(f"No compatible architecture path found; inspected: {rendered}")
