"""Separated deterministic RNG derivation with no global RNG state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

RNG_DOMAINS = frozenset(
    {
        "prompt-order",
        "condition-order",
        "intervention-direction",
        "sampling-token",
        "bootstrap",
        "judge-order",
        "human-review-order",
    }
)


def derive_seed(master_seed: int, domain: str, *components: Any) -> int:
    """Derive a 64-bit seed using SHA-256 and an explicit independent domain."""

    if domain not in RNG_DOMAINS:
        raise ValueError(f"Unknown RNG domain: {domain!r}")
    if master_seed < 0:
        raise ValueError("master_seed must be non-negative")
    payload = json.dumps(
        {"master_seed": master_seed, "domain": domain, "components": components},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def deterministic_order(
    master_seed: int, domain: str, values: Iterable[str], *components: Any
) -> list[str]:
    """Sort values by derived keys; adding RNG calls cannot affect the result."""

    return sorted(values, key=lambda value: derive_seed(master_seed, domain, *components, value))
