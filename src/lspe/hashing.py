"""Stable hashes and content identifiers for scientific artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    """Encode data in a stable, cross-process representation."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_id(scientific_inputs: dict[str, Any]) -> str:
    """Return a namespaced content address for a planned generation/artifact."""

    return f"sha256:{sha256_bytes(canonical_json(scientific_inputs))}"


def root_digest(entries: dict[str, str]) -> str:
    """Hash sorted relative-path/checksum pairs without filesystem ambiguity."""

    return sha256_bytes(canonical_json(sorted(entries.items())))
