"""Append-only, content-addressed run artifacts with atomic generation commits."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .hashing import canonical_json, content_id, sha256_bytes


@dataclass(frozen=True)
class CommitResult:
    generation_id: str
    committed: bool
    checksum: str


class RunStore:
    """A small durable store: individual journal entries plus a JSONL projection."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.journal = root / "journal"
        self.generations_path = root / "generations.jsonl"

    def initialize(self, manifest: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=False)
        self.journal.mkdir()
        self._atomic_write_json(self.root / "manifest.json", manifest)
        self.generations_path.touch()

    def completed_ids(self) -> set[str]:
        if not self.journal.exists():
            return set()
        return {
            f"sha256:{path.stem}" for path in self.journal.glob("*.json") if self._valid_entry(path)
        }

    def commit_generation(
        self, scientific_inputs: dict[str, Any], record: dict[str, Any]
    ) -> CommitResult:
        generation_id = content_id(scientific_inputs)
        normalized = {**record, "generation_id": generation_id}
        encoded = canonical_json(normalized)
        checksum = sha256_bytes(encoded)
        target = self.journal / f"{generation_id.removeprefix('sha256:')}.json"
        if target.exists():
            existing = target.read_bytes()
            if sha256_bytes(existing) != checksum:
                raise ValueError(f"Content ID collision with different record: {generation_id}")
            return CommitResult(generation_id, False, checksum)
        self._atomic_write_bytes(target, encoded)
        with self.generations_path.open("ab") as handle:
            handle.write(encoded + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return CommitResult(generation_id, True, checksum)

    @staticmethod
    def _valid_entry(path: Path) -> bool:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return True

    @staticmethod
    def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
        RunStore._atomic_write_bytes(path, canonical_json(value))

    @staticmethod
    def _atomic_write_bytes(path: Path, value: bytes) -> None:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
