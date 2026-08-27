"""Strict JSONL prompt loading with per-row content hashes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..hashing import sha256_bytes


class PromptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    prompt_id: str = Field(min_length=1)
    split: Literal["calibration", "pilot", "confirm", "controls"]
    task_type: str = Field(min_length=1)
    system_variant: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    response_schema: str = Field(min_length=1)
    validator: str = Field(min_length=1)
    expected: object | None
    tags: tuple[str, ...]

    @property
    def content_hash(self) -> str:
        return sha256_bytes(self.model_dump_json(exclude_none=False).encode("utf-8"))


def load_prompts(path: Path, expected_split: str | None = None) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = PromptRecord.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as error:
            raise ValueError(f"Invalid prompt at {path}:{number}: {error}") from error
        if expected_split is not None and record.split != expected_split:
            raise ValueError(
                f"Prompt {record.prompt_id} has split {record.split}, expected {expected_split}"
            )
        if record.prompt_id in seen:
            raise ValueError(f"Duplicate prompt ID in {path}: {record.prompt_id}")
        seen.add(record.prompt_id)
        records.append(record)
    if not records:
        raise ValueError(f"No prompt records in {path}")
    return records
