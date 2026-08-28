"""Deterministic held-out corpora for FNDE after functional mapping."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..hashing import sha256_bytes
from .mapping_data import load_network_map_dataset

Phase2Split = Literal["causal_map", "calibration", "pilot", "confirm", "replication", "controls"]

SPLIT_COUNTS: dict[Phase2Split, int] = {
    "causal_map": 48,
    "calibration": 36,
    "pilot": 24,
    "confirm": 60,
    "replication": 36,
    "controls": 36,
}


class Phase2Prompt(BaseModel):
    """One frozen Phase 2 prompt with a teacher-forced reference continuation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    prompt_id: str = Field(min_length=1)
    split: Phase2Split
    task_type: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    reference_continuation: str = Field(min_length=1)
    response_schema: str = Field(min_length=1)
    validator: str = Field(min_length=1)
    expected: object | None
    tags: tuple[str, ...]


def build_phase2_datasets(root: Path, *, force: bool = False) -> dict[str, int]:
    """Write all non-mapping Phase 2 splits with no generated external content."""

    paths = {split: root / f"{split}.jsonl" for split in SPLIT_COUNTS}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not force:
        raise FileExistsError("Refusing to replace existing Phase 2 data: " + ", ".join(existing))
    root.mkdir(parents=True, exist_ok=True)
    for split, count in SPLIT_COUNTS.items():
        rows = _records(split, count)
        paths[split].write_text(
            "".join(row.model_dump_json(exclude_none=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    return dict(SPLIT_COUNTS)


def load_phase2_dataset(path: Path, expected_split: Phase2Split) -> list[Phase2Prompt]:
    rows = [
        Phase2Prompt.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != SPLIT_COUNTS[expected_split]:
        raise ValueError(
            f"{expected_split} must contain {SPLIT_COUNTS[expected_split]} prompts, got {len(rows)}"
        )
    if any(row.split != expected_split for row in rows):
        raise ValueError(f"Dataset {path} contains a row from another split")
    if len({row.prompt_id for row in rows}) != len(rows):
        raise ValueError(f"Dataset {path} contains duplicate prompt IDs")
    return rows


def phase2_data_hashes(root: Path) -> dict[str, str]:
    """Return exact file hashes for every frozen Phase 2 split."""

    paths = {"network_map": root / "network_map.jsonl"}
    paths.update({split: root / f"{split}.jsonl" for split in SPLIT_COUNTS})
    return {split: sha256_bytes(path.read_bytes()) for split, path in paths.items()}


def audit_phase2_leakage(root: Path, maximum_similarity: float = 0.92) -> dict[str, object]:
    """Hard-fail exact, normalized, or high character-similarity cross-split leakage."""

    mapping = load_network_map_dataset(root / "network_map.jsonl")
    texts: list[tuple[str, str, str]] = [
        (row.split, row.prompt_id, row.prompt) for row in mapping
    ]
    for split in SPLIT_COUNTS:
        texts.extend(
            (row.split, row.prompt_id, row.prompt)
            for row in load_phase2_dataset(root / f"{split}.jsonl", split)
        )
    normalized: dict[str, tuple[str, str]] = {}
    for split, prompt_id, prompt in texts:
        key = _normalize(prompt)
        previous = normalized.get(key)
        if previous is not None and previous[0] != split:
            raise ValueError(f"Normalized prompt leakage: {previous[1]} and {prompt_id}")
        normalized[key] = (split, prompt_id)
    nearest = 0.0
    nearest_pair: tuple[str, str] | None = None
    for left, first in enumerate(texts):
        for second in texts[left + 1 :]:
            if first[0] == second[0]:
                continue
            ratio = SequenceMatcher(None, _normalize(first[2]), _normalize(second[2])).ratio()
            if ratio > nearest:
                nearest = ratio
                nearest_pair = (first[1], second[1])
            if ratio >= maximum_similarity:
                raise ValueError(
                    f"Cross-split near duplicate ({ratio:.3f}): {first[1]} and {second[1]}"
                )
    return {
        "prompt_count": len(texts),
        "normalized_duplicates": 0,
        "maximum_cross_split_character_similarity": nearest,
        "nearest_pair": nearest_pair,
        "threshold": maximum_similarity,
        "passed": True,
    }


def _records(split: Phase2Split, count: int) -> list[Phase2Prompt]:
    if split == "controls":
        return [_control_row(split, index) for index in range(count)]
    task_types = (
        "cross_domain_analogy",
        "conceptual_blend",
        "alternative_use",
        "evidence_hypothesis",
        "constrained_continuation",
        "remote_association",
    )
    return [
        _behavior_row(split, index, task_types[index % len(task_types)])
        for index in range(count)
    ]


def _behavior_row(split: Phase2Split, index: int, task_type: str) -> Phase2Prompt:
    serial = index + 1
    split_code = {
        "causal_map": 11,
        "calibration": 23,
        "pilot": 37,
        "confirm": 53,
        "replication": 71,
    }[split]
    subject = split_code * 100 + serial
    frame = {
        "causal_map": "Mechanism-screen notebook; isolate which component carries the answer",
        "calibration": "Dose-calibration worksheet; preserve the supplied continuation exactly",
        "pilot": "Exploratory design trial; compare candidate settings without choosing prose",
        "confirm": "Sealed confirmation packet; follow the registered schema without commentary",
        "replication": (
            "Independent replication file; reproduce the relation on a fresh architecture"
        ),
    }[split]
    prompts = {
        "cross_domain_analogy": (
            f"{frame} {subject}: connect seed banks to distributed backups. Return JSON fields "
            "source, target, mechanism, and limitation."
        ),
        "conceptual_blend": (
            f"{frame} {subject}: design a public bench combining shade capture with wayfinding. "
            "Return JSON fields concept, mechanism, factual_constraint, and tradeoff."
        ),
        "alternative_use": (
            f"{frame} {subject}: propose a feasible emergency use for a ceramic tile. Return JSON "
            "fields object, use, mechanism, and safety_limit."
        ),
        "evidence_hypothesis": (
            f"{frame} {subject}: a sensor drops only dawn readings; battery and storage tests "
            "pass. Return JSON fields hypothesis, evidence_link, and discriminating_test."
        ),
        "constrained_continuation": (
            f"{frame} {subject}: continue a design log where the bridge stays open and no motor is "
            "added. Return JSON fields continuation, constraint_used, and causal_link."
        ),
        "remote_association": (
            f"{frame} {subject}: relate compass, yeast, and queue. Return JSON fields "
            "bridge, link_compass, link_yeast, and link_queue."
        ),
    }
    reference = json.dumps(
        {"task": task_type, "case": subject, "relation": "mechanism links constraints to outcome"},
        sort_keys=True,
    )
    return Phase2Prompt(
        schema_version=1,
        prompt_id=f"fnde-{split}-{serial:03d}",
        split=split,
        task_type=task_type,
        prompt=prompts[task_type],
        reference_continuation=reference,
        response_schema="relational_json.v1",
        validator="relational_json",
        expected=None,
        tags=("phase2", split, "behavior"),
    )


def _control_row(split: Phase2Split, index: int) -> Phase2Prompt:
    serial = index + 1
    left = 200 + serial
    right = 3 + serial % 11
    expected = str(left * right)
    return Phase2Prompt(
        schema_version=1,
        prompt_id=f"fnde-controls-{serial:03d}",
        split=split,
        task_type="arithmetic",
        prompt=f"Control card {9000 + serial}: return only the integer result of {left} * {right}.",
        reference_continuation=expected,
        response_schema="text.v1",
        validator="exact_answer",
        expected=expected,
        tags=("phase2", "controls", "competence"),
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.casefold())).strip()
