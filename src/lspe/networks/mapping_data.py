"""Frozen, offline corpus for functional-network mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..hashing import sha256_bytes

MappingCategory = Literal["constrained", "factual", "narrative", "analogical", "code", "control"]
PairKind = Literal["paraphrase", "unrelated"]


class NetworkMapPrompt(BaseModel):
    """One immutable mapping prompt, including registered pairing metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    prompt_id: str = Field(min_length=1)
    split: Literal["network_map"]
    category: MappingCategory
    prompt: str = Field(min_length=1)
    pair_kind: PairKind | None = None
    pair_id: str | None = None
    pair_member: Literal["a", "b"] | None = None


def build_network_map_dataset(path: Path, *, force: bool = False) -> int:
    """Write the preregistered 200-prompt mapping corpus."""

    if path.exists() and not force:
        raise FileExistsError(f"Refusing to replace existing mapping dataset: {path}")
    rows = _mapping_records()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(row.model_dump_json(exclude_none=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def load_network_map_dataset(path: Path) -> list[NetworkMapPrompt]:
    rows = [
        NetworkMapPrompt.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 200:
        raise ValueError(
            f"Network mapping corpus must contain exactly 200 prompts, got {len(rows)}"
        )
    identifiers = [row.prompt_id for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Network mapping prompt IDs must be unique")
    _validate_pairs(rows)
    return rows


def network_map_hash(path: Path) -> str:
    """Hash the exact serialized mapping corpus."""

    return sha256_bytes(path.read_bytes())


def _mapping_records() -> list[NetworkMapPrompt]:
    categories: tuple[MappingCategory, ...] = (
        "constrained",
        "factual",
        "narrative",
        "analogical",
        "code",
        "control",
    )
    counts = (34, 34, 33, 33, 33, 33)
    paraphrase_pairs = (9, 9, 8, 8, 8, 8)
    rows: list[NetworkMapPrompt] = []
    leftovers: list[tuple[MappingCategory, int]] = []
    for category, count, pair_count in zip(categories, counts, paraphrase_pairs, strict=True):
        for pair_index in range(pair_count):
            subject = 2 * pair_index
            pair_id = f"para-{category}-{pair_index + 1:02d}"
            rows.extend(
                (
                    _row(category, subject, pair_id, "paraphrase", "a", paraphrase=False),
                    _row(category, subject, pair_id, "paraphrase", "b", paraphrase=True),
                )
            )
        leftovers.extend((category, index) for index in range(2 * pair_count, count))
    # Pair prompts from deliberately different task families as negative controls.
    for pair_index in range(25):
        first = leftovers[pair_index]
        second = leftovers[pair_index + 25]
        pair_id = f"unrelated-{pair_index + 1:02d}"
        rows.extend(
            (
                _row(*first, pair_id, "unrelated", "a"),
                _row(*second, pair_id, "unrelated", "b"),
            )
        )
    for category, index in leftovers[50:]:
        rows.append(_row(category, index, None, None, None))
    return sorted(rows, key=lambda row: row.prompt_id)


def _row(
    category: MappingCategory,
    index: int,
    pair_id: str | None,
    pair_kind: PairKind | None,
    pair_member: Literal["a", "b"] | None,
    *,
    paraphrase: bool = False,
) -> NetworkMapPrompt:
    prompt = _prompt(category, index, paraphrase)
    suffix = "b" if paraphrase else "a"
    return NetworkMapPrompt(
        schema_version=1,
        prompt_id=f"network-map-{category}-{index + 1:02d}-{suffix}",
        split="network_map",
        category=category,
        prompt=prompt,
        pair_kind=pair_kind,
        pair_id=pair_id,
        pair_member=pair_member,
    )


def _prompt(category: MappingCategory, index: int, paraphrase: bool) -> str:
    item = index + 1
    variants = {
        "constrained": (
            f"Write exactly three sentences about a lighthouse; sentence two must contain {item}.",
            "Describe a lighthouse in precisely three sentences, "
            f"putting {item} in the middle one.",
        ),
        "factual": (
            "Explain in two short sentences why odd number "
            f"{101 + 2 * index} has no even divisor.",
            f"In two brief sentences, state why {101 + 2 * index} cannot be divided "
            "by an even integer.",
        ),
        "narrative": (
            "Continue this scene in four sentences: the archivist opened "
            f"drawer {item} and heard rain.",
            "Add four sentences to this scene: rain sounded as the archivist "
            f"pulled open drawer {item}.",
        ),
        "analogical": (
            "Give one concrete analogy between a beehive and a transit system, "
            f"emphasizing route {item}.",
            f"Using route {item}, make a specific comparison between public transit and a beehive.",
        ),
        "code": (
            f"Write a Python function f{item}(xs) returning the {item % 5 + 1} "
            "smallest unique integers.",
            f"Define Python f{item}(xs) to produce unique integers ordered smallest first, "
            f"capped at {item % 5 + 1}.",
        ),
        "control": (
            f"Return only the integer result of {(item + 11)} * {(item % 7) + 3}.",
            f"Compute {(item % 7) + 3} times {(item + 11)} and output the integer alone.",
        ),
    }
    return variants[category][int(paraphrase)]


def _validate_pairs(rows: list[NetworkMapPrompt]) -> None:
    pairs: dict[tuple[PairKind, str], list[NetworkMapPrompt]] = {}
    for row in rows:
        metadata = (row.pair_kind, row.pair_id, row.pair_member)
        if all(value is None for value in metadata):
            continue
        if any(value is None for value in metadata):
            raise ValueError(f"Incomplete pair metadata for {row.prompt_id}")
        assert row.pair_kind is not None and row.pair_id is not None
        pairs.setdefault((row.pair_kind, row.pair_id), []).append(row)
    for key, members in pairs.items():
        if len(members) != 2 or {member.pair_member for member in members} != {"a", "b"}:
            raise ValueError(f"Pair {key} must contain exactly members a and b")
    counts = {kind: sum(key[0] == kind for key in pairs) for kind in ("paraphrase", "unrelated")}
    if counts != {"paraphrase": 50, "unrelated": 25}:
        raise ValueError(f"Mapping pair counts are invalid: {counts}")
