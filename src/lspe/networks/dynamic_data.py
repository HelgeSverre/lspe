"""Fresh trajectory corpus for the Dynamic Connectivity Flattening experiment."""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DynamicCategory = Literal[
    "constrained", "factual", "narrative", "analogical", "code", "open_association"
]


class DynamicMapPrompt(BaseModel):
    """One frozen prompt and its mapping-only fold assignment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    prompt_id: str = Field(min_length=1)
    split: Literal["dynamic_map"]
    category: DynamicCategory
    fold: Literal[0, 1, 2, 3]
    prompt: str = Field(min_length=1)


def build_dynamic_map_dataset(path: Path, *, force: bool = False) -> int:
    """Write the preregistered 96-prompt DCF mapping corpus."""

    if path.exists() and not force:
        raise FileExistsError(f"Refusing to replace existing DCF mapping dataset: {path}")
    rows = _records()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    return len(rows)


def load_dynamic_map_dataset(path: Path) -> list[DynamicMapPrompt]:
    """Load and validate the exact DCF corpus shape and fold balance."""

    rows = [
        DynamicMapPrompt.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 96:
        raise ValueError(f"DCF mapping corpus must contain 96 prompts, got {len(rows)}")
    if len({row.prompt_id for row in rows}) != len(rows):
        raise ValueError("DCF prompt IDs must be unique")
    counts = Counter((row.category, row.fold) for row in rows)
    if set(counts.values()) != {4} or len(counts) != 24:
        raise ValueError(f"DCF category/fold cells must each contain four prompts: {counts}")
    return rows


def audit_dynamic_map_leakage(
    dynamic_path: Path,
    comparison_paths: list[Path],
    maximum_similarity: float = 0.92,
) -> dict[str, object]:
    """Reject exact, normalized, and near-duplicate prompts from earlier splits."""

    current = load_dynamic_map_dataset(dynamic_path)
    earlier: list[tuple[str, str]] = []
    for path in comparison_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                earlier.append((str(row["prompt_id"]), str(row["prompt"])))
    normalized_earlier = {_normalize(prompt): prompt_id for prompt_id, prompt in earlier}
    nearest = 0.0
    nearest_pair: tuple[str, str] | None = None
    for row in current:
        normalized = _normalize(row.prompt)
        if normalized in normalized_earlier:
            raise ValueError(
                f"Normalized DCF leakage: {row.prompt_id} and "
                f"{normalized_earlier[normalized]}"
            )
        for earlier_id, earlier_prompt in earlier:
            similarity = SequenceMatcher(
                None, normalized, _normalize(earlier_prompt)
            ).ratio()
            if similarity > nearest:
                nearest = similarity
                nearest_pair = (row.prompt_id, earlier_id)
            if similarity >= maximum_similarity:
                raise ValueError(
                    f"DCF near duplicate ({similarity:.3f}): "
                    f"{row.prompt_id} and {earlier_id}"
                )
    return {
        "dynamic_prompt_count": len(current),
        "comparison_prompt_count": len(earlier),
        "normalized_duplicates": 0,
        "maximum_character_similarity": nearest,
        "nearest_pair": nearest_pair,
        "threshold": maximum_similarity,
        "passed": True,
    }


def _records() -> list[DynamicMapPrompt]:
    categories: tuple[DynamicCategory, ...] = (
        "constrained",
        "factual",
        "narrative",
        "analogical",
        "code",
        "open_association",
    )
    return [
        DynamicMapPrompt(
            schema_version=1,
            prompt_id=f"dcf-{category}-{index + 1:02d}",
            split="dynamic_map",
            category=category,
            fold=index % 4,
            prompt=_prompt(category, index),
        )
        for category in categories
        for index in range(16)
    ]


def _prompt(category: DynamicCategory, index: int) -> str:
    serial = 8401 + index * 29
    prompts = {
        "constrained": [
            f"Draft a six-line museum label for object R{serial}. Use the word 'hinge' "
            "once, make line three a question, and end with a concrete noun.",
            f"Describe garden R{serial} in four sentences. Begin each with a different letter, "
            "include one measurement, and do not use the word green.",
            f"Write five instructions for repairing device R{serial}. Step two must rhyme with "
            "step four and the final step must contain exactly six words.",
            f"Create a three-column markdown table for expedition R{serial} with four data rows. "
            "Use columns risk, evidence, and response; no cell may exceed five words.",
        ],
        "factual": [
            f"A sealed vial marked R{serial} warms from 18 C to 31 C while its volume stays "
            "fixed. Explain the expected pressure change and name the physical relation used.",
            f"Sample R{serial} has mass 240 g and displaces 80 mL of water. Calculate its density "
            "and explain whether it sinks in water.",
            f"Station R{serial} records a 12-second delay every 15 minutes. Compute the "
            "accumulated delay over five hours and state the assumption.",
            f"A plant in trial R{serial} receives blue light while an identical control receives "
            "darkness. Name the independent variable and one required control.",
        ],
        "narrative": [
            f"Continue for exactly six sentences: At platform R{serial}, the departure board "
            "listed a train that had arrived yesterday. Keep the explanation non-supernatural.",
            f"Continue scene R{serial} for six sentences: a baker finds a handwritten weather "
            "forecast inside an uncut loaf. Resolve it without coincidence.",
            f"Write the next six sentences of case R{serial}: the lighthouse flashes an unfamiliar "
            "pattern only when no ships are nearby. Keep the cause mechanical.",
            f"Complete vignette R{serial} in six sentences: every clock in the library is seven "
            "minutes fast except one. End with a mundane explanation.",
        ],
        "analogical": [
            f"Develop an analogy between a wetland and a fault-tolerant message queue for case "
            f"R{serial}. Identify the shared mechanism and two precise failure points.",
            f"Compare an immune response with a software incident team for case R{serial}. Explain "
            "the shared feedback loop and two places the analogy fails.",
            f"Use a river delta to explain database sharding in lesson R{serial}. Map three parts "
            "explicitly and identify two misleading correspondences.",
            f"Build an analogy between musical counterpoint and traffic control for case "
            f"R{serial}. State the coordination mechanism and two limitations.",
        ],
        "code": [
            f"Write Python function reconcile_{serial}(events) that keeps the newest event per "
            "key, breaks timestamp ties by lexicographic value, and returns keys sorted.",
            f"Write Python function windows_{serial}(values, size) returning every full sliding "
            "window whose sum is even, without mutating the input.",
            f"Implement Python function merge_{serial}(left, right) for sorted integer iterables; "
            "remove duplicates lazily and include type hints.",
            f"Define Python function runs_{serial}(text) that returns start, end, and character "
            "for each maximal repeated-character run, including singletons.",
        ],
        "open_association": [
            f"For design card R{serial}, combine a night market, error-correcting codes, and "
            "migratory birds into one feasible public service. Explain the mechanism and cost.",
            f"For concept R{serial}, combine public fountains, version control, and oral history "
            "into a feasible civic project. Explain operation, benefit, and failure mode.",
            f"Design R{serial} by combining cargo bicycles, fungal networks, and library lending. "
            "Describe a plausible service, its mechanism, and its main cost.",
            f"Invent service R{serial} using tide tables, neighborhood kitchens, and cryptographic "
            "signatures. Keep it physically feasible and explain one tradeoff.",
        ],
    }
    return prompts[category][index // 4]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.casefold())).strip()
