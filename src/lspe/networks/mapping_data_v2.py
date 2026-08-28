"""Fresh, balanced mapping corpus for the stronger FNDE v2 attempt."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from .mapping_data import MappingCategory, NetworkMapPrompt


def build_network_map_v2_dataset(path: Path, *, force: bool = False) -> int:
    """Write 240 prompts with 60 paraphrase and 30 unrelated pairs."""

    if path.exists() and not force:
        raise FileExistsError(f"Refusing to replace existing v2 mapping dataset: {path}")
    rows = _records()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(row.model_dump_json(exclude_none=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def load_network_map_v2_dataset(path: Path) -> list[NetworkMapPrompt]:
    rows = [
        NetworkMapPrompt.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 240:
        raise ValueError(f"FNDE v2 mapping corpus must contain 240 prompts, got {len(rows)}")
    categories = {category: 0 for category in _categories()}
    pairs: dict[tuple[str, str], list[NetworkMapPrompt]] = {}
    for row in rows:
        categories[row.category] += 1
        if row.pair_kind and row.pair_id:
            pairs.setdefault((row.pair_kind, row.pair_id), []).append(row)
    if set(categories.values()) != {40}:
        raise ValueError(f"FNDE v2 categories are unbalanced: {categories}")
    pair_counts = {
        kind: sum(pair_kind == kind for pair_kind, _ in pairs)
        for kind in ("paraphrase", "unrelated")
    }
    if pair_counts != {"paraphrase": 60, "unrelated": 30}:
        raise ValueError(f"FNDE v2 pair counts are invalid: {pair_counts}")
    if any(len(members) != 2 for members in pairs.values()):
        raise ValueError("Every FNDE v2 pair must contain exactly two prompts")
    return rows


def audit_v2_mapping_leakage(
    v2_path: Path, comparison_paths: list[Path], maximum_similarity: float = 0.92
) -> dict[str, object]:
    """Reject exact, normalized, and near-duplicate prompts from every earlier split."""

    v2 = load_network_map_v2_dataset(v2_path)
    earlier: list[tuple[str, str]] = []
    for path in comparison_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                earlier.append((str(row["prompt_id"]), str(row["prompt"])))
    earlier_normalized = {_normalize(prompt): prompt_id for prompt_id, prompt in earlier}
    nearest = 0.0
    nearest_pair: tuple[str, str] | None = None
    for row in v2:
        normalized = _normalize(row.prompt)
        if normalized in earlier_normalized:
            raise ValueError(
                f"Normalized v2 leakage: {row.prompt_id} and {earlier_normalized[normalized]}"
            )
        for earlier_id, earlier_prompt in earlier:
            ratio = SequenceMatcher(None, normalized, _normalize(earlier_prompt)).ratio()
            if ratio > nearest:
                nearest = ratio
                nearest_pair = (row.prompt_id, earlier_id)
            if ratio >= maximum_similarity:
                raise ValueError(
                    f"V2 near duplicate ({ratio:.3f}): {row.prompt_id} and {earlier_id}"
                )
    return {
        "v2_prompt_count": len(v2),
        "comparison_prompt_count": len(earlier),
        "normalized_duplicates": 0,
        "maximum_character_similarity": nearest,
        "nearest_pair": nearest_pair,
        "threshold": maximum_similarity,
        "passed": True,
    }


def _records() -> list[NetworkMapPrompt]:
    rows: list[NetworkMapPrompt] = []
    leftovers: list[tuple[MappingCategory, int]] = []
    for category in _categories():
        for pair_index in range(10):
            subject = pair_index
            pair_id = f"v2-para-{category}-{pair_index + 1:02d}"
            rows.extend(
                (
                    _row(category, subject, pair_id, "paraphrase", "a", False),
                    _row(category, subject, pair_id, "paraphrase", "b", True),
                )
            )
        leftovers.extend((category, index) for index in range(10, 30))
    for pair_index in range(30):
        first = leftovers[pair_index]
        second = leftovers[pair_index + 30]
        pair_id = f"v2-unrelated-{pair_index + 1:02d}"
        rows.extend(
            (
                _row(*first, pair_id, "unrelated", "a", False),
                _row(*second, pair_id, "unrelated", "b", False),
            )
        )
    for category, index in leftovers[60:]:
        rows.append(_row(category, index, None, None, None, False))
    return sorted(rows, key=lambda row: row.prompt_id)


def _row(
    category: MappingCategory,
    index: int,
    pair_id: str | None,
    pair_kind: Literal["paraphrase", "unrelated"] | None,
    pair_member: Literal["a", "b"] | None,
    paraphrase: bool,
) -> NetworkMapPrompt:
    return NetworkMapPrompt(
        schema_version=1,
        prompt_id=f"network-map-v2-{category}-{index + 1:02d}-{'b' if paraphrase else 'a'}",
        split="network_map",
        category=category,
        prompt=_prompt(category, index, paraphrase),
        pair_kind=pair_kind,
        pair_id=pair_id,
        pair_member=pair_member,
    )


def _prompt(category: MappingCategory, index: int, paraphrase: bool) -> str:
    serial = 6100 + index * 17
    variants = {
        "constrained": (
            f"Compose five lines about a windmill docket {serial}; line four must end with amber.",
            f"For windmill docket {serial}, write five lines and finish the fourth with amber.",
        ),
        "factual": (
            f"Using only the supplied fact that alloy sample {serial} expands when heated, explain "
            "one measurement consequence in three sentences.",
            f"In three sentences, infer a measurable result from this fact: alloy {serial} expands "
            "under heat."
        ),
        "narrative": (
            f"Write the next five sentences after courier {serial} finds a dry key "
            "inside a storm drain.",
            f"Courier {serial} discovers a key that stayed dry in a flooded drain; "
            "continue for five sentences.",
        ),
        "analogical": (
            "Build a precise analogy between fungal roots and freight hubs for "
            f"routing case {serial}, "
            "including one point where the analogy breaks.",
            f"For routing case {serial}, compare freight hubs with fungal roots "
            "and state one limitation.",
        ),
        "code": (
            f"Write Python function route_{serial}(pairs) that returns keys grouped "
            "by equal values, "
            "with keys sorted inside each group.",
            f"Define route_{serial}(pairs): group keys sharing a value and sort every key group.",
        ),
        "control": (
            f"Ledger {serial}: output only the integer obtained by subtracting "
            f"{index + 9} from {serial}.",
            f"For ledger {serial}, calculate {serial} minus {index + 9} and return "
            "only that integer.",
        ),
    }
    return variants[category][int(paraphrase)]


def _categories() -> tuple[MappingCategory, ...]:
    return ("constrained", "factual", "narrative", "analogical", "code", "control")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.casefold())).strip()
