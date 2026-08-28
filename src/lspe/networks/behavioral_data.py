"""Deterministic fresh datasets for the SCCF behavioral experiment."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..hashing import sha256_bytes

CALIBRATION_TOPICS = (
    "observatory",
    "canal lock",
    "ceramics studio",
    "seed bank",
    "weather station",
    "rail depot",
    "aquarium",
    "wind farm",
    "archive",
    "foundry",
    "orchard",
    "ferry terminal",
)
OPEN_TOPICS = (
    "lighthouse",
    "greenhouse",
    "clock tower",
    "train platform",
    "quarry",
    "planetarium",
    "bakery",
    "marsh",
    "radio studio",
    "courthouse",
    "vineyard",
    "shipyard",
)
ANALOGY_PAIRS = (
    ("mushroom mycelium", "urban package delivery"),
    ("coral spawning", "database replication"),
    ("bird migration", "hospital staffing"),
    ("pottery glazing", "software deployment"),
    ("avalanche control", "online moderation"),
    ("tide pools", "shared office scheduling"),
    ("ant trail repair", "emergency road routing"),
    ("music counterpoint", "warehouse robotics"),
    ("tree-ring growth", "personal budgeting"),
    ("kite steering", "electrical grid balancing"),
    ("fermentation", "community dispute resolution"),
    ("stage improvisation", "incident response"),
)
NARRATIVE_SEEDS = (
    ("a cartographer", "a map that updates forgotten promises"),
    ("a night watchmaker", "a clock that skips one stranger each day"),
    ("a retired diver", "a bell heard beneath a dry reservoir"),
    ("a window cleaner", "a reflection that arrives one hour early"),
    ("a seed librarian", "a packet labelled with tomorrow's weather"),
    ("a tram conductor", "a passenger holding an impossible transfer"),
    ("a bridge inspector", "footprints ending halfway across fresh paint"),
    ("a radio host", "a caller broadcasting from an abandoned frequency"),
    ("a museum guard", "a portrait that loses one object nightly"),
    ("a beekeeper", "honey tasting of a childhood room"),
    ("a locksmith", "a key warm from a door not yet built"),
    ("a weather clerk", "rain recorded inside only one house"),
)
CONSTRAINT_WORDS = (
    ("quartz", "harbor"),
    ("willow", "static"),
    ("velvet", "compass"),
    ("cinder", "orbit"),
    ("meadow", "hinge"),
    ("silver", "current"),
    ("mosaic", "thunder"),
    ("cedar", "mirror"),
    ("spiral", "orchard"),
    ("frost", "violin"),
    ("amber", "stairwell"),
    ("linen", "eclipse"),
)
FACTS = (
    ("What is 17 multiplied by 6? Return only the answer.", "102"),
    ("What is the chemical symbol for tungsten? Return only the answer.", "W"),
    ("How many sides does a dodecagon have? Return only the answer.", "12"),
    ("What is 144 divided by 12? Return only the answer.", "12"),
    ("Which planet is closest to the Sun? Return only the answer.", "Mercury"),
    ("What is the square root of 169? Return only the answer.", "13"),
    ("How many minutes are in three hours? Return only the answer.", "180"),
    ("What gas has the chemical formula CO2? Return only the answer.", "carbon dioxide"),
    ("What is 23 plus 58? Return only the answer.", "81"),
    ("How many vertices does a cube have? Return only the answer.", "8"),
    ("What is the Roman numeral for 50? Return only the answer.", "L"),
    ("What is 9 cubed? Return only the answer.", "729"),
)


def build_behavioral_datasets(root: Path, force: bool = False) -> dict[str, int]:
    """Write calibration, pilot, and confirmation corpora plus a leakage audit."""

    root.mkdir(parents=True, exist_ok=True)
    rows = {
        "calibration": [
            _calibration_row(index, topic) for index, topic in enumerate(CALIBRATION_TOPICS)
        ],
        "pilot": _behavior_rows("pilot", 0, 4),
        "confirm": _behavior_rows("confirm", 4, 12),
    }
    for split, records in rows.items():
        path = root / f"{split}.jsonl"
        if path.exists() and not force:
            raise FileExistsError(f"Behavioral dataset already exists: {path}")
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
        )
    audit = _leakage_audit(rows, root)
    (root / "leakage-audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return {split: len(records) for split, records in rows.items()}


def _behavior_rows(split: str, start: int, stop: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(start, stop):
        rows.extend(
            [
                _row(
                    split,
                    "open_association",
                    index,
                    f'Return exactly ten unique common nouns loosely associated with "{OPEN_TOPICS[index]}" as a JSON array. Favor surprising but defensible connections.',
                    "divergent_words",
                    None,
                ),
                _row(
                    split,
                    "analogical",
                    index,
                    f"Return JSON only: transfer one concrete principle from {ANALOGY_PAIRS[index][0]} to {ANALOGY_PAIRS[index][1]}. Use exactly the keys source_principle, target_application, and mechanism.",
                    "cross_domain_bridge",
                    None,
                ),
                _row(
                    split,
                    "narrative",
                    index,
                    f"Write a self-contained story of 60 to 180 words about {NARRATIVE_SEEDS[index][0]} discovering {NARRATIVE_SEEDS[index][1]}. Make the ending surprising but causally earned.",
                    "bounded_prose",
                    {"minimum_words": 60, "maximum_words": 180},
                ),
                _row(
                    split,
                    "constrained",
                    index,
                    f"Write exactly four lines, each 6 to 14 words, and include the words {CONSTRAINT_WORDS[index][0]} and {CONSTRAINT_WORDS[index][1]} somewhere.",
                    "constrained_creative",
                    {"required_words": list(CONSTRAINT_WORDS[index])},
                ),
                _row(split, "factual", index, FACTS[index][0], "exact_answer", FACTS[index][1]),
                _code_row(split, index),
            ]
        )
    return rows


def _calibration_row(index: int, topic: str) -> dict[str, Any]:
    return _row(
        "calibration",
        "calibration",
        index,
        f"Describe an unexpected but plausible connection between a {topic} and a public library in two sentences.",
        "bounded_prose",
        {"minimum_words": 12, "maximum_words": 80},
    )


def _code_row(split: str, index: int) -> dict[str, Any]:
    names = (
        "clamp",
        "rotate_left",
        "count_vowels",
        "pair_sums",
        "dedupe_ordered",
        "chunk_sizes",
        "median_three",
        "is_palindrome",
        "running_max",
        "word_lengths",
        "invert_pairs",
        "nearest_multiple",
    )
    cases: list[dict[str, Any]]
    prompts = (
        "Write only Python code defining clamp(value, low, high), returning value limited to the inclusive range.",
        "Write only Python code defining rotate_left(items, steps), returning a new list rotated left; handle empty lists.",
        "Write only Python code defining count_vowels(text), counting a, e, i, o, u case-insensitively.",
        "Write only Python code defining pair_sums(items), returning sums of adjacent non-overlapping pairs and preserving a final unpaired item.",
        "Write only Python code defining dedupe_ordered(items), removing duplicates while preserving first occurrence order.",
        "Write only Python code defining chunk_sizes(total, width), returning the sizes of consecutive chunks until total is consumed.",
        "Write only Python code defining median_three(a, b, c), returning the middle value.",
        "Write only Python code defining is_palindrome(text), ignoring case and non-alphanumeric characters.",
        "Write only Python code defining running_max(items), returning the maximum seen at each position.",
        "Write only Python code defining word_lengths(words), mapping each string to its length.",
        "Write only Python code defining invert_pairs(pairs), swapping each two-item pair.",
        "Write only Python code defining nearest_multiple(value, base), returning the nearest multiple with ties rounded upward.",
    )
    all_cases = (
        [([5, 0, 3], 3), ([-2, 0, 3], 0), ([2, 0, 3], 2)],
        [([[1, 2, 3, 4], 1], [2, 3, 4, 1]), ([[], 3], []), ([[1, 2, 3], 4], [2, 3, 1])],
        [(["Aeon"], 3), (["rhythm"], 0), (["Umbrella"], 3)],
        [([[1, 2, 3, 4, 5]], [3, 7, 5]), ([[]], []), ([[9]], [9])],
        [([[1, 2, 1, 3, 2]], [1, 2, 3]), ([[]], []), ([["a", "a"]], ["a"])],
        [([10, 4], [4, 4, 2]), ([0, 3], []), ([5, 8], [5])],
        [([3, 1, 2], 2), ([9, 9, 2], 9), ([-1, -3, -2], -2)],
        [(["A man, a plan, a canal: Panama!"], True), (["OpenAI"], False), ([""], True)],
        [([[3, 1, 4, 2]], [3, 3, 4, 4]), ([[-2, -1]], [-2, -1]), ([[]], [])],
        [([["a", "pear", ""]], [1, 4, 0]), ([[]], []), ([["hello"]], [5])],
        [([[[1, "a"], [2, "b"]]], [["a", 1], ["b", 2]]), ([[]], []), ([[[True, 0]]], [[0, True]])],
        [([14, 5], 15), ([12, 5], 10), ([7, 4], 8)],
    )
    cases = [{"args": args, "expected": expected} for args, expected in all_cases[index]]
    return _row(
        split,
        "code",
        index,
        prompts[index],
        "python_function",
        {"function_name": names[index], "cases": cases},
    )


def _row(
    split: str, category: str, index: int, prompt: str, validator: str, expected: Any
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "prompt_id": f"scbe-{split}-{category}-{index + 1:02d}",
        "split": split,
        "category": category,
        "prompt": prompt,
        "validator": validator,
        "expected": expected,
    }


def _leakage_audit(rows: dict[str, list[dict[str, Any]]], root: Path) -> dict[str, Any]:
    flat = [row for records in rows.values() for row in records]
    normalized = [" ".join(row["prompt"].casefold().split()) for row in flat]
    identifiers = [row["prompt_id"] for row in flat]
    exact_unique = len(normalized) == len(set(normalized)) and len(identifiers) == len(
        set(identifiers)
    )
    fingerprints = {
        row["prompt_id"]: sha256_bytes(text.encode())
        for row, text in zip(flat, normalized, strict=True)
    }
    prior_prompts = []
    for path in sorted(root.parent.rglob("*.jsonl")):
        if root in path.parents:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and isinstance(value.get("prompt"), str):
                prior_prompts.append(" ".join(value["prompt"].casefold().split()))
    prior_exact = set(prior_prompts)
    exact_prior_matches = [
        row["prompt_id"] for row, text in zip(flat, normalized, strict=True) if text in prior_exact
    ]
    maximum_jaccard = 0.0
    closest_pair: list[str] | None = None
    prior_grams = [_token_ngrams(text, 5) for text in prior_prompts]
    for row, text in zip(flat, normalized, strict=True):
        grams = _token_ngrams(text, 5)
        for prior_index, other in enumerate(prior_grams):
            union = grams | other
            similarity = len(grams & other) / len(union) if union else 0.0
            if similarity > maximum_jaccard:
                maximum_jaccard = similarity
                closest_pair = [row["prompt_id"], str(prior_index)]
    passed = exact_unique and not exact_prior_matches and maximum_jaccard < 0.80
    return {
        "schema_version": 1,
        "row_count": len(flat),
        "exact_unique": exact_unique,
        "prior_prompt_count": len(prior_prompts),
        "exact_prior_matches": exact_prior_matches,
        "maximum_prior_5gram_jaccard": maximum_jaccard,
        "maximum_allowed_prior_5gram_jaccard": 0.80,
        "closest_pair": closest_pair,
        "prompt_sha256": fingerprints,
        "passed": passed,
    }


def _token_ngrams(text: str, size: int) -> set[tuple[str, ...]]:
    tokens = text.split()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}
