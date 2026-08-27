"""Deterministic version-1 LSPE prompt corpus builder.

The corpus is generated from explicitly listed prompt ingredients so the
repository can reproduce its JSONL files without a network service or a
language-model-generated dataset.  Phase tags make the smaller smoke/pilot
subsets strict subsets of the separately held-out confirmatory corpus only
where the protocol explicitly permits that (smoke); pilot and confirm IDs
remain disjoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def build_default_datasets(root: Path, *, force: bool = False) -> dict[str, int]:
    """Write the checked-in corpus, refusing accidental replacement by default."""

    paths = {
        "calibration": root / "calibration.jsonl",
        "pilot": root / "pilot.jsonl",
        "confirm": root / "confirm.jsonl",
        "controls": root / "controls.jsonl",
    }
    if not force:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError("Refusing to replace existing datasets: " + ", ".join(existing))
    records = {
        "calibration": _calibration_records(),
        "pilot": _creative_records("pilot", 6, ("pilot",)),
        "confirm": _creative_records("confirm", 20, ("confirm",)),
        "controls": _control_records(),
    }
    root.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        paths[split].write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
    return {split: len(rows) for split, rows in records.items()}


def _record(
    prompt_id: str,
    split: str,
    task_type: str,
    prompt: str,
    response_schema: str,
    validator: str,
    expected: Any,
    tags: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_id": prompt_id,
        "split": split,
        "task_type": task_type,
        "system_variant": "neutral",
        "prompt": prompt,
        "response_schema": response_schema,
        "validator": validator,
        "expected": expected,
        "tags": list(tags),
    }


def _creative_records(prefix: str, count: int, phase_tags: tuple[str, ...]) -> list[dict[str, Any]]:
    subjects = [
        "harbor",
        "forest",
        "library",
        "workshop",
        "hospital",
        "school",
        "market",
        "museum",
        "garden",
        "theater",
        "kitchen",
        "train",
        "river",
        "mountain",
        "laboratory",
        "stadium",
        "bakery",
        "observatory",
        "farm",
        "archive",
    ][:count]
    objects = [
        "brick",
        "paperclip",
        "glass jar",
        "wooden spoon",
        "rubber band",
        "cardboard box",
        "bicycle wheel",
        "ceramic mug",
        "rope",
        "cork",
        "clothespin",
        "tin can",
        "ladder",
        "mirror",
        "chalk",
        "bucket",
        "umbrella",
        "magnet",
        "map",
        "spring",
    ][:count]
    domains = [
        ("beekeeping", "traffic engineering"),
        ("origami", "data backup"),
        ("theater lighting", "emergency triage"),
        ("gardening", "software testing"),
        ("sailing", "classroom scheduling"),
        ("bread baking", "battery management"),
        ("cartography", "music education"),
        ("bird migration", "warehouse routing"),
        ("weaving", "network security"),
        ("pottery", "team retrospectives"),
        ("meteorology", "restaurant service"),
        ("chess", "waste collection"),
        ("astronomy", "public transit"),
        ("woodworking", "clinical handoffs"),
        ("improvisation", "scientific peer review"),
        ("rowing", "project planning"),
        ("fishing", "inventory control"),
        ("calligraphy", "interface design"),
        ("composting", "risk management"),
        ("photography", "water conservation"),
    ][:count]
    required_words = [
        ("lantern", "tidal"),
        ("copper", "whisper"),
        ("maple", "signal"),
        ("window", "ember"),
        ("anchor", "violet"),
        ("ribbon", "north"),
        ("pebble", "echo"),
        ("meadow", "clock"),
        ("cedar", "orbit"),
        ("quartz", "thread"),
        ("harvest", "mirror"),
        ("saffron", "bridge"),
        ("frost", "compass"),
        ("thimble", "rain"),
        ("coral", "paper"),
        ("willow", "engine"),
        ("marble", "harbor"),
        ("feather", "valley"),
        ("candle", "island"),
        ("moss", "window"),
    ][:count]
    rows: list[dict[str, Any]] = []
    for index, subject in enumerate(subjects, 1):
        tags = (
            *phase_tags,
            "creativity",
            *(("smoke",) if index <= 4 and prefix == "pilot" else ()),
            *(("replicate",) if index <= 10 and prefix == "confirm" else ()),
        )
        rows.append(
            _record(
                f"{prefix}-words-{index:03d}",
                prefix,
                "divergent_words",
                "Return exactly ten mutually unrelated common nouns associated "
                f"with a {subject} as a JSON array.",
                "divergent_words.v1",
                "divergent_words",
                None,
                tags,
            )
        )
    for index, obj in enumerate(objects, 1):
        tags = (
            *phase_tags,
            "creativity",
            *(("replicate",) if index <= 10 and prefix == "confirm" else ()),
        )
        rows.append(
            _record(
                f"{prefix}-uses-{index:03d}",
                prefix,
                "alternative_uses",
                f"Give unusual physically plausible uses for a {obj} as a JSON array. "
                "Each item needs idea, mechanism, and feasibility.",
                "alternative_uses.v1",
                "alternative_uses",
                None,
                tags,
            )
        )
    for index, (source, target) in enumerate(domains, 1):
        tags = (
            *phase_tags,
            "creativity",
            *(("replicate",) if index <= 10 and prefix == "confirm" else ()),
        )
        rows.append(
            _record(
                f"{prefix}-bridge-{index:03d}",
                prefix,
                "cross_domain_bridge",
                "Return JSON only: propose one concrete mechanism that transfers "
                f"a principle from {source} to {target}.",
                "cross_domain_bridge.v1",
                "cross_domain_bridge",
                None,
                tags,
            )
        )
    for index, words in enumerate(required_words, 1):
        tags = (
            *phase_tags,
            "creativity",
            *(("replicate",) if index <= 10 and prefix == "confirm" else ()),
        )
        rows.append(
            _record(
                f"{prefix}-constraint-{index:03d}",
                prefix,
                "constrained_creative",
                "Write exactly four lines, each 6 to 14 words, and include the words "
                f"{words[0]} and {words[1]} somewhere.",
                "constrained_creative.v1",
                "constrained_creative",
                {"required_words": list(words)},
                tags,
            )
        )
    return rows


def _control_records() -> list[dict[str, Any]]:
    arithmetic = [(17 + n, 6 + (n % 7)) for n in range(20)]
    rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(arithmetic, 1):
        tags = (
            "confirm",
            *(("replicate",) if index <= 10 else ()),
            *(("pilot",) if index <= 8 else ()),
            *(("smoke",) if index <= 4 else ()),
            "control",
        )
        rows.append(
            _record(
                f"control-arithmetic-{index:03d}",
                "controls",
                "arithmetic",
                f"Return only the integer result of {left} * {right}.",
                "text.v1",
                "exact_answer",
                str(left * right),
                tags,
            )
        )
    for index in range(1, 6):
        tags = (
            "confirm",
            *(("replicate",) if index <= 2 else ()),
            *(("pilot",) if index <= 2 else ()),
            "control",
        )
        rows.append(
            _record(
                f"control-json-{index:03d}",
                "controls",
                "json_schema",
                f"Return JSON only with one key named answer and integer value {index}.",
                "json.v1",
                "json",
                {"answer": index},
                tags,
            )
        )
    for index, words in enumerate(
        (
            ("river", "stone"),
            ("cloud", "wheel"),
            ("book", "light"),
            ("field", "bell"),
            ("glass", "seed"),
            ("road", "leaf"),
            ("moon", "thread"),
            ("salt", "gate"),
            ("bird", "iron"),
            ("fire", "paper"),
        ),
        1,
    ):
        if index > 5:
            break
        tags = (
            "confirm",
            *(("replicate",) if index <= 2 else ()),
            *(("pilot",) if index <= 2 else ()),
            "control",
        )
        rows.append(
            _record(
                f"control-constraint-{index:03d}",
                "controls",
                "constraint",
                "Write exactly four lines, each 6 to 14 words, and include "
                f"{words[0]} and {words[1]}.",
                "constrained_creative.v1",
                "constrained_creative",
                {"required_words": list(words)},
                tags,
            )
        )
    code_cases = [
        ("add_one", [{"args": [1], "expected": 2}, {"args": [-3], "expected": -2}]),
        ("double", [{"args": [2], "expected": 4}, {"args": [-4], "expected": -8}]),
        ("is_even", [{"args": [2], "expected": True}, {"args": [3], "expected": False}]),
        ("square", [{"args": [5], "expected": 25}, {"args": [-2], "expected": 4}]),
        ("clamp_zero", [{"args": [-1], "expected": 0}, {"args": [4], "expected": 4}]),
        ("first_char", [{"args": ["fjord"], "expected": "f"}, {"args": ["oak"], "expected": "o"}]),
        ("count_letters", [{"args": ["fjord"], "expected": 5}, {"args": [""], "expected": 0}]),
        ("max_pair", [{"args": [2, 7], "expected": 7}, {"args": [-1, -3], "expected": -1}]),
        ("abs_difference", [{"args": [7, 2], "expected": 5}, {"args": [2, 7], "expected": 5}]),
        ("join_dash", [{"args": ["north", "star"], "expected": "north-star"}]),
    ]
    for index, (function_name, cases) in enumerate(code_cases, 1):
        tags = (
            "confirm",
            *(("replicate",) if index <= 6 else ()),
            *(("pilot",) if index <= 4 else ()),
            "control",
        )
        rows.append(
            _record(
                f"control-code-{index:03d}",
                "controls",
                "python_function",
                "Return Python source code only. Define exactly one pure function named "
                f"`{function_name}`. Do not import modules or read/write files.",
                "python.v1",
                "python_function",
                {"function_name": function_name, "cases": cases},
                tags,
            )
        )
    return rows


def _calibration_records() -> list[dict[str, Any]]:
    questions = [
        "Return exactly ten unrelated common nouns as a JSON array.",
        "Give unusual physically plausible uses for a cardboard tube as a JSON array.",
        "Return JSON only: transfer a gardening principle to queue management.",
        "Write exactly four lines containing lantern and river.",
        "Return only the integer result of 17 * 6.",
        "Return JSON only with one key named answer and integer value 5.",
        "Return exactly ten unrelated common nouns as a JSON array.",
        "Give unusual physically plausible uses for a cork as a JSON array.",
        "Return JSON only: transfer a weaving principle to data visualization.",
        "Write exactly four lines containing compass and rain.",
        "Return only the integer result of 19 + 23.",
        "Return JSON only with one key named answer and integer value 8.",
    ]
    validators = [
        "divergent_words",
        "alternative_uses",
        "cross_domain_bridge",
        "constrained_creative",
        "exact_answer",
        "json",
    ] * 2
    expected = [
        None,
        None,
        None,
        {"required_words": ["lantern", "river"]},
        "102",
        {"answer": 5},
        None,
        None,
        None,
        {"required_words": ["compass", "rain"]},
        "42",
        {"answer": 8},
    ]
    return [
        _record(
            f"calibration-{index:03d}",
            "calibration",
            "calibration",
            question,
            "text.v1",
            validator,
            value,
            ("calibration",),
        )
        for index, (question, validator, value) in enumerate(
            zip(questions, validators, expected, strict=True), 1
        )
    ]
