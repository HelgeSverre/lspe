"""Blinded, deterministic local-Qwen secondary judging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ModelConfig
from .fetch import fetch_model
from .models.factory import create_adapter
from .rng import derive_seed

RATINGS = ("novelty", "usefulness", "coherence", "constraint_adherence", "plausibility")
JUDGE_SYSTEM = (
    "You are a blinded evaluator. Return JSON only, with exactly the five required integer "
    "ratings from 1 (lowest) to 5 (highest). Do not mention hidden experimental conditions."
)


@dataclass(frozen=True)
class JudgeSummary:
    comparisons: int
    parsed: int
    parse_failures: int
    model_revision: str


def judge_run(
    run_dir: Path, judge_model: str, master_seed: int, offline: bool = False
) -> JudgeSummary:
    """Judge baseline/coherent and coherent/white pairs in both blinded orders."""

    generations = _read_jsonl(run_dir / "generations.jsonl")
    prompts = {
        str(row["prompt_id"]): row for row in _read_jsonl(run_dir / "prompts.snapshot.jsonl")
    }
    pairs = _pairs(generations, master_seed)
    model = ModelConfig(adapter="mlx_qwen3", repo_id=judge_model)
    fetched = fetch_model(model, offline=offline)
    runtime_model = model.model_copy(
        update={"revision": fetched.revision, "local_path": fetched.local_path}
    )
    adapter = create_adapter(runtime_model)
    records: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    parsed = 0
    try:
        adapter.load(runtime_model)
        for pair_index, (left, right) in enumerate(pairs):
            prompt = prompts.get(str(left["prompt_id"]))
            if prompt is None:
                raise ValueError(f"Judge pair references unknown prompt: {left['prompt_id']}")
            for variant in (0, 1):
                first, second = (left, right) if variant == 0 else (right, left)
                response = _greedy_judge_response(
                    adapter,
                    str(prompt["prompt"]),
                    str(first.get("output_text", "")),
                    str(second.get("output_text", "")),
                )
                assessment, error = _parse_assessment(response)
                parsed += assessment is not None
                label_a = f"J{pair_index:05d}A"
                label_b = f"J{pair_index:05d}B"
                records.append(
                    {
                        "pair_index": pair_index,
                        "variant": variant,
                        "first_label": label_a,
                        "second_label": label_b,
                        "task_type": prompt["task_type"],
                        "response": response,
                        "ratings": assessment,
                        "parse_failure": error,
                    }
                )
                key.extend(
                    [
                        {
                            "pair_index": pair_index,
                            "variant": variant,
                            "label": label_a,
                            "generation_id": first["generation_id"],
                        },
                        {
                            "pair_index": pair_index,
                            "variant": variant,
                            "label": label_b,
                            "generation_id": second["generation_id"],
                        },
                    ]
                )
    finally:
        adapter.unload()
    (run_dir / "judge.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    (run_dir / "judge-unblinding-key.json").write_text(
        json.dumps({"schema_version": 1, "items": key}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return JudgeSummary(len(records), parsed, len(records) - parsed, fetched.revision)


def _pairs(
    generations: list[dict[str, Any]], master_seed: int
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in generations:
        grouped.setdefault((str(row["prompt_id"]), int(row["generation_index"])), {})[
            str(row["condition"])
        ] = row
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for _, conditions in sorted(grouped.items()):
        for first, second in (("baseline", "coherent"), ("coherent", "white")):
            if first in conditions and second in conditions:
                pairs.append((conditions[first], conditions[second]))
    return sorted(
        pairs,
        key=lambda pair: derive_seed(
            master_seed, "judge-order", pair[0]["generation_id"], pair[1]["generation_id"]
        ),
    )


def _greedy_judge_response(adapter: Any, prompt: str, first: str, second: str) -> str:
    content = (
        "Rate these two candidate responses independently. Return exactly this JSON shape: "
        '{"A":{"novelty":1,"usefulness":1,"coherence":1,"constraint_adherence":1,'
        '"plausibility":1},"B":{"novelty":1,"usefulness":1,"coherence":1,'
        '"constraint_adherence":1,"plausibility":1}}.\n\n'
        f"Task:\n{prompt}\n\nCandidate A:\n{first}\n\nCandidate B:\n{second}"
    )
    token_ids = adapter.format_prompt(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": content}]
    )
    cache = adapter.make_cache()
    if len(token_ids) > 1:
        adapter.forward(token_ids[:-1], cache=cache)
    next_input = [token_ids[-1]]
    output: list[int] = []
    for _ in range(160):
        logits = np.asarray(adapter.forward(next_input, cache=cache).logits)
        token = int(np.argmax(logits.reshape(-1, logits.shape[-1])[-1]))
        output.append(token)
        if token in adapter.eos_token_ids():
            break
        next_input = [token]
    return adapter.decode(output)


def _parse_assessment(text: str) -> tuple[dict[str, dict[str, int]] | None, str | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, "INVALID_JSON"
    if not isinstance(payload, dict) or set(payload) != {"A", "B"}:
        return None, "INVALID_TOP_LEVEL_SCHEMA"
    result: dict[str, dict[str, int]] = {}
    for label in ("A", "B"):
        ratings = payload[label]
        if not isinstance(ratings, dict) or set(ratings) != set(RATINGS):
            return None, "INVALID_RATING_SCHEMA"
        if any(type(value) is not int or not 1 <= value <= 5 for value in ratings.values()):
            return None, "INVALID_RATING_RANGE"
        result[label] = {rating: int(ratings[rating]) for rating in RATINGS}
    return result, None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required judge artifact: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
