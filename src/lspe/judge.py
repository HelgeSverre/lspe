"""Blinded, deterministic local-Qwen secondary judging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ModelConfig
from .fetch import fetch_model
from .hashing import sha256_file
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


def reparse_judge_run(
    run_dir: Path, judge_model: str, offline: bool = False
) -> JudgeSummary:
    """Reparse already-generated blinded responses after a parser-only repair.

    This recovery path deliberately never calls the judge model.  It preserves
    the response text, labels, pair order, and unblinding key from the original
    run, changing only the deterministic ``ratings`` and ``parse_failure``
    fields derived from that text.
    """

    records = _read_jsonl(run_dir / "judge.jsonl")
    repaired = _reparse_records(records)
    model = ModelConfig(adapter="mlx_qwen3", repo_id=judge_model)
    fetched = fetch_model(model, offline=offline)
    (run_dir / "judge.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in repaired), encoding="utf-8"
    )
    parsed = sum(record["ratings"] is not None for record in repaired)
    return JudgeSummary(len(repaired), parsed, len(repaired) - parsed, fetched.revision)


def judge_behavioral_run(
    run_dir: Path,
    data_path: Path,
    judge_model: str,
    master_seed: int,
    offline: bool = False,
) -> JudgeSummary:
    """Judge SCBE SCCF/control pairs in both orders with hidden conditions."""

    generations = _read_jsonl(run_dir / "confirm-generations.jsonl")
    prompts = {row["prompt_id"]: row for row in _read_jsonl(data_path)}
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in generations:
        if row["category"] not in {"open_association", "analogical", "narrative"}:
            continue
        grouped.setdefault((row["prompt_id"], int(row["generation_index"])), {})[
            row["condition"]
        ] = row
    work = []
    for (prompt_id, generation_index), conditions in grouped.items():
        for comparison in ("baseline", "temp_match", "random_basis", "attn_noise"):
            work.append(
                (
                    prompt_id,
                    generation_index,
                    comparison,
                    conditions["sccf"],
                    conditions[comparison],
                )
            )
    work.sort(key=lambda item: derive_seed(master_seed, "judge-order", item[0], item[1], item[2]))
    model = ModelConfig(adapter="mlx_qwen3", repo_id=judge_model)
    fetched = fetch_model(model, offline=offline)
    runtime = model.model_copy(
        update={"revision": fetched.revision, "local_path": fetched.local_path}
    )
    adapter = create_adapter(runtime)
    output = run_dir / "behavioral-judge.jsonl"
    existing = _read_jsonl(output) if output.exists() else []
    completed = {
        (row["prompt_id"], row["generation_index"], row["comparison"], row["variant"])
        for row in existing
    }
    parsed = sum(row.get("ratings") is not None for row in existing)
    try:
        adapter.load(runtime)
        for prompt_id, generation_index, comparison, sccf, control in work:
            for variant in (0, 1):
                identity = (prompt_id, generation_index, comparison, variant)
                if identity in completed:
                    continue
                first, second = (sccf, control) if variant == 0 else (control, sccf)
                response = _greedy_judge_response(
                    adapter,
                    str(prompts[prompt_id]["prompt"]),
                    str(first["output_text"]),
                    str(second["output_text"]),
                )
                assessment, error = _parse_assessment(response)
                normalized = None
                if assessment is not None:
                    normalized = (
                        {"sccf": assessment["A"], "control": assessment["B"]}
                        if variant == 0
                        else {"sccf": assessment["B"], "control": assessment["A"]}
                    )
                    parsed += 1
                row = {
                    "schema_version": 1,
                    "prompt_id": prompt_id,
                    "category": prompts[prompt_id]["category"],
                    "generation_index": generation_index,
                    "comparison": comparison,
                    "variant": variant,
                    "response": response,
                    "ratings": normalized,
                    "parse_failure": error,
                }
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                existing.append(row)
                print(
                    json.dumps(
                        {"event": "scbe_judge", "complete": len(existing), "total": len(work) * 2}
                    )
                )
    finally:
        adapter.unload()
    summary = JudgeSummary(len(existing), parsed, len(existing) - parsed, fetched.revision)
    (run_dir / "behavioral-judge-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_revision": fetched.revision,
                "data_sha256": sha256_file(data_path),
                "generation_sha256": sha256_file(
                    run_dir / "confirm-generations.jsonl"
                ),
                "summary": asdict(summary),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _refresh_run_checksums(run_dir)
    return summary


def _refresh_run_checksums(run_dir: Path) -> None:
    entries = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "checksums.sha256"
    }
    (run_dir / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in entries.items())
    )


def _reparse_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return copies whose parse fields are regenerated from immutable text."""

    repaired: list[dict[str, Any]] = []
    for record in records:
        updated = dict(record)
        assessment, error = _parse_assessment(str(updated.get("response", "")))
        updated["ratings"] = assessment
        updated["parse_failure"] = error
        repaired.append(updated)
    return repaired


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
    # Qwen chat templates can decode their assistant terminator even when the
    # token is also configured as EOS.  It is transport framing, not part of
    # the requested JSON payload.  Strip only recognised *trailing* markers;
    # arbitrary prose or malformed JSON must still fail closed below.
    normalized = text.strip()
    for terminator in ("<|im_end|>", "<|endoftext|>"):
        if normalized.endswith(terminator):
            normalized = normalized[: -len(terminator)].rstrip()
    try:
        payload = json.loads(normalized)
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
