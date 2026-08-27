"""Deterministic validation plus independent local embedding scoring."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .columnar import write_parquet
from .metrics.degeneration import degeneration_metrics
from .metrics.deterministic import valid_semantic_diversity
from .tasks.loader import PromptRecord
from .tasks.validators import validate_response


@dataclass(frozen=True)
class ScoringSummary:
    rows: int
    prompt_condition_scores: int
    embedding_model: str


def score_run(
    run_dir: Path, embedding_model: str, embedding_revision: str | None = None
) -> ScoringSummary:
    """Score immutable raw generations after the subject model has been unloaded."""

    prompts = {
        record.prompt_id: record
        for record in _load_prompt_snapshot(run_dir / "prompts.snapshot.jsonl")
    }
    generations = _load_jsonl(run_dir / "generations.jsonl")
    texts = [str(row.get("output_text", "")) for row in generations]
    embeddings = _embed_texts(texts, embedding_model, embedding_revision)
    individual: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[tuple[bool, np.ndarray]]] = defaultdict(list)
    grouped_degradation: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row, embedding in zip(generations, embeddings, strict=True):
        prompt = prompts[row["prompt_id"]]
        validation = validate_response(
            prompt.validator, str(row.get("output_text", "")), prompt.expected
        )
        metrics = degeneration_metrics(row.get("output_token_ids", []))
        individual_row = {
            "generation_id": row["generation_id"],
            "prompt_id": prompt.prompt_id,
            "condition": row["condition"],
            "valid": validation.valid,
            "failure_code": validation.failure_code,
            "response_length": len(str(row.get("output_text", ""))),
            **metrics,
        }
        individual.append(individual_row)
        grouped[(prompt.prompt_id, row["condition"])].append((validation.valid, embedding))
        grouped_degradation[(prompt.prompt_id, row["condition"])].append(
            float(metrics["repeated_4gram_ratio"] > 0 or metrics["max_identical_run"] >= 8)
        )
    prompt_scores: list[dict[str, Any]] = []
    for (prompt_id, condition), rows in sorted(grouped.items()):
        prompt = prompts[prompt_id]
        valid, vectors = zip(*rows, strict=True)
        vsd = valid_semantic_diversity(list(valid), np.stack(vectors)) if len(rows) >= 2 else None
        prompt_scores.append(
            {
                "prompt_id": prompt_id,
                "split": prompt.split,
                "task_type": prompt.task_type,
                "tags": list(prompt.tags),
                "condition": condition,
                "n_generations": len(rows),
                "validity_rate": float(np.mean(valid)),
                "vsd": vsd,
                "degeneration_rate": float(np.mean(grouped_degradation[(prompt_id, condition)])),
            }
        )
    (run_dir / "scores.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in individual), encoding="utf-8"
    )
    (run_dir / "prompt-effects.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prompt_scores), encoding="utf-8"
    )
    write_parquet(run_dir / "scores.parquet", individual)
    write_parquet(run_dir / "prompt-effects.parquet", prompt_scores)
    return ScoringSummary(len(individual), len(prompt_scores), embedding_model)


def _embed_texts(texts: list[str], model_name: str, revision: str | None) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("Embedding scoring requires `uv sync --extra analysis`.") from error
    model = SentenceTransformer(model_name, revision=revision, trust_remote_code=False)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise RuntimeError("Embedding model returned non-finite or malformed vectors")
    return values


def _load_prompt_snapshot(path: Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for row in _load_jsonl(path):
        records.append(PromptRecord.model_validate(row))
    return records


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"Artifact contains no records: {path}")
    return rows
