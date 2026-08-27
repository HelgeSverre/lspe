"""Condition-blinded, deterministic human-review exports."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .rng import derive_seed


def export_human_review(run_dir: Path, master_seed: int, sample: int = 80) -> dict[str, int]:
    """Write a blinded stratified sample and a separately stored unblinding key."""

    if sample < 1:
        raise ValueError("Human-review sample must be positive")
    prompts = {
        str(row["prompt_id"]): row for row in _read_jsonl(run_dir / "prompts.snapshot.jsonl")
    }
    generations = _read_jsonl(run_dir / "generations.jsonl")
    if not generations:
        raise ValueError("Cannot export human review from an empty generation run")
    chosen = _stratified_sample(generations, prompts, sample, master_seed)
    labels = sorted(
        chosen,
        key=lambda row: derive_seed(master_seed, "human-review-order", row["generation_id"]),
    )
    blinded: list[dict[str, Any]] = []
    key: list[dict[str, str]] = []
    for index, row in enumerate(labels, 1):
        prompt = prompts[str(row["prompt_id"])]
        label = f"R{index:04d}"
        blinded.append(
            {
                "review_id": label,
                "prompt_id": row["prompt_id"],
                "task_type": prompt["task_type"],
                "prompt": prompt["prompt"],
                "response": row.get("output_text", ""),
                "stop_reason": row.get("stop_reason"),
                "validator": row.get("validator"),
            }
        )
        key.append(
            {
                "review_id": label,
                "generation_id": str(row["generation_id"]),
                "condition": str(row["condition"]),
            }
        )
    root = run_dir / "human-review"
    root.mkdir(exist_ok=True)
    (root / "review.json").write_text(
        json.dumps({"schema_version": 1, "items": blinded}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "unblinding-key.json").write_text(
        json.dumps({"schema_version": 1, "items": key}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "index.html").write_text(_render_html(blinded), encoding="utf-8")
    return {"sampled": len(blinded), "available": len(generations)}


def _stratified_sample(
    rows: list[dict[str, Any]], prompts: dict[str, dict[str, Any]], sample: int, master_seed: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        prompt = prompts.get(str(row["prompt_id"]))
        if prompt is None:
            raise ValueError(f"Generation references unknown prompt: {row['prompt_id']}")
        groups.setdefault((str(prompt["task_type"]), str(row["condition"])), []).append(row)
    selected = [
        min(
            group,
            key=lambda row: derive_seed(
                master_seed, "human-review-order", "stratum", row["generation_id"]
            ),
        )
        for _, group in sorted(groups.items())
    ]
    remaining = [row for row in rows if row not in selected]
    remaining.sort(
        key=lambda row: derive_seed(
            master_seed, "human-review-order", "remaining", row["generation_id"]
        )
    )
    return selected + remaining[: max(0, sample - len(selected))]


def _render_html(items: list[dict[str, Any]]) -> str:
    cards = "\n".join(
        "<article><h2>{}</h2><p><em>{}</em></p><h3>Prompt</h3><pre>{}</pre>"
        "<h3>Response</h3><pre>{}</pre></article>".format(
            html.escape(str(item["review_id"])),
            html.escape(str(item["task_type"])),
            html.escape(str(item["prompt"])),
            html.escape(str(item["response"])),
        )
        for item in items
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>LSPE blinded human review</title>"
        "<style>body{font-family:system-ui;max-width:70rem;margin:2rem auto}"
        "pre{white-space:pre-wrap;background:#f5f5f5;padding:1rem}</style></head><body>"
        "<h1>Condition-blinded LSPE review</h1>"
        "<p>Do not inspect the separately stored unblinding key while rating responses.</p>"
        f"{cards}</body></html>\n"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required human-review artifact: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
