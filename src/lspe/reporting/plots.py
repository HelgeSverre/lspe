"""Small, deterministic report plots with adjacent machine-readable tables."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def build_plots(run_dir: Path) -> list[str]:
    effects_path = run_dir / "prompt-effects.jsonl"
    if not effects_path.is_file():
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Report plots require `uv sync --extra analysis`.") from error
    rows = [
        json.loads(line) for line in effects_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if not rows:
        return []
    root = run_dir / "plots"
    root.mkdir(exist_ok=True)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_prompt: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
        by_prompt[str(row["prompt_id"])][str(row["condition"])] = row
    summaries = [
        {
            "condition": condition,
            "vsd_mean": _mean([row.get("vsd") for row in values]),
            "validity_mean": _mean([row.get("validity_rate") for row in values]),
            "degeneration_mean": _mean([row.get("degeneration_rate") for row in values]),
        }
        for condition, values in sorted(by_condition.items())
    ]
    written = []
    for metric, label in (
        ("vsd_mean", "Valid semantic diversity"),
        ("validity_mean", "Deterministic validity"),
        ("degeneration_mean", "Degeneration rate"),
    ):
        filename = metric.removesuffix("_mean")
        _write_table(root / f"{filename}-by-condition.json", summaries)
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar([row["condition"] for row in summaries], [row[metric] or 0.0 for row in summaries])
        axis.set_ylabel(label)
        axis.set_title(f"{label} by condition")
        figure.tight_layout()
        figure.savefig(root / f"{filename}-by-condition.png", dpi=160)
        plt.close(figure)
        written.append(filename)
    h1 = [
        float(values["coherent"]["vsd"]) - float(values["temp_match"]["vsd"])
        for values in by_prompt.values()
        if values.get("coherent", {}).get("vsd") is not None
        and values.get("temp_match", {}).get("vsd") is not None
    ]
    if h1:
        _write_table(
            root / "h1-prompt-effects.json",
            [{"coherent_minus_temp_match_vsd": value} for value in h1],
        )
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.hist(h1, bins=min(20, max(5, len(h1))))
        axis.axvline(0, color="black", linewidth=1)
        axis.set_title("Coherent minus entropy-matched VSD by prompt")
        figure.tight_layout()
        figure.savefig(root / "h1-prompt-effects.png", dpi=160)
        plt.close(figure)
        written.append("h1-prompt-effects")
    return written


def _mean(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
