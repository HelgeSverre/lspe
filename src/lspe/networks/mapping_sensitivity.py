"""Nested mapping-only density sensitivity after the primary graph gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..hashing import sha256_file
from .communities import adjusted_rand_index, density_threshold, spectral_communities
from .mapping_runner import _cka_adjacency, _stable_nonisolated_nodes

DEFAULT_DENSITIES = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20)


def run_nested_mapping_sensitivity(run_dir: Path) -> dict[str, Any]:
    """Select on mapping folds 0/2 and evaluate once on untouched folds 1/3."""

    manifest = json.loads((run_dir / "manifest.json").read_text())
    protocol = manifest["protocol"]
    normalized = np.load(run_dir / "component-activity-normalized.npy", mmap_mode="r")
    grams = np.load(run_dir / "cka-grams.npy", mmap_mode="r")
    row_count = normalized.shape[1]
    if row_count % 8:
        raise ValueError("Nested mapping sensitivity requires paired rows divisible by eight")
    full = np.asarray(grams @ grams.T, dtype=np.float64)
    np.fill_diagonal(full, 0.0)
    prompt_rows = np.arange(row_count).reshape(-1, 2)
    fold_graphs = [
        _cka_adjacency(normalized, prompt_rows[fold::4].reshape(-1)) for fold in range(4)
    ]
    candidates: list[dict[str, Any]] = []
    for density in DEFAULT_DENSITIES:
        graphs = [density_threshold(graph, density) for graph in [full, *fold_graphs]]
        eligible = _stable_nonisolated_nodes(graphs)
        subgraphs = [graph[np.ix_(eligible, eligible)] for graph in graphs]
        for count in protocol["community_counts"]:
            labels = [
                spectral_communities(
                    graph, count, seed=int(protocol["master_seed"]) + count
                )
                for graph in subgraphs[1:]
            ]
            candidates.append(
                {
                    "density": density,
                    "community_count": count,
                    "eligible_nodes": len(eligible),
                    "tuning_ari": adjusted_rand_index(labels[0], labels[2]),
                    "heldout_ari": adjusted_rand_index(labels[1], labels[3]),
                }
            )
    selected = select_nested_candidate(candidates)
    threshold = float(protocol["minimum_split_half_ari"])
    result = {
        "schema_version": 1,
        "purpose": "mapping_only_feasibility_and_falsification",
        "selection_folds": [0, 2],
        "heldout_folds": [1, 3],
        "densities": list(DEFAULT_DENSITIES),
        "candidates": candidates,
        "selected_on_tuning_only": selected,
        "required_ari": threshold,
        "tuning_gate_passed": selected["tuning_ari"] >= threshold,
        "heldout_gate_passed": selected["heldout_ari"] >= threshold,
        "passed": selected["tuning_ari"] >= threshold and selected["heldout_ari"] >= threshold,
        "decision": "CONTINUE" if selected["heldout_ari"] >= threshold else "STOP_MAPPING_UNSTABLE",
    }
    path = run_dir / "mapping-feasibility.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _refresh_checksums(run_dir)
    return result


def select_nested_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select without looking at held-out ARI."""

    if not candidates:
        raise ValueError("At least one mapping sensitivity candidate is required")
    return max(
        candidates,
        key=lambda row: (
            row["tuning_ari"],
            row["eligible_nodes"],
            -row["density"],
            -row["community_count"],
        ),
    )


def _refresh_checksums(run_dir: Path) -> None:
    checksums = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "checksums.sha256"
    }
    (run_dir / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items())
    )
