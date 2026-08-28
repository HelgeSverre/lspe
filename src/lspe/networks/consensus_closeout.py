"""Corrected held-out evaluation and closeout for FNDE v2 consensus mapping."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..hashing import sha256_file
from .communities import adjusted_rand_index, density_threshold, spectral_communities
from .consensus_runner import (
    ConsensusMappingProtocol,
    _category_rows,
    _cka_adjacency,
    _select_candidate,
)
from .mapping_closeout import verify_mapping_checksums
from .mapping_data_v2 import load_network_map_v2_dataset
from .mapping_runner import _stable_nonisolated_nodes


def reevaluate_consensus_heldout(
    run_dir: Path, data_path: Path
) -> dict[str, Any]:
    """Recompute the frozen selection with defined held-out non-isolated coverage."""

    protocol = ConsensusMappingProtocol()
    normalized = np.load(run_dir / "component-activity-normalized.npy", mmap_mode="r")
    prompts = load_network_map_v2_dataset(data_path)
    continuations = [
        json.loads(line)
        for line in (run_dir / "fixed-continuations.jsonl").read_text().splitlines()
        if line
    ]
    _, fold_rows = _category_rows(prompts, continuations)
    categories = sorted(fold_rows)
    fold_graphs = []
    for fold in range(4):
        fold_graphs.append(
            np.mean(
                [
                    _cka_adjacency(normalized, fold_rows[category][fold])
                    for category in categories
                ],
                axis=0,
            )
        )
        print(json.dumps({"event": "v2_heldout_fold", "complete": fold + 1, "total": 4}))
    candidates: list[dict[str, Any]] = []
    for density in protocol.graph_densities:
        graphs = [density_threshold(graph, density) for graph in fold_graphs]
        tuning_nodes = _stable_nonisolated_nodes([graphs[0], graphs[2]])
        tuning_graphs = [graph[np.ix_(tuning_nodes, tuning_nodes)] for graph in graphs]
        heldout_positions = _stable_nonisolated_nodes([tuning_graphs[1], tuning_graphs[3]])
        for count in protocol.community_counts:
            tuning_labels = [
                spectral_communities(graph, count, seed=protocol.master_seed + count)
                for graph in (tuning_graphs[0], tuning_graphs[2])
            ]
            heldout_labels = [
                spectral_communities(
                    graph[np.ix_(heldout_positions, heldout_positions)],
                    count,
                    seed=protocol.master_seed + count,
                )
                for graph in (tuning_graphs[1], tuning_graphs[3])
            ]
            candidates.append(
                {
                    "density": density,
                    "community_count": count,
                    "eligible_nodes": len(tuning_nodes),
                    "tuning_ari": adjusted_rand_index(*tuning_labels),
                    "heldout_ari": adjusted_rand_index(*heldout_labels),
                    "heldout_coverage": len(heldout_positions) / len(tuning_nodes),
                }
            )
    selected = _select_candidate(candidates)
    original = json.loads((run_dir / "communities.json").read_text())
    original_selected = original["selected_on_tuning_only"]
    same_selection = (
        selected["density"] == original_selected["density"]
        and selected["community_count"] == original_selected["community_count"]
        and np.isclose(
            selected["tuning_ari"], original_selected["tuning_ari"], atol=1e-12
        )
    )
    if not same_selection:
        raise RuntimeError("Corrected held-out evaluation changed tuning-only selection")
    sizes = Counter(int(label) for label in original["labels"].values())
    minimum_fraction = min(sizes.values()) / sum(sizes.values())
    gates = {
        "heldout_ari": selected["heldout_ari"] >= protocol.minimum_split_half_ari,
        "heldout_coverage": selected["heldout_coverage"]
        >= protocol.minimum_heldout_coverage,
        "community_balance": minimum_fraction >= protocol.minimum_community_fraction,
    }
    result = {
        "schema_version": 1,
        "source_commit": _source_commit(),
        "selection_unchanged": True,
        "selected_on_tuning_only": selected,
        "community_sizes": dict(sorted(sizes.items())),
        "minimum_community_fraction": minimum_fraction,
        "gates": gates,
        "passed": all(gates.values()),
        "decision": "CONTINUE_CAUSAL_SCREENING" if all(gates.values()) else "STOP_MAPPING_UNSTABLE",
        "candidates": candidates,
    }
    (run_dir / "v2-heldout-reevaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_checksums(run_dir)
    return result


def close_out_consensus_v2(run_dir: Path, data_path: Path) -> dict[str, Any]:
    """Verify and report the stronger v2 map's terminal gate result."""

    manifest = json.loads((run_dir / "manifest.json").read_text())
    original = json.loads((run_dir / "communities.json").read_text())
    corrected = json.loads((run_dir / "v2-heldout-reevaluation.json").read_text())
    raw = np.load(run_dir / "component-activity.npy", mmap_mode="r")
    patterns = np.load(run_dir / "attention-patterns.npy", mmap_mode="r")
    grams = np.load(run_dir / "task-cka-grams.npy", mmap_mode="r")
    checks = {
        "data_hash": sha256_file(data_path) == manifest["network_map_v2_sha256"],
        "activity_shape": raw.shape == (1152, 960, 2560),
        "attention_shape": patterns.shape == (1152, 960, 16),
        "task_cka_shape": grams.shape == (6, 1152, 25_600),
        "finite_activity_sample": bool(np.isfinite(raw[:, :2, :]).all()),
        "finite_attention": bool(np.isfinite(patterns).all()),
        "selection_unchanged": corrected["selection_unchanged"] is True,
        "heldout_gate_failed": corrected["gates"]["heldout_ari"] is False,
        "community_balance_failed": corrected["gates"]["community_balance"] is False,
        "stop_decision": corrected["decision"] == "STOP_MAPPING_UNSTABLE",
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"Invalid v2 closeout checks: {failed}")
    selected = corrected["selected_on_tuning_only"]
    report = {
        "schema_version": 1,
        "status": "MECHANISM_NOT_ACHIEVED",
        "stage_reached": "task_balanced_consensus_mapping",
        "later_stages_executed": False,
        "selected_density": selected["density"],
        "selected_community_count": selected["community_count"],
        "tuning_ari": selected["tuning_ari"],
        "heldout_ari": selected["heldout_ari"],
        "heldout_coverage": selected["heldout_coverage"],
        "community_sizes": corrected["community_sizes"],
        "minimum_community_fraction": corrected["minimum_community_fraction"],
        "median_assignment_probability": original["median_assignment_probability"],
        "pair_similarity": original["pair_similarity"],
        "verification": checks,
        "verified": True,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    (run_dir / "verification.json").write_text(
        json.dumps({"schema_version": 1, "checks": checks, "passed": True}, indent=2) + "\n"
    )
    _refresh_checksums(run_dir)
    if not verify_mapping_checksums(run_dir):
        raise RuntimeError("FNDE v2 checksum verification failed after closeout")
    return report


def _markdown(report: dict[str, Any]) -> str:
    return f"""# FNDE v2: stronger map, same stopping point

**Status: `{report['status']}`**

V2 used 240 fresh prompts, two-token greedy and sampled trajectories, 960 matched generated
positions, all 1,152 heads, and six equally weighted task-family graphs.

The candidate chosen without held-out outcomes used density `{report['selected_density']}` and
{report['selected_community_count']} communities. Tuning ARI was `{report['tuning_ari']:.3f}`.
On {100 * report['heldout_coverage']:.1f}% of the same heads that remained evaluable in both
untouched folds, held-out ARI was
`{report['heldout_ari']:.3f}` against the unchanged `0.700` requirement.

The final community sizes were {list(report['community_sizes'].values())}. The smallest contained
only `{100 * report['minimum_community_fraction']:.1f}%` of eligible heads, below the 5% balance
gate. Causal screening and CCAD were therefore not run.
"""


def _refresh_checksums(run_dir: Path) -> None:
    entries = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "checksums.sha256"
    }
    (run_dir / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in entries.items())
    )


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
