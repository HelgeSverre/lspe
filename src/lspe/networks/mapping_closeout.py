"""Verification and honest early-stop reporting for the FNDE mapping gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..hashing import sha256_file
from .mapping_data import network_map_hash


def close_out_mapping_failure(run_dir: Path, data_path: Path) -> dict[str, Any]:
    """Verify the stopped run and write machine-readable and human-readable reports."""

    manifest = json.loads((run_dir / "manifest.json").read_text())
    communities = json.loads((run_dir / "communities.json").read_text())
    sensitivity = json.loads((run_dir / "mapping-feasibility.json").read_text())
    geometry = manifest["geometry"]
    protocol = manifest["protocol"]
    node_count = len(protocol["selected_layers"]) * geometry["attention_heads"]
    continuation_rows = sum(
        1 for line in (run_dir / "fixed-continuations.jsonl").read_text().splitlines() if line
    )
    raw = np.load(run_dir / "component-activity.npy", mmap_mode="r")
    patterns = np.load(run_dir / "attention-patterns.npy", mmap_mode="r")
    grams = np.load(run_dir / "cka-grams.npy", mmap_mode="r")
    checks = {
        "mapping_data_hash": network_map_hash(data_path) == manifest["network_map_sha256"],
        "continuation_rows": continuation_rows == 400,
        "activity_shape": raw.shape
        == (node_count, continuation_rows, geometry["hidden_width"]),
        "attention_shape": patterns.shape
        == (node_count, continuation_rows, protocol["attention_bins"]),
        "cka_shape": grams.shape == (node_count, continuation_rows * continuation_rows),
        "finite_activity_sample": bool(np.isfinite(raw[:, :2, :]).all()),
        "finite_attention": bool(np.isfinite(patterns).all()),
        "primary_mapping_gate_failed": communities["passed"] is False,
        "nested_heldout_gate_failed": sensitivity["heldout_gate_passed"] is False,
        "nested_stop_decision": sensitivity["decision"] == "STOP_MAPPING_UNSTABLE",
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Cannot close out invalid mapping run: {failed}")
    result = {
        "schema_version": 1,
        "status": "MECHANISM_NOT_ACHIEVED",
        "stage_reached": "functional_mapping",
        "later_stages_executed": False,
        "stop_rule": "split-half adjusted Rand index must be at least 0.70",
        "primary_split_half_ari": communities["selected_statistics"]["split_half_ari"],
        "primary_density": protocol["graph_density"],
        "primary_communities": communities["selected_count"],
        "eligible_heads": communities["eligible_node_count"],
        "total_heads": node_count,
        "nested_selected": sensitivity["selected_on_tuning_only"],
        "verification": checks,
        "verified": True,
        "interpretation": (
            "The model showed non-random and paraphrase-sensitive head dependence, but the "
            "community partition did not reproduce across independent mapping partitions. "
            "The protocol therefore forbids causal screening, CCAD calibration, pilot, "
            "confirmation, and replication."
        ),
    }
    (run_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "report.md").write_text(_markdown_report(result), encoding="utf-8")
    (run_dir / "verification.json").write_text(
        json.dumps({"schema_version": 1, "checks": checks, "passed": True}, indent=2) + "\n",
        encoding="utf-8",
    )
    _refresh_checksums(run_dir)
    return result


def verify_mapping_checksums(run_dir: Path) -> bool:
    """Verify every file enumerated by the mapping checksum manifest."""

    for line in (run_dir / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if sha256_file(run_dir / name) != digest:
            return False
    return True


def _markdown_report(result: dict[str, Any]) -> str:
    nested = result["nested_selected"]
    return f"""# FNDE Phase 2: stopped at the network map

**Status: `{result['status']}`**

The model produced a real-looking functional graph, but not a stable enough one to drug.

The primary map retained {result['eligible_heads']} of {result['total_heads']} attention heads and
found {result['primary_communities']} communities. Its split-half ARI was
`{result['primary_split_half_ari']:.3f}`; the frozen gate required at least `0.700`.

A nested mapping-only sensitivity audit selected density `{nested['density']}` and
{nested['community_count']} communities using tuning folds only. It reached ARI
`{nested['tuning_ari']:.3f}` on those folds, then fell to `{nested['heldout_ari']:.3f}` on untouched
mapping folds.

That is the stopping condition doing its job. The dependence was non-random and paraphrases were
more similar than unrelated prompts, but the boundaries were not reproducible enough to support a
claim about temporarily reducing segregation between functional networks. Causal screening, CCAD,
pilot generation, confirmation, and replication were therefore not run.
"""


def _refresh_checksums(run_dir: Path) -> None:
    checksums = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "checksums.sha256"
    }
    (run_dir / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items())
    )
