"""Task-balanced consensus mapping for the stronger FNDE v2 attempt."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config
from ..fetch import fetch_model
from ..hashing import sha256_file
from ..models.mlx_qwen3 import MlxQwen3Adapter
from .communities import (
    adjusted_rand_index,
    degree_preserving_null_modularities,
    density_threshold,
    spectral_communities,
)
from .graph import weighted_modularity
from .mapping_data_v2 import load_network_map_v2_dataset
from .mapping_runner import (
    _collect_activity,
    _fixed_continuations,
    _memmap,
    _node_id,
    _pair_similarity,
    _stable_nonisolated_nodes,
    _write_mapping_projections,
)


@dataclass(frozen=True)
class ConsensusMappingProtocol:
    """Frozen choices from FNDE_V2_AMENDMENT.md."""

    master_seed: int = 8_104_229
    continuation_tokens: int = 2
    selected_layers: tuple[int, ...] = tuple(range(36))
    layer_batch_size: int = 6
    attention_bins: int = 16
    graph_densities: tuple[float, ...] = (
        0.005,
        0.01,
        0.015,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
    )
    community_counts: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    bootstrap_samples: int = 20
    null_samples: int = 100
    minimum_split_half_ari: float = 0.70
    minimum_assignment_probability: float = 0.80
    minimum_heldout_coverage: float = 0.80
    minimum_community_fraction: float = 0.05
    system_prompt: str = "Follow the requested format exactly. Give only the answer."


def run_consensus_mapping(
    *, model_config: Path, data_path: Path, run_dir: Path, offline: bool = True
) -> dict[str, Any]:
    """Execute observation, nested selection, and held-out consensus-map evaluation."""

    _require_clean_source()
    protocol = ConsensusMappingProtocol()
    prompts = load_network_map_v2_dataset(data_path)
    config = load_config(model_config)
    fetched = fetch_model(config.model, offline=offline)
    runtime_model = config.model.model_copy(
        update={"revision": fetched.revision, "local_path": fetched.local_path}
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter = MlxQwen3Adapter()
    adapter.load(runtime_model)
    try:
        geometry = adapter.attention_geometry()
        if tuple(protocol.selected_layers) != tuple(range(geometry["layers"])):
            raise RuntimeError("FNDE v2 requires observation of every model layer")
        continuations = _fixed_continuations(adapter, prompts, protocol, run_dir)
        _collect_activity(adapter, continuations, protocol, geometry, run_dir)
    finally:
        adapter.unload()
    result = _build_consensus_map(protocol, geometry, prompts, continuations, run_dir)
    manifest = {
        "schema_version": 2,
        "stage": "task_balanced_consensus_mapping",
        "source_commit": _source_commit(),
        "model_repo_id": fetched.repo_id,
        "model_revision": fetched.revision,
        "model_weight_files": fetched.weight_files,
        "network_map_v2_sha256": sha256_file(data_path),
        "protocol": asdict(protocol),
        "geometry": geometry,
        "result": result,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_checksums(run_dir)
    return manifest


def _build_consensus_map(
    protocol: ConsensusMappingProtocol,
    geometry: dict[str, int],
    prompts: list[Any],
    continuations: list[dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    raw = np.load(run_dir / "component-activity.npy", mmap_mode="r")
    patterns = np.load(run_dir / "attention-patterns.npy", mmap_mode="r")
    node_count, row_count, hidden_width = raw.shape
    normalized = _memmap(run_dir / "component-activity-normalized.npy", raw.shape)
    means = _memmap(run_dir / "component-means.npy", (node_count, hidden_width))
    scales = _memmap(run_dir / "component-scales.npy", (node_count, hidden_width))
    rms = _memmap(run_dir / "component-rms.npy", (node_count, row_count))
    for node in range(node_count):
        values = np.asarray(raw[node], dtype=np.float32)
        means[node] = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale[scale < 1e-8] = 1.0
        scales[node] = scale
        normalized[node] = (values - means[node]) / scale
        rms[node] = np.sqrt(np.mean(np.square(values), axis=1))
    for array in (normalized, means, scales, rms):
        array.flush()

    category_rows, category_prompt_rows = _category_rows(prompts, continuations)
    categories = sorted(category_rows)
    if any(len(category_rows[category]) != 160 for category in categories):
        raise RuntimeError("Every v2 task family must contribute exactly 160 observation rows")
    full, task_grams = _task_consensus_adjacency(
        normalized, category_rows, categories, run_dir / "task-cka-grams.npy"
    )
    fold_graphs = []
    for fold in range(4):
        task_graphs = [
            _cka_adjacency(normalized, category_prompt_rows[category][fold])
            for category in categories
        ]
        fold_graphs.append(np.mean(task_graphs, axis=0))

    candidates: list[dict[str, Any]] = []
    candidate_state: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}
    for density in protocol.graph_densities:
        graphs = [density_threshold(graph, density) for graph in [full, *fold_graphs]]
        tuning_eligible = _stable_nonisolated_nodes([graphs[1], graphs[3]])
        tuning_graphs = [graph[np.ix_(tuning_eligible, tuning_eligible)] for graph in graphs]
        for count in protocol.community_counts:
            tuning_labels = [
                spectral_communities(
                    graph, count, seed=protocol.master_seed + count
                )
                for graph in (tuning_graphs[1], tuning_graphs[3])
            ]
            heldout_graphs = (tuning_graphs[2], tuning_graphs[4])
            heldout_positions = _stable_nonisolated_nodes(list(heldout_graphs))
            heldout_coverage = len(heldout_positions) / len(tuning_eligible)
            if len(heldout_positions) <= count:
                heldout_ari = -1.0
            else:
                heldout_labels = [
                    spectral_communities(
                        graph[np.ix_(heldout_positions, heldout_positions)],
                        count,
                        seed=protocol.master_seed + count,
                    )
                    for graph in heldout_graphs
                ]
                heldout_ari = adjusted_rand_index(*heldout_labels)
            candidates.append(
                {
                    "density": density,
                    "community_count": count,
                    "eligible_nodes": len(tuning_eligible),
                    "tuning_ari": adjusted_rand_index(*tuning_labels),
                    "heldout_ari": heldout_ari,
                    "heldout_coverage": heldout_coverage,
                }
            )
            candidate_state[(density, count)] = (tuning_eligible, tuning_graphs[0])
    selected = _select_candidate(candidates)
    tuning_eligible, full_graph_on_tuning_nodes = candidate_state[
        (selected["density"], selected["community_count"])
    ]
    final_positions = _stable_nonisolated_nodes([full_graph_on_tuning_nodes])
    eligible = tuning_eligible[final_positions]
    community_graph = full_graph_on_tuning_nodes[np.ix_(final_positions, final_positions)]
    labels = spectral_communities(
        community_graph,
        selected["community_count"],
        seed=protocol.master_seed + selected["community_count"],
    )
    assignment = _consensus_bootstrap_assignment(
        task_grams,
        categories,
        eligible,
        community_graph,
        labels,
        protocol,
        selected["community_count"],
    )
    nulls = degree_preserving_null_modularities(
        community_graph,
        labels,
        samples=protocol.null_samples,
        seed=protocol.master_seed + 991,
    )
    layer_labels = np.repeat(protocol.selected_layers, geometry["attention_heads"])[eligible]
    kv_labels = np.tile(
        np.arange(geometry["attention_heads"]) % geometry["kv_heads"],
        len(protocol.selected_layers),
    )[eligible]
    mixed_layers = sum(
        len(set(labels[layer_labels == layer])) > 1 for layer in protocol.selected_layers
    )
    _, community_sizes = np.unique(labels, return_counts=True)
    minimum_community_fraction = float(np.min(community_sizes) / len(labels))
    pair_stability = _pair_similarity(raw, prompts)
    modularity = weighted_modularity(community_graph, labels)
    null_95 = float(np.quantile(nulls, 0.95))
    gates = {
        "heldout_ari": selected["heldout_ari"] >= protocol.minimum_split_half_ari,
        "heldout_coverage": selected["heldout_coverage"]
        >= protocol.minimum_heldout_coverage,
        "assignment_probability": float(np.median(assignment))
        >= protocol.minimum_assignment_probability,
        "paraphrase_over_unrelated": pair_stability["paraphrase"]
        > pair_stability["unrelated"],
        "not_layer_only": adjusted_rand_index(labels, layer_labels) < 0.90,
        "not_shared_kv_only": adjusted_rand_index(labels, kv_labels) < 0.90,
        "minimum_communities": selected["community_count"] >= 3,
        "community_balance": minimum_community_fraction
        >= protocol.minimum_community_fraction,
        "mixed_layers": mixed_layers >= 2,
        "null_modularity": modularity > null_95,
    }
    thresholded_full = density_threshold(full, selected["density"])
    _write_mapping_projections(
        protocol,
        geometry,
        raw,
        patterns,
        rms,
        full,
        thresholded_full,
        eligible,
        labels,
        assignment,
        run_dir,
    )
    result = {
        "status": "MAPPING_PASS" if all(gates.values()) else "MECHANISM_NOT_ACHIEVED",
        "selected_on_tuning_only": selected,
        "eligible_node_count": len(eligible),
        "excluded_node_count": node_count - len(eligible),
        "community_count": selected["community_count"],
        "modularity": modularity,
        "null_modularity_95th": null_95,
        "median_assignment_probability": float(np.median(assignment)),
        "pair_similarity": pair_stability,
        "layer_ari": adjusted_rand_index(labels, layer_labels),
        "shared_kv_ari": adjusted_rand_index(labels, kv_labels),
        "mixed_layer_count": mixed_layers,
        "community_sizes": [int(value) for value in community_sizes],
        "minimum_community_fraction": minimum_community_fraction,
        "labels": {
            _node_id(protocol, geometry["attention_heads"], int(node)): int(label)
            for node, label in zip(eligible, labels, strict=True)
        },
        "candidates": candidates,
        "gates": gates,
        "passed": all(gates.values()),
    }
    (run_dir / "communities.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _category_rows(
    prompts: list[Any], continuations: list[dict[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    rows_by_prompt: dict[str, list[int]] = {}
    for row in continuations:
        rows_by_prompt.setdefault(row["prompt_id"], []).append(int(row["row_index"]))
    categories = sorted({prompt.category for prompt in prompts})
    category_rows: dict[str, np.ndarray] = {}
    folds: dict[str, list[np.ndarray]] = {}
    for category in categories:
        members = [prompt for prompt in prompts if prompt.category == category]
        groups = [rows_by_prompt[prompt.prompt_id] for prompt in members]
        if any(len(group) != 4 for group in groups):
            raise RuntimeError("Every v2 prompt must own four observation rows")
        category_rows[category] = np.array(groups, dtype=np.int64).reshape(-1)
        fold_groups: list[list[list[int]]] = [[], [], [], []]
        fold_sizes = [0, 0, 0, 0]
        units: dict[str, list[list[int]]] = {}
        for prompt, group in zip(members, groups, strict=True):
            unit = (
                f"paraphrase:{prompt.pair_id}"
                if prompt.pair_kind == "paraphrase"
                else f"prompt:{prompt.prompt_id}"
            )
            units.setdefault(unit, []).append(group)
        ordered_units = sorted(
            units.items(),
            key=lambda item: (
                -len(item[1]),
                hashlib.sha256(item[0].encode()).hexdigest(),
            ),
        )
        for _, unit_groups in ordered_units:
            fold = min(range(4), key=lambda index: (fold_sizes[index], index))
            fold_groups[fold].extend(unit_groups)
            fold_sizes[fold] += len(unit_groups)
        if fold_sizes != [10, 10, 10, 10]:
            raise RuntimeError(f"Category {category} has unbalanced v2 folds: {fold_sizes}")
        folds[category] = [np.array(groups, dtype=np.int64).reshape(-1) for groups in fold_groups]
    return category_rows, folds


def _task_consensus_adjacency(
    activity: np.ndarray,
    category_rows: dict[str, np.ndarray],
    categories: list[str],
    gram_path: Path,
) -> tuple[np.ndarray, np.memmap]:
    node_count = activity.shape[0]
    sample_count = len(category_rows[categories[0]])
    grams = _memmap(gram_path, (len(categories), node_count, sample_count * sample_count))
    adjacency = np.zeros((node_count, node_count), dtype=np.float64)
    for category_index, category in enumerate(categories):
        _fill_gram_vectors(activity, category_rows[category], grams[category_index])
        task_graph = np.asarray(
            grams[category_index] @ grams[category_index].T, dtype=np.float64
        )
        np.fill_diagonal(task_graph, 0.0)
        adjacency += np.clip(task_graph, 0.0, 1.0) / len(categories)
        print(
            json.dumps(
                {
                    "event": "consensus_task_graph",
                    "category": category,
                    "complete": category_index + 1,
                    "total": len(categories),
                }
            )
        )
    grams.flush()
    return adjacency, grams


def _cka_adjacency(activity: np.ndarray, rows: np.ndarray) -> np.ndarray:
    grams = np.empty((activity.shape[0], len(rows) * len(rows)), dtype=np.float32)
    _fill_gram_vectors(activity, rows, grams)
    adjacency = np.asarray(grams @ grams.T, dtype=np.float64)
    np.fill_diagonal(adjacency, 0.0)
    return np.clip(adjacency, 0.0, 1.0)


def _fill_gram_vectors(activity: np.ndarray, rows: np.ndarray, target: np.ndarray) -> None:
    for node in range(activity.shape[0]):
        values = np.asarray(activity[node, rows], dtype=np.float32)
        kernel = values @ values.T
        kernel -= np.mean(kernel, axis=0, keepdims=True)
        kernel -= np.mean(kernel, axis=1, keepdims=True)
        kernel += np.mean(kernel)
        norm = np.linalg.norm(kernel)
        if norm <= 1e-12:
            raise RuntimeError(f"MAPPING_GATE_CONSTANT_ACTIVITY:{node}")
        target[node] = (kernel / norm).reshape(-1)


def _consensus_bootstrap_assignment(
    task_grams: np.ndarray,
    categories: list[str],
    eligible: np.ndarray,
    full_graph: np.ndarray,
    full_labels: np.ndarray,
    protocol: ConsensusMappingProtocol,
    count: int,
) -> np.ndarray:
    rng = np.random.default_rng(protocol.master_seed + 313)
    retained_edges = np.argwhere(np.triu(full_graph > 0, 1))
    matches = np.zeros(len(eligible), dtype=np.float64)
    valid_samples = 0
    for sample in range(protocol.bootstrap_samples):
        edge_weights = np.zeros(len(retained_edges), dtype=np.float64)
        for category_index, _ in enumerate(categories):
            prompt_indices = rng.integers(0, 40, size=40)
            rows = np.concatenate(
                [np.arange(4 * index, 4 * index + 4) for index in prompt_indices]
            )
            vectors = np.empty((len(eligible), len(rows) * len(rows)), dtype=np.float32)
            kernels = task_grams[category_index, eligible].reshape(-1, 160, 160)
            for node in range(len(eligible)):
                kernel = np.asarray(kernels[node][np.ix_(rows, rows)], dtype=np.float32)
                kernel -= np.mean(kernel, axis=0, keepdims=True)
                kernel -= np.mean(kernel, axis=1, keepdims=True)
                kernel += np.mean(kernel)
                vectors[node] = (kernel / max(np.linalg.norm(kernel), 1e-12)).reshape(-1)
            for offset in range(0, len(retained_edges), 256):
                edges = retained_edges[offset : offset + 256]
                edge_weights[offset : offset + len(edges)] += np.einsum(
                    "ij,ij->i", vectors[edges[:, 0]], vectors[edges[:, 1]]
                ) / len(categories)
        graph = np.zeros_like(full_graph)
        graph[retained_edges[:, 0], retained_edges[:, 1]] = np.clip(edge_weights, 0, 1)
        graph[retained_edges[:, 1], retained_edges[:, 0]] = np.clip(edge_weights, 0, 1)
        if np.any(np.sum(graph, axis=1) == 0):
            continue
        labels = spectral_communities(graph, count, seed=protocol.master_seed + sample)
        aligned = _align_labels(labels, full_labels)
        matches += aligned == full_labels
        valid_samples += 1
        print(
            json.dumps(
                {
                    "event": "consensus_bootstrap",
                    "complete": sample + 1,
                    "total": protocol.bootstrap_samples,
                }
            )
        )
    if valid_samples < int(np.ceil(0.95 * protocol.bootstrap_samples)):
        raise RuntimeError(f"MAPPING_GATE_BOOTSTRAP_INVALID:{valid_samples}")
    return matches / valid_samples


def _select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda row: (
            row["tuning_ari"],
            row["eligible_nodes"],
            -row["density"],
            -row["community_count"],
        ),
    )


def _align_labels(labels: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = sorted(int(value) for value in np.unique(labels))
    destinations = sorted(int(value) for value in np.unique(target))
    contingency = np.array(
        [
            [np.sum((labels == source) & (target == destination)) for destination in destinations]
            for source in values
        ]
    )

    def score(order: tuple[int, ...]) -> int:
        return int(sum(contingency[index, value] for index, value in enumerate(order)))

    assignment = max(permutations(range(len(destinations))), key=score)
    mapping = {source: destinations[assignment[index]] for index, source in enumerate(values)}
    return np.array([mapping[int(label)] for label in labels], dtype=target.dtype)


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


def _require_clean_source() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if status:
        raise RuntimeError("FNDE v2 mapping requires a clean committed source tree")
