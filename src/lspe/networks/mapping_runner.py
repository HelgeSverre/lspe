"""Resumable functional-network mapping for the frozen Qwen Phase 2 subject."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from ..columnar import write_parquet
from ..config import load_config
from ..fetch import fetch_model
from ..generation.sampler import sample_token
from ..hashing import sha256_file
from ..models.mlx_qwen3 import MlxQwen3Adapter
from ..rng import derive_seed
from .communities import (
    adjusted_rand_index,
    degree_preserving_null_modularities,
    density_threshold,
    spectral_communities,
)
from .graph import weighted_modularity
from .mapping_data import load_network_map_dataset, network_map_hash
from .observation import InMemoryHeadObserver


@dataclass(frozen=True)
class MappingProtocol:
    """Frozen mapping-only choices, never selected using behavioral outcomes."""

    master_seed: int = 4_872_031
    selected_layers: tuple[int, ...] = tuple(range(36))
    layer_batch_size: int = 6
    attention_bins: int = 16
    graph_density: float = 0.02
    community_counts: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    bootstrap_samples: int = 20
    null_samples: int = 100
    minimum_split_half_ari: float = 0.70
    minimum_assignment_probability: float = 0.80
    system_prompt: str = "Follow the requested format exactly. Give only the answer."


def run_functional_mapping(
    *,
    model_config: Path,
    data_path: Path,
    run_dir: Path,
    protocol: MappingProtocol | None = None,
    offline: bool = True,
) -> dict[str, Any]:
    """Collect component activity and build the registered baseline graph."""

    _require_clean_source()
    protocol = protocol or MappingProtocol()
    prompts = load_network_map_dataset(data_path)
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
        _validate_protocol(protocol, geometry)
        continuations = _fixed_continuations(adapter, prompts, protocol, run_dir)
        _collect_activity(adapter, continuations, protocol, geometry, run_dir)
    finally:
        adapter.unload()
    result = _build_graph_and_communities(protocol, geometry, prompts, run_dir)
    manifest = {
        "schema_version": 1,
        "stage": "functional_mapping",
        "source_commit": _source_commit(),
        "model_repo_id": fetched.repo_id,
        "model_revision": fetched.revision,
        "model_weight_files": fetched.weight_files,
        "network_map_sha256": network_map_hash(data_path),
        "protocol": asdict(protocol),
        "geometry": geometry,
        "result": result,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksums = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "checksums.sha256"
    }
    (run_dir / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()), encoding="utf-8"
    )
    return manifest


def _fixed_continuations(
    adapter: MlxQwen3Adapter,
    prompts: list[Any],
    protocol: MappingProtocol,
    run_dir: Path,
) -> list[dict[str, Any]]:
    path = run_dir / "fixed-continuations.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(rows) != 2 * len(prompts):
            raise RuntimeError("Existing fixed continuation plan is incomplete")
        return rows
    rows: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        tokens = adapter.format_prompt(
            [
                {"role": "system", "content": protocol.system_prompt},
                {"role": "user", "content": prompt.prompt},
            ]
        )
        logits = adapter.forward(tokens).logits[0, -1]
        greedy = int(np.argmax(logits))
        sampled = sample_token(
            logits,
            temperature=0.8,
            top_k=64,
            top_p=1.0,
            seed=derive_seed(protocol.master_seed, "sampling-token", prompt.prompt_id, 0),
            store_top_logprobs=8,
        ).token_id
        for mode, token_id in (("greedy", greedy), ("sampled", sampled)):
            rows.append(
                {
                    "row_index": len(rows),
                    "prompt_id": prompt.prompt_id,
                    "category": prompt.category,
                    "pair_kind": prompt.pair_kind,
                    "pair_id": prompt.pair_id,
                    "pair_member": prompt.pair_member,
                    "mode": mode,
                    "prompt_token_ids": tokens,
                    "continuation_token_id": token_id,
                    "continuation_text": adapter.decode([token_id]),
                }
            )
        if index % 20 == 0:
            print(json.dumps({"event": "mapping_continuations", "complete": index, "total": 200}))
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return rows


def _collect_activity(
    adapter: MlxQwen3Adapter,
    continuations: list[dict[str, Any]],
    protocol: MappingProtocol,
    geometry: dict[str, int],
    run_dir: Path,
) -> None:
    layer_count = len(protocol.selected_layers)
    head_count = geometry["attention_heads"]
    hidden_width = geometry["hidden_width"]
    node_count = layer_count * head_count
    row_count = len(continuations)
    raw_path = run_dir / "component-activity.npy"
    pattern_path = run_dir / "attention-patterns.npy"
    raw = _memmap(raw_path, (node_count, row_count, hidden_width), preserve=True)
    patterns = _memmap(
        pattern_path, (node_count, row_count, protocol.attention_bins), preserve=True
    )
    checkpoint_path = run_dir / "mapping-checkpoint.json"
    completed = set()
    if checkpoint_path.exists():
        completed = set(json.loads(checkpoint_path.read_text())["completed_layers"])
    layers = list(protocol.selected_layers)
    for offset in range(0, len(layers), protocol.layer_batch_size):
        group = layers[offset : offset + protocol.layer_batch_size]
        pending = [layer for layer in group if layer not in completed]
        if not pending:
            continue
        observer = InMemoryHeadObserver(
            last_position_only=True, attention_bins=protocol.attention_bins
        )
        adapter.wrap_attention_observer(observer, frozenset(pending))
        try:
            for index, row in enumerate(continuations, start=1):
                adapter.forward([*row["prompt_token_ids"], row["continuation_token_id"]])
                if index % 40 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "mapping_activity",
                                "layers": pending,
                                "complete": index,
                                "total": row_count,
                            }
                        )
                    )
        finally:
            adapter.unwrap_attention_observer()
        by_node = {activity.node.node_id: activity.values for activity in observer.activities()}
        pattern_by_node = {
            activity.node.node_id: activity.values for activity in observer.attention_patterns()
        }
        for layer in pending:
            layer_offset = layers.index(layer) * head_count
            for head in range(head_count):
                node_id = f"L{layer:03d}H{head:03d}"
                raw[layer_offset + head] = by_node[node_id]
                patterns[layer_offset + head] = pattern_by_node[node_id]
        raw.flush()
        patterns.flush()
        completed.update(pending)
        checkpoint_path.write_text(
            json.dumps({"completed_layers": sorted(completed)}, indent=2) + "\n"
        )


def _build_graph_and_communities(
    protocol: MappingProtocol,
    geometry: dict[str, int],
    prompts: list[Any],
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
    primary = _cka_adjacency(normalized, np.arange(row_count), run_dir / "cka-grams.npy")
    thresholded = density_threshold(primary, protocol.graph_density)
    prompt_rows = np.arange(row_count).reshape(-1, 2)
    halves = (prompt_rows[::2].reshape(-1), prompt_rows[1::2].reshape(-1))
    half_graphs = [
        density_threshold(_cka_adjacency(normalized, rows), protocol.graph_density)
        for rows in halves
    ]
    eligible = _stable_nonisolated_nodes([thresholded, *half_graphs])
    if len(eligible) < 3:
        raise RuntimeError("MAPPING_GATE_TOO_FEW_NONISOLATED_NODES")
    community_graph = thresholded[np.ix_(eligible, eligible)]
    community_half_graphs = [graph[np.ix_(eligible, eligible)] for graph in half_graphs]
    candidates: list[dict[str, Any]] = []
    for count in protocol.community_counts:
        full_labels = spectral_communities(
            community_graph, count, seed=protocol.master_seed + count
        )
        half_labels = [
            spectral_communities(graph, count, seed=protocol.master_seed + count)
            for graph in community_half_graphs
        ]
        candidates.append(
            {
                "count": count,
                "split_half_ari": adjusted_rand_index(*half_labels),
                "modularity": weighted_modularity(community_graph, full_labels),
                "labels": full_labels,
            }
        )
    selected = max(candidates, key=lambda row: (row["split_half_ari"], row["modularity"]))
    labels = selected.pop("labels")
    assignment = _bootstrap_assignment_probability(
        run_dir / "cka-grams.npy",
        row_count,
        community_graph,
        eligible,
        labels,
        protocol,
        selected["count"],
    )
    nulls = degree_preserving_null_modularities(
        community_graph,
        labels,
        samples=protocol.null_samples,
        seed=protocol.master_seed + 991,
    )
    pair_stability = _pair_similarity(raw, prompts)
    all_layer_labels = np.repeat(protocol.selected_layers, geometry["attention_heads"])
    all_kv_labels = np.tile(
        np.arange(geometry["attention_heads"]) % geometry["kv_heads"],
        len(protocol.selected_layers),
    )
    layer_labels = all_layer_labels[eligible]
    kv_labels = all_kv_labels[eligible]
    mixed_layers = sum(
        len(set(labels[layer_labels == layer])) > 1 for layer in protocol.selected_layers
    )
    null_95 = float(np.quantile(nulls, 0.95))
    gates = {
        "split_half_ari": selected["split_half_ari"] >= protocol.minimum_split_half_ari,
        "assignment_probability": float(np.median(assignment))
        >= protocol.minimum_assignment_probability,
        "paraphrase_over_unrelated": pair_stability["paraphrase"]
        > pair_stability["unrelated"],
        "not_layer_only": adjusted_rand_index(labels, layer_labels) < 0.90,
        "not_shared_kv_only": adjusted_rand_index(labels, kv_labels) < 0.90,
        "minimum_communities": selected["count"] >= 3,
        "mixed_layers": mixed_layers >= 2,
        "null_modularity": selected["modularity"] > null_95,
    }
    _write_mapping_projections(
        protocol,
        geometry,
        raw,
        patterns,
        rms,
        primary,
        thresholded,
        eligible,
        labels,
        assignment,
        run_dir,
    )
    communities = {
        "selected_count": selected["count"],
        "labels": {
            _node_id(protocol, geometry["attention_heads"], int(node)): int(label)
            for node, label in zip(eligible, labels, strict=True)
        },
        "eligible_node_count": len(eligible),
        "isolated_or_unstable_node_count": node_count - len(eligible),
        "isolated_or_unstable_nodes": [
            _node_id(protocol, geometry["attention_heads"], int(node))
            for node in sorted(set(range(node_count)) - set(int(value) for value in eligible))
        ],
        "candidate_statistics": [
            {key: value for key, value in row.items() if key != "labels"} for row in candidates
        ],
        "selected_statistics": selected,
        "median_assignment_probability": float(np.median(assignment)),
        "pair_similarity": pair_stability,
        "layer_ari": adjusted_rand_index(labels, layer_labels),
        "shared_kv_ari": adjusted_rand_index(labels, kv_labels),
        "mixed_layer_count": mixed_layers,
        "null_modularity_95th": null_95,
        "gates": gates,
        "passed": all(gates.values()),
    }
    (run_dir / "communities.json").write_text(
        json.dumps(communities, indent=2, sort_keys=True) + "\n"
    )
    return communities


def _cka_adjacency(
    activity: np.ndarray, rows: np.ndarray, gram_path: Path | None = None
) -> np.ndarray:
    node_count = activity.shape[0]
    sample_count = len(rows)
    if gram_path is None:
        grams = np.empty((node_count, sample_count * sample_count), dtype=np.float32)
    else:
        grams = _memmap(gram_path, (node_count, sample_count * sample_count))
    for node in range(node_count):
        values = np.asarray(activity[node, rows], dtype=np.float32)
        kernel = values @ values.T
        kernel -= np.mean(kernel, axis=0, keepdims=True)
        kernel -= np.mean(kernel, axis=1, keepdims=True)
        kernel += np.mean(kernel)
        norm = np.linalg.norm(kernel)
        if norm <= 1e-12:
            raise RuntimeError(f"MAPPING_GATE_CONSTANT_ACTIVITY:{node}")
        grams[node] = (kernel / norm).reshape(-1)
    if isinstance(grams, np.memmap):
        grams.flush()
    adjacency = np.asarray(grams @ grams.T, dtype=np.float64)
    np.fill_diagonal(adjacency, 0.0)
    return np.clip(adjacency, 0.0, 1.0)


def _bootstrap_assignment_probability(
    gram_path: Path,
    row_count: int,
    full_graph: np.ndarray,
    eligible_nodes: np.ndarray,
    full_labels: np.ndarray,
    protocol: MappingProtocol,
    count: int,
) -> np.ndarray:
    rng = np.random.default_rng(protocol.master_seed + 313)
    matches = np.zeros(len(full_labels), dtype=np.float64)
    valid_samples = 0
    all_kernels = np.load(gram_path, mmap_mode="r").reshape(-1, row_count, row_count)
    kernels = all_kernels[eligible_nodes]
    retained_edges = np.argwhere(np.triu(full_graph > 0, 1))
    for sample in range(protocol.bootstrap_samples):
        prompt_indices = rng.integers(0, row_count // 2, size=row_count // 2)
        rows = np.column_stack((2 * prompt_indices, 2 * prompt_indices + 1)).reshape(-1)
        gram_vectors = np.empty((len(full_labels), row_count * row_count), dtype=np.float32)
        for node in range(len(full_labels)):
            kernel = np.asarray(kernels[node][np.ix_(rows, rows)], dtype=np.float32)
            kernel -= np.mean(kernel, axis=0, keepdims=True)
            kernel -= np.mean(kernel, axis=1, keepdims=True)
            kernel += np.mean(kernel)
            gram_vectors[node] = (kernel / max(np.linalg.norm(kernel), 1e-12)).reshape(-1)
        graph = np.zeros_like(full_graph)
        for offset in range(0, len(retained_edges), 256):
            edges = retained_edges[offset : offset + 256]
            weights = np.einsum(
                "ij,ij->i", gram_vectors[edges[:, 0]], gram_vectors[edges[:, 1]]
            )
            graph[edges[:, 0], edges[:, 1]] = np.clip(weights, 0.0, 1.0)
            graph[edges[:, 1], edges[:, 0]] = np.clip(weights, 0.0, 1.0)
        if np.any(np.sum(graph, axis=1) == 0):
            continue
        labels = spectral_communities(graph, count, seed=protocol.master_seed + sample)
        aligned = _align_labels(labels, full_labels)
        matches += aligned == full_labels
        valid_samples += 1
        print(
            json.dumps(
                {
                    "event": "mapping_bootstrap",
                    "complete": sample + 1,
                    "total": protocol.bootstrap_samples,
                }
            )
        )
    if valid_samples < int(np.ceil(0.95 * protocol.bootstrap_samples)):
        raise RuntimeError(f"MAPPING_GATE_BOOTSTRAP_INVALID:{valid_samples}")
    return matches / valid_samples


def _align_labels(labels: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_values = sorted(int(value) for value in np.unique(labels))
    target_values = sorted(int(value) for value in np.unique(target))
    if len(source_values) != len(target_values):
        raise ValueError("Cannot align community labelings with different cluster counts")
    contingency = np.array(
        [
            [np.sum((labels == source) & (target == destination)) for destination in target_values]
            for source in source_values
        ]
    )
    def overlap(order: tuple[int, ...]) -> int:
        return int(
            sum(contingency[index, destination] for index, destination in enumerate(order))
        )

    assignment = max(permutations(range(len(target_values))), key=overlap)
    mapping = {
        source: target_values[assignment[index]] for index, source in enumerate(source_values)
    }
    return np.array([mapping[int(label)] for label in labels], dtype=target.dtype)


def _pair_similarity(activity: np.ndarray, prompts: list[Any]) -> dict[str, float]:
    by_pair: dict[tuple[str, str], list[int]] = {}
    for prompt_index, prompt in enumerate(prompts):
        if prompt.pair_kind and prompt.pair_id:
            by_pair.setdefault((prompt.pair_kind, prompt.pair_id), []).append(prompt_index)
    scores: dict[str, list[float]] = {"paraphrase": [], "unrelated": []}
    # Each prompt owns adjacent greedy and sampled rows; average both modes and all heads.
    summarized = np.mean(activity, axis=0).reshape(len(prompts), 2, -1).mean(axis=1)
    summarized /= np.maximum(np.linalg.norm(summarized, axis=1, keepdims=True), 1e-12)
    for (kind, _), members in by_pair.items():
        if len(members) == 2:
            scores[kind].append(float(summarized[members[0]] @ summarized[members[1]]))
    return {kind: float(np.mean(values)) for kind, values in scores.items()}


def _write_mapping_projections(
    protocol: MappingProtocol,
    geometry: dict[str, int],
    raw: np.ndarray,
    patterns: np.ndarray,
    rms: np.ndarray,
    primary: np.ndarray,
    thresholded: np.ndarray,
    eligible: np.ndarray,
    labels: np.ndarray,
    assignment: np.ndarray,
    run_dir: Path,
) -> None:
    head_count = geometry["attention_heads"]
    eligible_position = {int(node): index for index, node in enumerate(eligible)}
    component_rows = []
    for node in range(raw.shape[0]):
        layer = protocol.selected_layers[node // head_count]
        head = node % head_count
        component_rows.append(
            {
                "node_id": _node_id(protocol, head_count, node),
                "layer": layer,
                "head": head,
                "shared_kv_family": head % geometry["kv_heads"],
                "mean_rms": float(np.mean(rms[node])),
                "activity_variance": float(np.var(raw[node])),
                "attention_pattern_entropy": float(
                    np.mean(-np.sum(patterns[node] * np.log(patterns[node] + 1e-12), axis=1))
                ),
                "included": node in eligible_position,
                "community": (
                    int(labels[eligible_position[node]]) if node in eligible_position else -1
                ),
                "assignment_probability": (
                    float(assignment[eligible_position[node]])
                    if node in eligible_position
                    else None
                ),
            }
        )
    edge_rows = []
    for first, second in zip(*np.triu_indices(raw.shape[0], 1), strict=True):
        if thresholded[first, second] <= 0:
            continue
        edge_rows.append(
            {
                "source": _node_id(protocol, head_count, int(first)),
                "target": _node_id(protocol, head_count, int(second)),
                "linear_cka": float(primary[first, second]),
                "retained_weight": float(thresholded[first, second]),
                "mean_cosine": _mean_row_cosine(raw[first], raw[second]),
                "rms_correlation": _correlation(rms[first], rms[second]),
                "attention_js_similarity": _attention_js(
                    patterns[first], patterns[second]
                ),
                "same_layer": bool(first // head_count == second // head_count),
                "shared_kv": bool(first % geometry["kv_heads"] == second % geometry["kv_heads"]),
            }
        )
    write_parquet(run_dir / "component-map.parquet", component_rows)
    write_parquet(run_dir / "functional-graph.parquet", edge_rows)


def _stable_nonisolated_nodes(graphs: list[np.ndarray]) -> np.ndarray:
    """Return the fixed point of nodes connected in full and both split graphs."""

    eligible = np.arange(graphs[0].shape[0])
    while True:
        keep = np.ones(len(eligible), dtype=bool)
        for graph in graphs:
            keep &= np.sum(graph[np.ix_(eligible, eligible)], axis=1) > 0
        updated = eligible[keep]
        if np.array_equal(updated, eligible):
            return eligible
        eligible = updated


def _mean_row_cosine(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float32)
    y = np.asarray(second, dtype=np.float32)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    valid = denominator > 1e-12
    if not np.any(valid):
        return 0.0
    return float(np.mean(np.sum(x[valid] * y[valid], axis=1) / denominator[valid]))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _attention_js(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    x /= np.maximum(np.sum(x, axis=1, keepdims=True), 1e-12)
    y /= np.maximum(np.sum(y, axis=1, keepdims=True), 1e-12)
    midpoint = 0.5 * (x + y)
    first_term = np.sum(np.where(x > 0, x * np.log(x / midpoint), 0.0), axis=1)
    second_term = np.sum(np.where(y > 0, y * np.log(y / midpoint), 0.0), axis=1)
    divergence = 0.5 * (first_term + second_term)
    return float(np.clip(1.0 - np.mean(divergence) / np.log(2.0), 0.0, 1.0))


def _memmap(path: Path, shape: tuple[int, ...], preserve: bool = False) -> np.memmap:
    if preserve and path.exists():
        return np.load(path, mmap_mode="r+")
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def _node_id(protocol: MappingProtocol, head_count: int, node: int) -> str:
    layer = protocol.selected_layers[node // head_count]
    return f"L{layer:03d}H{node % head_count:03d}"


def _validate_protocol(protocol: MappingProtocol, geometry: dict[str, int]) -> None:
    invalid = set(protocol.selected_layers) - set(range(geometry["layers"]))
    if invalid:
        raise ValueError(f"Mapping protocol contains invalid layers: {sorted(invalid)}")
    if len(protocol.selected_layers) % protocol.layer_batch_size:
        raise ValueError("Selected layers must divide evenly into observation batches")


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _require_clean_source() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if status:
        raise RuntimeError("Functional mapping requires a clean committed source tree")
