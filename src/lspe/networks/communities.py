"""Deterministic graph thresholding, spectral communities, and stability statistics."""

from __future__ import annotations

from math import comb

import numpy as np

from .graph import weighted_modularity


def density_threshold(adjacency: np.ndarray, density: float) -> np.ndarray:
    """Retain the strongest registered fraction of undirected finite edges."""

    graph = _adjacency(adjacency)
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    upper = np.triu_indices(graph.shape[0], 1)
    weights = graph[upper]
    positive = np.flatnonzero(weights > 0)
    if not len(positive):
        raise ValueError("Cannot threshold a graph without positive edges")
    keep_count = max(1, int(np.ceil(density * len(positive))))
    order = positive[np.argsort(weights[positive], kind="stable")[-keep_count:]]
    result = np.zeros_like(graph)
    result[upper[0][order], upper[1][order]] = weights[order]
    return result + result.T


def spectral_communities(adjacency: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    """Cluster normalized-adjacency eigenvectors with deterministic farthest-point k-means."""

    graph = _adjacency(adjacency)
    if count < 2 or count >= graph.shape[0]:
        raise ValueError("community count must be between two and node_count - 1")
    degree = np.sum(graph, axis=1)
    if np.any(degree <= 0):
        raise ValueError("Spectral clustering requires no isolated nodes")
    inverse = 1.0 / np.sqrt(degree)
    normalized = inverse[:, None] * graph * inverse[None, :]
    _, vectors = np.linalg.eigh(normalized)
    embedding = vectors[:, -count:]
    # Remove arbitrary eigenvector signs before deterministic initialization.
    for column in range(embedding.shape[1]):
        pivot = int(np.argmax(np.abs(embedding[:, column])))
        if embedding[pivot, column] < 0:
            embedding[:, column] *= -1
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.maximum(norms, 1e-12)
    return _deterministic_kmeans(embedding, count, seed)


def adjusted_rand_index(first: np.ndarray, second: np.ndarray) -> float:
    """Adjusted Rand index without an external clustering dependency."""

    left = np.asarray(first)
    right = np.asarray(second)
    if left.ndim != 1 or right.shape != left.shape or len(left) < 2:
        raise ValueError("ARI requires equally sized one-dimensional label arrays")
    left_values, left_inverse = np.unique(left, return_inverse=True)
    right_values, right_inverse = np.unique(right, return_inverse=True)
    table = np.zeros((len(left_values), len(right_values)), dtype=np.int64)
    np.add.at(table, (left_inverse, right_inverse), 1)
    def pairs(values: np.ndarray) -> int:
        return sum(comb(int(value), 2) for value in values if value >= 2)
    both = pairs(table.ravel())
    left_pairs = pairs(np.sum(table, axis=1))
    right_pairs = pairs(np.sum(table, axis=0))
    total = comb(len(left), 2)
    expected = left_pairs * right_pairs / total
    maximum = 0.5 * (left_pairs + right_pairs)
    if maximum == expected:
        return 1.0
    return float((both - expected) / (maximum - expected))


def degree_preserving_null_modularities(
    adjacency: np.ndarray,
    communities: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Generate binary-degree-preserving edge-swap nulls with shuffled observed weights."""

    graph = _adjacency(adjacency)
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(seed)
    edges = [tuple(edge) for edge in np.argwhere(np.triu(graph > 0, 1))]
    weights = np.array([graph[edge] for edge in edges])
    results = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        swapped = set(edges)
        attempts = max(100, 20 * len(edges))
        edge_list = list(swapped)
        for _ in range(attempts):
            first_index, second_index = rng.integers(0, len(edge_list), size=2)
            if first_index == second_index:
                continue
            a, b = edge_list[first_index]
            c, d = edge_list[second_index]
            if len({a, b, c, d}) < 4:
                continue
            proposed = tuple(sorted((a, d))), tuple(sorted((c, b)))
            if proposed[0] == proposed[1] or any(edge in swapped for edge in proposed):
                continue
            swapped.remove((a, b))
            swapped.remove((c, d))
            swapped.update(proposed)
            edge_list[first_index], edge_list[second_index] = proposed
        randomized = np.zeros_like(graph)
        shuffled_weights = rng.permutation(weights)
        for edge, weight in zip(sorted(swapped), shuffled_weights, strict=True):
            randomized[edge] = weight
            randomized[edge[::-1]] = weight
        results[sample] = weighted_modularity(randomized, communities)
    return results


def _deterministic_kmeans(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = [int(rng.integers(0, len(values)))]
    while len(centers) < count:
        distances = np.min(
            np.stack([np.sum((values - values[index]) ** 2, axis=1) for index in centers]),
            axis=0,
        )
        distances[centers] = -1
        centers.append(int(np.argmax(distances)))
    centroid = values[centers].copy()
    labels = np.zeros(len(values), dtype=np.int64)
    for _ in range(200):
        updated = np.argmin(
            np.stack([np.sum((values - center) ** 2, axis=1) for center in centroid]),
            axis=0,
        )
        for empty_label in set(range(count)) - set(int(label) for label in updated):
            assigned_distance = np.sum((values - centroid[updated]) ** 2, axis=1)
            replacement = int(np.argmax(assigned_distance))
            updated[replacement] = empty_label
        if np.array_equal(updated, labels) and _ > 0:
            break
        labels = updated
        for label in range(count):
            members = values[labels == label]
            if len(members):
                centroid[label] = np.mean(members, axis=0)
    # Canonicalize arbitrary cluster numbers by first node occurrence.
    ordering = sorted(range(count), key=lambda label: int(np.flatnonzero(labels == label)[0]))
    remap = {old: new for new, old in enumerate(ordering)}
    return np.array([remap[int(label)] for label in labels], dtype=np.int64)


def _adjacency(value: np.ndarray) -> np.ndarray:
    graph = np.asarray(value, dtype=np.float64)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1] or graph.shape[0] < 3:
        raise ValueError("Adjacency must be a square matrix with at least three nodes")
    if not np.isfinite(graph).all() or np.any(graph < 0):
        raise ValueError("Adjacency must contain finite non-negative weights")
    if not np.allclose(graph, graph.T, rtol=0.0, atol=1e-10):
        raise ValueError("Adjacency must be symmetric")
    result = graph.copy()
    np.fill_diagonal(result, 0.0)
    return result
