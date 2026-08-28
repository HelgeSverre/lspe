"""Deterministic weighted-graph statistics for functional communities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def weighted_modularity(
    adjacency: np.ndarray, communities: Sequence[int], resolution: float = 1.0
) -> float:
    """Return Newman–Girvan modularity for an undirected weighted graph."""

    graph = np.asarray(adjacency, dtype=np.float64)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("Adjacency must be a square matrix")
    if graph.shape[0] != len(communities):
        raise ValueError("One community label is required for every graph node")
    if graph.shape[0] < 2:
        raise ValueError("Modularity requires at least two nodes")
    if not np.isfinite(graph).all() or np.any(graph < 0):
        raise ValueError("Adjacency must contain finite non-negative weights")
    if not np.allclose(graph, graph.T, rtol=0.0, atol=1e-12):
        raise ValueError("Adjacency must be symmetric")
    if not np.allclose(np.diag(graph), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("Adjacency diagonal must be zero")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    labels = np.asarray(communities)
    degree = np.sum(graph, axis=1)
    twice_weight = float(np.sum(degree))
    if twice_weight <= 0:
        raise ValueError("Modularity is undefined for a graph with no edges")
    expected = resolution * np.outer(degree, degree) / twice_weight
    same_community = labels[:, None] == labels[None, :]
    return float(np.sum((graph - expected) * same_community) / twice_weight)
