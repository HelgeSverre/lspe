"""Functional-network mapping primitives for the FNDE follow-up experiment."""

from .communities import (
    adjusted_rand_index,
    degree_preserving_null_modularities,
    density_threshold,
    spectral_communities,
)
from .dependence import (
    attention_js_similarity,
    linear_cka,
    mean_cosine_similarity,
    pairwise_linear_cka,
    rms_timeseries_correlation,
)
from .graph import weighted_modularity
from .mapping_data import NetworkMapPrompt, load_network_map_dataset, network_map_hash
from .nodes import HeadActivity, HeadNode
from .observation import InMemoryHeadObserver, dense_head_contributions

__all__ = [
    "HeadActivity",
    "HeadNode",
    "InMemoryHeadObserver",
    "dense_head_contributions",
    "attention_js_similarity",
    "adjusted_rand_index",
    "degree_preserving_null_modularities",
    "density_threshold",
    "linear_cka",
    "mean_cosine_similarity",
    "pairwise_linear_cka",
    "rms_timeseries_correlation",
    "spectral_communities",
    "weighted_modularity",
    "NetworkMapPrompt",
    "load_network_map_dataset",
    "network_map_hash",
]
