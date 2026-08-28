"""Functional-network mapping primitives for the FNDE follow-up experiment."""

from .dependence import linear_cka, pairwise_linear_cka
from .graph import weighted_modularity
from .mapping_data import NetworkMapPrompt, load_network_map_dataset, network_map_hash
from .nodes import HeadActivity, HeadNode
from .observation import InMemoryHeadObserver, dense_head_contributions

__all__ = [
    "HeadActivity",
    "HeadNode",
    "InMemoryHeadObserver",
    "dense_head_contributions",
    "linear_cka",
    "pairwise_linear_cka",
    "weighted_modularity",
    "NetworkMapPrompt",
    "load_network_map_dataset",
    "network_map_hash",
]
