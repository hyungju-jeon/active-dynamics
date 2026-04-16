"""Metrics package for Active Dynamics.

Expose base metric types and common implementations. Use :func:`metric_from_str`
to dynamically resolve metrics by name.
"""

from .base import BaseMetric, CompositeMetric, DiscountedMetric
from .information import (
    AmbiguityAwareEmbeddingFisherMetric,
    EmbeddingFisherMetric,
    FisherInformationMetric,
)

__all__ = [
    "BaseMetric",
    "EmbeddingFisherMetric",
    "AmbiguityAwareEmbeddingFisherMetric",
    "FisherInformationMetric",
    "CompositeMetric",
    "DiscountedMetric",
]

import importlib

_metric_map = {
    "reward": (".reward", "RewardMetric"),
    "goal-distance": (".reward", "GoalDistanceMetric"),
    "action": (".cost", "ActionCost"),
    "a-optimality": (".information", "AOptimality"),
    "d-optimality": (".information", "DOptimality"),
    "ensemble-disagreement": (".uncertainty", "EnsembleDisagreement"),
    "embedding-fisher": (".information", "EmbeddingFisherMetric"),
    "ambiguity-aware-embedding-fisher": (".information", "AmbiguityAwareEmbeddingFisherMetric"),
}


def metric_from_str(metric_str: str) -> type[BaseMetric]:
    if metric_str not in _metric_map:
        raise ImportError(f"Unknown metric: {metric_str}. Available: {list(_metric_map.keys())}")
    module_name, class_name = _metric_map[metric_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)
