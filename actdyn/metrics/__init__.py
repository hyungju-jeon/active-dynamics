from .base import BaseMetric, CompositeMetric, DiscountedMetric
from .information import FisherInformationMetric
import importlib

__all__ = [
    "metric_from_str",
    "BaseMetric",
    "FisherInformationMetric",
    "CompositeMetric",
    "DiscountedMetric",
]


_metric_map = {
    "reward": (".reward", "RewardMetric"),
    "goal-distance": (".reward", "GoalDistanceMetric"),
    "action": (".cost", "ActionCost"),
    "A-optimality": (".information", "AOptimality"),
    "D-optimality": (".information", "DOptimality"),
    "Ensemble_disagreement": (".uncertainty", "EnsembleDisagreement"),
}


def metric_from_str(metric_str: str, **kwargs) -> type[BaseMetric]:
    """
    Create a metric from string name.
    """
    if metric_str not in _metric_map:
        raise ImportError(
            f"Unknown metric: {metric_str}. Available: {list(_metric_map.keys())}"
        )
    module_name, class_name = _metric_map[metric_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)
