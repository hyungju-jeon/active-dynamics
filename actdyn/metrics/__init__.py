from .base import BaseMetric, CompositeSumCost
import importlib


_reward_map = {
    "sum": (".reward", "RewardMetric"),
    "discounted": (".reward", "DiscountedRewardMetric"),
    "goal-distance": (".reward", "GoalDistanceMetric"),
}

_cost_map = {
    "action": (".cost", "ActionCost"),
}

_information_map = {
    "fisher": (".information", "FisherInformationMetric"),
}


def reward_from_str(reward_str: str, **kwargs) -> BaseMetric:
    """
    Create a reward metric from string name.

    Args:
        reward_str: Name of the reward metric
        **kwargs: Additional arguments for the metric constructor

    Returns:
        BaseMetric: Instance of the specified reward metric

    Example:
        reward = reward_from_str('discounted', gamma=0.99)
    """
    if reward_str not in _reward_map:
        raise ImportError(
            f"Unknown reward metric: {reward_str}. Available: {list(_reward_map.keys())}"
        )
    module_name, class_name = _reward_map[reward_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)(**kwargs)


def cost_from_str(cost_str: str, **kwargs) -> BaseMetric:
    """
    Create a cost metric from string name.

    Args:
        cost_str: Name of the cost metric
        **kwargs: Additional arguments for the metric constructor

    Returns:
        BaseMetric: Instance of the specified cost metric

    Example:
        cost = cost_from_str('action', weight=0.1)
    """
    if cost_str not in _cost_map:
        raise ImportError(
            f"Unknown cost metric: {cost_str}. Available: {list(_cost_map.keys())}"
        )
    module_name, class_name = _cost_map[cost_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)(**kwargs)


def information_from_str(info_str: str, **kwargs) -> BaseMetric:
    """
    Create an information metric from string name.

    Args:
        info_str: Name of the information metric
        **kwargs: Additional arguments for the metric constructor

    Returns:
        BaseMetric: Instance of the specified information metric

    Example:
        fim = information_from_str('fisher', dynamics=model, decoder=decoder)
    """
    if info_str not in _information_map:
        raise ImportError(
            f"Unknown information metric: {info_str}. Available: {list(_information_map.keys())}"
        )
    module_name, class_name = _information_map[info_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)(**kwargs)
