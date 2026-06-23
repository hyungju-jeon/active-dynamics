from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from actdyn.policy.baseline_thompson import BaselineThompsonPolicy
from actdyn.policy.baseline_ucb import BaselineUCBPolicy


@pytest.mark.parametrize(
    ("policy_cls", "kwargs"),
    [
        (BaselineThompsonPolicy, {"seed": 0}),
        (BaselineUCBPolicy, {"beta": 0.0}),
    ],
)
def test_linear_bandit_baselines_share_candidate_update(policy_cls, kwargs) -> None:
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    policy = policy_cls(action_space=action_space, **kwargs)

    action, cost = policy.get_action(
        torch.ones(1, 2),
        candidate_actions=[np.asarray([-1.0, 0.0]), np.asarray([1.0, 0.0])],
    )
    info = policy.update(
        {
            "reward": 1.0,
            "state": torch.ones(1, 2),
            "action": torch.tensor([[[1.0, 0.0]]]),
        }
    )

    assert action.shape == (1, 1, 2)
    assert cost.shape == (1,)
    assert info["posterior_trace"] > 0.0
    assert not np.allclose(policy._posterior_mean(), 0.0)
