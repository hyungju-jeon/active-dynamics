"""Linear UCB baseline for active learning benchmarks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import gymnasium as gym
import torch

from ._linear_bandit import _LinearBanditPolicy


class BaselineUCBPolicy(_LinearBanditPolicy):
    """Contextual linear bandit with upper-confidence bound action selection."""

    def __init__(
        self,
        action_space: gym.Space,
        beta: float = 1.0,
        ridge: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            action_space=action_space,
            ridge=ridge,
            policy_name="BaselineUCBPolicy",
            **kwargs,
        )
        self.beta = float(beta)

    def get_action(self, state: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_actions: Iterable[np.ndarray] | None = kwargs.get("candidate_actions")
        candidates, feature_mat = self._candidate_features(state, candidate_actions)
        a_inv = np.linalg.inv(self._a_mat)
        mu = self._posterior_mean()
        means = feature_mat @ mu
        conf = np.sqrt(np.einsum("bi,ij,bj->b", feature_mat, a_inv, feature_mat))
        ucb_scores = means + self.beta * conf
        return self._action_from_scores(candidates, ucb_scores)
