"""Linear Thompson sampling baseline for active learning benchmarks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import gymnasium as gym
import torch

from ._linear_bandit import _LinearBanditPolicy


class BaselineThompsonPolicy(_LinearBanditPolicy):
    """Contextual linear bandit with Thompson sampling over candidate actions."""

    def __init__(
        self,
        action_space: gym.Space,
        prior_var: float = 0.2,
        ridge: float = 1.0,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(
            action_space=action_space,
            ridge=ridge,
            policy_name="BaselineThompsonPolicy",
            **kwargs,
        )
        self.prior_var = float(prior_var)
        self._rng = np.random.default_rng(seed)

    def _posterior_cov(self) -> np.ndarray:
        a_inv = np.linalg.inv(self._a_mat)
        return self.prior_var * a_inv

    def get_action(self, state: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_actions: Iterable[np.ndarray] | None = kwargs.get("candidate_actions")
        candidates, feature_mat = self._candidate_features(state, candidate_actions)
        sampled_theta = self._rng.multivariate_normal(self._posterior_mean(), self._posterior_cov())
        scores = feature_mat @ sampled_theta
        return self._action_from_scores(candidates, scores)
