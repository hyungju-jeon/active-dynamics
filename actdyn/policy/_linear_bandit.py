"""Shared contextual linear bandit posterior updates for simple baselines."""

from __future__ import annotations

from collections.abc import Iterable

import gymnasium as gym
import numpy as np
import torch

from .base import BasePolicy


class _LinearBanditPolicy(BasePolicy):
    """Contextual linear bandit over a fixed finite candidate-action set."""

    def __init__(
        self,
        action_space: gym.Space,
        *,
        ridge: float = 1.0,
        policy_name: str,
        **kwargs,
    ):
        super().__init__(action_space=action_space, **kwargs)
        if not isinstance(self.action_space, gym.spaces.Box):
            raise TypeError(f"{policy_name} requires a gymnasium.spaces.Box action space")

        self.ridge = float(ridge)
        self._feature_dim = 5
        self._a_mat = np.eye(self._feature_dim, dtype=np.float64) * self.ridge
        self._b_vec = np.zeros(self._feature_dim, dtype=np.float64)

        self._low = np.asarray(self.action_space.low, dtype=np.float64).reshape(-1)
        self._high = np.asarray(self.action_space.high, dtype=np.float64).reshape(-1)
        self._action_dim = int(np.prod(self.action_space.shape))

    def _candidate_actions(self) -> np.ndarray:
        candidates: list[np.ndarray] = [np.zeros(self._action_dim, dtype=np.float64)]

        for dim in range(self._action_dim):
            pos = np.zeros(self._action_dim, dtype=np.float64)
            neg = np.zeros(self._action_dim, dtype=np.float64)
            pos[dim] = self._high[dim] if np.isfinite(self._high[dim]) else 1.0
            neg[dim] = self._low[dim] if np.isfinite(self._low[dim]) else -1.0
            candidates.extend([neg, pos])

        all_high = np.where(np.isfinite(self._high), self._high, 1.0)
        all_low = np.where(np.isfinite(self._low), self._low, -1.0)
        candidates.extend([all_low, all_high])
        return np.stack(candidates, axis=0)

    def _features(self, state: torch.Tensor | np.ndarray, action: np.ndarray) -> np.ndarray:
        s_val = float(torch.as_tensor(state).float().mean().item())
        a_mean = float(np.mean(action))
        a_norm = float(np.linalg.norm(action))
        return np.asarray([1.0, s_val, a_mean, a_norm, s_val * a_mean], dtype=np.float64)

    def _posterior_mean(self) -> np.ndarray:
        return np.linalg.solve(self._a_mat, self._b_vec)

    def _candidate_features(
        self,
        state: torch.Tensor,
        candidate_actions: Iterable[np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = (
            self._candidate_actions()
            if candidate_actions is None
            else np.asarray(list(candidate_actions), dtype=np.float64)
        )
        feature_mat = np.stack([self._features(state, action) for action in candidates], axis=0)
        return candidates, feature_mat

    def _action_from_scores(
        self, candidates: np.ndarray, scores: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        best_idx = int(np.argmax(scores))
        action = torch.as_tensor(candidates[best_idx], dtype=torch.float32, device=self.device)
        return action.view(1, 1, -1), torch.as_tensor([-scores[best_idx]], device=self.device)

    def update(self, batch: dict) -> dict[str, float]:
        reward = float(batch.get("reward", 0.0))
        state = batch.get("state")
        action = batch.get("action")
        if state is None or action is None:
            return {}

        action_vec = np.asarray(
            torch.as_tensor(action).detach().cpu().numpy(), dtype=np.float64
        ).reshape(-1)
        phi = self._features(state, action_vec)
        self._a_mat += np.outer(phi, phi)
        self._b_vec += reward * phi

        return {"posterior_trace": float(np.trace(np.linalg.inv(self._a_mat)))}
