"""Pseudo-random binary sequence (PRBS) benchmark baseline."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
import torch

from .base import BasePolicy


def _prbs_selection_order(num_items: int, budget: int) -> np.ndarray:
    if num_items <= 0 or budget <= 0:
        return np.zeros((0,), dtype=np.int64)
    anchors = np.linspace(0.0, float(num_items - 1), int(budget))
    order: list[int] = []
    seen: set[int] = set()
    for idx in np.round(anchors).astype(int).tolist():
        idx_i = int(np.clip(idx, 0, num_items - 1))
        if idx_i not in seen:
            seen.add(idx_i)
            order.append(idx_i)
    for idx in range(num_items):
        if idx not in seen:
            order.append(idx)
    return np.asarray(order[: min(num_items, budget)], dtype=np.int64)


class BaselinePRBSPolicy(BasePolicy):
    """PRBS policy that holds binary-valued actions for fixed intervals."""

    def __init__(
        self,
        action_space: gym.Space,
        hold_steps: int = 5,
        amplitude: float = 1.0,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(action_space=action_space, **kwargs)
        self.hold_steps = max(1, int(hold_steps))
        self.amplitude = float(amplitude)
        self._step = 0
        self._rng = np.random.default_rng(seed)

        self._low = np.asarray(getattr(self.action_space, "low", -1.0), dtype=np.float32)
        self._high = np.asarray(getattr(self.action_space, "high", 1.0), dtype=np.float32)
        self._bounded = np.all(np.isfinite(self._low)) and np.all(np.isfinite(self._high))

        action_dim = int(np.prod(self.action_space.shape))
        self._current = np.zeros(action_dim, dtype=np.float32)

    def _sample_action(self) -> np.ndarray:
        signs = self._rng.choice([-1.0, 1.0], size=self._current.shape[0]).astype(np.float32)

        if self._bounded:
            center = 0.5 * (self._high.reshape(-1) + self._low.reshape(-1))
            radius = 0.5 * (self._high.reshape(-1) - self._low.reshape(-1))
            action = center + signs * radius * self.amplitude
            return np.clip(action, self._low.reshape(-1), self._high.reshape(-1)).astype(np.float32)

        return (signs * self.amplitude).astype(np.float32)

    def get_action(self, state: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        del state, kwargs
        if self._step % self.hold_steps == 0:
            self._current = self._sample_action()
        self._step += 1

        action = torch.as_tensor(self._current, dtype=torch.float32, device=self.device)
        return action.view(1, 1, -1), torch.zeros(1, device=self.device)
