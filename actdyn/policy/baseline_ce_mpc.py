"""Lightweight cross-entropy MPC benchmark baseline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import gymnasium as gym
import torch

from .base import BasePolicy


class BaselineCEMPCPolicy(BasePolicy):
    """Cross-entropy search over open-loop action sequences."""

    def __init__(
        self,
        action_space: gym.Space,
        horizon: int = 8,
        num_samples: int = 64,
        num_iterations: int = 3,
        num_elite: int = 8,
        alpha: float = 0.25,
        init_std: float = 0.5,
        action_penalty: float = 0.05,
        **kwargs,
    ):
        super().__init__(action_space=action_space, **kwargs)
        self.horizon = max(1, int(horizon))
        self.num_samples = max(4, int(num_samples))
        self.num_iterations = max(1, int(num_iterations))
        self.num_elite = max(2, min(int(num_elite), self.num_samples))
        self.alpha = float(alpha)
        self.init_std = float(init_std)
        self.action_penalty = float(action_penalty)

        if not isinstance(self.action_space, gym.spaces.Box):
            raise TypeError("BaselineCEMPCPolicy requires a gymnasium.spaces.Box action space")

        self.action_dim = int(np.prod(self.action_space.shape))
        self._low = torch.as_tensor(self.action_space.low, dtype=torch.float32, device=self.device).view(
            1, 1, -1
        )
        self._high = torch.as_tensor(
            self.action_space.high, dtype=torch.float32, device=self.device
        ).view(1, 1, -1)

        self._bounded = torch.isfinite(self._low).all() and torch.isfinite(self._high).all()
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def _init_distribution(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._bounded:
            mean = 0.5 * (self._low + self._high).expand(1, self.horizon, -1).clone()
            std = 0.5 * (self._high - self._low).expand(1, self.horizon, -1).clone()
            std = torch.clamp(std * self.init_std, min=1e-3)
        else:
            mean = torch.zeros(1, self.horizon, self.action_dim, device=self.device)
            std = torch.full_like(mean, fill_value=max(self.init_std, 1e-3))
        return mean, std

    def _sample(self, mean: torch.Tensor, std: torch.Tensor, num_samples: int) -> torch.Tensor:
        samples = torch.randn(num_samples, self.horizon, self.action_dim, device=self.device)
        actions = samples * std + mean
        if self._bounded:
            actions = torch.clamp(actions, self._low, self._high)
        return actions

    def _default_objective(self, action_sequences: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        integrated = torch.cumsum(action_sequences, dim=1)
        excitation = integrated.var(dim=1).mean(dim=1)
        energy = (action_sequences**2).mean(dim=(1, 2))
        if self.horizon > 1:
            smoothness = ((action_sequences[:, 1:] - action_sequences[:, :-1]) ** 2).mean(dim=(1, 2))
        else:
            smoothness = torch.zeros_like(energy)

        state_scale = 1.0 + float(torch.as_tensor(state).abs().mean().item())
        return -(state_scale * excitation) + self.action_penalty * energy + 0.1 * smoothness

    def get_action(self, state: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        objective_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor | np.ndarray] | None = kwargs.get(
            "objective_fn"
        )

        if self._mean is None or self._std is None:
            self._mean, self._std = self._init_distribution()

        mean = self._mean
        std = self._std
        best_sequence = mean[0]
        best_cost = torch.tensor(float("inf"), device=self.device)

        for _ in range(self.num_iterations):
            sampled = self._sample(mean=mean, std=std, num_samples=self.num_samples)

            if objective_fn is None:
                costs = self._default_objective(sampled, state)
            else:
                values = objective_fn(sampled, state)
                costs = torch.as_tensor(values, dtype=torch.float32, device=self.device)

            if costs.ndim != 1 or costs.shape[0] != sampled.shape[0]:
                raise ValueError(
                    f"objective_fn must return shape ({sampled.shape[0]},), got {tuple(costs.shape)}"
                )

            elite_idx = torch.topk(-costs, k=self.num_elite).indices
            elite = sampled[elite_idx]
            new_mean = elite.mean(dim=0, keepdim=True)
            new_std = elite.std(dim=0, keepdim=True).clamp(min=1e-3)

            mean = (1.0 - self.alpha) * new_mean + self.alpha * mean
            std = (1.0 - self.alpha) * new_std + self.alpha * std

            iter_best = torch.argmin(costs)
            if costs[iter_best] < best_cost:
                best_cost = costs[iter_best]
                best_sequence = sampled[iter_best]

        shifted = mean.clone()
        shifted[:, :-1] = mean[:, 1:]
        self._mean = shifted
        self._std = std

        return best_sequence.unsqueeze(0), best_cost.view(1)
