"""Literature-style Receding Horizon Curiosity baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .base import BasePolicy
from .rhc_blr import RFFBayesianLinearDynamics, RHCObjective
from .rhc_planner import RhcMultipleShootingPlanner


def _to_numpy_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float64)
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    return np.asarray(arr, dtype=np.float64).reshape(-1)


@dataclass
class RhcDiagnostics:
    objective: str
    model_samples: int
    model_lengthscale: float
    episode_updates: int
    planner_cost: float


class RecedingHorizonCuriosityPolicy(BasePolicy):
    """Receding-horizon curiosity with Bayesian RFF dynamics and episodic replanning.

    This follows the structure of Schultheis et al. (CoRL 2020):
    - open-loop multiple-shooting planning over a fixed episode horizon
    - Bayesian linear regression on random Fourier features of ``(x, u)``
    - update the internal model only after executing the entire planned episode
    """

    requires_observed_state = True

    def __init__(
        self,
        *,
        action_space,
        horizon: int,
        device: str = "cpu",
        objective: str = "rhc_us",
        num_features: int = 64,
        prior_precision: float = 1.0,
        obs_noise_var: float = 1e-3,
        lengthscale: float = 1.0,
        optimize_lengthscale: bool = True,
        planner_maxiter: int = 500,
        seed: int = 0,
        **_: Any,
    ) -> None:
        super().__init__(action_space=action_space, chunk=max(1, int(horizon)), device=device)
        self.horizon = max(1, int(horizon))
        objective_name = str(objective)
        if objective_name in {"us", "mvr"}:
            objective_name = f"rhc_{objective_name}"
        self.objective = objective_name
        if self.objective not in {"rhc_us", "rhc_mvr"}:
            raise ValueError(f"Unsupported RHC objective {self.objective!r}")
        self.num_features = int(num_features)
        self.prior_precision = float(prior_precision)
        self.obs_noise_var = float(obs_noise_var)
        self.lengthscale = float(lengthscale)
        self.optimize_lengthscale = bool(optimize_lengthscale)
        self.planner_maxiter = int(max(planner_maxiter, 1))
        self._rng = np.random.default_rng(int(seed))
        self._internal_model: RFFBayesianLinearDynamics | None = None
        self._planner: RhcMultipleShootingPlanner | None = None
        self._episode_inputs: list[np.ndarray] = []
        self._episode_deltas: list[np.ndarray] = []
        self._episode_updates = 0
        self.last_update_info: dict[str, float | int | str] = {}

    def beginning_of_rollout(self, state: torch.Tensor):
        self.count = 0
        self.action_list = []
        self._episode_inputs = []
        self._episode_deltas = []

    def get_action(self, state: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        observed_state = kwargs.get("observed_state", state)
        x0 = _to_numpy_vector(observed_state)
        if x0.size == 0:
            raise ValueError("RHC requires an observed state to plan from")
        self._ensure_model(x0)
        assert self._planner is not None
        plan = self._planner.plan(x0=x0, objective=self.objective)
        action_tensor = torch.as_tensor(
            plan.actions[None, :, :],
            dtype=torch.float32,
            device=self.device,
        )
        cost_tensor = torch.as_tensor([[plan.cost]], dtype=torch.float32, device=self.device)
        self.last_update_info = {
            "objective": self.objective,
            "model_samples": int(self._internal_model.num_samples if self._internal_model else 0),
            "model_lengthscale": float(
                self._internal_model.lengthscale if self._internal_model else self.lengthscale
            ),
            "episode_updates": int(self._episode_updates),
            "planner_cost": float(plan.cost),
        }
        return action_tensor, cost_tensor

    def update(self, batch) -> dict[str, float | int | str]:
        if not isinstance(batch, Mapping):
            return self.last_update_info
        if "env_state" not in batch or "next_env_state" not in batch or "action" not in batch:
            return self.last_update_info
        env_state = _to_numpy_vector(batch["env_state"])
        next_env_state = _to_numpy_vector(batch["next_env_state"])
        action = _to_numpy_vector(batch["action"])
        if env_state.size == 0 or next_env_state.size == 0 or action.size == 0:
            return self.last_update_info
        self._ensure_model(env_state)
        xu = np.concatenate([env_state, action], axis=0)
        delta = next_env_state - env_state
        self._episode_inputs.append(xu)
        self._episode_deltas.append(delta)
        if len(self._episode_inputs) >= self.horizon:
            assert self._internal_model is not None
            self._internal_model.add_episode(
                np.stack(self._episode_inputs, axis=0),
                np.stack(self._episode_deltas, axis=0),
            )
            self._episode_inputs = []
            self._episode_deltas = []
            self._episode_updates += 1
            self.last_update_info = {
                "objective": self.objective,
                "model_samples": int(self._internal_model.num_samples),
                "model_lengthscale": float(self._internal_model.lengthscale),
                "episode_updates": int(self._episode_updates),
                "rhc_episode_index": float(self._episode_updates),
                "planner_cost": float(self.cost),
            }
        return self.last_update_info

    def _ensure_model(self, state: np.ndarray) -> None:
        if self._internal_model is not None:
            return
        state_dim = int(state.shape[0])
        action_dim = int(np.prod(self.action_space.shape))
        self._internal_model = RFFBayesianLinearDynamics(
            input_dim=state_dim + action_dim,
            output_dim=state_dim,
            num_features=self.num_features,
            prior_precision=self.prior_precision,
            obs_noise_var=self.obs_noise_var,
            lengthscale=self.lengthscale,
            seed=int(self._rng.integers(0, 2**31 - 1)),
            optimize_lengthscale=self.optimize_lengthscale,
        )
        self._planner = RhcMultipleShootingPlanner(
            model=self._internal_model,
            action_low=np.asarray(self.action_space.low, dtype=np.float64),
            action_high=np.asarray(self.action_space.high, dtype=np.float64),
            horizon=self.horizon,
            planner_maxiter=self.planner_maxiter,
            rng=self._rng,
        )
