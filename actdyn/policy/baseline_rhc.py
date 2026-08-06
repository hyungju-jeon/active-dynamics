"""Paper-faithful RHC baseline adapted to horizon-as-episode execution.

This keeps the RHC structure from Schultheis et al. while adapting the outer
loop to the repo's online driver: one executed planning horizon is treated as one
episode, and the Bayesian dynamics model is updated exactly once at the end of
that horizon.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from actdyn.models.planning_surrogates import LocalRBFBayesianLinearDynamics, RFFBayesianLinearDynamics

from .base import BasePolicy
from .rhc_planner import RhcMultipleShootingPlanner


def _to_numpy_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float64)
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _symmetric_bounds(bound: float | np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(bound, dtype=np.float64)
    if arr.ndim == 0:
        mag = np.repeat(abs(float(arr)), dim)
    else:
        flat = arr.reshape(-1)
        if flat.shape[0] != dim:
            raise ValueError(f"Expected state_bound with {dim} entries, got {flat.shape[0]}")
        mag = np.abs(flat)
    low = -np.maximum(mag, 1e-6)
    high = np.maximum(mag, 1e-6)
    return low, high


class RecedingHorizonCuriosityPolicy(BasePolicy):
    """Exact RHC with open-loop planning and horizon-end Bayesian updates."""

    requires_observed_state = True

    def __init__(
        self,
        *,
        action_space,
        horizon: int,
        device: str = 'cpu',
        objective: str = 'rhc_us',
        num_features: int = 64,
        prior_precision: float = 1e-8,
        beta: float | None = None,
        obs_noise_var: float | None = None,
        bandwidth_init: float | np.ndarray = 1.0,
        lengthscale: float | None = None,
        optimize_hyperparams: bool = True,
        optimize_lengthscale: bool | None = None,
        planner_maxiter: int = 500,
        seed: int = 0,
        warm_start: bool = False,
        surrogate_kind: str = 'rff',
        rbf_grid_points: int = 3,
        rbf_epsilon: float = 0.01,
        state_bound: float | np.ndarray = 5.0,
        rbf_include_bias: bool = False,
        **_: Any,
    ) -> None:
        super().__init__(action_space=action_space, chunk=max(1, int(horizon)), device=device)
        self.horizon = max(1, int(horizon))
        objective_name = str(objective)
        if objective_name in {'us', 'mvr'}:
            objective_name = f'rhc_{objective_name}'
        if objective_name not in {'rhc_us', 'rhc_mvr'}:
            raise ValueError(f'Unsupported RHC objective {objective_name!r}')
        self.objective = objective_name
        self.num_features = int(num_features)
        self.prior_precision = float(prior_precision)
        if beta is None:
            if obs_noise_var is None:
                beta = 1.0
            else:
                beta = 1.0 / float(max(obs_noise_var, 1e-12))
        self.beta = float(max(beta, 1e-12))
        if lengthscale is not None:
            bandwidth_init = lengthscale
        self.bandwidth_init = bandwidth_init
        self.optimize_hyperparams = bool(
            optimize_hyperparams if optimize_lengthscale is None else optimize_lengthscale
        )
        self.planner_maxiter = int(max(planner_maxiter, 1))
        self.warm_start = bool(warm_start)
        self.surrogate_kind = str(surrogate_kind)
        self.rbf_grid_points = int(max(rbf_grid_points, 1))
        self.rbf_epsilon = float(np.clip(rbf_epsilon, 1e-8, 0.5))
        self.state_bound = state_bound
        self.rbf_include_bias = bool(rbf_include_bias)
        self._rng = np.random.default_rng(int(seed))
        self._internal_model: RFFBayesianLinearDynamics | LocalRBFBayesianLinearDynamics | None = None
        self._planner: RhcMultipleShootingPlanner | None = None
        self._episode_inputs: list[np.ndarray] = []
        self._episode_deltas: list[np.ndarray] = []
        self._episode_updates = 0
        self.last_update_info: dict[str, float | int | str | bool] = {
            'parameter_posterior_updated': False,
            'objective': self.objective,
            'surrogate_kind': self.surrogate_kind,
            'model_samples': 0,
            'model_lengthscale': float(np.mean(np.asarray(bandwidth_init, dtype=np.float64)))
            if not np.isscalar(bandwidth_init)
            else float(bandwidth_init),
            'episode_updates': 0,
            'planner_cost': 0.0,
        }

    def beginning_of_rollout(self, state: torch.Tensor):
        self.count = 0
        self.action_list = []
        self._episode_inputs = []
        self._episode_deltas = []

    def get_action(self, state: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        observed_state = kwargs.get('observed_state', state)
        x0 = _to_numpy_vector(observed_state)
        if x0.size == 0:
            raise ValueError('RHC requires an observed state to plan from')
        self._ensure_model(x0)
        assert self._planner is not None
        plan = self._planner.plan(x0=x0, objective=self.objective)
        action_tensor = torch.as_tensor(plan.actions[None, :, :], dtype=torch.float32, device=self.device)
        cost_tensor = torch.as_tensor([[plan.cost]], dtype=torch.float32, device=self.device)
        assert self._internal_model is not None
        self.last_update_info = {
            'parameter_posterior_updated': False,
            'objective': self.objective,
            'surrogate_kind': self.surrogate_kind,
            'model_samples': int(self._internal_model.num_samples),
            'model_lengthscale': float(self._internal_model.lengthscale),
            'episode_updates': int(self._episode_updates),
            'planner_cost': float(plan.cost),
        }
        return action_tensor, cost_tensor

    def update(self, batch) -> dict[str, float | int | str | bool]:
        if not isinstance(batch, Mapping):
            return self.last_update_info
        if 'env_state' not in batch or 'next_env_state' not in batch:
            return self.last_update_info
        env_state = _to_numpy_vector(batch['env_state'])
        next_env_state = _to_numpy_vector(batch['next_env_state'])
        action_value = batch.get('env_action', batch.get('action'))
        action = _to_numpy_vector(action_value)
        if env_state.size == 0 or next_env_state.size == 0 or action.size == 0:
            return self.last_update_info
        self._ensure_model(env_state)
        xu = np.concatenate([env_state, action], axis=0)
        delta = next_env_state - env_state
        self._episode_inputs.append(xu)
        self._episode_deltas.append(delta)
        updated = False
        if len(self._episode_inputs) >= self.horizon:
            assert self._internal_model is not None
            self._internal_model.add_episode(
                np.stack(self._episode_inputs, axis=0),
                np.stack(self._episode_deltas, axis=0),
            )
            self._episode_inputs = []
            self._episode_deltas = []
            self._episode_updates += 1
            updated = True
        assert self._internal_model is not None
        self.last_update_info = {
            'parameter_posterior_updated': updated,
            'objective': self.objective,
            'surrogate_kind': self.surrogate_kind,
            'model_samples': int(self._internal_model.num_samples),
            'model_lengthscale': float(self._internal_model.lengthscale),
            'episode_updates': int(self._episode_updates),
            'rhc_episode_index': float(self._episode_updates),
            'planner_cost': float(self.cost),
        }
        return self.last_update_info

    def _ensure_model(self, state: np.ndarray) -> None:
        if self._internal_model is not None:
            return
        state_dim = int(state.shape[0])
        action_low = np.asarray(self.action_space.low, dtype=np.float64).reshape(-1)
        action_high = np.asarray(self.action_space.high, dtype=np.float64).reshape(-1)
        action_dim = int(action_low.shape[0])
        if self.surrogate_kind == 'local_rbf':
            state_low, state_high = _symmetric_bounds(self.state_bound, state_dim)
            self._internal_model = LocalRBFBayesianLinearDynamics(
                input_dim=state_dim + action_dim,
                output_dim=state_dim,
                input_low=np.concatenate((state_low, action_low), axis=0),
                input_high=np.concatenate((state_high, action_high), axis=0),
                grid_points=self.rbf_grid_points,
                epsilon=self.rbf_epsilon,
                beta=self.beta,
                prior_precision=self.prior_precision,
                include_bias=self.rbf_include_bias,
            )
        elif self.surrogate_kind == 'rff':
            self._internal_model = RFFBayesianLinearDynamics(
                input_dim=state_dim + action_dim,
                output_dim=state_dim,
                num_features=self.num_features,
                bandwidth_init=self.bandwidth_init,
                beta=self.beta,
                prior_precision=self.prior_precision,
                seed=int(self._rng.integers(0, 2**31 - 1)),
                optimize_hyperparams=self.optimize_hyperparams,
            )
        else:
            raise ValueError(f'Unsupported surrogate_kind {self.surrogate_kind!r}')
        self._planner = RhcMultipleShootingPlanner(
            model=self._internal_model,
            action_low=action_low,
            action_high=action_high,
            horizon=self.horizon,
            planner_maxiter=self.planner_maxiter,
            warm_start=self.warm_start,
            rng=self._rng,
        )
