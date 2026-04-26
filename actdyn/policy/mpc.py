import copy
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from actdyn.environment import action
import torch
import colorednoise


from .base import BaseMPC
from actdyn.utils.rollout import RolloutBuffer


def _clone_for_worker(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clone_for_worker(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_for_worker(v) for v in value)
    if hasattr(value, "detach"):
        return value.detach().clone()
    return copy.deepcopy(value)


@dataclass
class _AsyncPlanResult:
    actions: torch.Tensor
    cost: torch.Tensor
    predicted_boundary_state: torch.Tensor
    runtime_sec: float
    status: str
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    elite_actions: torch.Tensor | None = None
    elite_costs_traj: torch.Tensor | None = None


class MpcICem(BaseMPC):
    mean: np.ndarray
    std: np.ndarray
    model_evals_per_timestep: int
    elite_samples: RolloutBuffer

    def __init__(
        self,
        num_samples: int = 32,
        num_iterations: int = 10,
        num_elite: int = 10,
        alpha: float = 0.1,
        init_std: float = 0.5,
        noise_beta: float = 1.0,
        factor_decrease_num: float = 1.25,
        frac_prev_elites: float = 0.2,
        frac_elites_reused: float = 0.3,
        use_mean_actions: bool = True,
        shift_elites: bool = True,
        keep_elites: bool = True,
        action_constraint: str = "box",
        action_radius: float | None = None,
        coarse_dt_factor: int = 1,
        coarse_action_mapping: str = "hold",
        coarse_mapping_opt_steps: int = 25,
        coarse_mapping_opt_lr: float = 0.05,
        **kwargs,
    ):

        # Call parent constructor with correct arguments
        super().__init__(num_samples=num_samples, **kwargs)

        # Set ICEM-specific parameters
        self.alpha = alpha
        self.num_elites = num_elite
        self.num_iterations = num_iterations
        self.init_std = init_std
        self.noise_beta = noise_beta
        self.factor_decrease_num = factor_decrease_num
        self.frac_prev_elites = frac_prev_elites
        self.frac_elites_reused = frac_elites_reused
        self.use_mean_actions = use_mean_actions
        self.shift_elites = shift_elites
        self.keep_elites = keep_elites
        self.action_constraint = str(action_constraint or "box").strip().lower()
        if self.action_constraint not in {"box", "radius"}:
            raise ValueError(f"Unsupported action_constraint={self.action_constraint!r}")
        if action_radius is None and self.action_bounds is not None:
            low = torch.as_tensor(self.action_bounds[0], dtype=torch.float32, device=self.device).reshape(-1)
            high = torch.as_tensor(self.action_bounds[1], dtype=torch.float32, device=self.device).reshape(-1)
            action_radius = float(torch.min(torch.minimum(low.abs(), high.abs())).item())
        self.action_radius = None if action_radius is None else float(max(action_radius, 1e-8))
        self.coarse_dt_factor = max(1, int(coarse_dt_factor))
        self.coarse_action_mapping = str(coarse_action_mapping or "hold").strip().lower()
        if self.coarse_action_mapping not in {"hold", "endpoint_opt"}:
            raise ValueError(f"Unsupported coarse_action_mapping={self.coarse_action_mapping!r}")
        self.coarse_mapping_opt_steps = int(max(coarse_mapping_opt_steps, 0))
        self.coarse_mapping_opt_lr = float(max(coarse_mapping_opt_lr, 0.0))

        self.was_reset = False

    def beginning_of_rollout(self, state: torch.Tensor):
        super().beginning_of_rollout(state=state)
        self.mean = self.get_init_mean()
        self.std = self.get_init_std()
        self.elite_actions = None
        self.elite_costs_traj = None
        self.was_reset = True

        self.model_evals_per_timestep = (
            sum(
                [
                    max(
                        self.num_elites * 2,
                        int(self.num_samples / (self.factor_decrease_num**i)),
                    )
                    for i in range(0, self.num_iterations)
                ]
            )
            * self.horizon
        )

        print(
            f"iCEM using {self.model_evals_per_timestep} evaluations per step "
            f"and {self.model_evals_per_timestep / self.horizon} trajectories per step"
        )

    def end_of_rollout(self, total_time, total_return, mode):
        super().end_of_rollout(total_time, total_return, mode)

    def get_init_mean(self):
        if self.action_bounds is not None:
            mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
            for dim in range(self.action_dim):
                mean[:, dim] = torch.tensor(
                    (self.action_bounds[1][dim] + self.action_bounds[0][dim]) / 2.0,
                    device=self.device,
                )
            return mean
        return torch.zeros(self.horizon, self.action_dim, device=self.device)

    def get_init_std(self):
        if self.action_bounds is not None:
            std = torch.ones(self.horizon, self.action_dim, device=self.device)
            for dim in range(self.action_dim):
                std[:, dim] = torch.tensor(
                    (self.action_bounds[1][dim] - self.action_bounds[0][dim]) / 2.0 * self.init_std,
                    device=self.device,
                )
            return std
        return self.init_std * torch.ones(self.horizon, self.action_dim, device=self.device)

    def _project_actions(self, actions: torch.Tensor) -> torch.Tensor:
        projected = actions
        if self.action_bounds is not None:
            low = torch.as_tensor(
                self.action_bounds[0], dtype=projected.dtype, device=projected.device
            ).view(*([1] * (projected.ndim - 1)), -1)
            high = torch.as_tensor(
                self.action_bounds[1], dtype=projected.dtype, device=projected.device
            ).view(*([1] * (projected.ndim - 1)), -1)
            projected = torch.maximum(torch.minimum(projected, high), low)
        if self.action_constraint == "radius" and self.action_radius is not None:
            norms = torch.linalg.norm(projected, dim=-1, keepdim=True)
            scale = torch.clamp(self.action_radius / norms.clamp_min(1e-8), max=1.0)
            projected = projected * scale
        return projected

    def sample_action_sequences(self, num_samples):
        # Generate action sequences with colored noise
        if self.noise_beta > 0 and self.horizon > 1:
            samples = torch.tensor(
                colorednoise.powerlaw_psd_gaussian(
                    self.noise_beta, size=(num_samples, self.action_dim, self.horizon)
                ),
                device=self.device,
                dtype=torch.float32,
            ).transpose(1, 2)
        else:
            samples = torch.randn(num_samples, self.horizon, self.action_dim, device=self.device)
        actions = samples * self.std + self.mean
        actions = self._project_actions(actions)
        return actions

    def simulate(self, initial_state: torch.Tensor, actions: torch.Tensor):
        # Simulated trajectories using mean prediction. Use no_grad to avoid
        # accumulating autograd history for planning.
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    a_enc = self.model.action_encoder(actions)
                    simulated_paths = self.model.predict(a_enc)
            else:
                a_enc = self.model.action_encoder(actions)
                simulated_paths = self.model.predict(a_enc)
            simulated_paths = torch.cat(
                [self.model._state.repeat(simulated_paths.shape[0], 1, 1), simulated_paths], dim=-2
            )
        rollout = RolloutBuffer(device=self.device)
        rollout.add_dict(
            {
                "action": actions,
                "encoded_action": a_enc,
                "model_state": simulated_paths[:, :-1],
                "next_model_state": simulated_paths[:, 1:],
            }
        )
        return rollout

    @staticmethod
    def _set_model_planning_state(model: Any, state: torch.Tensor) -> None:
        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=getattr(model, "device", state.device if isinstance(state, torch.Tensor) else "cpu"),
        )
        if state_t.ndim == 1:
            state_t = state_t.reshape(1, 1, -1)
        elif state_t.ndim == 2:
            state_t = state_t.unsqueeze(1) if state_t.shape[0] == 1 else state_t.unsqueeze(0)
        model._state = state_t.detach().clone()
        if isinstance(getattr(model, "z", None), dict) and "m" in model.z:
            model.z["m"] = state_t.detach().clone()

    def _encode_actions_for_state(self, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.model.action_encoder is None:
            return actions
        try:
            state_seq = state.expand(actions.shape[0], actions.shape[1], state.shape[-1])
            return self.model.action_encoder(actions, state_seq)
        except TypeError:
            return self.model.action_encoder(actions)

    def _rollout_from_state(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-2] == 0:
            return state.detach().clone()
        with torch.no_grad():
            encoded_actions = self._encode_actions_for_state(actions, state)
            _samples, next_states, _vars = self.model.dynamics.sample_forward(
                init_z=state.detach().clone(),
                action=encoded_actions,
                k_step=actions.shape[-2],
                add_noise=False,
                return_traj=True,
            )
        return next_states[-1].detach().clone()

    def _set_planning_dt_scale(self, scale: float):
        targets = []
        for obj in (self.model, getattr(self.model, "dynamics", None)):
            if obj is not None and hasattr(obj, "dt"):
                targets.append((obj, float(getattr(obj, "dt"))))
                setattr(obj, "dt", float(getattr(obj, "dt")) * float(scale))
        return targets

    @staticmethod
    def _restore_planning_dt(targets) -> None:
        for obj, original_dt in targets:
            setattr(obj, "dt", original_dt)

    def _shift_warm_start(self, steps: int) -> None:
        shifted_mean = self.mean.clone()
        shift = min(max(1, int(steps)), self.horizon)
        if shift < self.horizon:
            shifted_mean[:-shift] = self.mean[shift:]
        self.mean = shifted_mean
        self.std = self.get_init_std()

    def _encoded_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.model.action_encoder is None:
            return actions
        return self.model.action_encoder(actions)

    def _predict_action_trajectory(self, actions: torch.Tensor) -> torch.Tensor:
        encoded_actions = self._encoded_actions(actions)
        _samples, next_states, _vars = self.model.dynamics.sample_forward(
            init_z=self.model._state,
            action=encoded_actions,
            k_step=actions.shape[-2],
            add_noise=False,
            return_traj=True,
        )
        return torch.cat(next_states, dim=-2)

    def _coarse_actions_to_hold(self, coarse_actions: torch.Tensor) -> torch.Tensor:
        factor = self.coarse_dt_factor
        return coarse_actions.repeat_interleave(factor, dim=-2)[..., : self.chunk, :]

    def _map_coarse_actions_endpoint_opt(
        self, coarse_actions: torch.Tensor, coarse_targets: torch.Tensor
    ) -> torch.Tensor:
        hold_actions = self._coarse_actions_to_hold(coarse_actions).detach()
        if self.coarse_mapping_opt_steps <= 0 or self.coarse_mapping_opt_lr <= 0.0:
            return hold_actions

        fine_actions = hold_actions.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([fine_actions], lr=self.coarse_mapping_opt_lr)
        factor = self.coarse_dt_factor
        n_coarse = coarse_actions.shape[-2]
        start_state = self.model._state.detach()
        target_states = coarse_targets.detach()
        hold_detached = hold_actions.detach()

        for _ in range(self.coarse_mapping_opt_steps):
            optimizer.zero_grad()
            candidate = self._project_actions(fine_actions)
            fine_traj = self._predict_action_trajectory(candidate)
            endpoint_loss = torch.zeros((), dtype=fine_traj.dtype, device=fine_traj.device)
            for coarse_idx in range(n_coarse):
                fine_count = min((coarse_idx + 1) * factor, self.chunk)
                if fine_count <= coarse_idx * factor:
                    continue
                target = target_states[:, coarse_idx : coarse_idx + 1, :]
                block_len = fine_count - coarse_idx * factor
                if block_len < factor:
                    prev_target = (
                        start_state
                        if coarse_idx == 0
                        else target_states[:, coarse_idx - 1 : coarse_idx, :]
                    )
                    ratio = float(block_len) / float(factor)
                    target = prev_target + ratio * (target - prev_target)
                endpoint_loss = endpoint_loss + torch.mean(
                    (fine_traj[:, fine_count - 1 : fine_count, :] - target) ** 2
                )

            deviation_loss = torch.mean((candidate - hold_detached) ** 2)
            smooth_loss = torch.zeros((), dtype=fine_traj.dtype, device=fine_traj.device)
            if candidate.shape[-2] > 1:
                smooth_loss = torch.mean((candidate[:, 1:, :] - candidate[:, :-1, :]) ** 2)
            loss = endpoint_loss + 1e-3 * deviation_loss + 1e-3 * smooth_loss
            action_grad = torch.autograd.grad(loss, fine_actions, allow_unused=True)[0]
            if action_grad is None:
                break
            fine_actions.grad = action_grad
            optimizer.step()
            with torch.no_grad():
                fine_actions.copy_(self._project_actions(fine_actions))

        return self._project_actions(fine_actions.detach())

    def _map_coarse_actions(
        self, coarse_actions: torch.Tensor, coarse_targets: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.coarse_action_mapping == "hold":
            return self._coarse_actions_to_hold(coarse_actions)
        if coarse_targets is None:
            raise ValueError("endpoint_opt coarse action mapping requires coarse_targets")
        return self._map_coarse_actions_endpoint_opt(coarse_actions, coarse_targets)

    def _run_icem_search(self, state, *, shift_steps: int = 1, debug=False, **kwargs):
        if not self.was_reset:
            self.beginning_of_rollout(state)

        best_cost = float("inf")
        best_first_action = None
        costs = [float("inf")]

        current_num_samples = self.num_samples
        for iter in range(self.num_iterations):
            # Decay of sample size
            if iter > 0:
                current_num_samples = max(
                    self.num_elites * 2,
                    int(current_num_samples / self.factor_decrease_num),
                )

            # Sample actions from distribution
            actions = self.sample_action_sequences(current_num_samples)

            # Adding mean actions as candidate at the last iteration
            if self.use_mean_actions and iter == self.num_iterations - 1:
                actions[0] = self._project_actions(self.mean.unsqueeze(0)).squeeze(0)

            # Shifting elites over time
            has_elite = self.elite_actions is not None and len(self.elite_actions) > 0
            if iter == 0 and self.shift_elites and has_elite:
                elites_actions = self.elite_actions
                reused_actions = elites_actions[:, 1:]
                num_elites = int(reused_actions.shape[0] * self.frac_elites_reused)
                reused_actions = reused_actions[:num_elites]
                last_actions = self.sample_action_sequences(num_elites)[:, -1:]
                elites_actions = torch.cat([reused_actions, last_actions], dim=-2)
                actions = torch.cat([actions, elites_actions], dim=0)

            # Simulate and Compute Cost (no_grad to avoid retaining graphs)
            rollout = self.simulate(state, actions)
            with torch.no_grad():
                costs = self.metric(rollout, **kwargs).reshape(-1)
                costs = torch.nan_to_num(costs, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
                if not torch.isfinite(costs).any():
                    # If every candidate is invalid, avoid random saturated actions.
                    # Fall back to the current mean plan (typically near-zero and bounded).
                    fallback_action = self._project_actions(self.mean.unsqueeze(0).clone())
                    fallback_cost = torch.full((1,), 1e12, device=self.device)
                    return fallback_action, fallback_cost

            # Keep elites from previous iteration
            if iter > 0 and self.keep_elites:
                num_elites_to_keep = int(len(self.elite_actions) * self.frac_elites_reused)
                if num_elites_to_keep > 0:
                    prev_elites_actions = self.elite_actions
                    prev_elite_costs = self.elite_costs_traj
                    # Ensure prev_elites_actions has the same shape as actions except for the batch dimension
                    assert (
                        actions.shape[1:] == prev_elites_actions.shape[1:]
                    ), f"Shape mismatch: actions {actions.shape}, prev_elites_actions {prev_elites_actions.shape}"
                    actions = torch.cat([actions, prev_elites_actions[:num_elites_to_keep]], dim=0)
                    # Ensure cost dimensions match except for batch dimension
                    if costs.shape[1:] != prev_elite_costs.shape[1:]:
                        prev_elite_costs = prev_elite_costs.view(
                            prev_elite_costs.shape[0], *costs.shape[1:]
                        )
                    costs = torch.cat([costs, prev_elite_costs[:num_elites_to_keep]], dim=0)

            # Get elite samples
            elite_idxs = torch.topk(-costs, self.num_elites, dim=0)[1]
            self.elite_actions = actions[elite_idxs]
            self.elite_costs_traj = costs[elite_idxs]

            # Update best first action if we found better solution
            min_cost_idx = elite_idxs[0]
            if costs[min_cost_idx] < best_cost:
                best_cost = costs[min_cost_idx]
                best_first_action = actions[min_cost_idx, 0]

            # Update mean/std using a numerically stable variance estimate.
            # `unbiased=False` avoids NaNs when num_elites == 1.
            new_mean = self.elite_actions.mean(dim=0).to(self.device)
            new_std = self.elite_actions.std(dim=0, unbiased=False).to(self.device)
            new_mean = torch.nan_to_num(new_mean, nan=0.0, posinf=0.0, neginf=0.0)
            new_std = torch.nan_to_num(new_std, nan=0.0, posinf=self.init_std, neginf=0.0)

            self.mean = (1 - self.alpha) * new_mean + self.alpha * self.mean
            self.std = (1 - self.alpha) * new_std + self.alpha * self.std

            # Print cost for debugging
            if self.verbose:
                print(
                    f"iter {iter}:{current_num_samples} "
                    f"--- best cost: {costs.min()} "
                    f"--- mean: {costs.mean()} "
                    f"--- worst: {costs.max()} "
                )

        self._shift_warm_start(shift_steps)

        return (actions[min_cost_idx].unsqueeze(0), costs[min_cost_idx].unsqueeze(0).detach())

    def get_action(self, state, debug=False, **kwargs):
        if self.coarse_dt_factor <= 1:
            return self._run_icem_search(state, debug=debug, **kwargs)

        n_coarse_execute = int(np.ceil(float(self.chunk) / float(self.coarse_dt_factor)))
        if self.horizon < n_coarse_execute:
            raise ValueError(
                "coarse_dt_factor and chunk require "
                f"{n_coarse_execute} coarse actions, but horizon is only {self.horizon}"
            )

        targets = self._set_planning_dt_scale(self.coarse_dt_factor)
        coarse_plan = None
        coarse_cost = None
        coarse_targets = None
        try:
            coarse_plan, coarse_cost = self._run_icem_search(
                state, shift_steps=n_coarse_execute, debug=debug, **kwargs
            )
            coarse_actions = coarse_plan[:, :n_coarse_execute, :]
            if self.coarse_action_mapping == "endpoint_opt":
                with torch.no_grad():
                    coarse_targets = self._predict_action_trajectory(coarse_actions)
        finally:
            self._restore_planning_dt(targets)

        assert coarse_plan is not None and coarse_cost is not None
        coarse_actions = coarse_plan[:, :n_coarse_execute, :]
        fine_actions = self._map_coarse_actions(coarse_actions, coarse_targets=coarse_targets)
        return fine_actions, coarse_cost


class AsyncMpcICem(MpcICem):
    """Double-buffered asynchronous iCEM planner.

    The live control loop only swaps action buffers. Background planning runs on
    a deep-copied policy/model snapshot so iCEM warm starts and temporary dt
    scaling do not mutate the live planner.
    """

    def __init__(
        self,
        *,
        async_stale_tolerance: float = 0.5,
        async_stale_refine_iterations: int = 2,
        async_worker_backend: str = "thread",
        async_start_after_first_plan: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.async_stale_tolerance = float(max(async_stale_tolerance, 0.0))
        self.async_stale_refine_iterations = int(max(async_stale_refine_iterations, 0))
        self.async_worker_backend = str(async_worker_backend or "thread").strip().lower()
        if self.async_worker_backend != "thread":
            raise ValueError("AsyncMpcICem v1 only supports async_worker_backend='thread'")
        self.async_start_after_first_plan = bool(async_start_after_first_plan)

        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1)
        self._planning_future: Future | None = None
        self._current_buffer: torch.Tensor | None = None
        self._buffer_index = 0
        self._predicted_boundary_state: torch.Tensor | None = None
        self.last_plan_status: dict[str, Any] = self._empty_plan_status()
        self.last_update_info: dict[str, Any] = dict(self.last_plan_status)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_executor"] = None
        state["_planning_future"] = None
        state["_current_buffer"] = None
        state["_predicted_boundary_state"] = None
        return state

    def _empty_plan_status(self) -> dict[str, Any]:
        return {
            "async_plan_ready": False,
            "async_plan_used": False,
            "async_plan_stale": False,
            "async_boundary_mismatch": 0.0,
            "async_refined": False,
            "async_blocking_fallback": False,
            "async_plan_runtime_sec": 0.0,
            "async_plan_status": "idle",
        }

    def _reset_executor(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._planning_future = None

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._planning_future = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def beginning_of_rollout(self, state: torch.Tensor):
        super().beginning_of_rollout(state)
        self._reset_executor()
        self._current_buffer = None
        self._buffer_index = 0
        self._predicted_boundary_state = None
        self.last_plan_status = self._empty_plan_status()
        self.last_update_info = dict(self.last_plan_status)

    def end_of_rollout(self, *args, **kwargs):
        self.close()
        return super().end_of_rollout(*args, **kwargs)

    def _normalize_action_buffer(self, actions: torch.Tensor) -> torch.Tensor:
        action_seq = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if action_seq.ndim == 2:
            action_seq = action_seq.unsqueeze(0)
        if action_seq.shape[-2] == 1:
            action_seq = action_seq.repeat(1, self.chunk, 1)
        elif action_seq.shape[-2] >= self.chunk:
            action_seq = action_seq[:, : self.chunk, :]
        else:
            raise ValueError(
                f"AsyncMpcICem received action sequence length {action_seq.shape[-2]}, "
                f"but chunk is {self.chunk}"
            )
        return action_seq.detach().clone()

    def _set_current_buffer(self, actions: torch.Tensor, cost: torch.Tensor) -> None:
        self._current_buffer = self._normalize_action_buffer(actions)
        self.action_list = [a.unsqueeze(0) for a in self._current_buffer[0]]
        self._buffer_index = 0
        self.cost = torch.as_tensor(cost).squeeze().item()

    def _sync_plan(
        self,
        state: torch.Tensor,
        *,
        num_iterations: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._set_model_planning_state(self.model, state)
        original_iterations = self.num_iterations
        if num_iterations is not None:
            self.num_iterations = max(1, int(num_iterations))
        try:
            return MpcICem.get_action(self, state, **kwargs)
        finally:
            self.num_iterations = original_iterations

    def _boundary_mismatch(
        self, real_state: torch.Tensor, predicted_state: torch.Tensor | None
    ) -> float:
        if predicted_state is None:
            return float("inf")
        real = torch.as_tensor(real_state, dtype=torch.float32, device=self.device).reshape(-1)
        pred = torch.as_tensor(predicted_state, dtype=torch.float32, device=self.device).reshape(-1)
        denom = max(1.0, float(torch.linalg.norm(real).item()))
        return float(torch.linalg.norm(real - pred).item() / denom)

    def _apply_worker_warm_start(self, result: _AsyncPlanResult) -> None:
        for attr in ("mean", "std", "elite_actions", "elite_costs_traj"):
            value = getattr(result, attr, None)
            if value is not None:
                setattr(self, attr, value.detach().clone().to(self.device))

    def _consume_completed_future(self) -> _AsyncPlanResult | None:
        if self._planning_future is None or not self._planning_future.done():
            return None
        future = self._planning_future
        self._planning_future = None
        return future.result()

    def _make_snapshot_planner(self, predicted_boundary_state: torch.Tensor):
        try:
            planner = copy.deepcopy(self)
        except Exception as exc:
            raise RuntimeError(
                "AsyncMpcICem could not deepcopy the policy/model snapshot. "
                "Use policy_type='mpc-icem' or make the model deepcopy-compatible."
            ) from exc
        planner.close()
        planner._executor = None
        planner._planning_future = None
        planner._current_buffer = None
        planner._predicted_boundary_state = None
        planner._set_model_planning_state(planner.model, predicted_boundary_state)
        return planner

    @staticmethod
    def _background_plan_worker(
        planner,
        predicted_boundary_state: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> _AsyncPlanResult:
        start = time.perf_counter()
        actions, cost = MpcICem.get_action(planner, predicted_boundary_state, **kwargs)
        runtime = time.perf_counter() - start
        return _AsyncPlanResult(
            actions=actions.detach().clone(),
            cost=cost.detach().clone(),
            predicted_boundary_state=predicted_boundary_state.detach().clone(),
            runtime_sec=float(runtime),
            status="completed",
            mean=getattr(planner, "mean", None),
            std=getattr(planner, "std", None),
            elite_actions=getattr(planner, "elite_actions", None),
            elite_costs_traj=getattr(planner, "elite_costs_traj", None),
        )

    def _launch_background_plan(self, state: torch.Tensor, kwargs: dict[str, Any]) -> None:
        if self._current_buffer is None or self._planning_future is not None:
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        remaining = self._current_buffer[:, self._buffer_index :, :]
        if remaining.shape[-2] == 0:
            predicted_boundary = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            if predicted_boundary.ndim == 1:
                predicted_boundary = predicted_boundary.reshape(1, 1, -1)
        else:
            predicted_boundary = self._rollout_from_state(
                torch.as_tensor(state, dtype=torch.float32, device=self.device),
                remaining,
            )
        self._predicted_boundary_state = predicted_boundary.detach().clone()
        worker_kwargs = _clone_for_worker(kwargs)
        worker_kwargs["observed_state"] = predicted_boundary.detach().clone()
        planner = self._make_snapshot_planner(predicted_boundary)
        self._planning_future = self._executor.submit(
            self._background_plan_worker,
            planner,
            predicted_boundary.detach().clone(),
            worker_kwargs,
        )

    def _activate_new_chunk(self, state: torch.Tensor, kwargs: dict[str, Any]) -> None:
        status = self._empty_plan_status()
        result: _AsyncPlanResult | None = None
        if self._current_buffer is None:
            actions, cost = self._sync_plan(state, **kwargs)
            status["async_blocking_fallback"] = True
            status["async_plan_status"] = "initial_blocking"
            self._set_current_buffer(actions, cost)
        elif self._planning_future is not None and self._planning_future.done():
            try:
                result = self._consume_completed_future()
                status["async_plan_ready"] = result is not None
            except Exception as exc:
                result = None
                status["async_blocking_fallback"] = True
                status["async_plan_status"] = f"worker_error:{type(exc).__name__}"
            if result is not None:
                mismatch = self._boundary_mismatch(state, result.predicted_boundary_state)
                status["async_boundary_mismatch"] = mismatch
                status["async_plan_runtime_sec"] = float(result.runtime_sec)
                if mismatch <= self.async_stale_tolerance:
                    self._apply_worker_warm_start(result)
                    self._set_current_buffer(result.actions, result.cost)
                    status["async_plan_used"] = True
                    status["async_plan_status"] = "used_ready"
                else:
                    actions, cost = self._sync_plan(
                        state,
                        num_iterations=self.async_stale_refine_iterations,
                        **kwargs,
                    )
                    self._set_current_buffer(actions, cost)
                    status["async_plan_stale"] = True
                    status["async_refined"] = True
                    status["async_plan_status"] = "refined_stale"
        if self._current_buffer is not None and self._buffer_index >= self.chunk:
            actions, cost = self._sync_plan(state, **kwargs)
            self._set_current_buffer(actions, cost)
            status["async_blocking_fallback"] = True
            if status["async_plan_status"] == "idle":
                status["async_plan_status"] = "blocking_fallback"

        self.last_plan_status = status
        self.last_update_info = dict(status)

    def __call__(self, state, **kwargs) -> torch.Tensor:
        if not self.was_reset:
            self.beginning_of_rollout(state)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_t.ndim == 1:
            state_t = state_t.reshape(1, 1, -1)
        elif state_t.ndim == 2:
            state_t = state_t.unsqueeze(1) if state_t.shape[0] == 1 else state_t.unsqueeze(0)

        at_boundary = self._current_buffer is None or self._buffer_index >= self.chunk
        if at_boundary:
            self._activate_new_chunk(state_t, dict(kwargs))

        if self.async_start_after_first_plan or self.count > 0:
            self._launch_background_plan(state_t, dict(kwargs))

        assert self._current_buffer is not None
        action = self._current_buffer[:, self._buffer_index : self._buffer_index + 1, :]
        self._buffer_index += 1
        self.count += 1
        return action

    def update(self, batch) -> dict[str, Any]:
        return dict(self.last_plan_status)
