import copy
import time
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from actdyn.environment import action
import torch
import colorednoise


from .base import BaseMPC
from actdyn.utils.torch_utils import safe_cholesky, symmetrize
from actdyn.utils.rollout import RolloutBuffer


def _clone_for_worker(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clone_for_worker(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_for_worker(v) for v in value)
    if hasattr(value, "detach"):
        return value.detach().clone()
    return copy.deepcopy(value)


def _move_tensors_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_tensors_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(v, device) for v in value)
    return value


def _move_snapshot_to_device(planner: Any, device: torch.device) -> None:
    planner.device = device
    if callable(getattr(planner.model, "to", None)):
        planner.model.to(device)
    planner.model.device = device
    for attr in ("e", "z"):
        if hasattr(planner.model, attr):
            setattr(planner.model, attr, _move_tensors_to_device(getattr(planner.model, attr), device))
    for attr in (
        "_state",
        "_theta_score_block",
        "_theta_info_block",
        "_theta_sensitivity",
        "_last_theta_score_block_applied",
        "_last_theta_info_block_applied",
    ):
        value = getattr(planner.model, attr, None)
        if torch.is_tensor(value):
            setattr(planner.model, attr, value.to(device))
    if callable(getattr(planner.model, "set_params", None)) and hasattr(planner.model, "e"):
        planner.model.set_params(planner.model.e["m"])

    for attr in ("mean", "std", "elite_actions", "elite_costs_traj"):
        value = getattr(planner, attr, None)
        if torch.is_tensor(value):
            setattr(planner, attr, value.to(device))

    for metric in getattr(planner.metric, "metric_list", [planner.metric]):
        metric.device = device
        for attr in ("current_cost", "I", "member_weights", "weights"):
            value = getattr(metric, attr, None)
            if torch.is_tensor(value):
                setattr(metric, attr, value.to(device))
        if hasattr(metric, "ensemble_members"):
            metric.ensemble_members = _move_tensors_to_device(metric.ensemble_members, device)
    planner.metric.device = device
    if torch.is_tensor(getattr(planner.metric, "weights", None)):
        planner.metric.weights = planner.metric.weights.to(device)


def _rollout_has_model_state(rollout: Any) -> bool:
    return (
        rollout is not None
        and len(rollout) > 0
        and callable(getattr(rollout, "get", None))
        and rollout.get("model_state") is not None
    )


def _update_metric_from_rollout(metric: Any, rollout: Any) -> bool:
    if not _rollout_has_model_state(rollout):
        return False
    for item in getattr(metric, "metric_list", [metric]):
        update = getattr(item, "update", None)
        if callable(update):
            update(rollout)
    return True


@dataclass
class _AsyncPlanResult:
    actions: torch.Tensor
    cost: torch.Tensor
    predicted_boundary_state: torch.Tensor
    runtime_sec: float
    status: str
    model_update_version: int = 0
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    elite_actions: torch.Tensor | None = None
    elite_costs_traj: torch.Tensor | None = None
    reanchor_count: int = 0
    reanchor_mismatch: float = 0.0


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
        adaptive_replanning: bool = False,
        adaptive_replan_min_interval: int = 1,
        adaptive_replan_state_error_threshold: float | None = None,
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
        self.adaptive_replanning = bool(adaptive_replanning)
        self.force_replan_on_parameter_update = self.adaptive_replanning
        self.adaptive_replan_min_interval = max(1, int(adaptive_replan_min_interval))
        self.adaptive_replan_state_error_threshold = (
            None
            if adaptive_replan_state_error_threshold is None
            else float(adaptive_replan_state_error_threshold)
        )

        self.was_reset = False
        self._current_replan_interval = int(self.chunk)
        self._chunk_step = 0
        self._planned_state_trace = None
        self._last_action_index = None
        self._last_state_tracking_error = None
        self._force_replan_next = False
        self._force_replan_reason = None
        self.last_update_info: dict[str, Any] = {}
        self.last_plan_info: dict[str, Any] = {
            "plan_executed": False,
            "plan_reason": "none",
        }
        self._foreground_active = None
        self._yield_to_foreground = False

    def beginning_of_rollout(self, state: torch.Tensor):
        super().beginning_of_rollout(state=state)
        self.mean = self.get_init_mean()
        self.std = self.get_init_std()
        self.elite_actions = None
        self.elite_costs_traj = None
        self.was_reset = True
        self.action_list = []
        self._current_replan_interval = int(self.chunk)
        self._chunk_step = 0
        self._planned_state_trace = None
        self._last_action_index = None
        self._last_state_tracking_error = None
        self._force_replan_next = False
        self._force_replan_reason = None
        self.last_update_info = {}
        self.last_plan_info = {"plan_executed": False, "plan_reason": "none"}

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
        with torch.inference_mode():
            if actions.device.type == "cuda":
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
        with torch.inference_mode():
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

    def _coarse_actions_to_hold(
        self, coarse_actions: torch.Tensor, fine_steps: int | None = None
    ) -> torch.Tensor:
        factor = self.coarse_dt_factor
        steps = self.chunk if fine_steps is None else max(1, int(fine_steps))
        return coarse_actions.repeat_interleave(factor, dim=-2)[..., :steps, :]

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
        self,
        coarse_actions: torch.Tensor,
        coarse_targets: torch.Tensor | None = None,
        fine_steps: int | None = None,
    ) -> torch.Tensor:
        if self.coarse_action_mapping == "hold":
            return self._coarse_actions_to_hold(coarse_actions, fine_steps=fine_steps)
        if coarse_targets is None:
            raise ValueError("endpoint_opt coarse action mapping requires coarse_targets")
        return self._map_coarse_actions_endpoint_opt(coarse_actions, coarse_targets)

    def request_replan(self, reason: str) -> None:
        """Force the next control query to run iCEM before returning an action."""
        self._force_replan_next = True
        self._force_replan_reason = str(reason)

    def _normalize_action_sequence(self, actions: torch.Tensor, interval: int) -> torch.Tensor:
        interval = max(1, int(interval))
        action_seq = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if action_seq.ndim == 2:
            action_seq = action_seq.unsqueeze(0)
        if action_seq.shape[-2] == 1:
            action_seq = action_seq.repeat(1, interval, 1)
        elif action_seq.shape[-2] >= interval:
            action_seq = action_seq[:, :interval, :]
        else:
            raise ValueError(
                f"Action sequence length {action_seq.shape[-2]} is less than interval {interval}"
            )
        return action_seq.detach().clone()

    def _set_action_chunk(self, actions: torch.Tensor, cost: torch.Tensor, interval: int) -> None:
        self._current_replan_interval = max(1, int(interval))
        sequence = self._normalize_action_sequence(actions, self._current_replan_interval)
        self.action_list = [a.unsqueeze(0) for a in sequence[0]]
        self.cost = torch.as_tensor(cost).squeeze().item()
        self._chunk_step = 0
        with torch.no_grad():
            self._planned_state_trace = self._predict_action_trajectory(sequence).detach().clone()

    def _state_tracking_error(self, filtered_state: torch.Tensor) -> float | None:
        if self._planned_state_trace is None or self._last_action_index is None:
            return None
        if self._last_action_index >= self._planned_state_trace.shape[-2]:
            return None
        z_hat = torch.as_tensor(filtered_state, dtype=torch.float32, device=self.device)
        if z_hat.ndim == 1:
            z_hat = z_hat.reshape(1, 1, -1)
        elif z_hat.ndim == 2:
            z_hat = z_hat.unsqueeze(1) if z_hat.shape[0] == 1 else z_hat.unsqueeze(0)
        z_plan = self._planned_state_trace[:, self._last_action_index : self._last_action_index + 1]
        dz = z_hat.shape[-1]
        delta = (z_hat - z_plan).reshape(z_hat.shape[0], dz, 1)

        P = getattr(self.model, "z", {}).get("P")
        if P is None:
            P = torch.eye(dz, device=self.device).unsqueeze(0)
        P = torch.as_tensor(P, dtype=torch.float32, device=self.device)
        if P.ndim == 4:
            P = P.squeeze(1)
        elif P.ndim == 2:
            P = P.unsqueeze(0)
        if P.shape[0] == 1 and delta.shape[0] > 1:
            P = P.expand(delta.shape[0], -1, -1)
        eye = torch.eye(dz, device=self.device).unsqueeze(0).expand(P.shape[0], -1, -1)
        cov = symmetrize(P + 1e-6 * eye)
        chol = safe_cholesky(cov)
        quad = delta.transpose(-1, -2) @ torch.cholesky_solve(delta, chol)
        err = quad.reshape(-1) / float(dz)
        return float(torch.nan_to_num(err.mean(), nan=0.0, posinf=1e6, neginf=0.0).item())

    def _latest_next_model_state(self, batch) -> torch.Tensor | None:
        if isinstance(batch, dict):
            value = batch.get("next_model_state")
        elif hasattr(batch, "get"):
            value = batch.get("next_model_state")
        else:
            value = None
        if value is None:
            return None
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim >= 3:
            return tensor[:, -1:]
        if tensor.ndim == 2:
            return tensor.unsqueeze(1)
        if tensor.ndim == 1:
            return tensor.reshape(1, 1, -1)
        return None

    def _run_icem_search(self, state, *, shift_steps: int = 1, debug=False, **kwargs):
        if not self.was_reset:
            self.beginning_of_rollout(state)

        best_cost = float("inf")
        best_first_action = None
        costs = [float("inf")]

        foreground_active = None
        if bool(getattr(self, "_yield_to_foreground", False)):
            foreground_active = getattr(self, "_foreground_active", None)
            for metric in getattr(self.metric, "metric_list", [self.metric]):
                setattr(metric, "_foreground_active", foreground_active)
        maybe_reanchor = getattr(self, "_maybe_reanchor_worker_plan", None)

        current_num_samples = self.num_samples
        for iter in range(self.num_iterations):
            while foreground_active is not None and foreground_active.is_set():
                time.sleep(0.0005)
            if callable(maybe_reanchor):
                state = maybe_reanchor(state, kwargs)

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

            def score_actions(candidate_actions: torch.Tensor) -> torch.Tensor:
                rollout = self.simulate(state, candidate_actions)
                with torch.no_grad():
                    candidate_costs = self.metric(rollout, **kwargs).reshape(-1)
                    return torch.nan_to_num(
                        candidate_costs,
                        nan=float("inf"),
                        posinf=float("inf"),
                        neginf=float("inf"),
                    )

            # Simulate and compute cost. If a background worker receives a newer
            # live boundary while scoring, re-score the same candidate set before
            # selecting elites so stale candidate costs are not mixed with fresh state.
            costs = score_actions(actions)
            if callable(maybe_reanchor):
                reanchor_count = int(getattr(self, "_worker_reanchor_count", 0))
                state = maybe_reanchor(state, kwargs)
                if int(getattr(self, "_worker_reanchor_count", 0)) > reanchor_count:
                    costs = score_actions(actions)
            if not torch.isfinite(costs).any():
                # If every candidate is invalid, avoid random saturated actions.
                # Fall back to the current mean plan (typically near-zero and bounded).
                fallback_action = self._project_actions(self.mean.unsqueeze(0).clone())
                fallback_cost = torch.full((1,), 1e12, device=self.device)
                return fallback_action, fallback_cost

            # Keep elites from previous iteration
            if iter > 0 and self.keep_elites and self.elite_actions is not None:
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
        fine_steps = int(self.chunk)
        if bool(getattr(self, "_return_planning_tail", False)):
            fine_steps = min(int(self.chunk) * 2, int(self.horizon) * int(self.coarse_dt_factor))
        n_coarse_return = int(np.ceil(float(fine_steps) / float(self.coarse_dt_factor)))
        coarse_actions = coarse_plan[:, : max(n_coarse_execute, n_coarse_return), :]
        fine_actions = self._map_coarse_actions(
            coarse_actions, coarse_targets=coarse_targets, fine_steps=fine_steps
        )
        return fine_actions, coarse_cost

    def __call__(self, state, **kwargs) -> torch.Tensor:
        if not self.was_reset:
            self.beginning_of_rollout(state)

        need_plan = (
            not self.action_list
            or self._chunk_step >= self._current_replan_interval
            or bool(self._force_replan_next)
        )
        plan_info = {"plan_executed": False, "plan_reason": "none"}
        if need_plan:
            forced_reason = self._force_replan_reason if self._force_replan_next else None
            actions, cost = self.get_action(state, **kwargs)
            self._set_action_chunk(actions, cost, int(self.chunk))
            self._force_replan_next = False
            self._force_replan_reason = None
            plan_info = {
                "plan_executed": True,
                "plan_reason": "cadence" if forced_reason is None else forced_reason,
            }
            self.last_update_info = {
                **self.last_update_info,
                "adaptive_replan_triggered": bool(forced_reason),
                "adaptive_replan_reason": "cadence" if forced_reason is None else forced_reason,
                "adaptive_replan_interval": int(self._current_replan_interval),
                "adaptive_state_tracking_error": self._last_state_tracking_error,
            }
        self.last_plan_info = plan_info

        action = self.action_list[self._chunk_step]
        self._last_action_index = int(self._chunk_step)
        self._chunk_step += 1
        self.count += 1
        return action


    def update(self, batch) -> dict[str, Any]:
        next_state = self._latest_next_model_state(batch)
        if next_state is None:
            return dict(self.last_update_info)

        state_error = self._state_tracking_error(next_state)
        self._last_state_tracking_error = state_error
        replan_triggered = False
        replan_reason = "none"
        if (
            self.adaptive_replanning
            and self.adaptive_replan_state_error_threshold is not None
            and state_error is not None
            and self._chunk_step >= min(self.adaptive_replan_min_interval, self.chunk)
            and state_error > self.adaptive_replan_state_error_threshold
        ):
            self.request_replan("state_tracking_error")
            replan_triggered = True
            replan_reason = "state_tracking_error"

        self.last_update_info = {
            "adaptive_replan_triggered": replan_triggered,
            "adaptive_replan_reason": replan_reason,
            "adaptive_replan_interval": int(self._current_replan_interval),
            "adaptive_state_tracking_error": state_error,
        }
        return dict(self.last_update_info)


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
        async_worker_iterations: int | None = None,
        async_worker_full_interval: int | None = None,
        async_worker_backend: str = "thread",
        async_worker_device: str | None = None,
        async_start_after_first_plan: bool = True,
        async_refine_on_parameter_update: bool = True,
        async_reanchor_live_state: bool = False,
        async_reanchor_tolerance: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.async_stale_tolerance = float(max(async_stale_tolerance, 0.0))
        self.async_stale_refine_iterations = int(max(async_stale_refine_iterations, 0))
        self.async_worker_iterations = (
            None if async_worker_iterations is None else max(1, int(async_worker_iterations))
        )
        self.async_worker_full_interval = (
            None if async_worker_full_interval is None else max(1, int(async_worker_full_interval))
        )
        self._async_background_plan_count = 0
        self._return_planning_tail = True
        self.async_worker_backend = str(async_worker_backend or "thread").strip().lower()
        if self.async_worker_backend != "thread":
            raise ValueError("AsyncMpcICem only supports async_worker_backend='thread'")
        self.async_worker_device = (
            None
            if async_worker_device is None or str(async_worker_device).strip() == ""
            else torch.device(str(async_worker_device))
        )
        self.async_start_after_first_plan = bool(async_start_after_first_plan)
        self.async_refine_on_parameter_update = bool(async_refine_on_parameter_update)
        self.async_reanchor_live_state = bool(async_reanchor_live_state)
        self.async_reanchor_tolerance = float(max(async_reanchor_tolerance, 0.0))
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1)
        self._foreground_active = threading.Event()
        self._async_anchor_lock = threading.Lock()
        self._async_anchor: dict[str, Any] = {"seq": 0, "boundary": None}
        self._worker_anchor_seq = 0
        self._worker_reanchor_count = 0
        self._worker_reanchor_mismatch = 0.0
        self._planning_future: Future | None = None
        self._current_buffer: torch.Tensor | None = None
        self._buffer_index = 0
        self._predicted_boundary_state: torch.Tensor | None = None
        self.last_plan_status: dict[str, Any] = self._empty_plan_status()
        self.last_update_info: dict[str, Any] = dict(self.last_plan_status)

    @property
    def updates_metric_in_background(self) -> bool:
        return False

    def set_foreground_active(self, active: bool) -> None:
        if self._foreground_active is None:
            self._foreground_active = threading.Event()
        if active:
            self._foreground_active.set()
        else:
            self._foreground_active.clear()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_executor"] = None
        state["_foreground_active"] = None
        state["_async_anchor_lock"] = None
        state["_async_anchor"] = {"seq": 0, "boundary": None}
        state["_yield_to_foreground"] = False
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
            "async_model_version_mismatch": False,
            "async_plan_model_version": -1,
            "async_live_model_version": -1,
            "async_refined": False,
            "async_blocking_fallback": False,
            "async_plan_runtime_sec": 0.0,
            "async_plan_status": "idle",
            "async_reanchor_count": 0,
            "async_reanchor_mismatch": 0.0,
        }

    def _cancel_planning_future(self) -> None:
        if self._planning_future is not None:
            cancel = getattr(self._planning_future, "cancel", None)
            if callable(cancel):
                cancel()
        self._planning_future = None

    def _reset_executor(self) -> None:
        self._cancel_planning_future()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = ThreadPoolExecutor(max_workers=1)

    def close(self) -> None:
        self._cancel_planning_future()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

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
        self._async_background_plan_count = 0
        self._foreground_active = threading.Event()
        self._async_anchor_lock = threading.Lock()
        self._async_anchor = {"seq": 0, "boundary": None}
        self._worker_anchor_seq = 0
        self._worker_reanchor_count = 0
        self._worker_reanchor_mismatch = 0.0
        self._yield_to_foreground = False
        self.last_plan_status = self._empty_plan_status()
        self.last_update_info = dict(self.last_plan_status)
        self.last_plan_info = {"plan_executed": False, "plan_reason": "none"}

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
            action_seq = action_seq
        else:
            raise ValueError(
                f"AsyncMpcICem received action sequence length {action_seq.shape[-2]}, "
                f"but chunk is {self.chunk}"
            )
        return action_seq.detach().clone()

    def _async_execution_interval(self) -> int:
        if self.adaptive_replanning and not self.async_refine_on_parameter_update:
            return 1
        return int(self.chunk)

    def _async_buffer_boundary(self) -> int:
        if self._current_buffer is None:
            return self._async_execution_interval()
        return min(self._async_execution_interval(), int(self._current_buffer.shape[-2]))

    def _set_current_buffer(
        self,
        actions: torch.Tensor,
        cost: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> None:
        if state is not None:
            self._set_model_planning_state(self.model, state)
        self._current_buffer = self._normalize_action_buffer(actions)
        self.action_list = [a.unsqueeze(0) for a in self._current_buffer[0]]
        self._buffer_index = 0
        self._chunk_step = 0
        self._last_action_index = None
        self.cost = torch.as_tensor(cost).squeeze().item()
        if (
            self.adaptive_replanning
            and self.adaptive_replan_state_error_threshold is not None
            and self.async_refine_on_parameter_update
        ):
            with torch.no_grad():
                self._planned_state_trace = self._predict_action_trajectory(
                    self._current_buffer
                ).detach().clone()
        else:
            self._planned_state_trace = None

    def _sync_plan(
        self,
        state: torch.Tensor,
        *,
        num_iterations: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = dict(kwargs)
        _update_metric_from_rollout(self.metric, kwargs.pop("recent_rollout", None))
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

    def _async_live_boundary(self, state: torch.Tensor) -> torch.Tensor:
        if self._current_buffer is None:
            boundary = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            buffer_boundary = self._async_buffer_boundary()
            remaining_end = (
                buffer_boundary if self._buffer_index < buffer_boundary else self._buffer_index
            )
            remaining = self._current_buffer[:, self._buffer_index : remaining_end, :]
            if remaining.shape[-2] == 0:
                boundary = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            else:
                boundary = self._rollout_from_state(
                    torch.as_tensor(state, dtype=torch.float32, device=self.device),
                    remaining,
                )
        if boundary.ndim == 1:
            boundary = boundary.reshape(1, 1, -1)
        elif boundary.ndim == 2:
            boundary = boundary.unsqueeze(1) if boundary.shape[0] == 1 else boundary.unsqueeze(0)
        return boundary.detach().clone()

    def _publish_async_anchor(
        self,
        boundary: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> None:
        if not self.async_reanchor_live_state:
            return
        if self._async_anchor_lock is None:
            self._async_anchor_lock = threading.Lock()
        with self._async_anchor_lock:
            self._async_anchor["seq"] = int(self._async_anchor.get("seq", 0)) + 1
            self._async_anchor["boundary"] = boundary.detach().clone()

    def _maybe_reanchor_worker_plan(
        self,
        state: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        if (
            not self.async_reanchor_live_state
            or not bool(getattr(self, "_yield_to_foreground", False))
            or self._async_anchor_lock is None
        ):
            return state
        with self._async_anchor_lock:
            seq = int(self._async_anchor.get("seq", 0))
            if seq <= self._worker_anchor_seq:
                return state
            boundary = self._async_anchor.get("boundary")
            self._worker_anchor_seq = seq
        if boundary is None:
            return state
        boundary_t = torch.as_tensor(boundary, dtype=torch.float32, device=self.device)
        mismatch = self._boundary_mismatch(getattr(self.model, "_state", state), boundary_t)
        if mismatch <= self.async_reanchor_tolerance:
            return state
        self._set_model_planning_state(self.model, boundary_t)
        kwargs["observed_state"] = boundary_t.detach().clone()
        self.elite_actions = None
        self.elite_costs_traj = None
        self._worker_reanchor_count += 1
        self._worker_reanchor_mismatch = mismatch
        return boundary_t

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
            planner = copy.copy(self)
            planner.model = copy.deepcopy(self.model)
            planner.metric = copy.deepcopy(self.metric, {id(self.model): planner.model})
            for attr in ("mean", "std", "elite_actions", "elite_costs_traj"):
                value = getattr(self, attr, None)
                if torch.is_tensor(value):
                    value = value.detach().clone()
                else:
                    value = copy.deepcopy(value)
                setattr(planner, attr, value)
        except Exception as exc:
            raise RuntimeError(
                "AsyncMpcICem could not copy the policy/model snapshot. "
                "Use policy_type='mpc-icem' or make the model snapshot-compatible."
            ) from exc
        planner._executor = None
        planner._foreground_active = self._foreground_active
        planner._async_anchor_lock = self._async_anchor_lock
        planner._async_anchor = self._async_anchor
        planner._yield_to_foreground = True
        planner._planning_future = None
        planner._current_buffer = None
        planner._predicted_boundary_state = None
        planner._worker_anchor_seq = int(self._async_anchor.get("seq", 0))
        planner._worker_reanchor_count = 0
        planner._worker_reanchor_mismatch = 0.0
        planner.last_plan_status = dict(self.last_plan_status)
        planner.last_update_info = dict(self.last_update_info)
        if self.async_worker_device is not None:
            target_device = torch.device(self.async_worker_device)
            if torch.device(planner.device) != target_device:
                _move_snapshot_to_device(planner, target_device)
            predicted_boundary_state = predicted_boundary_state.to(planner.device)
        planner._set_model_planning_state(planner.model, predicted_boundary_state)
        return planner


    @staticmethod
    def _background_plan_worker(
        planner,
        predicted_boundary_state: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> _AsyncPlanResult:
        start = time.perf_counter()
        kwargs = dict(kwargs)
        worker_device = torch.device(getattr(planner, "device", "cpu"))
        predicted_boundary_state = torch.as_tensor(
            predicted_boundary_state, dtype=torch.float32, device=worker_device
        )
        kwargs = _move_tensors_to_device(kwargs, worker_device)
        model_update_version = int(
            kwargs.get("parameter_update_version", kwargs.get("model_update_version", 0))
        )
        _update_metric_from_rollout(planner.metric, kwargs.pop("recent_rollout", None))
        actions, cost = MpcICem.get_action(planner, predicted_boundary_state, **kwargs)
        runtime = time.perf_counter() - start
        final_boundary_state = getattr(planner.model, "_state", predicted_boundary_state)
        return _AsyncPlanResult(
            actions=actions.detach().clone(),
            cost=cost.detach().clone(),
            predicted_boundary_state=final_boundary_state.detach().clone(),
            runtime_sec=float(runtime),
            status="completed",
            model_update_version=model_update_version,
            mean=getattr(planner, "mean", None),
            std=getattr(planner, "std", None),
            elite_actions=getattr(planner, "elite_actions", None),
            elite_costs_traj=getattr(planner, "elite_costs_traj", None),
            reanchor_count=int(getattr(planner, "_worker_reanchor_count", 0)),
            reanchor_mismatch=float(getattr(planner, "_worker_reanchor_mismatch", 0.0)),
        )

    def request_replan(self, reason: str) -> None:
        reason = str(reason)
        if reason == "parameter_update":
            return
        if reason == "state_tracking_error" and not self.async_refine_on_parameter_update:
            return
        super().request_replan(reason)

    def _launch_background_plan(self, state: torch.Tensor, kwargs: dict[str, Any]) -> None:
        if self._current_buffer is None:
            return
        predicted_boundary = None
        if self.async_reanchor_live_state:
            predicted_boundary = self._async_live_boundary(state)
            self._publish_async_anchor(predicted_boundary, kwargs)
        if self._planning_future is not None:
            return
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        if predicted_boundary is None:
            predicted_boundary = self._async_live_boundary(state)
        self._predicted_boundary_state = predicted_boundary.detach().clone()
        worker_kwargs = dict(kwargs)
        worker_kwargs.pop("recent_rollout", None)
        worker_kwargs = _clone_for_worker(worker_kwargs)
        worker_kwargs["observed_state"] = predicted_boundary.detach().clone()
        planner = self._make_snapshot_planner(predicted_boundary)
        self._async_background_plan_count += 1
        use_full_iterations = (
            self.async_worker_full_interval is not None
            and self._async_background_plan_count % self.async_worker_full_interval == 0
        )
        if self.async_worker_iterations is not None and not use_full_iterations:
            planner.num_iterations = int(self.async_worker_iterations)
        assert self._executor is not None
        self._planning_future = self._executor.submit(
            self._background_plan_worker,
            planner,
            predicted_boundary.detach().clone(),
            worker_kwargs,
        )

    def _activate_new_chunk(self, state: torch.Tensor, kwargs: dict[str, Any]) -> None:
        status = self._empty_plan_status()
        plan_info = {"plan_executed": False, "plan_reason": "none"}
        live_model_version = int(
            kwargs.get("parameter_update_version", kwargs.get("model_update_version", 0))
        )
        status["async_live_model_version"] = live_model_version
        result: _AsyncPlanResult | None = None
        forced_reason = self._force_replan_reason if self._force_replan_next else None
        if forced_reason is not None:
            if self._planning_future is not None and not self._planning_future.done():
                self._cancel_planning_future()
            else:
                self._planning_future = None
            actions, cost = self._sync_plan(state, **kwargs)
            plan_info = {"plan_executed": True, "plan_reason": forced_reason}
            status["async_blocking_fallback"] = True
            status["async_plan_status"] = f"forced_{forced_reason}"
            self._set_current_buffer(actions, cost, state=state)
            self._force_replan_next = False
            self._force_replan_reason = None
        elif self._current_buffer is None:
            actions, cost = self._sync_plan(state, **kwargs)
            plan_info = {"plan_executed": True, "plan_reason": "initial_blocking"}
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
                message = str(exc).replace("\n", " ")[:120]
                status["async_plan_status"] = f"worker_error:{type(exc).__name__}:{message}"
            if result is not None:
                mismatch = self._boundary_mismatch(state, result.predicted_boundary_state)
                plan_model_version = int(getattr(result, "model_update_version", 0))
                version_mismatch = plan_model_version != live_model_version
                status["async_boundary_mismatch"] = mismatch
                status["async_plan_model_version"] = plan_model_version
                status["async_model_version_mismatch"] = version_mismatch
                status["async_plan_runtime_sec"] = float(result.runtime_sec)
                status["async_reanchor_count"] = int(getattr(result, "reanchor_count", 0))
                status["async_reanchor_mismatch"] = float(
                    getattr(result, "reanchor_mismatch", 0.0)
                )
                parameter_stale = version_mismatch and self.async_refine_on_parameter_update
                if mismatch <= self.async_stale_tolerance and not parameter_stale:
                    self._apply_worker_warm_start(result)
                    self._set_current_buffer(result.actions, result.cost, state=state)
                    status["async_plan_used"] = True
                    status["async_plan_status"] = (
                        "used_ready_parameter_stale" if version_mismatch else "used_ready"
                    )
                else:
                    self._apply_worker_warm_start(result)
                    actions, cost = self._sync_plan(
                        state,
                        num_iterations=self.async_stale_refine_iterations,
                        **kwargs,
                    )
                    plan_info = {"plan_executed": True, "plan_reason": "refined_stale"}
                    self._set_current_buffer(actions, cost, state=state)
                    self._predicted_boundary_state = None
                    status["async_plan_stale"] = True
                    status["async_refined"] = True
                    status["async_plan_status"] = "refined_stale"
        if self._current_buffer is not None:
            exhausted = self._buffer_index >= self._current_buffer.shape[-2]
            waiting = self._planning_future is not None and not self._planning_future.done()
            if waiting and not exhausted and status["async_plan_status"] == "idle":
                status["async_plan_status"] = "waiting_ready_tail"
            elif exhausted:
                if waiting:
                    hold = self._current_buffer[:, -1:, :].detach().clone()
                    self._current_buffer = hold
                    self.action_list = [hold[0, 0].unsqueeze(0)]
                    self._buffer_index = 0
                    self._chunk_step = 0
                    if status["async_plan_status"] == "idle":
                        status["async_plan_status"] = "waiting_ready_hold"
                else:
                    actions, cost = self._sync_plan(state, **kwargs)
                    plan_info = {"plan_executed": True, "plan_reason": "blocking_fallback"}
                    self._set_current_buffer(actions, cost, state=state)
                    status["async_blocking_fallback"] = True
                    if status["async_plan_status"] == "idle":
                        status["async_plan_status"] = "blocking_fallback"

        self.last_plan_status = status
        self.last_update_info = dict(status)
        self.last_plan_info = plan_info

    def __call__(self, state, **kwargs) -> torch.Tensor:
        defer_background_launch = bool(kwargs.pop("defer_background_launch", False))
        if not self.was_reset:
            self.beginning_of_rollout(state)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_t.ndim == 1:
            state_t = state_t.reshape(1, 1, -1)
        elif state_t.ndim == 2:
            state_t = state_t.unsqueeze(1) if state_t.shape[0] == 1 else state_t.unsqueeze(0)

        buffer_boundary = self._async_buffer_boundary()
        at_boundary = (
            self._current_buffer is None
            or self._buffer_index >= buffer_boundary
            or bool(self._force_replan_next)
        )
        if at_boundary:
            self._activate_new_chunk(state_t, dict(kwargs))
        else:
            self.last_plan_info = {"plan_executed": False, "plan_reason": "none"}

        if defer_background_launch and self.async_reanchor_live_state:
            self._publish_async_anchor(self._async_live_boundary(state_t), kwargs)

        if not defer_background_launch and (self.async_start_after_first_plan or self.count > 0):
            self._launch_background_plan(state_t, dict(kwargs))

        assert self._current_buffer is not None
        self._last_action_index = int(self._buffer_index)
        action = self._current_buffer[:, self._buffer_index : self._buffer_index + 1, :]
        self._buffer_index += 1
        self._chunk_step = int(self._buffer_index)
        self.count += 1
        return action

    def launch_background_plan(self, state, **kwargs) -> None:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_t.ndim == 1:
            state_t = state_t.reshape(1, 1, -1)
        elif state_t.ndim == 2:
            state_t = state_t.unsqueeze(1) if state_t.shape[0] == 1 else state_t.unsqueeze(0)
        if self.async_start_after_first_plan or self.count > 0:
            self._launch_background_plan(state_t, dict(kwargs))

    def update(self, batch) -> dict[str, Any]:
        if self.adaptive_replanning and not self.async_refine_on_parameter_update:
            update_info = {
                "adaptive_replan_triggered": False,
                "adaptive_replan_reason": "none",
                "adaptive_replan_interval": int(self._current_replan_interval),
                "adaptive_state_tracking_error": None,
            }
        else:
            update_info = MpcICem.update(self, batch)
        merged = {**self.last_plan_status, **update_info}
        self.last_update_info = merged
        return dict(merged)
