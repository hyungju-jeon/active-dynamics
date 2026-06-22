#!/usr/bin/env python3
"""Render saved online localized RBF tracking results."""

from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
from threading import Lock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import colorednoise
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from actdyn.environment.vectorfield import VectorFieldEnv, build_vectorfield
from actdyn.utils.video import figure_to_rgb_array, write_video_frames

try:
    from .local_rbf_asymmetric_basin import LocalGridRBFDynamics, fit_ridge, grid_states
except ImportError:
    from local_rbf_asymmetric_basin import LocalGridRBFDynamics, fit_ridge, grid_states

DEFAULT_OUTPUT = REPO_ROOT / "results/rbf/assets/asymmetric_basin_online_rbf_behavior.mp4"
DEFAULT_TRACK = REPO_ROOT / "results/rbf/tracks/asymmetric_basin_online_rbf_track.npz"
DEFAULT_DYNAMICS_TYPE = "asymmetric_basin"
DEFAULT_TBME_DYN_PARAMS = "-1.2,-0.8,0.5,1.1"


def _set_torch_threads(args: argparse.Namespace) -> None:
    threads = int(args.torch_num_threads)
    if threads > 0:
        torch.set_num_threads(threads)


class OnlineRBFWeightUpdater:
    """Online local update for fixed-grid RBF speed weights.

    Centers and length scale are fixed. Only ``model.weights`` is updated, so
    the surrogate parameter is the center-wise 2D drift/speed.
    """

    def __init__(
        self,
        model: LocalGridRBFDynamics,
        *,
        lr: float,
        ridge: float,
        prior_precision: float,
        eig_beta: float,
        weight_diffusion: float,
        smoothing_radius: int,
        smoothing_strength: float,
        smoothing_precision_threshold: float,
    ) -> None:
        self.model = model
        self.lr = float(lr)
        self.ridge = float(max(ridge, 1e-12))
        self.eig_beta = float(max(eig_beta, 1e-12))
        self.weight_diffusion = float(max(weight_diffusion, 0.0))
        self.smoothing_strength = float(max(smoothing_strength, 0.0))
        self.smoothing_precision_threshold = float(max(smoothing_precision_threshold, 1e-12))
        self.smoothing_indices, self.smoothing_valid = self._smoothing_stencil(int(smoothing_radius))
        self.precision = torch.full_like(model.weights.data, float(max(prior_precision, 1e-12)))
        self.num_updates = 0

    def _smoothing_stencil(self, radius: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if radius <= 0:
            return None, None
        g = int(self.model.grid_points)
        axis = torch.arange(g, dtype=torch.long, device=self.model.device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        centers = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
        offsets = torch.arange(-radius, radius + 1, dtype=torch.long, device=self.model.device)
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        delta = torch.stack([oy.reshape(-1), ox.reshape(-1)], dim=1)
        delta = delta[torch.sum(torch.abs(delta), dim=1) <= radius]
        neighbors = centers[:, None, :] + delta[None, :, :]
        valid = ((neighbors >= 0) & (neighbors < g)).all(dim=-1)
        safe = neighbors.clamp(0, g - 1)
        return safe[:, :, 0] * g + safe[:, :, 1], valid

    @torch.no_grad()
    def diffuse_prior(self) -> None:
        if self.weight_diffusion <= 0.0:
            return
        var = 1.0 / self.precision.clamp_min(1e-12)
        self.precision = 1.0 / (var + self.weight_diffusion)

    @torch.no_grad()
    def update(self, state: torch.Tensor, target_drift: torch.Tensor) -> float:
        """Update only weights whose RBF centers are active at ``state``.

        Args:
            state: One observed latent state with shape ``(state_dim,)``.
            target_drift: Drift estimated from the observed transition, shape
                ``(state_dim,)``.

        Returns:
            Euclidean prediction error before the update.
        """

        state = state.reshape(1, -1)
        pred = self.model(state)[0]
        indices, values, valid = self.model.local_feature_entries(state)
        active = indices[0, valid[0]]
        phi = values[0, valid[0]].unsqueeze(-1)
        if active.numel() == 0:
            return 0.0
        error = target_drift.reshape(-1) - pred.reshape(-1)
        denom = self.ridge + torch.sum(phi.reshape(-1) ** 2)
        self.model.weights.data[active] += self.lr * phi * error.reshape(1, -1) / denom
        self.precision[active] += self.eig_beta * phi.square().expand(-1, self.model.state_dim)
        self.smooth_weights(exclude=active)
        self.model.network.weights = self.model.weights
        self.num_updates += 1
        return float(torch.linalg.norm(error).item())

    @torch.no_grad()
    def smooth_weights(self, exclude: torch.Tensor | None = None) -> None:
        if self.smoothing_strength <= 0.0 or self.smoothing_indices is None or self.smoothing_valid is None:
            return
        idx = self.smoothing_indices
        valid = self.smoothing_valid.unsqueeze(-1)
        weights = self.model.weights.data[idx]
        precision = self.precision[idx].clamp_min(1e-12) * valid
        denom = precision.sum(dim=1).clamp_min(1e-12)
        smoothed = (precision * weights).sum(dim=1) / denom
        center_precision = self.precision.mean(dim=1, keepdim=True)
        gate = ((self.smoothing_precision_threshold - center_precision) / self.smoothing_precision_threshold).clamp(0.0, 1.0)
        strength = min(self.smoothing_strength, 1.0) * gate
        if exclude is not None and exclude.numel() > 0:
            strength[exclude.reshape(-1)] = 0.0
        self.model.weights.data.add_(strength * (smoothed - self.model.weights.data))


def _dyn_params(args: argparse.Namespace) -> list[float]:
    values = [float(x) for x in str(args.dyn_params).split(",") if str(x).strip()]
    if not values:
        raise ValueError("--dyn-params must contain at least one comma-separated value")
    return values


def _true_drift(args: argparse.Namespace, states: torch.Tensor) -> torch.Tensor:
    vf = build_vectorfield(
        str(args.dynamics_type),
        _dyn_params(args),
        dynamics_alpha=float(args.dynamics_alpha),
        device=states.device,
    )
    with torch.no_grad():
        return vf.compute(states)


def _bounds(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.tensor([args.state_low, args.state_low], dtype=torch.float32, device=device)
    high = torch.tensor([args.state_high, args.state_high], dtype=torch.float32, device=device)
    return low, high


def _new_env(args: argparse.Namespace, device: torch.device) -> VectorFieldEnv:
    return VectorFieldEnv(
        dynamics_type=str(args.dynamics_type),
        d_state=2,
        Q=float(args.state_noise),
        dt=float(args.dt),
        device=str(device),
        action_bounds=(-float(args.action_max), float(args.action_max)),
        state_bounds=(float(args.state_low), float(args.state_high)),
        initial_state=args.initial_state,
        dyn_params=_dyn_params(args),
        boundary_enabled=True,
        boundary_type=str(args.boundary_type),
        boundary_radius=float(args.boundary_radius),
        boundary_barrier_enabled=bool(args.boundary_barrier_enabled),
        boundary_projection_enabled=True,
        boundary_barrier_width=float(args.boundary_barrier_width),
        boundary_barrier_strength=float(args.boundary_barrier_strength),
        boundary_barrier_temperature=float(args.boundary_barrier_temperature),
        alpha=float(args.dynamics_alpha),
    )


def _new_surrogate(args: argparse.Namespace, device: torch.device) -> LocalGridRBFDynamics:
    low, high = _bounds(args, device)
    model = LocalGridRBFDynamics(
        state_low=low,
        state_high=high,
        grid_points=args.grid_points,
        lengthscale=args.lengthscale,
        active_radius=args.active_radius,
        dt=args.dt,
        device=device,
    )
    if bool(args.initial_fit_rbf):
        train_states = grid_states(low=low, high=high, points=int(args.initial_fit_grid_points))
        fit_ridge(model, train_states, _true_drift(args, train_states), ridge=float(args.initial_fit_ridge))
    else:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(args.seed) + 1729)
        init = torch.randn(model.weights.shape, generator=gen, dtype=torch.float32)
        model.weights.data.copy_(float(args.rbf_weight_init_std) * init.to(device))
        model.network.weights = model.weights
    return model


def _project_action_radius(actions: torch.Tensor, action_max: float) -> torch.Tensor:
    """Project actions to the disk ``||u_t||_2 <= action_max``.

    Args:
        actions: Tensor with shape ``(..., 2)``.
        action_max: Maximum Euclidean action norm.

    Returns:
        Tensor with the same shape as ``actions``.
    """

    norm = torch.linalg.norm(actions, dim=-1, keepdim=True).clamp_min(1e-12)
    return actions * torch.clamp(float(action_max) / norm, max=1.0)


def _fixed_action_sequences(args: argparse.Namespace, device: torch.device, horizon: int | None = None) -> torch.Tensor:
    """Return zero and constant-direction action sequences for iCEM seeding.

    Returns:
        Tensor with shape ``(1 + num_direction_actions, horizon, 2)``.
    """

    horizon = int(args.planning_horizon if horizon is None else max(1, horizon))
    angles = torch.linspace(0.0, 2.0 * torch.pi, int(args.num_direction_actions) + 1)[:-1]
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1) * float(args.action_max)
    constant = directions[:, None, :].expand(-1, horizon, -1)
    zero = torch.zeros(1, horizon, 2, dtype=torch.float32)
    return torch.cat([zero, constant], dim=0).to(device)


def _planner_horizon(args: argparse.Namespace, planner_cache: dict[str, torch.Tensor]) -> int:
    value = planner_cache.get("planning_horizon")
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return max(1, int(round(float(value.item()))))
    return max(1, int(args.planning_horizon))


def _adaptive_planning_horizon(
    args: argparse.Namespace,
    model: LocalGridRBFDynamics,
    prior_precision: torch.Tensor,
    state: torch.Tensor,
) -> int:
    """Choose a coarse iCEM horizon from active-center posterior precision."""

    default_max = int(np.ceil(float(args.replan_interval) / max(1, int(args.coarse_dt_factor))))
    max_horizon = default_max if args.adaptive_horizon_max is None else int(args.adaptive_horizon_max)
    max_horizon = max(1, min(max_horizon, int(args.planning_horizon)))
    min_horizon = max(1, min(int(args.adaptive_horizon_min), max_horizon))
    if not bool(args.adaptive_planning_horizon) or min_horizon >= max_horizon:
        return max_horizon

    indices, _, valid = model.local_feature_entries(state.reshape(1, -1))
    active = indices[0, valid[0]]
    if active.numel() == 0:
        return min_horizon

    local_precision = float(prior_precision[active].mean().item())
    prior = max(float(args.prior_precision), 1e-12)
    threshold = max(float(args.adaptive_horizon_precision_threshold), prior * (1.0 + 1e-6))
    fraction = (np.log(max(local_precision, prior)) - np.log(prior)) / (np.log(threshold) - np.log(prior))
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return int(round(min_horizon + fraction * (max_horizon - min_horizon)))


@torch.no_grad()
def _rollout_action_sequences(
    args: argparse.Namespace,
    model: LocalGridRBFDynamics,
    state: torch.Tensor,
    actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Roll out coarse action sequences under the current surrogate dynamics.

    Args:
        state: Current latent state with shape ``(2,)``.
        actions: Candidate controls with shape ``(batch, planning_horizon, 2)``.

    Returns:
        Coarse paths with shape ``(batch, planning_horizon + 1, 2)``.
    """

    batch, horizon, _ = actions.shape
    states = state.reshape(1, -1).expand(batch, -1).clone()
    trajectory = [states.clone()]
    dt = float(args.dt) * float(args.coarse_dt_factor)
    for t in range(horizon):
        drift = model(states)
        states = (states + dt * (drift + actions[:, t])).clamp(low, high)
        trajectory.append(states.clone())
    return torch.stack(trajectory, dim=1)


def _shift_action_sequences(sequences: torch.Tensor, shift: int, fill: torch.Tensor) -> torch.Tensor:
    """Shift iCEM warm-start sequences after executing coarse actions.

    Args:
        sequences: Tensor with shape ``(..., planning_horizon, 2)``.
        shift: Number of coarse actions already executed.
        fill: Tail value with shape ``(2,)``.

    Returns:
        Shifted tensor with the same shape as ``sequences``.
    """

    shift = int(max(0, shift))
    if shift == 0:
        return sequences
    horizon = sequences.shape[-2]
    tail_shape = (*sequences.shape[:-2], min(shift, horizon), sequences.shape[-1])
    tail = fill.reshape(*([1] * (len(tail_shape) - 1)), sequences.shape[-1]).expand(tail_shape)
    if shift >= horizon:
        return tail.clone()
    return torch.cat([sequences[..., shift:, :], tail], dim=-2)


def _block_weight_eig_score(
    args: argparse.Namespace,
    model: LocalGridRBFDynamics,
    prior_precision: torch.Tensor,
    path: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Score a path by propagating local RBF weight sensitivities.

    This follows the same idea as the repo RBF Fisher path: the observation at
    later states is informative about a weight only through ``dz_t / dw``.  The
    recursion uses the coarse planning step ``dt * coarse_dt_factor``.
    """

    indices, values, valid = model.local_feature_entries(path[:-1])
    horizon = indices.shape[0]
    action_cost = float(args.action_penalty) * torch.sum(actions ** 2)
    flat_valid = valid.reshape(-1)
    if not bool(flat_valid.any()):
        return -action_cost

    centers = torch.unique(indices.reshape(-1)[flat_valid], sorted=False)
    local_dim = int(centers.numel() * model.state_dim)
    center_pos = torch.full((model.centers.shape[0],), -1, dtype=torch.long, device=model.device)
    center_pos[centers] = torch.arange(centers.numel(), dtype=torch.long, device=model.device)

    dt = float(args.dt) * float(args.coarse_dt_factor)
    state_dim = int(model.state_dim)
    eye_state = torch.eye(state_dim, dtype=torch.float32, device=model.device)
    sensitivity = torch.zeros(state_dim, local_dim, dtype=torch.float32, device=model.device)
    design = torch.empty(horizon * state_dim, local_dim, dtype=torch.float32, device=model.device)
    for t in range(horizon):
        active = indices[t, valid[t]]
        phi = values[t, valid[t]]
        if active.numel() > 0:
            z_t = path[t].reshape(1, -1)
            dphi_dz = -2.0 * float(model.gamma) * phi.unsqueeze(-1) * (z_t - model.centers[active])
            drift_jac = model.weights[active].T @ dphi_dz
            sensitivity = (eye_state + dt * drift_jac) @ sensitivity

            local_centers = center_pos[active]
            increments = dt * phi
            for d in range(state_dim):
                sensitivity[d, local_centers * state_dim + d] += increments
        design[t * state_dim : (t + 1) * state_dim].copy_((float(args.eig_gamma) ** (0.5 * t)) * sensitivity)
    inv_std = torch.rsqrt(prior_precision[centers].reshape(-1).clamp_min(1e-12))
    weighted = torch.nan_to_num(design * inv_std.unsqueeze(0), nan=0.0, posinf=1e6, neginf=-1e6)
    if bool(args.eig_float64):
        weighted = weighted.double()
    eye = torch.eye(weighted.shape[0], dtype=weighted.dtype, device=model.device)
    gram = eye + float(args.eig_beta) * (weighted @ weighted.T)
    gram = 0.5 * (gram + gram.T)
    chol = None
    for jitter in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        try:
            chol = torch.linalg.cholesky(gram + jitter * eye)
            break
        except torch.linalg.LinAlgError:
            continue
    if chol is None:
        return -action_cost
    return torch.log(torch.diagonal(chol).clamp_min(1e-12)).sum() - action_cost


def _numpy_planning_arrays(model: LocalGridRBFDynamics) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        model.state_low.detach().cpu().numpy(),
        model.grid_spacing.detach().cpu().numpy(),
        model.grid_offsets.detach().cpu().numpy(),
        model.weights.detach().cpu().numpy(),
        model.centers.detach().cpu().numpy(),
    )


def _numpy_local_feature_entries(
    args: argparse.Namespace,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    paths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_low, grid_spacing, grid_offsets, _, centers_all = arrays
    grid_points = int(args.grid_points)
    gamma = 0.5 / float(args.lengthscale) ** 2
    z = paths[:, :-1, :]
    nearest = np.rint((z - state_low[None, None, :]) / grid_spacing[None, None, :]).astype(np.int64)
    multi_index = nearest[:, :, None, :] + grid_offsets[None, None, :, :]
    valid = np.all((multi_index >= 0) & (multi_index < grid_points), axis=-1)
    safe = np.clip(multi_index, 0, grid_points - 1)
    indices = safe[:, :, :, 0] * grid_points + safe[:, :, :, 1]
    centers = centers_all[indices]
    values = np.exp(-np.sum((z[:, :, None, :] - centers) ** 2, axis=-1) * gamma).astype(np.float32)
    values *= valid.astype(np.float32)
    return indices, values, valid


def _block_weight_eig_score_numpy_from_entries(
    args: argparse.Namespace,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    prior_precision: np.ndarray,
    path: np.ndarray,
    actions: np.ndarray,
    path_indices: np.ndarray,
    path_values: np.ndarray,
    path_valid: np.ndarray,
) -> float:
    _, _, _, weights, centers_all = arrays
    gamma = 0.5 / float(args.lengthscale) ** 2
    dt = float(args.dt) * float(args.coarse_dt_factor)
    state_dim = 2
    action_cost = float(args.action_penalty) * float(np.sum(actions * actions))

    flat = path_indices[path_valid]
    if flat.size == 0:
        return -action_cost
    centers = np.unique(flat)
    center_pos = np.full(centers_all.shape[0], -1, dtype=np.int64)
    center_pos[centers] = np.arange(centers.size, dtype=np.int64)
    local_dim = int(centers.size * state_dim)

    inv_std = np.reciprocal(np.sqrt(np.maximum(prior_precision[centers].reshape(-1), 1e-12))).astype(np.float32)
    sensitivity = np.zeros((state_dim, local_dim), dtype=np.float32)
    weighted = np.empty((path_indices.shape[0] * state_dim, local_dim), dtype=np.float32)
    eye_state = np.eye(state_dim, dtype=np.float32)
    for t, (indices, values, valid) in enumerate(zip(path_indices, path_values, path_valid)):
        active = indices[valid]
        phi = values[valid]
        if active.size > 0:
            z = path[t]
            active_centers = centers_all[active]
            dphi_dz = -2.0 * gamma * phi[:, None] * (z[None, :] - active_centers)
            drift_jac = weights[active].T @ dphi_dz
            sensitivity = (eye_state + dt * drift_jac) @ sensitivity
            increments = dt * phi
            local_centers = center_pos[active]
            base = local_centers * state_dim
            sensitivity[0, base] += increments
            sensitivity[1, base + 1] += increments
        weighted[t * state_dim : (t + 1) * state_dim] = (
            (float(args.eig_gamma) ** (0.5 * t)) * sensitivity * inv_std[None, :]
        )

    weighted = np.nan_to_num(weighted, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    if bool(args.eig_float64):
        weighted = weighted.astype(np.float64)
    gram = np.eye(weighted.shape[0], dtype=weighted.dtype) + float(args.eig_beta) * (weighted @ weighted.T)
    gram = 0.5 * (gram + gram.T)
    eye = np.eye(gram.shape[0], dtype=gram.dtype)
    for jitter in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        try:
            chol = np.linalg.cholesky(gram + jitter * eye)
            return float(np.log(np.maximum(np.diag(chol), 1e-12)).sum() - action_cost)
        except np.linalg.LinAlgError:
            continue
    return -action_cost


def _block_weight_eig_score_numpy(
    args: argparse.Namespace,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    prior_precision: np.ndarray,
    path: np.ndarray,
    actions: np.ndarray,
) -> float:
    indices, values, valid = _numpy_local_feature_entries(args, arrays, path[None, :, :])
    return _block_weight_eig_score_numpy_from_entries(
        args,
        arrays,
        prior_precision,
        path,
        actions,
        indices[0],
        values[0],
        valid[0],
    )

def _candidate_proxy_scores(
    args: argparse.Namespace,
    model: LocalGridRBFDynamics,
    prior_precision: torch.Tensor,
    paths: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Cheap uncertainty score used only to preselect exact-EIG candidates."""

    indices, values, valid = model.local_feature_entries(paths[:, :-1])
    center_precision = prior_precision.mean(dim=1).clamp_min(1e-12)
    uncertainty = torch.reciprocal(center_precision[indices])
    discount = torch.as_tensor(
        [float(args.eig_gamma) ** t for t in range(values.shape[1])],
        dtype=values.dtype,
        device=values.device,
    ).reshape(1, -1, 1)
    score = (discount * values.square() * uncertainty * valid.to(values.dtype)).sum(dim=(1, 2))
    return score - float(args.action_penalty) * torch.sum(actions ** 2, dim=(1, 2))


@torch.no_grad()
def _fine_planned_path(
    args: argparse.Namespace,
    model: LocalGridRBFDynamics,
    state: torch.Tensor,
    coarse_actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Roll out the selected coarse plan at every environment time step."""

    z = state.reshape(-1).clone()
    path = [z.clone()]
    for action in coarse_actions:
        for _ in range(max(1, int(args.coarse_dt_factor))):
            drift = model(z.reshape(1, -1))[0]
            z = (z + float(args.dt) * (drift + action)).clamp(low, high)
            path.append(z.clone())
    return torch.stack(path, dim=0)


@torch.no_grad()
def _coarse_controls_for_shifted_paths(
    args: argparse.Namespace,
    model: LocalGridRBFDynamics,
    state: torch.Tensor,
    target_paths: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    """Fit coarse controls to shifted coarse paths from an updated state.

    Args:
        target_paths: Tensor with shape ``(batch, horizon + 1, state_dim)``.

    Returns:
        Coarse controls with shape ``(batch, horizon, state_dim)``.
    """

    if target_paths.shape[1] <= 1:
        return torch.empty(
            target_paths.shape[0],
            0,
            model.state_dim,
            dtype=torch.float32,
            device=model.device,
        )

    z = state.reshape(1, -1).to(device=model.device, dtype=torch.float32).expand(target_paths.shape[0], -1).clone()
    target = target_paths.detach().to(device=model.device, dtype=torch.float32)
    shifted_target = (target + (z[:, None, :] - target[:, :1, :])).clamp(low, high)
    coarse_factor = max(1, int(args.coarse_dt_factor))
    coarse_dt = float(args.dt) * float(coarse_factor)
    controls = []
    for t in range(1, shifted_target.shape[1]):
        next_target = shifted_target[:, t]
        drift = model(z)
        action = (next_target - z) / coarse_dt - drift
        action = _project_action_radius(action, float(args.action_max))
        for _ in range(coarse_factor):
            z = (z + float(args.dt) * (model(z) + action)).clamp(low, high)
        controls.append(action)
    return torch.stack(controls, dim=1)


@torch.no_grad()
def _plan_eig_action(
    args: argparse.Namespace,
    model: LocalGridRBFDynamics,
    prior_precision: torch.Tensor,
    state: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    generator: torch.Generator,
    np_generator: np.random.Generator,
    planner_cache: dict[str, torch.Tensor],
    live_snapshot: dict[str, object] | None = None,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Choose an environment-time action chunk with iCEM over block EIG."""

    horizon = _planner_horizon(args, planner_cache)
    action_max = float(args.action_max)
    sample_count = max(0, int(args.num_action_sequences))
    iterations = max(1, int(args.icem_iterations))
    elite_count = max(1, int(args.icem_elites))
    alpha = float(np.clip(float(args.icem_alpha), 0.0, 1.0))
    reuse_frac = float(np.clip(float(args.icem_reuse_frac), 0.0, 1.0))
    shift_steps_value = float(args.replan_interval)
    cached_shift_steps = planner_cache.get("shift_steps")
    if isinstance(cached_shift_steps, torch.Tensor) and cached_shift_steps.numel() == 1:
        shift_steps_value = float(cached_shift_steps.item())
    coarse_shift = int(np.ceil(max(0.0, shift_steps_value) / max(1, int(args.coarse_dt_factor))))
    init_std = max(action_max * float(args.icem_init_std), 1e-6)

    mean = torch.zeros(horizon, 2, dtype=torch.float32, device=model.device)
    std = torch.full_like(mean, init_std)
    zero_fill = torch.zeros(2, dtype=torch.float32, device=model.device)
    std_fill = torch.full((2,), init_std, dtype=torch.float32, device=model.device)

    cached_mean = planner_cache.get("mean")
    if isinstance(cached_mean, torch.Tensor) and cached_mean.shape == mean.shape:
        mean = _shift_action_sequences(cached_mean.to(model.device), coarse_shift, zero_fill)
    cached_std = planner_cache.get("std")
    if isinstance(cached_std, torch.Tensor) and cached_std.shape == std.shape:
        std = _shift_action_sequences(cached_std.to(model.device), coarse_shift, std_fill).clamp_min(1e-3)

    reused_elites = torch.empty(0, horizon, 2, dtype=torch.float32, device=model.device)
    cached_elites = planner_cache.get("elites")
    if isinstance(cached_elites, torch.Tensor) and cached_elites.shape[1:] == (horizon, 2):
        reused_count = int(cached_elites.shape[0] * reuse_frac)
        if reused_count > 0:
            reused_elites = _shift_action_sequences(cached_elites[:reused_count].to(model.device), coarse_shift, zero_fill)

    fixed_actions = _fixed_action_sequences(args, model.device, horizon)
    best_actions = fixed_actions[0].clone()
    best_score = -float("inf")
    elites = torch.empty(0, horizon, 2, dtype=torch.float32, device=model.device)

    use_numpy_score = str(args.eig_score_backend).lower() == "numpy" and model.state_dim == 2
    numpy_arrays = _numpy_planning_arrays(model) if use_numpy_score else None
    precision_np = prior_precision.detach().cpu().numpy() if use_numpy_score else None
    live_version = -1

    def refresh_live_snapshot() -> bool:
        nonlocal state, prior_precision, numpy_arrays, precision_np, live_version, best_score, mean, elites
        snapshot = _read_async_live_snapshot(live_snapshot)
        if snapshot is None:
            return False
        snap_state, snap_weights, snap_precision, version = snapshot
        if int(version) == live_version:
            return False
        old_state = state
        mean_path = _rollout_action_sequences(args, model, old_state, mean.reshape(1, horizon, -1), low, high)
        elite_paths = None
        if elites.shape[0] > 0:
            elite_paths = _rollout_action_sequences(args, model, old_state, elites, low, high)
        state = snap_state.to(device=model.device, dtype=torch.float32)
        if bool(args.async_live_refine_parameters):
            model.weights.data.copy_(snap_weights.to(device=model.device, dtype=torch.float32))
            model.network.weights = model.weights
            prior_precision = snap_precision.to(device=model.device, dtype=torch.float32)
            if use_numpy_score:
                numpy_arrays = _numpy_planning_arrays(model)
                precision_np = prior_precision.detach().cpu().numpy()
        mean = _coarse_controls_for_shifted_paths(args, model, state, mean_path, low, high)[0]
        if elite_paths is not None:
            elites = _coarse_controls_for_shifted_paths(args, model, state, elite_paths, low, high)
        live_version = int(version)
        best_score = -float("inf")
        _mark_async_live_refresh(live_snapshot)
        return True

    def score_candidates(candidate_paths: torch.Tensor, candidate_actions: torch.Tensor, score_indices: torch.Tensor) -> torch.Tensor:
        score_list = [int(b) for b in score_indices]
        if use_numpy_score:
            assert numpy_arrays is not None and precision_np is not None
            paths_np = candidate_paths[score_indices].detach().cpu().numpy()
            actions_np = candidate_actions[score_indices].detach().cpu().numpy()
            entry_indices, entry_values, entry_valid = _numpy_local_feature_entries(args, numpy_arrays, paths_np)
            scores_np = [
                _block_weight_eig_score_numpy_from_entries(
                    args,
                    numpy_arrays,
                    precision_np,
                    paths_np[i],
                    actions_np[i],
                    entry_indices[i],
                    entry_values[i],
                    entry_valid[i],
                )
                for i in range(paths_np.shape[0])
            ]
            return torch.as_tensor(scores_np, dtype=torch.float32, device=model.device)
        return torch.stack([
            _block_weight_eig_score(args, model, prior_precision, candidate_paths[b], candidate_actions[b])
            for b in score_list
        ])

    factor_decrease = max(float(args.icem_factor_decrease_num), 1.0)
    for iteration in range(iterations):
        refresh_live_snapshot()
        current_sample_count = sample_count
        if iteration > 0 and sample_count > 0:
            current_sample_count = max(elite_count * 2, int(sample_count / (factor_decrease**iteration)))
        if current_sample_count > 0:
            if float(args.icem_noise_beta) > 0.0 and horizon > 1:
                noise_np = colorednoise.powerlaw_psd_gaussian(
                    float(args.icem_noise_beta),
                    size=(current_sample_count, 2, horizon),
                    random_state=np_generator,
                )
                noise = torch.as_tensor(noise_np, dtype=torch.float32, device=model.device).transpose(1, 2)
            else:
                noise = torch.randn(current_sample_count, horizon, 2, generator=generator, dtype=torch.float32).to(model.device)
            sampled_actions = _project_action_radius(mean.unsqueeze(0) + std.unsqueeze(0) * noise, action_max)
        else:
            sampled_actions = torch.empty(0, horizon, 2, dtype=torch.float32, device=model.device)

        candidates = [sampled_actions]
        if bool(args.icem_use_mean_actions) and iteration == iterations - 1:
            candidates.insert(0, _project_action_radius(mean.unsqueeze(0), action_max))
        if iteration == 0:
            candidates.insert(0, fixed_actions)
            if reused_elites.shape[0] > 0:
                candidates.insert(0, reused_elites)
        elif elites.shape[0] > 0:
            keep_count = max(1, int(elites.shape[0] * reuse_frac))
            candidates.insert(0, elites[:keep_count])
        actions = torch.cat([candidate for candidate in candidates if candidate.shape[0] > 0], dim=0)

        coarse_paths = _rollout_action_sequences(args, model, state, actions, low, high)
        score_indices = torch.arange(actions.shape[0], device=model.device)
        exact_topk = int(args.eig_score_topk)
        if exact_topk > 0 and actions.shape[0] > exact_topk:
            k_exact = min(actions.shape[0], max(exact_topk, elite_count))
            proxy_scores = _candidate_proxy_scores(args, model, prior_precision, coarse_paths, actions)
            score_indices = torch.topk(proxy_scores, k=k_exact).indices
        exact_scores = score_candidates(coarse_paths, actions, score_indices)
        scores = torch.full((actions.shape[0],), -float("inf"), dtype=exact_scores.dtype, device=model.device)
        scores[score_indices] = exact_scores
        iteration_best = int(torch.argmax(scores).item())
        iteration_score = float(scores[iteration_best].item())
        if iteration_score > best_score:
            best_score = iteration_score
            best_actions = actions[iteration_best].clone()

        k = min(elite_count, actions.shape[0])
        elite_idx = torch.topk(scores, k=k).indices
        elites = actions[elite_idx].clone()
        elite_mean = elites.mean(dim=0)
        elite_std = elites.std(dim=0, unbiased=False).clamp_min(1e-3)
        mean = _project_action_radius(alpha * mean + (1.0 - alpha) * elite_mean, action_max)
        std = alpha * std + (1.0 - alpha) * elite_std


    planner_cache["mean"] = mean.detach().clone()
    planner_cache["std"] = std.detach().clone()
    planner_cache["elites"] = elites.detach().clone()

    fine_path = _fine_planned_path(args, model, state, best_actions, low, high)
    fine_actions = best_actions.repeat_interleave(max(1, int(args.coarse_dt_factor)), dim=0)
    return fine_actions, best_score, fine_path


def _diffuse_precision(precision: torch.Tensor, weight_diffusion: float, steps: int) -> torch.Tensor:
    out = precision.detach().clone()
    for _ in range(max(0, int(steps))):
        if float(weight_diffusion) <= 0.0:
            break
        var = 1.0 / out.clamp_min(1e-12)
        out = 1.0 / (var + float(weight_diffusion))
    return out


def _predict_boundary_state(
    args: argparse.Namespace,
    model: LocalGridRBFDynamics,
    state: torch.Tensor,
    env_actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> torch.Tensor:
    z = state.reshape(-1).detach().clone()
    for action in env_actions:
        drift = model(z.reshape(1, -1))[0]
        z = (z + float(args.dt) * (drift + action)).clamp(low, high)
    return z


@torch.no_grad()
def _track_shifted_path_controls(
    args: argparse.Namespace,
    model: LocalGridRBFDynamics,
    state: torch.Tensor,
    target_path: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refit visible controls so a stale plan starts at the updated state.

    ``target_path`` has shape ``(T + 1, state_dim)``.  The path geometry is
    shifted to start at ``state``; controls are then recomputed under the
    current surrogate dynamics.
    """

    z = state.reshape(-1).detach().clone()
    target = target_path.detach().to(device=model.device, dtype=torch.float32)
    if target.shape[0] <= 1:
        empty = torch.empty(0, model.state_dim, dtype=torch.float32, device=model.device)
        return empty, z.reshape(1, -1)

    shift = z - target[0].reshape(-1)
    shifted_target = (target + shift.reshape(1, -1)).clamp(low, high)
    actions = []
    path = [z.clone()]
    for next_target in shifted_target[1:]:
        drift = model(z.reshape(1, -1))[0]
        action = (next_target.reshape(-1) - z) / float(args.dt) - drift
        action = _project_action_radius(action.reshape(1, -1), float(args.action_max))[0]
        z = (z + float(args.dt) * (drift + action)).clamp(low, high)
        actions.append(action)
        path.append(z.clone())
    return torch.stack(actions, dim=0), torch.stack(path, dim=0)


def _score_fine_plan_current_eig(
    args: argparse.Namespace,
    model: LocalGridRBFDynamics,
    prior_precision: torch.Tensor,
    fine_path: torch.Tensor,
    fine_actions: torch.Tensor,
) -> float:
    """Score an env-time plan under the current posterior on the coarse grid."""

    factor = max(1, int(args.coarse_dt_factor))
    horizon = min((fine_path.shape[0] - 1) // factor, fine_actions.shape[0] // factor)
    if horizon <= 0:
        return -float("inf")
    coarse_path = fine_path[: horizon * factor + 1 : factor]
    coarse_actions = fine_actions[: horizon * factor].reshape(horizon, factor, -1).mean(dim=1)
    return float(_block_weight_eig_score(args, model, prior_precision, coarse_path, coarse_actions).item())


def _clone_planner_cache(cache: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in cache.items() if torch.is_tensor(value)}


def _clone_planning_model(model: LocalGridRBFDynamics) -> LocalGridRBFDynamics:
    clone = LocalGridRBFDynamics(
        state_low=model.state_low.detach().clone(),
        state_high=model.state_high.detach().clone(),
        grid_points=int(model.grid_points),
        lengthscale=float(model.lengthscale),
        active_radius=int(model.active_radius),
        dt=float(model.dt),
        device=model.device,
    )
    clone.weights.data.copy_(model.weights.detach())
    clone.network.weights = clone.weights
    return clone


def _relative_tensor_change(current: torch.Tensor, reference: torch.Tensor) -> float:
    denom = max(1.0, float(torch.linalg.norm(reference).item()))
    return float(torch.linalg.norm(current.detach() - reference.detach().to(current.device)).item()) / denom


def _new_async_live_snapshot() -> dict[str, object]:
    return {"lock": Lock(), "version": 0, "refreshes": 0}


def _publish_async_live_snapshot(
    snapshot: dict[str, object] | None,
    state: torch.Tensor,
    model: LocalGridRBFDynamics,
    precision: torch.Tensor,
) -> None:
    if snapshot is None:
        return
    lock = snapshot["lock"]
    with lock:
        snapshot["state"] = state.detach().cpu().clone()
        snapshot["weights"] = model.weights.detach().cpu().clone()
        snapshot["precision"] = precision.detach().cpu().clone()
        snapshot["version"] = int(snapshot.get("version", 0)) + 1


def _read_async_live_snapshot(snapshot: dict[str, object] | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
    if snapshot is None:
        return None
    lock = snapshot["lock"]
    with lock:
        if "state" not in snapshot or "weights" not in snapshot or "precision" not in snapshot:
            return None
        return (
            snapshot["state"].clone(),
            snapshot["weights"].clone(),
            snapshot["precision"].clone(),
            int(snapshot["version"]),
        )


def _count_async_live_refresh(snapshot: dict[str, object] | None) -> int:
    if snapshot is None:
        return 0
    lock = snapshot["lock"]
    with lock:
        return int(snapshot.get("refreshes", 0))


def _mark_async_live_refresh(snapshot: dict[str, object] | None) -> None:
    if snapshot is None:
        return
    lock = snapshot["lock"]
    with lock:
        snapshot["refreshes"] = int(snapshot.get("refreshes", 0)) + 1


def _plan_eig_action_seeded(
    args: argparse.Namespace,
    model: LocalGridRBFDynamics,
    prior_precision: torch.Tensor,
    state: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    planner_cache: dict[str, torch.Tensor],
    seed: int,
    torch_rng_state: torch.Tensor | None = None,
    numpy_rng_state: dict | None = None,
    isolate_torch_rng: bool = False,
    live_snapshot: dict[str, object] | None = None,
) -> tuple[torch.Tensor, float, torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    plan_args = copy.copy(args)
    if args.async_worker_iterations is not None:
        plan_args.icem_iterations = int(max(1, args.async_worker_iterations))
    generator = torch.Generator(device="cpu")
    if torch_rng_state is None:
        generator.manual_seed(int(seed))
    else:
        generator.set_state(torch_rng_state.cpu())
    np_generator = np.random.default_rng(int(seed))
    if numpy_rng_state is not None:
        np_generator.bit_generator.state = copy.deepcopy(numpy_rng_state)
    cache = _clone_planner_cache(planner_cache)
    if bool(isolate_torch_rng):
        with torch.random.fork_rng(devices=[]):
            actions, score, path = _plan_eig_action(
                plan_args,
                model,
                prior_precision,
                state,
                low,
                high,
                generator,
                np_generator,
                cache,
                live_snapshot,
            )
    else:
        actions, score, path = _plan_eig_action(
            plan_args,
            model,
            prior_precision,
            state,
            low,
            high,
            generator,
            np_generator,
            cache,
            live_snapshot,
        )
    return (
        actions.detach(),
        float(score),
        path.detach(),
        cache,
        path[0].detach(),
        model.weights.detach().clone(),
        prior_precision.detach().clone(),
        generator.get_state(),
        copy.deepcopy(np_generator.bit_generator.state),
    )


def _submit_async_plan(
    args: argparse.Namespace,
    executor: ThreadPoolExecutor,
    model: LocalGridRBFDynamics,
    prior_precision: torch.Tensor,
    state: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    planner_cache: dict[str, torch.Tensor],
    seed: int,
    torch_rng_state: torch.Tensor | None = None,
    numpy_rng_state: dict | None = None,
    isolate_torch_rng: bool = False,
    live_snapshot: dict[str, object] | None = None,
) -> Future:
    return executor.submit(
        _plan_eig_action_seeded,
        args,
        _clone_planning_model(model),
        prior_precision.detach().clone(),
        state.detach().clone(),
        low.detach().clone(),
        high.detach().clone(),
        _clone_planner_cache(planner_cache),
        int(seed),
        None if torch_rng_state is None else torch_rng_state.detach().clone(),
        None if numpy_rng_state is None else copy.deepcopy(numpy_rng_state),
        bool(isolate_torch_rng),
        live_snapshot,
    )


def _active_center_count_and_mark(model: LocalGridRBFDynamics, state: torch.Tensor, visited: torch.Tensor) -> int:
    """Mark centers whose weights can be updated from the visited ``state``."""

    indices, _, valid = model.local_feature_entries(state.reshape(1, -1))
    active = indices[0, valid[0]]
    visited[active] = True
    return int(active.numel())


def _simulate_online(
    args: argparse.Namespace, model: LocalGridRBFDynamics
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[torch.Tensor], list[torch.Tensor], np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    device = model.device
    low, high = _bounds(args, device)
    torch.manual_seed(int(args.seed))
    env = _new_env(args, device)
    true_state, _ = env.reset(seed=int(args.seed), options={"initial_state": args.initial_state})
    true_state = true_state.to(device=device, dtype=torch.float32).clamp(low, high)
    env.state = true_state.clone()
    rbf_state = true_state.clone()
    updater = OnlineRBFWeightUpdater(
        model,
        lr=args.online_lr,
        ridge=args.online_ridge,
        prior_precision=args.prior_precision,
        eig_beta=args.eig_beta,
        weight_diffusion=args.rbf_weight_diffusion,
        smoothing_radius=args.weight_smoothing_radius,
        smoothing_strength=args.weight_smoothing_strength,
        smoothing_precision_threshold=args.weight_smoothing_precision_threshold,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.planning_seed))
    np_generator = np.random.default_rng(int(args.planning_seed))
    visited_centers = torch.zeros(model.centers.shape[0], dtype=torch.bool, device=device)

    true_states = [true_state.detach().cpu().numpy()]
    rbf_states = [rbf_state.detach().cpu().numpy()]
    actions = [np.zeros(model.state_dim, dtype=np.float32)]
    update_errors = [0.0]
    active_counts = [0]
    visited_fractions = [0.0]
    eig_scores = [0.0]
    weight_snapshots = [model.weights.detach().cpu().clone()]
    information_snapshots = [updater.precision.detach().mean(dim=1).cpu().clone()]
    planned_paths = [true_state.reshape(1, -1).detach().cpu().numpy()]

    planned_actions = torch.zeros(1, model.state_dim, dtype=torch.float32, device=device)
    planned_eig_score = 0.0
    planned_path = true_state.reshape(1, -1).detach().clone()
    planner_cache: dict[str, torch.Tensor] = {}
    replan_interval = max(1, int(args.replan_interval))
    adaptive_horizon_enabled = bool(args.adaptive_planning_horizon)
    adaptive_enabled = bool(args.adaptive_cadence or adaptive_horizon_enabled)
    adaptive_min_interval = max(1, int(args.adaptive_replan_min_interval))
    plan_capacity = max(1, int(args.planning_horizon) * max(1, int(args.coarse_dt_factor)))
    adaptive_max_interval = args.adaptive_replan_max_interval
    if adaptive_max_interval is None:
        adaptive_max_interval = replan_interval
    adaptive_max_interval = min(plan_capacity, max(adaptive_min_interval, int(adaptive_max_interval)))
    adaptive_threshold = args.adaptive_replan_state_error_threshold
    adaptive_threshold = None if adaptive_threshold is None else float(adaptive_threshold)
    chunk_step = 0
    plan_start_step = 0
    plan_interval = replan_interval
    adaptive_stats = {"replans": 0, "cadence": 0, "state_error": 0, "tail": 0}
    planned_horizons: list[int] = []
    state_tracking_error = 0.0

    update_interval = max(1, int(args.update_interval))
    async_executor = ThreadPoolExecutor(max_workers=1) if bool(args.async_planning) else None
    async_future: Future | None = None
    async_plan_count = 0
    async_live_snapshot: dict[str, object] | None = None
    async_stats = {
        "launched": 0,
        "used": 0,
        "stale": 0,
        "waiting": 0,
        "blocking": 0,
        "refined": 0,
        "path_refined": 0,
        "path_tracked": 0,
        "live_refreshed": 0,
        "eig_stale": 0,
    }
    async_mismatches: list[float] = []
    async_model_mismatches: list[float] = []
    async_global_rng_state: torch.Tensor | None = None
    async_launch_lead_arg = args.async_launch_lead_steps

    def effective_async_launch_lead() -> int:
        if async_launch_lead_arg is None:
            return int(plan_interval if adaptive_horizon_enabled else 0)
        return max(0, int(async_launch_lead_arg))

    def effective_async_model_stale_tolerance() -> float:
        if args.async_model_stale_tolerance is not None:
            return float(args.async_model_stale_tolerance)
        if adaptive_horizon_enabled and effective_async_launch_lead() > 0:
            return float("inf")
        return 0.0

    def configure_planner_cache(
        cache: dict[str, torch.Tensor],
        plan_state: torch.Tensor,
        planning_precision: torch.Tensor,
        shift_steps: float,
    ) -> None:
        horizon = _adaptive_planning_horizon(args, model, planning_precision, plan_state)
        cache["planning_horizon"] = torch.as_tensor(float(horizon), dtype=torch.float32)
        cache["shift_steps"] = torch.as_tensor(float(max(0.0, shift_steps)), dtype=torch.float32)

    def record_plan_horizon(cache: dict[str, torch.Tensor], actions: torch.Tensor) -> None:
        planned_horizons.append(_planner_horizon(args, cache))

    def plan_execution_interval(actions: torch.Tensor) -> int:
        if bool(args.adaptive_cadence):
            cap = adaptive_max_interval
        elif adaptive_horizon_enabled:
            cap = int(actions.shape[0])
        else:
            cap = replan_interval
        return max(1, min(int(cap), int(actions.shape[0])))

    def submit_next_plan(
        boundary_state: torch.Tensor,
        launch_step: int,
        planning_precision: torch.Tensor | None = None,
    ) -> Future | None:
        nonlocal async_plan_count, async_global_rng_state, async_live_snapshot
        if async_executor is None or launch_step + 1 >= int(args.steps):
            return None
        async_plan_count += 1
        async_stats["launched"] += 1
        launch_lead = effective_async_launch_lead()
        async_global_rng_state = torch.random.get_rng_state() if launch_lead == 0 else None
        seed = int(args.planning_seed) + 1000003 + 7919 * async_plan_count + int(launch_step)
        cache = _clone_planner_cache(planner_cache)
        plan_precision = updater.precision if planning_precision is None else planning_precision
        configure_planner_cache(cache, boundary_state, plan_precision, float(max(0, plan_interval)))
        async_live_snapshot = _new_async_live_snapshot() if bool(args.async_live_refine) else None
        _publish_async_live_snapshot(async_live_snapshot, boundary_state, model, plan_precision)
        return _submit_async_plan(
            args,
            async_executor,
            model,
            updater.precision if planning_precision is None else planning_precision,
            boundary_state,
            low,
            high,
            cache,
            seed,
            generator.get_state(),
            copy.deepcopy(np_generator.bit_generator.state),
            launch_lead == 0,
            async_live_snapshot,
        )

    try:
        for step_idx in range(int(args.steps)):
            updater.diffuse_prior()
            prev_state = true_state.detach().clone()
            if chunk_step < planned_path.shape[0]:
                denom = max(1.0, float(torch.linalg.norm(prev_state).item()))
                state_tracking_error = float(torch.linalg.norm(prev_state - planned_path[chunk_step].to(device)).item()) / denom
            else:
                state_tracking_error = float("inf")
            plan_tail_exhausted = chunk_step >= planned_actions.shape[0]
            if adaptive_enabled:
                state_error_due = (
                    adaptive_threshold is not None
                    and chunk_step >= adaptive_min_interval
                    and state_tracking_error > adaptive_threshold
                )
                replan_reason = "tail" if plan_tail_exhausted else "cadence"
                if state_error_due:
                    replan_reason = "state_error"
                replan_due = step_idx == 0 or plan_tail_exhausted or state_error_due or chunk_step >= adaptive_max_interval
            else:
                replan_reason = "cadence"
                replan_due = step_idx % replan_interval == 0
            if replan_due:
                adaptive_stats[replan_reason] = adaptive_stats.get(replan_reason, 0) + 1
                adaptive_stats["replans"] += 1
                used_new_plan = False
                if async_executor is not None and step_idx > 0:
                    if async_future is not None:
                        if not async_future.done():
                            async_stats["waiting"] += 1
                        try:
                            (
                                async_actions,
                                async_score,
                                async_path,
                                async_cache,
                                async_state,
                                async_weights,
                                async_precision,
                                async_torch_rng_state,
                                async_numpy_rng_state,
                            ) = async_future.result()
                            if async_global_rng_state is not None:
                                torch.random.set_rng_state(async_global_rng_state)
                                async_global_rng_state = None
                            denom = max(1.0, float(torch.linalg.norm(prev_state).item()))
                            mismatch = torch.linalg.norm(prev_state - async_state.to(device)).item() / denom
                            model_mismatch = max(
                                _relative_tensor_change(model.weights, async_weights.to(device)),
                                _relative_tensor_change(updater.precision, async_precision.to(device)),
                            )
                            async_mismatches.append(float(mismatch))
                            async_model_mismatches.append(float(model_mismatch))
                            if (
                                mismatch <= float(args.async_stale_tolerance)
                                and model_mismatch <= effective_async_model_stale_tolerance()
                            ):
                                planned_actions = async_actions.to(device)
                                planned_eig_score = float(async_score)
                                planned_path = async_path.to(device)
                                planner_cache = {k: v.to(device) for k, v in async_cache.items()}
                                generator.set_state(async_torch_rng_state.cpu())
                                np_generator.bit_generator.state = copy.deepcopy(async_numpy_rng_state)
                                refine_iterations = int(max(0, args.async_refine_iterations))
                                if refine_iterations > 0:
                                    refine_args = copy.copy(args)
                                    refine_args.icem_iterations = refine_iterations
                                    configure_planner_cache(planner_cache, prev_state, updater.precision, 0.0)
                                    planned_actions, planned_eig_score, planned_path = _plan_eig_action(
                                        refine_args,
                                        model,
                                        updater.precision,
                                        prev_state,
                                        low,
                                        high,
                                        generator,
                                        np_generator,
                                        planner_cache,
                                    )
                                    async_stats["refined"] += 1
                                else:
                                    planned_actions, planned_path = _track_shifted_path_controls(
                                        args, model, prev_state, planned_path, low, high
                                    )
                                    async_stats["path_refined"] += 1
                                eig_ok = True
                                min_eig_frac = max(0.0, float(args.async_min_current_eig_frac))
                                if min_eig_frac > 0.0:
                                    current_eig_score = _score_fine_plan_current_eig(
                                        args, model, updater.precision, planned_path, planned_actions
                                    )
                                    if current_eig_score < min_eig_frac * max(float(async_score), 1e-12):
                                        eig_ok = False
                                        async_stats["eig_stale"] += 1
                                    else:
                                        planned_eig_score = current_eig_score
                                if eig_ok:
                                    chunk_step = 0
                                    plan_start_step = step_idx
                                    plan_interval = plan_execution_interval(planned_actions)
                                    record_plan_horizon(planner_cache, planned_actions)
                                    async_stats["used"] += 1
                                    used_new_plan = True
                            else:
                                async_stats["stale"] += 1
                        finally:
                            async_future = None
                            async_stats["live_refreshed"] += _count_async_live_refresh(async_live_snapshot)
                            async_live_snapshot = None
                    if not used_new_plan:
                        async_stats["blocking"] += 1
                        configure_planner_cache(planner_cache, prev_state, updater.precision, float(max(0, chunk_step)))
                        planned_actions, planned_eig_score, planned_path = _plan_eig_action(
                            args, model, updater.precision, prev_state, low, high, generator, np_generator, planner_cache
                        )
                        chunk_step = 0
                        plan_start_step = step_idx
                        plan_interval = plan_execution_interval(planned_actions)
                        record_plan_horizon(planner_cache, planned_actions)
                        used_new_plan = True
                if async_executor is None or step_idx == 0:
                    if async_executor is not None:
                        async_stats["blocking"] += 1
                    configure_planner_cache(planner_cache, prev_state, updater.precision, float(max(0, chunk_step)))
                    planned_actions, planned_eig_score, planned_path = _plan_eig_action(
                        args, model, updater.precision, prev_state, low, high, generator, np_generator, planner_cache
                    )
                    chunk_step = 0
                    plan_start_step = step_idx
                    plan_interval = plan_execution_interval(planned_actions)
                    record_plan_horizon(planner_cache, planned_actions)
                    used_new_plan = True
                launch_lead = effective_async_launch_lead()
                if async_executor is not None and async_future is None and launch_lead >= plan_interval:
                    boundary_idx = min(plan_interval, planned_path.shape[0] - 1)
                    planning_precision = _diffuse_precision(
                        updater.precision,
                        float(args.rbf_weight_diffusion),
                        plan_interval,
                    )
                    async_future = submit_next_plan(planned_path[boundary_idx].detach(), step_idx, planning_precision)
            path_track_blend = float(np.clip(float(args.async_path_track_blend), 0.0, 1.0))
            if (
                async_executor is not None
                and path_track_blend > 0.0
                and not replan_due
                and chunk_step < planned_actions.shape[0]
                and chunk_step + 1 < planned_path.shape[0]
            ):
                tracking_actions, _tracking_path = _track_shifted_path_controls(
                    args,
                    model,
                    prev_state,
                    planned_path[chunk_step : chunk_step + 2],
                    low,
                    high,
                )
                if tracking_actions.shape[0] > 0:
                    planned_actions[chunk_step] = _project_action_radius(
                        (
                            (1.0 - path_track_blend) * planned_actions[chunk_step]
                            + path_track_blend * tracking_actions[0]
                        ).reshape(1, -1),
                        float(args.action_max),
                    )[0]
                    # Keep planned_path fixed as the state-error target; only
                    # the visible next control is adjusted between replans.
                    async_stats["path_tracked"] += 1
            action = planned_actions[min(chunk_step, planned_actions.shape[0] - 1)]
            eig_score = planned_eig_score
            chunk_step += 1
            next_state, *_ = env.step(action)
            true_state = next_state.to(device=device, dtype=torch.float32).clamp(low, high)
            env.state = true_state.clone()

            with torch.no_grad():
                rbf_drift = model(rbf_state.unsqueeze(0))[0]
            if float(args.surrogate_drift_clip) > 0.0:
                rbf_drift = torch.clamp(rbf_drift, -float(args.surrogate_drift_clip), float(args.surrogate_drift_clip))
            rbf_state = (rbf_state + float(args.dt) * (rbf_drift + action)).clamp(low, high)
            correction = float(args.surrogate_state_correction)
            if correction > 0.0:
                rbf_state = (rbf_state + correction * (true_state - rbf_state)).clamp(low, high)

            observed_drift = (true_state - prev_state) / float(args.dt) - action
            state_err = updater.update(prev_state, observed_drift) if step_idx % update_interval == 0 else 0.0
            active_count = _active_center_count_and_mark(model, prev_state, visited_centers)

            if async_future is not None and async_live_snapshot is not None:
                next_step = step_idx + 1
                next_boundary = plan_start_step + plan_interval
                steps_until_boundary = max(0, next_boundary - next_step)
                live_interval = max(1, int(args.async_live_refine_interval))
                if steps_until_boundary <= live_interval or next_step % live_interval == 0:
                    planning_precision = _diffuse_precision(
                        updater.precision,
                        float(args.rbf_weight_diffusion),
                        steps_until_boundary + 1,
                    )
                    _publish_async_live_snapshot(async_live_snapshot, true_state, model, planning_precision)

            true_states.append(true_state.detach().cpu().numpy())
            rbf_states.append(rbf_state.detach().cpu().numpy())
            actions.append(action.detach().cpu().numpy().astype(np.float32))
            update_errors.append(state_err)
            active_counts.append(active_count)
            visited_fractions.append(float(visited_centers.float().mean().item()))
            eig_scores.append(float(eig_score))
            weight_snapshots.append(model.weights.detach().cpu().clone())
            information_snapshots.append(updater.precision.detach().mean(dim=1).cpu().clone())
            planned_paths.append(planned_path.detach().cpu().numpy())
            launch_lead = effective_async_launch_lead()
            if async_executor is not None and async_future is None and launch_lead < plan_interval:
                next_step = step_idx + 1
                next_boundary = plan_start_step + plan_interval
                steps_until_boundary = next_boundary - next_step
                if 0 <= steps_until_boundary <= launch_lead and next_boundary < int(args.steps):
                    remaining = planned_actions[chunk_step : chunk_step + steps_until_boundary]
                    boundary_state = _predict_boundary_state(args, model, true_state, remaining, low, high)
                    planning_precision = _diffuse_precision(
                        updater.precision,
                        float(args.rbf_weight_diffusion),
                        steps_until_boundary + 1,
                    )
                    async_future = submit_next_plan(boundary_state.detach(), step_idx, planning_precision)
    finally:
        if async_future is not None:
            async_future.cancel()
        if async_executor is not None:
            async_executor.shutdown(wait=True, cancel_futures=True)
    if adaptive_enabled:
        horizon_msg = ""
        if planned_horizons:
            horizon_msg = (
                f" horizon_min={min(planned_horizons)}"
                f" horizon_max={max(planned_horizons)}"
                f" horizon_mean={float(np.mean(planned_horizons)):.2f}"
            )
        print(
            "adaptive_planning "
            + " ".join(f"{key}={value}" for key, value in adaptive_stats.items())
            + horizon_msg
            + f" last_state_error={state_tracking_error:.3f}",
            flush=True,
        )
    if async_executor is not None:
        mismatch_msg = ""
        if async_mismatches:
            mismatch_msg = (
                f" mismatch_mean={float(np.mean(async_mismatches)):.3f}"
                f" mismatch_max={float(np.max(async_mismatches)):.3f}"
            )
        if async_model_mismatches:
            mismatch_msg += (
                f" model_mismatch_mean={float(np.mean(async_model_mismatches)):.3f}"
                f" model_mismatch_max={float(np.max(async_model_mismatches)):.3f}"
            )
        print(
            "async_planning "
            + " ".join(f"{key}={value}" for key, value in async_stats.items())
            + mismatch_msg,
            flush=True,
        )

    return (
        np.asarray(true_states),
        np.asarray(rbf_states),
        np.asarray(actions),
        np.asarray(update_errors, dtype=np.float32),
        weight_snapshots,
        information_snapshots,
        np.asarray(active_counts, dtype=np.int32),
        np.asarray(visited_fractions, dtype=np.float32),
        np.asarray(eig_scores, dtype=np.float32),
        planned_paths,
    )


def _grid(args: argparse.Namespace, device: torch.device):
    axis = np.linspace(float(args.state_low), float(args.state_high), int(args.grid_size), dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    pts = torch.as_tensor(points, dtype=torch.float32, device=device)
    return xx, yy, pts


@torch.no_grad()
def _true_field(args: argparse.Namespace, model: LocalGridRBFDynamics):
    xx, yy, pts = _grid(args, model.device)
    true_vel = _true_drift(args, pts).cpu().numpy()
    shape = xx.shape
    return xx, yy, true_vel[:, 0].reshape(shape), true_vel[:, 1].reshape(shape)


@torch.no_grad()
def _rbf_field(
    model: LocalGridRBFDynamics, weights: torch.Tensor, grid_points: torch.Tensor, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    old_weights = model.weights.data.detach().clone()
    try:
        model.weights.data.copy_(weights.to(model.device))
        vel = model(grid_points).detach().cpu().numpy().reshape(shape[0], shape[1], 2)
    finally:
        model.weights.data.copy_(old_weights)
    return vel[:, :, 0], vel[:, :, 1]


def _r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1, 2)
    denom = max(float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2)), 1e-12)
    return 1.0 - float(np.sum((y - pred) ** 2)) / denom


def _information_limits(parameter_information: np.ndarray) -> tuple[float, float]:
    values = np.asarray(parameter_information, dtype=np.float32)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1e-12, 1.0
    vmin = max(float(np.nanmin(values)), 1e-12)
    vmax = max(float(np.nanpercentile(values, 99.5)), vmin * 1.01)
    return vmin, vmax


def _predictive_r2_trace(
    model: LocalGridRBFDynamics,
    weights: list[torch.Tensor],
    grid: tuple[np.ndarray, np.ndarray, torch.Tensor],
    true_field: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    interval: int,
) -> np.ndarray:
    """Compute global vector-field R2 over time from stored weight snapshots."""

    xx, _, grid_points = grid
    target = np.stack([true_field[2], true_field[3]], axis=-1)
    values = np.full(len(weights), np.nan, dtype=np.float32)
    step = max(1, int(interval))
    indices = list(range(0, len(weights), step))
    if indices[-1] != len(weights) - 1:
        indices.append(len(weights) - 1)
    for idx in indices:
        rbf_u, rbf_v = _rbf_field(model, weights[idx], grid_points, xx.shape)
        values[idx] = _r2_score(target, np.stack([rbf_u, rbf_v], axis=-1))

    last = float(values[0])
    for idx in range(len(values)):
        if np.isfinite(values[idx]):
            last = float(values[idx])
        else:
            values[idx] = last
    return values


def _draw_path(ax: plt.Axes, path: np.ndarray, idx: int, *, color: str, label: str) -> None:
    points = np.asarray(path[: idx + 1, :2], dtype=np.float32)
    if len(points) > 900:
        # ponytail: coarse draw path; raise the cap if per-step trajectory detail matters.
        points = points[np.unique(np.linspace(0, len(points) - 1, 900, dtype=np.int64))]
    if len(points) < 2:
        ax.scatter(points[:, 0], points[:, 1], s=18, color=color, zorder=5, label=label)
        return
    segments = np.stack([points[:-1], points[1:]], axis=1)
    age = len(segments) - 1 - np.arange(len(segments))
    alpha = np.clip(1.0 - age / 240.0, 0.18, 1.0)
    ax.add_collection(
        LineCollection(segments, colors=[(color, float(a)) for a in alpha], linewidths=1.2, zorder=4)
    )
    ax.scatter(points[-1:, 0], points[-1:, 1], s=22, color=color, zorder=6, label=label)


def _style_phase_axis(ax: plt.Axes, args: argparse.Namespace, title: str) -> None:
    ax.set_title(title, fontsize=8.5)
    ax.set_xlim(float(args.state_low), float(args.state_high))
    ax.set_ylim(float(args.state_low), float(args.state_high))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_centers(ax: plt.Axes, model: LocalGridRBFDynamics) -> None:
    centers = model.centers.detach().cpu().numpy()
    ax.scatter(centers[:, 0], centers[:, 1], s=2.0, color="#C7CCD1", alpha=0.45, linewidths=0, zorder=1)


def _draw_active_centers(ax: plt.Axes, model: LocalGridRBFDynamics, state: np.ndarray) -> int:
    z = torch.as_tensor(state, dtype=torch.float32, device=model.device).unsqueeze(0)
    indices, _, valid = model.local_feature_entries(z)
    active = indices[0, valid[0]].detach().cpu()
    centers = model.centers.detach().cpu()[active].numpy()
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="s",
        s=22,
        facecolor="none",
        edgecolor="#FF7F0E",
        linewidth=1.0,
        alpha=0.98,
        zorder=5,
        label="active centers",
    )
    return int(active.numel())


def _draw_action(ax: plt.Axes, state: np.ndarray, action: np.ndarray) -> None:
    norm = float(np.linalg.norm(action))
    if norm <= 1e-12:
        return
    direction = action[:2] / norm
    length = min(0.65, 0.55 * norm / max(norm, 1.0))
    ax.arrow(
        float(state[0]),
        float(state[1]),
        float(length * direction[0]),
        float(length * direction[1]),
        color="#2CA02C",
        width=0.014,
        head_width=0.13,
        length_includes_head=True,
        alpha=0.95,
        zorder=7,
    )


def _setup_figure(dpi: int):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), dpi=int(dpi), constrained_layout=True)
    return fig, axes


def _draw_frame(
    axes,
    *,
    args: argparse.Namespace,
    model: LocalGridRBFDynamics,
    grid: tuple[np.ndarray, np.ndarray, torch.Tensor],
    true_field: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    weights: torch.Tensor,
    parameter_information: np.ndarray,
    true_states: np.ndarray,
    rbf_states: np.ndarray,
    actions: np.ndarray,
    update_errors: np.ndarray,
    active_counts: np.ndarray,
    visited_fractions: np.ndarray,
    eig_scores: np.ndarray,
    planned_paths: list[np.ndarray],
    predictive_r2: np.ndarray,
    idx: int,
    information_vmin: float,
    information_vmax: float,
) -> None:
    ax_true, ax_rbf = axes
    for ax in axes:
        ax.clear()
    xx, yy, grid_points = grid
    _, _, true_u, true_v = true_field
    rbf_u, rbf_v = _rbf_field(model, weights, grid_points, xx.shape)
    info = np.asarray(parameter_information[min(idx, len(parameter_information) - 1)], dtype=np.float32)
    raw_info_grid = info.reshape(model.grid_points, model.grid_points).T
    visited_info = raw_info_grid > float(args.prior_precision) * (1.0 + 1e-6)
    info_grid = np.maximum(raw_info_grid, 1e-12)
    info_alpha = np.where(visited_info, 0.38, 0.0)

    ax_true.streamplot(xx, yy, true_u, true_v, color="#8A8F98", density=1.0, linewidth=0.45, arrowsize=0.55)
    _draw_path(ax_true, true_states, idx, color="#1F77B4", label="true states")
    _draw_action(ax_true, true_states[idx], actions[idx])
    _style_phase_axis(ax_true, args, f"EIG-controlled true {args.dynamics_type}")

    ax_rbf.imshow(
        info_grid,
        extent=(float(args.state_low), float(args.state_high), float(args.state_low), float(args.state_high)),
        origin="lower",
        cmap="viridis",
        norm=LogNorm(vmin=max(float(information_vmin), 1e-12), vmax=max(float(information_vmax), 1e-12)),
        alpha=info_alpha,
        interpolation="nearest",
        zorder=0,
    )
    _draw_centers(ax_rbf, model)
    ax_rbf.streamplot(xx, yy, rbf_u, rbf_v, color="#4F5965", density=1.0, linewidth=0.42, arrowsize=0.50)
    _draw_path(ax_rbf, true_states, idx, color="#1F77B4", label="visited states")
    _draw_path(ax_rbf, rbf_states, idx, color="#D62728", label="inferred states")
    active_count = _draw_active_centers(ax_rbf, model, true_states[idx])
    _style_phase_axis(ax_rbf, args, f"online RBF weights, active centers={active_count}")

    weight_norm = float(torch.linalg.norm(weights).item())
    ax_rbf.text(
        0.02,
        0.03,
        f"predictive R2={float(predictive_r2[idx]):.3f}\n"
        f"visited centers={100.0 * float(visited_fractions[idx]):.1f}%\n"
        f"planned EIG={float(eig_scores[idx]):.2f}\n"
        f"weight ||W||={weight_norm:.2f}\n"
        f"max I_theta={float(np.nanmax(info_grid)):.2g}\n"
        f"update error={float(update_errors[idx]):.2f}\n"
        f"local params={int(active_counts[idx]) * model.state_dim}",
        transform=ax_rbf.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
        zorder=9,
    )

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=5.4, frameon=False, handlelength=1.2)
    ax_true.figure.suptitle(f"step {idx:04d} / {len(true_states) - 1}", fontsize=8.5)


def _frame_indices(n_steps: int, stride: int) -> list[int]:
    if int(stride) < 1:
        raise ValueError("stride must be >= 1")
    indices = list(range(0, int(n_steps), int(stride)))
    if indices[-1] != int(n_steps) - 1:
        indices.append(int(n_steps) - 1)
    return indices


def _make_scene(args: argparse.Namespace):
    device = torch.device(args.device)
    model = _new_surrogate(args, device)
    (
        true_states,
        rbf_states,
        actions,
        update_errors,
        weight_snapshots,
        information_snapshots,
        active_counts,
        visited_fractions,
        eig_scores,
        planned_paths,
    ) = _simulate_online(args, model)
    grid = _grid(args, device)
    true_field = _true_field(args, model)
    predictive_r2 = _predictive_r2_trace(
        model,
        weight_snapshots,
        grid,
        true_field,
        interval=args.r2_interval,
    )
    speed = np.sqrt(true_field[2] ** 2 + true_field[3] ** 2)
    error_vmax = args.error_vmax or float(np.nanpercentile(speed, 99.0))
    reached = np.flatnonzero(predictive_r2 >= float(args.min_r2))
    if bool(args.require_min_r2) and reached.size == 0:
        raise RuntimeError(
            f"Predictive R2 never reached {float(args.min_r2):.3f}; final={float(predictive_r2[-1]):.3f}."
        )
    return (
        model,
        true_states,
        rbf_states,
        actions,
        update_errors,
        weight_snapshots,
        information_snapshots,
        active_counts,
        visited_fractions,
        eig_scores,
        planned_paths,
        grid,
        true_field,
        predictive_r2,
        error_vmax,
    )


def _print_r2_summary(predictive_r2: np.ndarray, min_r2: float, visited_fractions: np.ndarray) -> None:
    reached = np.flatnonzero(predictive_r2 >= float(min_r2))
    first_hit = int(reached[0]) if reached.size else -1
    checkpoints = [0, 25, 100, 250, 500, 1000, 2000, len(predictive_r2) - 1]
    checkpoints = sorted({idx for idx in checkpoints if 0 <= idx < len(predictive_r2)})
    trace = ", ".join(f"{idx}:{float(predictive_r2[idx]):.3f}" for idx in checkpoints)
    cover = ", ".join(f"{idx}:{100.0 * float(visited_fractions[idx]):.1f}%" for idx in checkpoints)
    print(
        f"predictive_r2 start={float(predictive_r2[0]):.3f} "
        f"final={float(predictive_r2[-1]):.3f} max={float(np.max(predictive_r2)):.3f} "
        f"first_ge_{float(min_r2):.2f}={first_hit}",
        flush=True,
    )
    print(f"predictive_r2_trace {trace}", flush=True)
    print(f"visited_centers_trace {cover}", flush=True)


def _jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    out = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def save_track(args: argparse.Namespace, output: Path) -> Path:
    _set_torch_threads(args)
    scene = _make_scene(args)
    (
        _model,
        true_states,
        rbf_states,
        actions,
        update_errors,
        weights,
        parameter_information,
        active_counts,
        visited_fractions,
        eig_scores,
        _planned_paths,
        _grid,
        _true_field,
        predictive_r2,
        error_vmax,
    ) = scene
    _print_r2_summary(predictive_r2, float(args.min_r2), visited_fractions)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        config=json.dumps(_jsonable_args(args), sort_keys=True),
        true_states=true_states,
        rbf_states=rbf_states,
        actions=actions,
        update_errors=update_errors,
        weights=np.stack([w.detach().cpu().numpy() for w in weights], axis=0),
        parameter_information=np.stack([i.detach().cpu().numpy() for i in parameter_information], axis=0),
        active_counts=active_counts,
        visited_fractions=visited_fractions,
        eig_scores=eig_scores,
        predictive_r2=predictive_r2,
        error_vmax=np.asarray(error_vmax, dtype=np.float32),
    )
    print(f"track={output}", flush=True)
    return output


def _load_track_scene(track: Path, render_args: argparse.Namespace):
    if not track.exists():
        raise FileNotFoundError(f"track file not found: {track}")
    data = np.load(track, allow_pickle=False)
    config = json.loads(str(data["config"].item()))
    args = build_parser().parse_args([])
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    for key in ("device", "dpi", "fps", "stride", "step", "mode", "output", "track"):
        setattr(args, key, getattr(render_args, key))
    if render_args.error_vmax is not None:
        args.error_vmax = render_args.error_vmax

    device = torch.device(args.device)
    model = _new_surrogate(args, device)
    grid = _grid(args, device)
    true_field = _true_field(args, model)
    weights = [torch.as_tensor(w, dtype=torch.float32) for w in data["weights"]]
    if "parameter_information" in data.files:
        parameter_information = data["parameter_information"]
    else:
        parameter_information = np.full((len(weights), model.centers.shape[0]), float(args.prior_precision), dtype=np.float32)
    error_vmax = float(args.error_vmax) if args.error_vmax is not None else float(data["error_vmax"].item())
    return (
        model,
        data["true_states"],
        data["rbf_states"],
        data["actions"],
        data["update_errors"],
        weights,
        parameter_information,
        data["active_counts"],
        data["visited_fractions"],
        data["eig_scores"],
        [],
        grid,
        true_field,
        data["predictive_r2"],
        error_vmax,
    )


def render_frame(args: argparse.Namespace, output: Path) -> Path:
    _set_torch_threads(args)
    scene = _load_track_scene(args.track.resolve(), args)
    (
        model,
        true_states,
        rbf_states,
        actions,
        update_errors,
        weights,
        parameter_information,
        active_counts,
        visited_fractions,
        eig_scores,
        planned_paths,
        grid,
        true_field,
        predictive_r2,
        error_vmax,
    ) = scene
    _print_r2_summary(predictive_r2, float(args.min_r2), visited_fractions)
    information_vmin, information_vmax = _information_limits(parameter_information)
    idx = int(np.clip(args.step, 0, len(true_states) - 1))
    fig, axes = _setup_figure(args.dpi)
    _draw_frame(
        axes,
        args=args,
        model=model,
        grid=grid,
        true_field=true_field,
        weights=weights[idx],
        parameter_information=parameter_information,
        true_states=true_states,
        rbf_states=rbf_states,
        actions=actions,
        update_errors=update_errors,
        active_counts=active_counts,
        visited_fractions=visited_fractions,
        eig_scores=eig_scores,
        planned_paths=planned_paths,
        predictive_r2=predictive_r2,
        idx=idx,
        information_vmin=information_vmin,
        information_vmax=information_vmax,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def render_video(args: argparse.Namespace, output: Path) -> Path:
    _set_torch_threads(args)
    scene = _load_track_scene(args.track.resolve(), args)
    (
        model,
        true_states,
        rbf_states,
        actions,
        update_errors,
        weights,
        parameter_information,
        active_counts,
        visited_fractions,
        eig_scores,
        planned_paths,
        grid,
        true_field,
        predictive_r2,
        error_vmax,
    ) = scene
    _print_r2_summary(predictive_r2, float(args.min_r2), visited_fractions)
    information_vmin, information_vmax = _information_limits(parameter_information)
    indices = _frame_indices(len(true_states), args.stride)
    fig, axes = _setup_figure(args.dpi)
    output.parent.mkdir(parents=True, exist_ok=True)

    def frames():
        for n, idx in enumerate(indices, start=1):
            _draw_frame(
                axes,
                args=args,
                model=model,
                grid=grid,
                true_field=true_field,
                weights=weights[idx],
                parameter_information=parameter_information,
                true_states=true_states,
                rbf_states=rbf_states,
                actions=actions,
                update_errors=update_errors,
                active_counts=active_counts,
                visited_fractions=visited_fractions,
                eig_scores=eig_scores,
                planned_paths=planned_paths,
                predictive_r2=predictive_r2,
                idx=idx,
                information_vmin=information_vmin,
                information_vmax=information_vmax,
            )
            yield figure_to_rgb_array(fig)
            if n == 1 or n == len(indices) or n % 50 == 0:
                print(f"frame {n}/{len(indices)} step={idx}", flush=True)

    try:
        write_video_frames(frames(), output, fps=float(args.fps))
    finally:
        plt.close(fig)
    print(f"mp4={output}", flush=True)
    return output


def self_test() -> None:
    args = build_parser().parse_args(
        [
            "--mode",
            "frame",
            "--steps",
            "40",
            "--grid-points",
            "9",
            "--grid-size",
            "15",
            "--lengthscale",
            "0.45",
            "--active-radius",
            "2",
            "--planning-horizon",
            "4",
            "--num-action-sequences",
            "12",
            "--num-direction-actions",
            "4",
            "--icem-iterations",
            "2",
            "--icem-elites",
            "4",
            "--min-r2",
            "0.8",
            "--r2-interval",
            "10",
            "--dpi",
            "70",
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        args.track = Path(tmp) / "track.npz"
        save_track(args, args.track)
        scene = _load_track_scene(args.track, args)
    (
        model,
        true_states,
        rbf_states,
        actions,
        update_errors,
        weights,
        parameter_information,
        active_counts,
        visited_fractions,
        eig_scores,
        planned_paths,
        grid,
        true_field,
        predictive_r2,
        error_vmax,
    ) = scene
    assert true_states.shape == rbf_states.shape == (41, 2)
    assert actions.shape == (41, 2)
    assert int(active_counts.max()) <= model.max_active_centers
    assert float(torch.linalg.norm(weights[-1]).item()) > 0.0
    assert float(visited_fractions[-1]) > 0.0
    assert np.isfinite(eig_scores).all()
    assert np.isfinite(predictive_r2).all()
    assert np.isfinite(parameter_information).all()
    information_vmin, information_vmax = _information_limits(parameter_information)
    fig, axes = _setup_figure(args.dpi)
    _draw_frame(
        axes,
        args=args,
        model=model,
        grid=grid,
        true_field=true_field,
        weights=weights[20],
        parameter_information=parameter_information,
        true_states=true_states,
        rbf_states=rbf_states,
        actions=actions,
        update_errors=update_errors,
        active_counts=active_counts,
        visited_fractions=visited_fractions,
        eig_scores=eig_scores,
        planned_paths=planned_paths,
        predictive_r2=predictive_r2,
        idx=20,
        information_vmin=information_vmin,
        information_vmax=information_vmax,
    )
    frame = figure_to_rgb_array(fig)
    plt.close(fig)
    assert frame.ndim == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render online localized RBF vector-field learning.")
    parser.add_argument("--mode", choices=["frame", "video"], default="video")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--track", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--step", type=int, default=3200)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--state-low", type=float, default=-5.0)
    parser.add_argument("--state-high", type=float, default=5.0)
    parser.add_argument("--grid-points", type=int, default=81)
    parser.add_argument("--grid-size", type=int, default=55)
    parser.add_argument("--lengthscale", type=float, default=0.6)
    parser.add_argument("--active-radius", type=int, default=None)
    parser.add_argument("--planning-horizon", type=int, default=40)
    parser.add_argument("--coarse-dt-factor", type=int, default=10)
    parser.add_argument("--replan-interval", type=int, default=20)
    parser.add_argument("--adaptive-cadence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adaptive-planning-horizon", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adaptive-horizon-min", type=int, default=1)
    parser.add_argument("--adaptive-horizon-max", type=int, default=None)
    parser.add_argument("--adaptive-horizon-precision-threshold", type=float, default=1.0)
    parser.add_argument("--adaptive-replan-min-interval", type=int, default=20)
    parser.add_argument("--adaptive-replan-max-interval", type=int, default=None)
    parser.add_argument("--adaptive-replan-state-error-threshold", type=float, default=None)
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--num-action-sequences", type=int, default=96)
    parser.add_argument("--num-direction-actions", type=int, default=8)
    parser.add_argument("--icem-iterations", type=int, default=4)
    parser.add_argument("--icem-elites", type=int, default=16)
    parser.add_argument("--icem-alpha", type=float, default=0.2)
    parser.add_argument("--icem-init-std", type=float, default=1.0)
    parser.add_argument("--icem-reuse-frac", type=float, default=0.3)
    parser.add_argument("--icem-noise-beta", type=float, default=1.0)
    parser.add_argument("--icem-factor-decrease-num", type=float, default=1.25)
    parser.add_argument("--icem-use-mean-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--async-planning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--async-worker-iterations", type=int, default=None)
    parser.add_argument("--async-refine-iterations", type=int, default=0)
    parser.add_argument("--async-launch-lead-steps", type=int, default=None)
    parser.add_argument("--async-stale-tolerance", type=float, default=1.0)
    parser.add_argument("--async-model-stale-tolerance", type=float, default=None)
    parser.add_argument("--async-min-current-eig-frac", type=float, default=0.0)
    parser.add_argument("--async-live-refine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--async-live-refine-interval", type=int, default=5)
    parser.add_argument("--async-live-refine-parameters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--async-path-track-blend", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planning-seed", type=int, default=0)
    parser.add_argument("--eig-gamma", type=float, default=0.96)
    parser.add_argument("--eig-score-topk", type=int, default=0)
    parser.add_argument("--eig-score-backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--eig-float64", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eig-beta", type=float, default=40.0)
    parser.add_argument("--action-penalty", type=float, default=0.002)
    parser.add_argument("--prior-precision", type=float, default=1e-2)
    parser.add_argument("--r2-interval", type=int, default=20)
    parser.add_argument("--online-lr", type=float, default=0.9)
    parser.add_argument("--online-ridge", type=float, default=1e-3)
    parser.add_argument("--rbf-weight-init-std", type=float, default=1e-3)
    parser.add_argument("--initial-fit-rbf", action="store_true")
    parser.add_argument("--initial-fit-grid-points", type=int, default=61)
    parser.add_argument("--initial-fit-ridge", type=float, default=1e-4)
    parser.add_argument("--rbf-weight-diffusion", type=float, default=1e-5)
    parser.add_argument("--weight-smoothing-radius", type=int, default=0)
    parser.add_argument("--weight-smoothing-strength", type=float, default=0.0)
    parser.add_argument("--weight-smoothing-precision-threshold", type=float, default=1.0)
    parser.add_argument("--surrogate-drift-clip", type=float, default=25.0)
    parser.add_argument("--surrogate-state-correction", type=float, default=0.05)
    parser.add_argument("--min-r2", type=float, default=0.8)
    parser.add_argument("--require-min-r2", action="store_true")
    parser.add_argument("--dynamics-alpha", type=float, default=1.0)
    parser.add_argument("--dynamics-type", default=DEFAULT_DYNAMICS_TYPE)
    parser.add_argument("--dyn-params", default=DEFAULT_TBME_DYN_PARAMS)
    parser.add_argument("--state-noise", type=float, default=0.1)
    parser.add_argument("--boundary-type", default="radial")
    parser.add_argument("--boundary-radius", type=float, default=8.0)
    parser.add_argument("--boundary-barrier-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--boundary-barrier-width", type=float, default=0.5)
    parser.add_argument("--boundary-barrier-strength", type=float, default=5.0)
    parser.add_argument("--boundary-barrier-temperature", type=float, default=0.1)
    parser.add_argument("--action-max", type=float, default=1.0)
    parser.add_argument("--initial-state", type=float, nargs=2, default=(-2.0, 0.55))
    parser.add_argument("--error-vmax", type=float, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser
