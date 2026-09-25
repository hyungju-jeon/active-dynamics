from __future__ import annotations

"""
Helper functions for validating reconstruction results in Active Dynamics framework.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt

from actdyn.models.base import BaseModel
from actdyn.utils.rollout import RolloutBuffer, Rollout
from actdyn.utils.torch_utils import to_np
from actdyn.utils.plotting import create_subplot


def _trajectory_state_indices(
    state_indices: tuple[int, ...] | list[int] | None,
    *,
    state_dim: int,
    device,
) -> torch.Tensor | None:
    if state_indices is None:
        return None
    indices = tuple(int(index) for index in state_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("state_indices must contain unique coordinate indices")
    if min(indices) < 0 or max(indices) >= int(state_dim):
        raise ValueError(
            f"state_indices must lie in [0, {int(state_dim) - 1}], got {indices}"
        )
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def compute_model_r2(
    model: BaseModel = None,
    rollout: Union[Rollout, RolloutBuffer, Dict] = None,
    k_max: int = 10,
    n_idx: int = 200,
    n_samples: int = 100,
    fig_path: Optional[str] = None,
    show_fig: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute averaged k-step R^2 prediction scores at random starting indices
    """
    torch.manual_seed(0)
    dynamics = model.dynamics
    action_encoder = model.action_encoder
    decoder = model.decoder

    z = model.encoder(rollout["next_obs"], rollout["action"], n_samples=n_samples)[0]
    u = rollout["action"]
    y = rollout["next_obs"]

    B, T, D = y.shape
    y_mean = y.mean(dim=(1), keepdim=True)

    start_idx = torch.randint(0, T - k_max - 1, (n_idx,))
    # If model is provided, run full r2 computation

    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for t_idx in start_idx:
            y_true_list.append(y[:, t_idx : t_idx + k_max + 1, :])  # (B, k, D)
            z_pred_list = [z[..., t_idx : t_idx + 1, :]]
            for k in range(k_max):
                u_enc = action_encoder(
                    u[..., t_idx + 1 + k, :].unsqueeze(-2), z_pred_list[-1]
                )
                z_pred_list.append(
                    dynamics.sample_forward(
                        z_pred_list[-1], action=u_enc, k_step=1, return_traj=False
                    )[0]
                )

            z_pred = torch.cat(z_pred_list, dim=-2)  # (S, B, k+1, D)
            y_pred = (
                decoder(z_pred) if decoder is not None else z_pred
            )  # (S, B, k+1, D)
            y_pred = y_pred.mean(dim=0)  # (B, k+1, D)
            y_pred_list.append(y_pred)
            del z_pred, y_pred, z_pred_list, u_enc

    y_true = torch.stack(y_true_list, dim=0)  # (n_idx, B, k, D)
    y_pred = torch.stack(y_pred_list, dim=0)  # (n_idx, B, k, D)
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)  # (k, D)
    ss_tot = ((y_true - y_mean) ** 2).sum(dim=0)  # (k, D)

    r2_mat = 1 - ss_res / (ss_tot + 1e-6)  # (B, k, D)
    r2_mean_mat = to_np(torch.mean(r2_mat, dim=0))  # (k, D)
    r2_std_mat = to_np(torch.std(r2_mat, dim=0))  # (k, D)

    if fig_path is not None or show_fig:
        fig, axs = create_subplot(r2_mat)
        for i in range(r2_mean_mat.shape[1]):
            axs[i].plot(range(0, k_max + 1), r2_mean_mat[:, i])
            axs[i].fill_between(
                range(0, k_max + 1),
                r2_mean_mat[:, i] - r2_std_mat[:, i],
                r2_mean_mat[:, i] + r2_std_mat[:, i],
                alpha=0.3,
            )
            axs[i].set_title(f"Dimension {i + 1}")
            axs[i].set_xlabel("Prediction Steps")
            axs[i].set_ylabel(r"$R^2$")
            y_min = max(-3, min(-0.1, np.min(r2_mean_mat[:, i])))
            axs[i].set_ylim([y_min, 1.1])
            axs[i].grid(True)
        plt.tight_layout()
        if fig_path is not None:
            plt.savefig(fig_path)
        if show_fig:
            plt.show()
        else:
            plt.close(fig)

    # cleanup
    if "cuda" in str(z.device):
        del z, u, y, y_pred, y_true
        torch.cuda.empty_cache()

    return to_np(r2_mat), r2_mean_mat, r2_std_mat


def trajectory_r2_vectorfield(
    e_est: torch.Tensor,
    e_true: torch.Tensor,
    *,
    true_dynamics_type: str,
    true_full_params: np.ndarray,
    estimator_dynamics_type: str,
    estimator_full_params: np.ndarray,
    true_min_embedding_dim: int,
    estimator_min_embedding_dim: int,
    dt: float,
    dynamics_alpha: float,
    horizon: int,
    n_starts: int,
    rng: np.random.Generator,
    device,
    state_noise: float = 0.0,
    state_dim: int = 2,
    state_low: np.ndarray | list[float] | tuple[float, ...] | None = None,
    state_high: np.ndarray | list[float] | tuple[float, ...] | None = None,
    state_indices: tuple[int, ...] | list[int] | None = None,
    coordinate_balanced: bool = False,
) -> float:
    """Compare true and estimated latent trajectory rollouts.

    Starts have shape ``(n_starts, state_dim)`` and trajectories have shape
    ``(n_starts, horizon + 1, state_dim)``.  When ``state_noise > 0``, both rollouts
    use independent process-noise increments with the same ``sqrt(Q * dt)``
    scaling used by ``VectorFieldEnv.step``. ``state_indices`` restricts evaluation
    to named coordinates; ``coordinate_balanced`` averages one R2 per selected
    coordinate instead of pooling coordinates with different physical scales.
    """
    from actdyn.environment.vectorfield import pad_embedding_to_params, residual_torch

    low = -3.0 if state_low is None else np.asarray(state_low, dtype=np.float64)
    high = 3.0 if state_high is None else np.asarray(state_high, dtype=np.float64)
    starts = torch.as_tensor(
        rng.uniform(low=low, high=high, size=(n_starts, int(state_dim))),
        dtype=torch.float32,
        device=device,
    )
    e_true_batch = e_true.reshape(1, -1).repeat(n_starts, 1)
    e_est_batch = e_est.reshape(1, -1).repeat(n_starts, 1)
    noise_scale = float(max(0.0, state_noise) * dt) ** 0.5

    def _rollout(
        z0: torch.Tensor,
        embedding: torch.Tensor,
        *,
        dynamics_type: str,
        full_params: np.ndarray,
        min_embedding_dim: int,
    ) -> torch.Tensor:
        z = z0.clone()
        dyn_params = pad_embedding_to_params(
            embedding, full_params=full_params, min_embedding_dim=min_embedding_dim
        )
        traj = [z]
        for step in range(int(horizon)):
            drift = residual_torch(
                dynamics_type,
                z,
                dyn_params,
                dynamics_alpha=float(dynamics_alpha),
            )
            z = z + float(dt) * drift
            if noise_scale > 0.0:
                z = z + torch.as_tensor(
                    rng.normal(loc=0.0, scale=noise_scale, size=tuple(z.shape)),
                    dtype=z.dtype,
                    device=z.device,
                )
            traj.append(z)
        return torch.stack(traj, dim=1)

    with torch.no_grad():
        traj_true = _rollout(
            starts,
            e_true_batch,
            dynamics_type=true_dynamics_type,
            full_params=np.asarray(true_full_params, dtype=np.float32),
            min_embedding_dim=int(true_min_embedding_dim),
        )
        traj_est = _rollout(
            starts,
            e_est_batch,
            dynamics_type=estimator_dynamics_type,
            full_params=np.asarray(estimator_full_params, dtype=np.float32),
            min_embedding_dim=int(estimator_min_embedding_dim),
        )
        indices = _trajectory_state_indices(
            state_indices, state_dim=int(state_dim), device=device
        )
        if indices is not None:
            traj_true = torch.index_select(traj_true, dim=-1, index=indices)
            traj_est = torch.index_select(traj_est, dim=-1, index=indices)
        if coordinate_balanced:
            sse = torch.sum((traj_true - traj_est) ** 2, dim=(0, 1))
            true_mean = torch.mean(traj_true, dim=(0, 1), keepdim=True)
            sst = torch.sum((traj_true - true_mean) ** 2, dim=(0, 1))
            coordinate_r2 = torch.where(
                sst <= 1e-12, torch.zeros_like(sst), 1.0 - sse / sst
            )
            return float(torch.mean(coordinate_r2).item())
        y_true = traj_true.reshape(-1)
        y_est = traj_est.reshape(-1)
        sse = torch.sum((y_true - y_est) ** 2)
        sst = torch.sum((y_true - torch.mean(y_true)) ** 2)
        return 0.0 if float(sst.item()) <= 1e-12 else float((1.0 - sse / sst).item())


def trajectory_r2_vectorfield_many(
    e_estimates: torch.Tensor,
    e_true: torch.Tensor,
    *,
    true_dynamics_type: str,
    true_full_params: np.ndarray,
    estimator_dynamics_type: str,
    estimator_full_params: np.ndarray,
    true_min_embedding_dim: int,
    estimator_min_embedding_dim: int,
    dt: float,
    dynamics_alpha: float,
    horizon: int,
    n_starts: int,
    rng: np.random.Generator,
    device,
    state_noise: float = 0.0,
    state_dim: int = 2,
    state_low: np.ndarray | list[float] | tuple[float, ...] | None = None,
    state_high: np.ndarray | list[float] | tuple[float, ...] | None = None,
    state_indices: tuple[int, ...] | list[int] | None = None,
    coordinate_balanced: bool = False,
) -> np.ndarray:
    """Compute pooled trajectory R2 for many estimated embeddings.

    ``e_estimates`` has shape ``(M, E)``. Each output compares ``n_starts``
    stochastic rollouts of shape ``(horizon + 1, state_dim)``. By default the
    score pools all values. ``coordinate_balanced`` instead averages one R2 per
    coordinate after applying ``state_indices``.
    """
    from actdyn.environment.vectorfield import pad_embedding_to_params, residual_torch

    e_estimates = torch.as_tensor(e_estimates, dtype=torch.float32, device=device)
    e_true = torch.as_tensor(e_true, dtype=torch.float32, device=device)
    if e_estimates.ndim != 2:
        raise ValueError(
            f"e_estimates must have shape (M, E), got {tuple(e_estimates.shape)}."
        )
    n_eval = int(e_estimates.shape[0])
    starts_np: list[np.ndarray] = []
    true_noise_np: list[np.ndarray] = []
    est_noise_np: list[np.ndarray] = []
    noise_scale = float(max(0.0, state_noise) * dt) ** 0.5
    low = -3.0 if state_low is None else np.asarray(state_low, dtype=np.float64)
    high = 3.0 if state_high is None else np.asarray(state_high, dtype=np.float64)
    for _ in range(n_eval):
        starts_np.append(
            rng.uniform(low=low, high=high, size=(n_starts, int(state_dim)))
        )
        if noise_scale > 0.0 and int(horizon) > 0:
            true_noise_np.append(
                rng.normal(
                    loc=0.0,
                    scale=noise_scale,
                    size=(int(horizon), n_starts, int(state_dim)),
                )
            )
            est_noise_np.append(
                rng.normal(
                    loc=0.0,
                    scale=noise_scale,
                    size=(int(horizon), n_starts, int(state_dim)),
                )
            )
    starts = torch.as_tensor(np.stack(starts_np), dtype=torch.float32, device=device)
    true_noise = (
        torch.as_tensor(np.stack(true_noise_np), dtype=torch.float32, device=device)
        if true_noise_np
        else None
    )
    est_noise = (
        torch.as_tensor(np.stack(est_noise_np), dtype=torch.float32, device=device)
        if est_noise_np
        else None
    )
    e_true_batch = e_true.reshape(1, 1, -1).repeat(n_eval, n_starts, 1)
    e_est_batch = e_estimates.reshape(n_eval, 1, -1).repeat(1, n_starts, 1)

    def _rollout(
        z0: torch.Tensor,
        embedding: torch.Tensor,
        *,
        dynamics_type: str,
        full_params: np.ndarray,
        min_embedding_dim: int,
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        z = z0.clone()
        dyn_params = pad_embedding_to_params(
            embedding, full_params=full_params, min_embedding_dim=min_embedding_dim
        )
        traj = [z]
        for step in range(int(horizon)):
            drift = residual_torch(
                dynamics_type,
                z,
                dyn_params,
                dynamics_alpha=float(dynamics_alpha),
            )
            z = z + float(dt) * drift
            if noise is not None:
                z = z + noise[:, step]
            traj.append(z)
        return torch.stack(traj, dim=1)

    with torch.no_grad():
        traj_true = _rollout(
            starts,
            e_true_batch,
            dynamics_type=true_dynamics_type,
            full_params=np.asarray(true_full_params, dtype=np.float32),
            min_embedding_dim=int(true_min_embedding_dim),
            noise=true_noise,
        )
        traj_est = _rollout(
            starts,
            e_est_batch,
            dynamics_type=estimator_dynamics_type,
            full_params=np.asarray(estimator_full_params, dtype=np.float32),
            min_embedding_dim=int(estimator_min_embedding_dim),
            noise=est_noise,
        )
        indices = _trajectory_state_indices(
            state_indices, state_dim=int(state_dim), device=device
        )
        if indices is not None:
            traj_true = torch.index_select(traj_true, dim=-1, index=indices)
            traj_est = torch.index_select(traj_est, dim=-1, index=indices)
        if coordinate_balanced:
            sse = torch.sum((traj_true - traj_est) ** 2, dim=(1, 2))
            true_mean = torch.mean(traj_true, dim=(1, 2), keepdim=True)
            sst = torch.sum((traj_true - true_mean) ** 2, dim=(1, 2))
            coordinate_r2 = torch.where(
                sst <= 1e-12, torch.zeros_like(sst), 1.0 - sse / sst
            )
            r2 = torch.mean(coordinate_r2, dim=-1)
        else:
            sse = torch.sum((traj_true - traj_est) ** 2, dim=(1, 2, 3))
            true_mean = torch.mean(traj_true, dim=(1, 2, 3), keepdim=True)
            sst = torch.sum((traj_true - true_mean) ** 2, dim=(1, 2, 3))
            r2 = torch.where(sst <= 1e-12, torch.zeros_like(sst), 1.0 - sse / sst)
    return r2.cpu().numpy()


def basin_switch_cost_many(
    e_estimates: torch.Tensor,
    *,
    estimator_dynamics_type: str,
    estimator_full_params: np.ndarray,
    estimator_min_embedding_dim: int,
    e_true: torch.Tensor,
    true_dynamics_type: str,
    true_full_params: np.ndarray,
    true_min_embedding_dim: int,
    source_state: np.ndarray | list[float] | tuple[float, ...],
    target_state: np.ndarray | list[float] | tuple[float, ...],
    dt: float,
    dynamics_alpha: float,
    horizon: int,
    action_max: float,
    terminal_weight: float = 10.0,
    iterations: int = 200,
    learning_rate: float = 0.05,
    device="cpu",
) -> dict[str, np.ndarray]:
    """Score how cheaply each estimated model can switch the true system between basins.

    For each embedding estimate in ``e_estimates`` (shape ``(M, E)``) an
    open-loop input sequence ``u_{0:H-1}`` with ``|u_t| <= action_max`` is planned
    on the estimated dynamics ``f_hat`` by minimizing the control cost

        J(u) = dt * sum_t ||u_t||^2 + terminal_weight * ||z_H - z_target||^2,
        z_{t+1} = z_t + dt * (f_hat(z_t) + u_t),   z_0 = z_source,

    where ``z_H`` is the state after ``horizon`` driven steps followed by
    ``horizon`` passive steps on ``f_hat``, so the input only has to carry the
    state across the separatrix and the estimated dynamics finish the switch.
    The bound is enforced by ``u_t = action_max * tanh(v_t)`` with Adam on ``v``
    from ``v = 0``. The planned sequence is then applied open loop to the true
    dynamics from ``z_source`` with the same driven-plus-passive schedule. The
    switch succeeds when the settled true state is closer to ``z_target`` than
    to ``z_source``. The computation is deterministic given its arguments (no
    noise, zero initialization).

    Returns arrays of shape ``(M,)``: ``energy`` (``dt * sum ||u_t||^2`` of the
    applied input), ``success`` (0/1), and ``terminal_distance`` (distance of the
    settled true state to ``z_target``).
    """
    from actdyn.environment.vectorfield import pad_embedding_to_params, residual_torch

    e_estimates = torch.as_tensor(e_estimates, dtype=torch.float32, device=device)
    if e_estimates.ndim != 2:
        raise ValueError(
            f"e_estimates must have shape (M, E), got {tuple(e_estimates.shape)}."
        )
    n_eval = int(e_estimates.shape[0])
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    source = torch.as_tensor(np.asarray(source_state, dtype=np.float32), device=device)
    target = torch.as_tensor(np.asarray(target_state, dtype=np.float32), device=device)
    state_dim = int(source.shape[-1])
    est_params = pad_embedding_to_params(
        e_estimates,
        full_params=np.asarray(estimator_full_params, dtype=np.float32),
        min_embedding_dim=int(estimator_min_embedding_dim),
    )
    true_params = pad_embedding_to_params(
        torch.as_tensor(e_true, dtype=torch.float32, device=device).reshape(1, -1),
        full_params=np.asarray(true_full_params, dtype=np.float32),
        min_embedding_dim=int(true_min_embedding_dim),
    ).expand(n_eval, -1)

    def _rollout(
        inputs: torch.Tensor, *, dynamics_type: str, dyn_params: torch.Tensor, steps: int
    ) -> torch.Tensor:
        z = source.reshape(1, state_dim).expand(n_eval, state_dim)
        for step in range(int(steps)):
            drift = residual_torch(
                dynamics_type, z, dyn_params, dynamics_alpha=float(dynamics_alpha)
            )
            u = inputs[:, step] if step < inputs.shape[1] else torch.zeros_like(z)
            z = z + float(dt) * (drift + u)
        return z

    v = torch.zeros((n_eval, horizon, state_dim), dtype=torch.float32, device=device)
    v.requires_grad_(True)
    optimizer = torch.optim.Adam([v], lr=float(learning_rate))
    with torch.enable_grad():
        for _ in range(int(iterations)):
            optimizer.zero_grad()
            u = float(action_max) * torch.tanh(v)
            z_end = _rollout(
                u,
                dynamics_type=estimator_dynamics_type,
                dyn_params=est_params,
                steps=2 * horizon,
            )
            energy = float(dt) * torch.sum(u * u, dim=(1, 2))
            loss = torch.sum(
                energy + float(terminal_weight) * torch.sum((z_end - target) ** 2, dim=-1)
            )
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        u = float(action_max) * torch.tanh(v)
        energy = float(dt) * torch.sum(u * u, dim=(1, 2))
        z_settled = _rollout(
            u, dynamics_type=true_dynamics_type, dyn_params=true_params, steps=2 * horizon
        )
        dist_target = torch.linalg.norm(z_settled - target, dim=-1)
        dist_source = torch.linalg.norm(z_settled - source, dim=-1)
        success = (dist_target < dist_source).to(torch.float32)
    return {
        "energy": energy.detach().cpu().numpy().astype(np.float64),
        "success": success.detach().cpu().numpy().astype(np.float64),
        "terminal_distance": dist_target.detach().cpu().numpy().astype(np.float64),
    }


def rollout_r2_on_trajectories(
    e_estimates: torch.Tensor,
    *,
    dynamics_type: str,
    full_params: np.ndarray,
    min_embedding_dim: int,
    states: np.ndarray | torch.Tensor,
    inputs: np.ndarray | torch.Tensor,
    dt: float,
    horizon: int,
    stride: int,
    dynamics_alpha: float = 1.0,
    chunk_size: int = 64,
    device="cpu",
) -> np.ndarray:
    """Pooled multi-step rollout R2 of estimated models on recorded trajectories.

    Args:
        e_estimates: Shape (M, E), one embedding estimate per row.
        states: Shape (K, T+1, d), recorded latent trajectories; ``states[:, t+1]``
            is the state after the bin driven by ``inputs[:, t]``.
        inputs: Shape (K, T, d), total drive applied during each bin.
        horizon, stride: Rollout length and spacing of rollout starts, in bins.

    For each trajectory k and start s in {0, stride, ...} with s + horizon <= T,
    the model rolls ``z_{j+1} = z_j + dt (f(z_j; theta) + u_j)`` from the recorded
    ``z_s`` with the recorded inputs, and ``z_{s+1:s+horizon}`` is compared with the
    recording. ``R2 = 1 - SSE / SST`` with SST about the mean of all compared
    targets, pooled over coordinates, starts, and trajectories.

    For input-dependent fields the step is ``z_j + dt f(z_j, u_j; theta)``
    (:func:`actdyn.environment.vectorfield.drift_torch`); for the others the two
    forms are the same arithmetic.

    Returns:
        Shape (M,) float64 array of R2 values. Deterministic given its arguments.
    """
    from actdyn.environment.vectorfield import drift_torch, pad_embedding_to_params

    z = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
    u = torch.as_tensor(np.asarray(inputs), dtype=torch.float32, device=device)
    if z.ndim != 3 or u.ndim != 3 or z.shape[1] != u.shape[1] + 1 or z.shape[0] != u.shape[0]:
        raise ValueError(f"states {tuple(z.shape)} must be (K, T+1, d) for inputs {tuple(u.shape)}.")
    horizon, stride = int(horizon), int(stride)
    n_steps = int(u.shape[1])
    if horizon <= 0 or stride <= 0 or horizon > n_steps:
        raise ValueError(f"need 0 < horizon <= T={n_steps} and stride > 0.")
    starts = list(range(0, n_steps - horizon + 1, stride))
    z0 = torch.cat([z[:, s] for s in starts], dim=0)                               # (S, d)
    u_seg = torch.cat([u[:, s : s + horizon] for s in starts], dim=0)              # (S, H, d)
    target = torch.cat([z[:, s + 1 : s + horizon + 1] for s in starts], dim=0)     # (S, H, d)
    sst = torch.sum((target - target.mean()) ** 2)
    e_all = torch.as_tensor(e_estimates, dtype=torch.float32, device=device)
    if e_all.ndim != 2:
        raise ValueError(f"e_estimates must have shape (M, E), got {tuple(e_all.shape)}.")
    full = np.asarray(full_params, dtype=np.float32)
    out = np.empty(int(e_all.shape[0]), dtype=np.float64)
    n_seg = int(z0.shape[0])
    with torch.no_grad():
        for lo in range(0, int(e_all.shape[0]), int(chunk_size)):
            e = e_all[lo : lo + int(chunk_size)]
            m = int(e.shape[0])
            params = pad_embedding_to_params(e, full_params=full, min_embedding_dim=int(min_embedding_dim))
            params = params.repeat_interleave(n_seg, dim=0)                         # (m*S, P)
            zk = z0.repeat(m, 1)
            sse = torch.zeros(m, dtype=torch.float64, device=device)
            for j in range(horizon):
                drift = drift_torch(dynamics_type, zk, params, u_seg[:, j].repeat(m, 1), dynamics_alpha=float(dynamics_alpha))
                zk = torch.nan_to_num(zk + float(dt) * drift, nan=0.0, posinf=1e6, neginf=-1e6)
                err = (zk - target[:, j].repeat(m, 1)).reshape(m, n_seg, -1)
                sse += torch.sum(err.double() ** 2, dim=(1, 2))
            out[lo : lo + m] = (1.0 - sse / sst.double()).cpu().numpy()
    return out



def project_to_budget(
    u: torch.Tensor, budget: torch.Tensor, *, dt: float, action_max: float
) -> torch.Tensor:
    """Project input sequences (..., H, d) onto |u_t| <= action_max and dt sum_t |u_t|^2 <= budget.

    Each step is first clipped radially to ``action_max``; a sequence whose energy
    still exceeds its budget (``budget`` broadcasts over the leading dims) is then
    scaled by ``sqrt(budget / E)``. Scaling down keeps the step bound, so the result
    satisfies both constraints.
    """
    norms = torch.linalg.norm(u, dim=-1, keepdim=True)
    u = u * torch.clamp(float(action_max) / norms.clamp_min(1e-8), max=1.0)
    energy = float(dt) * torch.sum(u * u, dim=(-2, -1))
    scale = torch.sqrt(budget / torch.maximum(energy, budget).clamp_min(1e-12))
    return u * scale.unsqueeze(-1).unsqueeze(-1)


def update_decision_statistic(
    best: torch.Tensor, other_decided: torch.Tensor, lead: torch.Tensor, *, objective: str, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """One time step of the first-passage statistic of :func:`icem_reversal_plan`.

    ``best`` is the largest target lead so far (start at ``-inf``); with
    ``objective="first_decision"`` leads after the other pool first led by more
    than ``threshold`` (``other_decided``) no longer count. All shapes match.
    """
    if objective == "first_decision":
        other_decided = other_decided | (lead < -float(threshold))
        lead = torch.where(other_decided, torch.full_like(lead, -float("inf")), lead)
    elif objective != "reach":
        raise ValueError(f"unknown first-passage objective {objective!r}")
    return torch.maximum(best, lead), other_decided


def icem_reversal_plan(
    transition,
    z0: np.ndarray | torch.Tensor,
    drive: np.ndarray | torch.Tensor,
    *,
    dt: float,
    window: int,
    coarse_factor: int,
    budget: float,
    action_max: float,
    noise_var: float,
    objective: str,
    success_gap: float,
    mc_samples: int = 16,
    mc_seed: int = 0,
    action_seed: int = 0,
    num_samples: int = 64,
    num_elites: int = 8,
    num_iterations: int = 8,
    init_std: float = 0.5,
    alpha: float = 0.1,
    noise_beta: float = 1.0,
    factor_decrease: float = 1.25,
    frac_elites_reused: float = 0.3,
    sharpness: float = 4.0,
    seed_candidates: np.ndarray | torch.Tensor | None = None,
    device="cpu",
) -> dict[str, np.ndarray]:
    """Plan an input sequence that makes pool 2 win, with iCEM.

    Args:
        transition: Learner transition, a map of float32 tensors (N, d), (N, d_u)
            to (N, d); the pools are the state coordinates ``z_1, z_2``.
        z0: Shape (d,), the state the plan starts from.
        drive: Shape (T, d_u), known exogenous input over the whole rollout, T >= window.
        window, coarse_factor: The control acts for ``window`` bins as
            ``window / coarse_factor`` held values (coarse resolution, as in the
            PALDI planner's ``coarse_action_mapping: hold``).
        budget: Energy bound ``dt sum_t |u_t|^2`` over the fine window.

    The search follows ``actdyn.policy.mpc.MpcICem``: colored-noise samples
    (exponent ``noise_beta``) around a mean that starts at zero with spread
    ``init_std * action_max``; the sample count shrinks by ``factor_decrease`` per
    iteration (at least ``2 * num_elites``); a fraction of the previous elites is
    kept; the mean is added as a candidate in the last iteration; mean and spread
    move to the elite statistics with smoothing ``alpha``. Every candidate is
    projected by :func:`project_to_budget` before scoring.

    The score of a candidate is the model-predicted soft success probability
    ``mean_s sigmoid(sharpness (G - success_gap))`` over ``mc_samples`` rollouts
    ``z_{k+1} = transition(z_k, drive_k + u_k) + sqrt(noise_var dt) xi`` with noise
    shared by all candidates (common random numbers from ``mc_seed``). With the
    pool-2 lead ``gap = z_2 - z_1``, the statistic ``G`` is set by ``objective``:

    * ``"reach"``: ``max_t gap_t``, so success means pool 2 leads by more than
      ``success_gap`` at some time;
    * ``"first_decision"``: ``max_t gap_t`` over the times before pool 1 first
      leads by more than ``success_gap``, so success means pool 2 makes the
      first decision.

    ``seed_candidates`` (shape (C, window / coarse_factor, d_u), coarse held values)
    join the first iteration's samples, after the budget projection, so the plan
    scores at least as well as the best of them under the model. Colored noise
    uses ``numpy.random.default_rng(action_seed)``, so the plan is deterministic.

    Returns:
        ``u`` (window, d_u) planned fine inputs, ``soft_probability`` and
        ``probability`` (the hard success fraction) of the plan, and its ``energy``.
    """
    import colorednoise

    if objective not in ("reach", "first_decision"):
        raise ValueError(f"unknown objective {objective!r}")
    w = torch.as_tensor(np.asarray(drive), dtype=torch.float32, device=device)
    z_start = torch.as_tensor(np.asarray(z0), dtype=torch.float32, device=device).reshape(-1)
    d, d_u, n_t = int(z_start.shape[0]), int(w.shape[-1]), int(w.shape[0])
    m = 1  # one model; the leading axis keeps the batch layout of the search
    window, factor = int(window), int(coarse_factor)
    if window % factor != 0 or n_t < window:
        raise ValueError("window must be a multiple of coarse_factor and covered by drive.")
    hc = window // factor
    b = torch.tensor(float(budget), device=device)
    coarse_dt = float(dt) * factor
    gen = torch.Generator(device="cpu").manual_seed(int(mc_seed))
    xi = torch.randn(n_t, int(mc_samples), d, generator=gen).to(device) * float(np.sqrt(max(noise_var, 0.0) * dt))
    rng = np.random.default_rng(int(action_seed))
    s_mc = int(mc_samples)

    def score(u_coarse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """u_coarse (m, K, hc, d_u) -> soft and hard reversal probability, each (m, K)."""
        k = int(u_coarse.shape[1])
        u_fine = u_coarse.repeat_interleave(factor, dim=2)                       # (m, K, window, d_u)
        zk = z_start.expand(m, k, s_mc, d).clone()
        best = torch.full((m, k, s_mc), -float("inf"), device=device)
        other_decided = torch.zeros((m, k, s_mc), dtype=torch.bool, device=device)
        for t in range(n_t):
            u_t = w[t].expand(m, k, s_mc, d_u)
            if t < window:
                u_t = u_t + u_fine[:, :, t].unsqueeze(2)
            zk = transition(zk.reshape(-1, d), u_t.reshape(-1, d_u)).view(m, k, s_mc, d) + xi[t]
            best, other_decided = update_decision_statistic(
                best, other_decided, zk[..., 1] - zk[..., 0], objective=objective, threshold=float(success_gap)
            )
        soft = torch.sigmoid(float(sharpness) * (best - float(success_gap))).mean(dim=-1)
        hard = (best > float(success_gap)).float().mean(dim=-1)
        return soft, hard

    mean = torch.zeros(m, hc, d_u, device=device)
    std = torch.full((m, hc, d_u), float(init_std) * float(action_max), device=device)
    elites, elite_scores = None, None
    best_u = torch.zeros(m, hc, d_u, device=device)
    best_score = torch.full((m,), -float("inf"), device=device)
    n_samples = int(num_samples)
    with torch.no_grad():
        for it in range(int(num_iterations)):
            if it > 0:
                n_samples = max(2 * int(num_elites), int(n_samples / float(factor_decrease)))
            if float(noise_beta) > 0 and hc > 1:
                noise = colorednoise.powerlaw_psd_gaussian(float(noise_beta), size=(m, n_samples, d_u, hc), random_state=rng)
                noise = torch.as_tensor(noise, dtype=torch.float32, device=device).transpose(-1, -2)
            else:
                noise = torch.as_tensor(rng.standard_normal((m, n_samples, hc, d_u)), dtype=torch.float32, device=device)
            cand = mean.unsqueeze(1) + std.unsqueeze(1) * noise
            if it == int(num_iterations) - 1:
                cand[:, 0] = mean
            if it == 0 and seed_candidates is not None:
                seeds_c = torch.as_tensor(np.asarray(seed_candidates), dtype=torch.float32, device=device)
                cand = torch.cat([cand, seeds_c.unsqueeze(0).expand(m, -1, -1, -1)], dim=1)
            cand = project_to_budget(cand, b, dt=coarse_dt, action_max=float(action_max))
            soft, _hard = score(cand)
            if it > 0 and elites is not None:
                n_keep = int(elites.shape[1] * float(frac_elites_reused))
                if n_keep > 0:
                    cand = torch.cat([cand, elites[:, :n_keep]], dim=1)
                    soft = torch.cat([soft, elite_scores[:, :n_keep]], dim=1)
            top = torch.topk(soft, int(num_elites), dim=1)
            idx = top.indices
            elites = torch.gather(cand, 1, idx.view(m, -1, 1, 1).expand(-1, -1, hc, d_u))
            elite_scores = top.values
            improved = elite_scores[:, 0] > best_score
            best_score = torch.where(improved, elite_scores[:, 0], best_score)
            best_u = torch.where(improved.view(m, 1, 1), elites[:, 0], best_u)
            mean = (1.0 - float(alpha)) * elites.mean(dim=1) + float(alpha) * mean
            std = (1.0 - float(alpha)) * elites.std(dim=1, unbiased=False) + float(alpha) * std
        soft, hard = score(best_u.unsqueeze(1))
    u_fine = best_u.repeat_interleave(factor, dim=1)[0]
    return {
        "u": u_fine.cpu().numpy().astype(np.float64),
        "soft_probability": float(soft[0, 0]),
        "probability": float(hard[0, 0]),
        "energy": float(float(dt) * torch.sum(u_fine * u_fine)),
    }


def poisson_ekf_update(
    mean: np.ndarray,
    cov: np.ndarray,
    counts: np.ndarray,
    drive: np.ndarray,
    *,
    transition,
    readout_weight: np.ndarray,
    readout_bias: np.ndarray,
    dt: float,
    process_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One step of the agents' state filter for a transition ``z' = transition(z, u)``.

    The state part of ``FilteringEmbedding.update_posterior_embedding``:
    prediction ``m- = transition(m, u)`` and
    ``P- = F P F^T + process_var dt I + 1e-6 I`` with ``F = d transition / dz (m)``,
    then the information-form update for ``counts ~ Poisson(lambda)``,
    ``lambda = dt exp(C m- + b)``, linearized at ``m-``:
    ``P+ = (P-^{-1} + C^T diag(lambda) C)^{-1}``, ``m+ = m- + P+ C^T (y - lambda)``.

    Args:
        mean, cov: Shapes (d,) and (d, d), belief before the bin.
        counts: Shape (n,), spike counts of the bin.
        drive: Shape (d_u,), total input applied during the bin.
        transition: Maps float64 tensors (d,), (d_u,) to (d,).
        readout_weight, readout_bias: ``C`` (n, d) and ``b`` (n,).
        process_var: ``softplus(logvar)`` of the learner's dynamics (per unit time).

    Returns:
        ``(mean, cov)`` after the bin, float64 arrays.
    """
    m = torch.as_tensor(np.asarray(mean), dtype=torch.float64).reshape(-1)
    d = int(m.shape[0])
    u = torch.as_tensor(np.asarray(drive), dtype=torch.float64).reshape(-1)
    eye = torch.eye(d, dtype=torch.float64)
    F = torch.autograd.functional.jacobian(lambda z: transition(z, u), m)
    m_pred = transition(m, u).detach()
    P = torch.as_tensor(np.asarray(cov), dtype=torch.float64)
    P_pred = F @ P @ F.T + (float(process_var) * float(dt) + 1e-6) * eye
    P_pred = 0.5 * (P_pred + P_pred.T)
    C = torch.as_tensor(np.asarray(readout_weight), dtype=torch.float64)
    b = torch.as_tensor(np.asarray(readout_bias), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(counts), dtype=torch.float64)
    rate = (float(dt) * torch.exp(C @ m_pred + b)).clamp_min(1e-6)
    info = C.T @ (rate.unsqueeze(-1) * C)
    P_post = torch.linalg.inv(torch.linalg.inv(P_pred) + info)
    P_post = 0.5 * (P_post + P_post.T)
    m_post = m_pred + P_post @ (C.T @ (y - rate))
    return m_post.numpy(), P_post.numpy()
