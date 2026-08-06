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
