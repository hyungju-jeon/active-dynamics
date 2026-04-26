from __future__ import annotations

"""
Helper functions for validating reconstruction results in Active Dynamics framework.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from einops import repeat, rearrange

from actdyn.models.base import BaseModel
from actdyn.utils.rollout import RolloutBuffer, Rollout
from actdyn.utils.helper import to_np
from actdyn.utils.visualize import create_subplot


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
                u_enc = action_encoder(u[..., t_idx + 1 + k, :].unsqueeze(-2), z_pred_list[-1])
                z_pred_list.append(
                    dynamics.sample_forward(
                        z_pred_list[-1], action=u_enc, k_step=1, return_traj=False
                    )[0]
                )

            z_pred = torch.cat(z_pred_list, dim=-2)  # (S, B, k+1, D)
            y_pred = decoder(z_pred) if decoder is not None else z_pred  # (S, B, k+1, D)
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
            axs[i].set_title(f"Dimension {i+1}")
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


# TODO : Do not use
def compute_kstep_r2(
    model: BaseModel = None,
    rollout: Union[Rollout, RolloutBuffer, Dict] = None,
    z=None,
    u=None,
    y=None,
    action_encoder=None,
    dynamics=None,
    k_max: int = 10,
    n_idx: int = 100,
    n_samples: int = 32,
    fig_path: Optional[str] = None,
    show_fig: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute k-step R^2 prediction scores

    Args:
        model: SeqVae model with dynamics and action_encoder
        rollout: Rollout data containing 'latent_state', 'action', and 'obs'
        k_max: Maximum prediction steps
        n_idx: Number of random starting indices to evaluate
        n_samples: Number of samples for stochastic prediction
        pred_window: Length of prediction window
        fig_path: Optional path to save R^2 plot
        show_fig: Whether to display the figure
    """
    # If model is provided, run full r2 computation
    if model is not None:
        dynamics = model.dynamics
        action_encoder = model.action_encoder
        decoder = model.decoder

        z = (
            model.encoder(rollout["next_obs"], rollout["action"], n_samples=n_samples)[0]
            if z is None
            else z
        )
        u = rollout["action"][:, 1:] if u is None else u
        u_enc = action_encoder(u) if action_encoder is not None else u
        y = rollout["next_obs"] if y is None else y

    with torch.no_grad():
        z_pred_list = dynamics.sample_forward(z, action=u_enc, k_step=k_max, return_traj=True)[0]
        z_pred = [z_pred_list[k][..., : -k_max + k, :] for k in range(0, k_max)]
        z_pred += [z_pred_list[-1][..., :, :]]  # (k, S, B, T-k, D)
        del z_pred_list
        z_pred = torch.stack(z_pred, dim=0)  # (k, S, B, T-k, D)
        y_pred = decoder(z_pred) if decoder is not None else z_pred  # (k, S, B, T-k, D)
        del z_pred
        y_pred = y_pred.mean(dim=1)  # (k, B, T-k, D)

    r2_mat, r2_mean_mat, r2_std_mat = _compute_average_r2(y_pred, y, k_max, n_idx)

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
            axs[i].set_title(f"Dimension {i+1}")
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
        torch.cuda.empty_cache()
        del z, u, y, y_pred

    return r2_mat, r2_mean_mat, r2_std_mat


# TODO : Do not use
def _compute_average_r2(
    y_pred,
    y,
    k_max: int = 10,
    n_idx: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ """
    torch.manual_seed(0)
    B, T, D = y.shape
    y_mean = y.mean(dim=(1), keepdim=True)

    start_idx = torch.randint(0, T - k_max - 1, (n_idx,))
    r2_list = []

    y_true_list = []
    y_pred_list = []
    for t_idx in start_idx:
        y_true_i = y[:, t_idx : t_idx + k_max + 1, :]  # (B, k, D)
        y_pred_i = rearrange(y_pred[..., t_idx, :], "k B D -> B k D")  # (B, k, D)
        y_true_list.append(y_true_i)
        y_pred_list.append(y_pred_i)

    y_true = torch.stack(y_true_list, dim=0)  # (n_idx, B, k, D)
    y_pred = torch.stack(y_pred_list, dim=0)  # (n_idx, B, k, D)
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)  # (k, D)
    ss_tot = ((y_true - y_mean) ** 2).sum(dim=0)  # (k, D)

    r2 = 1 - ss_res / (ss_tot + 1e-6)  # (B, k, D)
    r2_mean = torch.mean(r2, dim=0)  # (k, D)
    r2_std = torch.std(r2, dim=0)  # (k, D)

    return (
        r2.cpu().numpy(),
        r2_mean.cpu().numpy(),
        r2_std.cpu().numpy(),
    )


# def validate_elbo(experiment: Experiment, rb: Rollout | RolloutBuffer, writer: SummaryWriter):
#     B, T, D = rb["obs"].shape
#     loss, log_like, kl_d = experiment.agent.model_env.model.compute_elbo(
#         y=rb["next_obs"],
#         u=rb["action"],
#         idx=None,
#         n_samples=16,
#         beta=1,
#         k_steps=1,
#     )
#     writer.add_scalar("validation/ELBO", -loss.item() / T, 0)
#     writer.add_scalar("validation/log_like", log_like.item() / T, 0)
#     writer.add_scalar("validation/kl_d", kl_d.item() / T, 0)


# def validate_kstep_r2(experiment: Experiment, rb: Rollout | RolloutBuffer, writer: SummaryWriter):
#     r2_fig_path = Path(experiment.results_dir) / "validation_kstep.png"
#     r2m, _ = compute_kstep_r2(
#         model=experiment.agent.model_env.model,
#         rollout=rb,
#         k_max=10,
#         n_samples=50,
#         fig_path=r2_fig_path,
#     )
#     for i in range(r2m.shape[1]):
#         for k in range(r2m.shape[0]):
#             writer.add_scalar(f"validation/kstep/r2_mean_{i}", r2m[k, i], k + 1)



def _pad_embedding_to_params(
    embedding: torch.Tensor,
    *,
    full_params: np.ndarray,
    min_embedding_dim: int,
) -> torch.Tensor:
    if embedding.shape[-1] < int(min_embedding_dim):
        raise ValueError(
            f"Embedding must have at least {min_embedding_dim} coordinates, got shape {tuple(embedding.shape)}."
        )
    full = torch.as_tensor(full_params, dtype=embedding.dtype, device=embedding.device)
    if embedding.shape[-1] >= full.shape[0]:
        return embedding[..., : full.shape[0]]
    tail = full[embedding.shape[-1]:]
    if embedding.ndim == 1:
        return torch.cat((embedding, tail), dim=0)
    tail = tail.reshape(*([1] * (embedding.ndim - 1)), -1).expand(*embedding.shape[:-1], -1)
    return torch.cat((embedding, tail), dim=-1)


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
) -> float:
    from actdyn.environment.vectorfield import residual_torch

    starts = torch.as_tensor(
        rng.uniform(low=-3.0, high=3.0, size=(n_starts, 2)),
        dtype=torch.float32,
        device=device,
    )
    e_true_batch = e_true.reshape(1, -1).repeat(n_starts, 1)
    e_est_batch = e_est.reshape(1, -1).repeat(n_starts, 1)

    def _rollout(
        z0: torch.Tensor,
        embedding: torch.Tensor,
        *,
        dynamics_type: str,
        full_params: np.ndarray,
        min_embedding_dim: int,
    ) -> torch.Tensor:
        z = z0.clone()
        dyn_params = _pad_embedding_to_params(
            embedding, full_params=full_params, min_embedding_dim=min_embedding_dim
        )
        traj = [z]
        for _ in range(int(horizon)):
            drift = residual_torch(
                dynamics_type,
                z,
                dyn_params,
                dynamics_alpha=float(dynamics_alpha),
            )
            z = z + float(dt) * drift
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
        y_true = traj_true.reshape(-1)
        y_est = traj_est.reshape(-1)
        sse = torch.sum((y_true - y_est) ** 2)
        sst = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 0.0 if float(sst.item()) <= 1e-12 else float((1.0 - sse / sst).item())

