from __future__ import annotations

"""
Tensor, PyTorch, and numerical utility functions
"""

import torch
import numpy as np
from typing import Callable, Dict, Sequence

eps = 1e-12


Belief = Dict[str, torch.Tensor]
Transition = Dict[str, torch.Tensor]


# -------------------------------------------------------------
# General Helpers
# -------------------------------------------------------------
def format_list(x):
    if isinstance(x, torch.Tensor):
        x = to_np(x.reshape(-1)).tolist()
        fstr = ", ".join([f"{val:.3f}" for val in x])
        return "(" + fstr + ")"
    else:
        return f"{x:.3f}"


def make_uniform_sampler(low: list[float] | float, high: list[float] | float, dim: int):
    if isinstance(low, (int, float)):
        low = [float(low)] * dim
    if isinstance(high, (int, float)):
        high = [float(high)] * dim
    if len(low) != dim or len(high) != dim:
        raise ValueError(f"low/high length must match dim={dim}, got {len(low)} and {len(high)}")

    low_t = torch.tensor(low, dtype=torch.float32).reshape(1, dim)
    span_t = torch.tensor(high, dtype=torch.float32).reshape(1, dim) - low_t

    def _sampler(N: int):
        return low_t + span_t * torch.rand(N, dim)

    return _sampler


def jacobian_wrt_param(fn: Callable, inputs: Sequence[torch.Tensor], argnum: int) -> torch.Tensor:
    """Compute Jacobian of ``fn(*inputs)`` with respect to a selected argument."""
    has_time = inputs[0].ndim == 3
    if has_time:
        batch, T, in_dim = inputs[0].shape
    else:
        batch, in_dim = inputs[0].shape
        T = 1

    inputs_list = [
        t.reshape(batch * T, -1).requires_grad_(True) if not t.requires_grad else t for t in inputs
    ]

    out = fn(*inputs_list)
    if out.ndim == 1:
        out = out.unsqueeze(0)
    _, out_dim = out.shape

    in_dim = inputs_list[argnum].shape[-1]
    J = torch.zeros(batch, T, out_dim, in_dim, device=out.device, dtype=out.dtype)

    for i in range(out_dim):
        grad_outputs = torch.zeros_like(out)
        grad_outputs[:, i] = 1.0
        (gi,) = torch.autograd.grad(
            out,
            inputs_list[argnum],
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
        )
        J[..., i, :] = gi.reshape(batch, T, in_dim)

    return J.reshape(batch, T, out_dim, in_dim)


# -------------------------------------------------------------
# Torch Helpers
# -------------------------------------------------------------
def to_np(x: torch.Tensor) -> np.ndarray:
    """Converts a PyTorch tensor to a NumPy array."""
    return x.cpu().detach().numpy()


def safe_cholesky(M, jitter=1e-6, max_tries=7, growth=10.0):
    M = symmetrize(torch.nan_to_num(M, nan=0.0, posinf=1e6, neginf=-1e6))
    I = torch.eye(M.size(-1), device=M.device).expand_as(M)
    j = 0.0
    for _ in range(max_tries):
        chol, info = torch.linalg.cholesky_ex(M + j * I)
        if torch.count_nonzero(info).item() == 0:
            return chol
        j = jitter if j == 0.0 else j * growth

    # Final fallback: iterative SPD projection with escalating eigenvalue floor.
    floor = max(float(j), 1e-4)
    M_work = M
    for _ in range(4):
        evals, evecs = torch.linalg.eigh((M_work + floor * I).double())
        evals = evals.clamp_min(floor)
        M_spd = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
        M_spd = symmetrize(torch.nan_to_num(M_spd, nan=0.0, posinf=1e6, neginf=-1e6)).to(M.dtype)
        chol, info = torch.linalg.cholesky_ex(M_spd + floor * I)
        if torch.count_nonzero(info).item() == 0:
            return chol
        floor *= growth
        M_work = M_spd

    try:
        return torch.linalg.cholesky(M_spd + floor * I)
    except RuntimeError:
        # Conservative final fallback: keep only positive diagonal mass.
        diag = torch.diagonal(M, dim1=-2, dim2=-1)
        diag = torch.nan_to_num(diag, nan=floor, posinf=1e6, neginf=floor).clamp_min(floor)
        return torch.linalg.cholesky(torch.diag_embed(diag + floor))


def symmetrize(M):
    return 0.5 * (M + M.transpose(-1, -2))


def attenuated_state_information(prior_cov: torch.Tensor, state_info: torch.Tensor) -> torch.Tensor:
    """Return the Schur-complement information transferred through uncertain state.

    For prior covariance ``P`` and state Fisher information ``J``, this computes
    ``J - J (P^{-1} + J)^{-1} J`` with input and output shape ``(..., d, d)``.
    This is equivalent to ``J (I + P J)^{-1}`` without requiring ``J`` to be
    invertible.
    """
    prior_cov = symmetrize(
        torch.nan_to_num(prior_cov, nan=0.0, posinf=1e6, neginf=-1e6)
    )
    state_info = symmetrize(
        torch.nan_to_num(state_info, nan=0.0, posinf=1e6, neginf=-1e6)
    )
    prior_precision = torch.cholesky_inverse(safe_cholesky(prior_cov))
    joint_precision = safe_cholesky(prior_precision + state_info)
    missing_info = state_info @ torch.cholesky_solve(state_info, joint_precision)
    return symmetrize(state_info - missing_info)


def activation_from_str(activation_str: str):
    """Convert a string to a PyTorch activation function."""
    if activation_str is None:
        return None
    if isinstance(activation_str, str):
        activation_str = activation_str.lower()
        if activation_str == "relu":
            return torch.nn.ReLU()
        elif activation_str == "tanh":
            return torch.nn.Tanh()
        elif activation_str == "sigmoid":
            return torch.nn.Sigmoid()
        elif activation_str == "leakyrelu":
            return torch.nn.LeakyReLU()
        elif activation_str == "leaky_relu":
            return torch.nn.LeakyReLU()
        elif activation_str == "elu":
            return torch.nn.ELU()
        else:
            raise ValueError(f"Unknown activation function: {activation_str}")


# @torch.no_grad()
# def generate_trajectory(self, x0, n_steps, action=None):
#     if x0.ndim == 2:
#         if x0.shape[0] == 1:
#             x0 = x0.unsqueeze(0)
#         else:
#             x0 = x0.unsqueeze(1)
#     B, T, D = x0.shape
#     if action is None:
#         action = torch.zeros(B, n_steps, D, device=self.device)
#     traj = [x0]
#     for i in range(n_steps):
#         traj.append(
#             traj[i]
#             + (self.compute_dynamics(traj[i]) + action[:, i].unsqueeze(1)) * self.dt
#             + torch.randn_like(traj[i]) * torch.sqrt(torch.tensor(self.var * self.dt))
#         )
#     return torch.cat(traj, dim=1)
