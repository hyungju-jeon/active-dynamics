from __future__ import annotations

"""
General helper functions and constants
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


def safe_cholesky(M, jitter=1e-6, max_tries=5, growth=10.0):
    I = torch.eye(M.size(-1), device=M.device).expand_as(M)
    j = 0.0
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(M + j * I)
        except RuntimeError:
            j = jitter if j == 0.0 else j * growth
    return torch.linalg.cholesky(M + j * I)


def symmetrize(M):
    return 0.5 * (M + M.transpose(-1, -2))


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
