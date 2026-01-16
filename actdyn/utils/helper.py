"""
General helper functions and constants
"""

import torch
import numpy as np
from typing import Dict

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
    if isinstance(low, float):
        low = [low] * dim
    if isinstance(high, float):
        high = [high] * dim

    def _sampler(N: int):
        return torch.stack(
            [low[i] + (high[i] - low[i]) * torch.rand(N) for i in range(dim)], dim=-1
        )

    return _sampler


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
