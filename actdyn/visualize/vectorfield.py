"""Vector-field evaluation and plotting primitives."""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt


def create_grid(x_range=2, n_grid=50, device="cpu"):
    """Create a grid of points in the specified range."""
    x = torch.linspace(-x_range, x_range, n_grid, device=device)
    y = torch.linspace(-x_range, x_range, n_grid, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="xy")  # [H, W]
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return grid, xx, yy


@torch.no_grad()
def compute_vector_field(
    dynamics, x_range=2.5, n_grid=50, tform=(None, None), is_residual=True, device="cpu"
):
    """
    Produces a vector field for a given dynamical system
    :param queries: N by dx torch tensor of query points where each row is a query
    :param dynamics: function handle for dynamics
    """
    xy, X, Y = create_grid(x_range=x_range, n_grid=n_grid, device=device)
    if hasattr(dynamics, "device"):
        xy = xy.to(dynamics.device)
    else:
        xy = xy.to(device)
    if tform[0] is not None:
        xy = (tform[0] @ xy.T).T + tform[1]

    vel = torch.zeros(xy.shape, device=device)
    with torch.no_grad():
        for n in range(xy.shape[0]):
            vel[n, :] = dynamics(xy[[n]])
            if not is_residual:
                vel[n, :] = vel[n, :] - xy[[n]].to(device)

    U = vel[:, 0].reshape(X.shape[0], X.shape[1])
    V = vel[:, 1].reshape(Y.shape[0], Y.shape[1])
    return X, Y, U, V


def plot_vector_field(dynamics, ax=None, title=None, streamplot_kwargs=None, **kwargs):
    X, Y, U, V = compute_vector_field(dynamics, **kwargs)
    X, Y, U, V = X.cpu().numpy(), Y.cpu().numpy(), U.cpu().numpy(), V.cpu().numpy()
    speed = np.sqrt(U**2 + V**2)
    stream_kwargs = dict(streamplot_kwargs or {})
    color_provided = "color" in stream_kwargs
    color = stream_kwargs.pop("color", speed)
    stream_kwargs.setdefault("linewidth", 0.5)
    stream_kwargs.setdefault("density", 2)
    if not color_provided:
        stream_kwargs.setdefault("cmap", "viridis")

    if ax is not None:
        plt.sca(ax)
    else:
        plt.figure(figsize=(8, 8))
    plt.streamplot(
        X,
        Y,
        U,
        V,
        color=color,
        **stream_kwargs,
    )
    title = "Vector Field of Latent Dynamics" if title is None else title
    if ax is None:
        # plt.colorbar(label="Speed", aspect=20)
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.title(title)
        # plt.axis("off")
        plt.axis("equal")
        plt.tight_layout()
        plt.colorbar(label="Speed", aspect=20)


@torch.no_grad()
def evaluate_vector_field_grid(
    dynamics,
    grid_points: np.ndarray,
    shape: tuple[int, int],
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a 2D vector field on flattened grid points.

    grid_points has shape (H * W, 2). Returned components each have shape
    (H, W), matching shape.
    """
    pts = torch.as_tensor(grid_points, dtype=torch.float32, device=device)
    if hasattr(dynamics, "device"):
        pts = pts.to(dynamics.device)
    vel = dynamics(pts).detach().cpu().numpy().reshape(shape[0], shape[1], 2)
    return vel[:, :, 0], vel[:, :, 1]


def vector_field_l2_error(
    true_u: np.ndarray,
    true_v: np.ndarray,
    inferred_u: np.ndarray,
    inferred_v: np.ndarray,
) -> np.ndarray:
    """Return pointwise L2 vector-field error on a shared plotting grid."""
    return np.sqrt(
        (np.asarray(inferred_u) - np.asarray(true_u)) ** 2
        + (np.asarray(inferred_v) - np.asarray(true_v)) ** 2
    )


class RbfVectorFieldDynamics:
    """Evaluate a sparse local RBF vector field on arbitrary query points."""

    def __init__(
        self,
        *,
        centers,
        axis,
        weights,
        width: float,
        support_radius: int,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.centers = torch.as_tensor(centers, dtype=torch.float32, device=self.device)
        self.axis = torch.as_tensor(axis, dtype=torch.float32, device=self.device)
        self.weights = torch.nan_to_num(
            torch.as_tensor(weights, dtype=torch.float32, device=self.device),
            nan=0.0,
            posinf=1e3,
            neginf=-1e3,
        ).clamp(-1e3, 1e3)
        self.width = float(max(width, 1e-6))
        self.support_radius = int(support_radius)
        n_axis = int(self.axis.numel())
        grid_i, grid_j = torch.meshgrid(
            torch.arange(n_axis, device=self.device),
            torch.arange(n_axis, device=self.device),
            indexing="ij",
        )
        self.center_i = grid_i.reshape(-1)
        self.center_j = grid_j.reshape(-1)

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        flat = state.reshape(-1, state.shape[-1])
        dx = torch.abs(flat[:, 0:1] - self.axis.view(1, -1))
        dy = torch.abs(flat[:, 1:2] - self.axis.view(1, -1))
        x_idx = torch.argmin(dx, dim=1)
        y_idx = torch.argmin(dy, dim=1)
        mask = (
            torch.abs(self.center_i.view(1, -1) - x_idx.view(-1, 1))
            + torch.abs(self.center_j.view(1, -1) - y_idx.view(-1, 1))
        ) <= self.support_radius
        scaled = (flat.unsqueeze(1) - self.centers.unsqueeze(0)) / self.width
        phi = torch.exp(-0.5 * torch.sum(scaled * scaled, dim=-1)) * mask.to(torch.float32)
        out = torch.nan_to_num(
            phi @ self.weights,
            nan=0.0,
            posinf=1e3,
            neginf=-1e3,
        ).clamp(-1e3, 1e3)
        return out.reshape(*state.shape[:-1], self.weights.shape[-1])
