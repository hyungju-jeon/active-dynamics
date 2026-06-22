#!/usr/bin/env python3
"""Localized RBF surrogate for the Duffing vector field.

The surrogate uses :class:`actdyn.models.dynamics.RBFDynamics` with fixed grid
centers and a fixed Gaussian length scale,

    phi_i(z) = exp(-||z - c_i||^2 / (2 ell^2)).

At each latent state ``z`` only the centers in a small grid stencil around
``z`` are evaluated.  This keeps the local parameter block for EIG small even
when the full center grid is dense.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn.functional import softplus

from actdyn.environment.vectorfield import build_vectorfield
from actdyn.models.dynamics import RBFDynamics
from actdyn.utils.torch_utils import eps


class LocalGridRBFDynamics(RBFDynamics):
    """RBF dynamics with fixed length scale and local grid-stencil features.

    Args:
        state_low: Lower latent-state bounds with shape ``(state_dim,)``.
        state_high: Upper latent-state bounds with shape ``(state_dim,)``.
        grid_points: Number of centers per latent dimension.
        lengthscale: Fixed Gaussian length scale ``ell``.
        active_radius: Optional Manhattan radius of the local grid stencil. If
            ``None``, use ``ceil(lengthscale / min_grid_spacing)``.

    Inputs:
        ``state`` has shape ``(..., state_dim)``.

    Outputs:
        ``compute_param(state)`` returns mean and variance with shape
        ``(..., state_dim)``.  The local weight Jacobian has shape
        ``(..., state_dim, active_centers * state_dim)``.
    """

    def __init__(
        self,
        *,
        state_low: list[float] | tuple[float, ...] | torch.Tensor,
        state_high: list[float] | tuple[float, ...] | torch.Tensor,
        grid_points: int = 81,
        lengthscale: float = 0.6,
        active_radius: int | None = None,
        alpha: float = 1.0,
        dt: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> None:
        device = torch.device(device)
        low = torch.as_tensor(state_low, dtype=torch.float32, device=device).reshape(-1)
        high = torch.as_tensor(state_high, dtype=torch.float32, device=device).reshape(-1)
        if low.shape != high.shape:
            raise ValueError(f"state_low and state_high must match, got {low.shape} and {high.shape}")
        if torch.any(high <= low):
            raise ValueError("state_high must be greater than state_low in every dimension")
        if int(grid_points) < 2:
            raise ValueError("grid_points must be at least 2")
        if float(lengthscale) <= 0.0:
            raise ValueError("lengthscale must be positive")

        state_dim = int(low.numel())
        axes = [
            torch.linspace(float(low[d]), float(high[d]), int(grid_points), device=device)
            for d in range(state_dim)
        ]
        mesh = torch.meshgrid(*axes, indexing="ij")
        centers = torch.stack([m.reshape(-1) for m in mesh], dim=1)
        gamma = 0.5 / float(lengthscale) ** 2

        super().__init__(
            state_dim=state_dim,
            alpha=float(alpha),
            gamma=gamma,
            centers=centers,
            device=device,
            dt=float(dt),
            is_residual=True,
        )
        self.network.weights = self.weights
        self.grid_points = int(grid_points)
        self.lengthscale = float(lengthscale)

        spacing = (high - low) / float(self.grid_points - 1)
        if active_radius is None:
            active_radius = int(math.ceil(self.lengthscale / float(spacing.min().item())))
        self.active_radius = int(max(active_radius, 0))
        strides = torch.tensor(
            [self.grid_points ** (state_dim - 1 - i) for i in range(state_dim)],
            dtype=torch.long,
            device=device,
        )
        offset_axis = torch.arange(
            -self.active_radius,
            self.active_radius + 1,
            dtype=torch.long,
            device=device,
        )
        offset_mesh = torch.meshgrid(*([offset_axis] * state_dim), indexing="ij")
        offsets = torch.stack([m.reshape(-1) for m in offset_mesh], dim=1)
        offsets = offsets[torch.sum(torch.abs(offsets), dim=1) <= self.active_radius]

        self.register_buffer("state_low", low)
        self.register_buffer("state_high", high)
        self.register_buffer("grid_spacing", spacing)
        self.register_buffer("grid_strides", strides)
        self.register_buffer("grid_offsets", offsets)

    @property
    def max_active_centers(self) -> int:
        return int(self.grid_offsets.shape[0])

    def _flatten_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state_dim={self.state_dim}, got shape {tuple(state.shape)}")
        return state.reshape(-1, self.state_dim), state.shape[:-1]

    def local_feature_entries(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return local center indices, RBF values, and validity mask.

        Args:
            state: Latent states with shape ``(..., state_dim)``.

        Returns:
            ``indices``, ``values``, and ``valid`` with shape
            ``(..., max_active_centers)``.  Invalid boundary-stencil entries have
            zero value and should be ignored by local EIG code.
        """

        flat_state, leading_shape = self._flatten_state(state)
        nearest = torch.round((flat_state - self.state_low) / self.grid_spacing).long()
        multi_index = nearest[:, None, :] + self.grid_offsets[None, :, :]
        valid = ((multi_index >= 0) & (multi_index < self.grid_points)).all(dim=-1)
        safe_multi_index = multi_index.clamp(0, self.grid_points - 1)
        indices = (safe_multi_index * self.grid_strides).sum(dim=-1)

        centers = self.centers[indices]
        squared_dist = ((flat_state[:, None, :] - centers) ** 2).sum(dim=-1)
        values = self.alpha * torch.exp(-squared_dist * self.gamma)
        values = values * valid.to(values.dtype)
        return (
            indices.reshape(*leading_shape, -1),
            values.reshape(*leading_shape, -1),
            valid.reshape(*leading_shape, -1),
        )

    def rbf(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate sparse local RBF features as a dense feature row.

        The dense return shape matches ``RBFDynamics.rbf`` for compatibility,
        but only the local stencil entries are computed and nonzero.
        """

        indices, values, _ = self.local_feature_entries(state)
        leading_shape = indices.shape[:-1]
        flat_indices = indices.reshape(-1, indices.shape[-1])
        flat_values = values.reshape_as(flat_indices).to(dtype=torch.float32)
        features = torch.zeros(
            flat_indices.shape[0],
            self.centers.shape[0],
            dtype=flat_values.dtype,
            device=flat_values.device,
        )
        features.scatter_add_(1, flat_indices, flat_values)
        return features.reshape(*leading_shape, self.centers.shape[0])

    def compute_param(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict drift using only weights attached to the local center stencil."""

        indices, values, _ = self.local_feature_entries(state)
        leading_shape = indices.shape[:-1]
        flat_indices = indices.reshape(-1, indices.shape[-1])
        flat_values = values.reshape(-1, values.shape[-1])
        local_weights = self.weights[flat_indices]
        mean = (flat_values.unsqueeze(-1) * local_weights).sum(dim=1)
        mean = mean.reshape(*leading_shape, self.state_dim)
        var = (softplus(self.logvar) + eps).expand_as(mean)
        return mean, var

    def local_weight_parameter_indices(self, state: torch.Tensor) -> torch.Tensor:
        """Return flattened weight-parameter ids active at each state.

        The ids use center-major order: ``center_index * state_dim + output_dim``.
        Shape is ``(..., max_active_centers * state_dim)``.
        """

        indices, _, _ = self.local_feature_entries(state)
        offsets = torch.arange(self.state_dim, dtype=torch.long, device=indices.device)
        return (indices.unsqueeze(-1) * self.state_dim + offsets).reshape(*indices.shape[:-1], -1)

    def local_weight_jacobian(self, state: torch.Tensor) -> torch.Tensor:
        """Differentiate local drift with respect to active local RBF weights.

        Args:
            state: Latent states with shape ``(..., state_dim)``.

        Returns:
            Jacobian with shape ``(..., state_dim, max_active_centers * state_dim)``.
        """

        _, values, _ = self.local_feature_entries(state)
        leading_shape = values.shape[:-1]
        flat_values = values.reshape(-1, values.shape[-1])
        eye = torch.eye(self.state_dim, dtype=flat_values.dtype, device=flat_values.device)
        jac = flat_values[:, None, :, None] * eye[None, :, None, :]
        return jac.reshape(*leading_shape, self.state_dim, -1)


def duffing_drift(
    states: torch.Tensor,
    *,
    dynamics_alpha: float = 1.0,
    dyn_params: list[float] | None = None,
) -> torch.Tensor:
    """Evaluate the true Duffing vector field.

    Args:
        states: Latent states with shape ``(n, 2)``.
        dynamics_alpha: Vector-field scale used by the environment.
        dyn_params: Optional ``[a, b, c]`` override.

    Returns:
        Drift targets with shape ``(n, 2)``.
    """

    vf = build_vectorfield(
        "duffing",
        dyn_params,
        dynamics_alpha=float(dynamics_alpha),
        device=states.device,
    )
    with torch.no_grad():
        return vf.compute(states)


def sample_uniform_states(
    n_samples: int,
    *,
    low: torch.Tensor,
    high: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Sample latent states uniformly from the experiment box.

    Args:
        n_samples: Number of states to sample.
        low: Lower bounds with shape ``(state_dim,)``.
        high: Upper bounds with shape ``(state_dim,)``.
        seed: CPU PRNG seed.

    Returns:
        State tensor with shape ``(n_samples, state_dim)``.
    """

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    unit = torch.rand(int(n_samples), low.numel(), generator=gen, dtype=torch.float32)
    return low + (high - low) * unit.to(device=low.device)


def grid_states(*, low: torch.Tensor, high: torch.Tensor, points: int) -> torch.Tensor:
    """Build an evaluation grid with shape ``(points ** state_dim, state_dim)``."""

    axes = [
        torch.linspace(float(low[d]), float(high[d]), int(points), device=low.device)
        for d in range(low.numel())
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=1)


def fit_ridge(
    model: LocalGridRBFDynamics,
    states: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge: float = 1e-4,
) -> None:
    """Fit RBF weights by ridge regression.

    Args:
        model: Localized RBF dynamics model.
        states: Training states with shape ``(n, state_dim)``.
        targets: Drift targets with shape ``(n, state_dim)``.
        ridge: Diagonal ridge penalty for the global weight solve.
    """

    features = model.rbf(states)
    eye = torch.eye(features.shape[-1], dtype=features.dtype, device=features.device)
    gram = features.T @ features + float(ridge) * eye
    rhs = features.T @ targets
    weights = torch.linalg.solve(gram, rhs)
    model.weights.data.copy_(weights)
    model.network.weights = model.weights


def _r2_score(target: torch.Tensor, prediction: torch.Tensor) -> float:
    residual = ((target - prediction) ** 2).sum()
    centered = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum().clamp_min(eps)
    return float((1.0 - residual / centered).item())


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    low = torch.tensor([args.state_low, args.state_low], dtype=torch.float32, device=device)
    high = torch.tensor([args.state_high, args.state_high], dtype=torch.float32, device=device)
    dyn_params = None if args.dyn_params is None else [float(x) for x in args.dyn_params.split(",")]
    if dyn_params is not None and len(dyn_params) != 3:
        raise ValueError("--dyn-params must contain three comma-separated Duffing values")

    model = LocalGridRBFDynamics(
        state_low=low,
        state_high=high,
        grid_points=args.grid_points,
        lengthscale=args.lengthscale,
        active_radius=args.active_radius,
        alpha=1.0,
        dt=args.dt,
        device=device,
    )

    train_states = sample_uniform_states(args.train_samples, low=low, high=high, seed=args.seed)
    train_targets = duffing_drift(
        train_states,
        dynamics_alpha=args.dynamics_alpha,
        dyn_params=dyn_params,
    )
    fit_ridge(model, train_states, train_targets, ridge=args.ridge)

    eval_states = grid_states(low=low, high=high, points=args.eval_grid_points)
    eval_targets = duffing_drift(
        eval_states,
        dynamics_alpha=args.dynamics_alpha,
        dyn_params=dyn_params,
    )
    with torch.no_grad():
        eval_prediction = model(eval_states)
    _, _, valid = model.local_feature_entries(eval_states)
    active_counts = valid.sum(dim=-1).float()

    summary = {
        "environment": "duffing",
        "train_samples": int(args.train_samples),
        "eval_grid_points": int(args.eval_grid_points),
        "num_centers": int(model.centers.shape[0]),
        "max_active_centers": int(model.max_active_centers),
        "mean_active_centers": float(active_counts.mean().item()),
        "max_active_weight_parameters": int(model.max_active_centers * model.state_dim),
        "lengthscale": float(model.lengthscale),
        "grid_points": int(model.grid_points),
        "active_radius": int(model.active_radius),
        "ridge": float(args.ridge),
        "dt": float(args.dt),
        "dynamics_alpha": float(args.dynamics_alpha),
        "mse": float(((eval_targets - eval_prediction) ** 2).mean().item()),
        "r2": _r2_score(eval_targets, eval_prediction),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    torch.save(
        {
            "centers": model.centers.detach().cpu(),
            "weights": model.weights.detach().cpu(),
            "summary": summary,
        },
        output_dir / "surrogate.pt",
    )
    return summary


def self_test() -> None:
    device = torch.device("cpu")
    low = torch.tensor([-2.0, -2.0], dtype=torch.float32, device=device)
    high = torch.tensor([2.0, 2.0], dtype=torch.float32, device=device)
    model = LocalGridRBFDynamics(
        state_low=low,
        state_high=high,
        grid_points=11,
        lengthscale=0.45,
        active_radius=2,
        device=device,
    )
    z = torch.tensor([[0.0, 0.0], [1.8, -1.8]], dtype=torch.float32)
    indices, values, valid = model.local_feature_entries(z)
    assert indices.shape == values.shape == valid.shape == (2, 13)
    assert int(valid.sum(dim=-1).max().item()) <= model.max_active_centers
    assert model.rbf(z).shape == (2, model.centers.shape[0])
    assert model.local_weight_jacobian(z).shape == (2, 2, 26)

    train_states = sample_uniform_states(192, low=low, high=high, seed=7)
    train_targets = duffing_drift(train_states)
    model.weights.data.zero_()
    before = ((model(train_states) - train_targets) ** 2).mean()
    fit_ridge(model, train_states, train_targets, ridge=1e-5)
    after = ((model(train_states) - train_targets) ** 2).mean()
    assert after < 0.8 * before, (float(before.item()), float(after.item()))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a localized RBFDynamics surrogate on Duffing."
    )
    parser.add_argument("--output-dir", default="results/rbf/duffing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--eval-grid-points", type=int, default=41)
    parser.add_argument("--state-low", type=float, default=-5.0)
    parser.add_argument("--state-high", type=float, default=5.0)
    parser.add_argument("--grid-points", type=int, default=81)
    parser.add_argument("--lengthscale", type=float, default=0.6)
    parser.add_argument("--active-radius", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dynamics-alpha", type=float, default=1.0)
    parser.add_argument(
        "--dyn-params",
        default="-0.5,-0.75,0.1",
        help="Optional comma-separated [a,b,c].",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.self_test:
        self_test()
        return
    summary = run_experiment(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
