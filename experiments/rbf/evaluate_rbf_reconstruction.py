#!/usr/bin/env python3
"""Sweep fixed-grid RBF settings for reconstructing a vector field."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from actdyn.environment.vectorfield import build_vectorfield

try:
    from .local_rbf_asymmetric_basin import LocalGridRBFDynamics, fit_ridge, grid_states
except ImportError:
    from local_rbf_asymmetric_basin import LocalGridRBFDynamics, fit_ridge, grid_states


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _drift(args: argparse.Namespace, states: torch.Tensor) -> torch.Tensor:
    vf = build_vectorfield(
        args.dynamics_type,
        _floats(args.dyn_params),
        dynamics_alpha=float(args.dynamics_alpha),
        device=states.device,
    )
    with torch.no_grad():
        return vf.compute(states)


def _r2(target: torch.Tensor, prediction: torch.Tensor) -> float:
    residual = torch.sum((target - prediction) ** 2)
    centered = torch.sum((target - target.mean(dim=0, keepdim=True)) ** 2).clamp_min(1e-12)
    return float((1.0 - residual / centered).item())


def evaluate(args: argparse.Namespace, grid_points: int, lengthscale: float, ridge: float) -> dict[str, float | int | str]:
    device = torch.device(args.device)
    low = torch.full((2,), float(args.state_low), dtype=torch.float32, device=device)
    high = torch.full((2,), float(args.state_high), dtype=torch.float32, device=device)
    model = LocalGridRBFDynamics(
        state_low=low,
        state_high=high,
        grid_points=int(grid_points),
        lengthscale=float(lengthscale),
        active_radius=args.active_radius,
        dt=args.dt,
        device=device,
    )
    train_states = grid_states(low=low, high=high, points=args.train_grid_points)
    eval_states = grid_states(low=low, high=high, points=args.eval_grid_points)
    train_targets = _drift(args, train_states)
    eval_targets = _drift(args, eval_states)
    fit_ridge(model, train_states, train_targets, ridge=float(ridge))
    with torch.no_grad():
        pred = model(eval_states)
    _, _, valid = model.local_feature_entries(eval_states)
    return {
        "dynamics_type": args.dynamics_type,
        "grid_points": int(grid_points),
        "num_centers": int(model.centers.shape[0]),
        "lengthscale": float(lengthscale),
        "active_radius": int(model.active_radius),
        "max_active_centers": int(model.max_active_centers),
        "mean_active_centers": float(valid.sum(dim=-1).float().mean().item()),
        "ridge": float(ridge),
        "train_grid_points": int(args.train_grid_points),
        "eval_grid_points": int(args.eval_grid_points),
        "mse": float(torch.mean((eval_targets - pred) ** 2).item()),
        "r2": _r2(eval_targets, pred),
    }




def plot_fit(args: argparse.Namespace, grid_points: int, lengthscale: float, ridge: float, output: Path) -> None:
    """Plot a single fitted RBF vector field against the true vector field."""

    device = torch.device(args.device)
    low = torch.full((2,), float(args.state_low), dtype=torch.float32, device=device)
    high = torch.full((2,), float(args.state_high), dtype=torch.float32, device=device)
    model = LocalGridRBFDynamics(
        state_low=low,
        state_high=high,
        grid_points=int(grid_points),
        lengthscale=float(lengthscale),
        active_radius=args.active_radius,
        dt=args.dt,
        device=device,
    )
    train_states = grid_states(low=low, high=high, points=args.train_grid_points)
    fit_ridge(model, train_states, _drift(args, train_states), ridge=float(ridge))

    n = int(args.plot_grid_points)
    axis = torch.linspace(float(args.state_low), float(args.state_high), n, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    states = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    with torch.no_grad():
        true = _drift(args, states).reshape(n, n, 2)
        pred = model(states).reshape_as(true)
    error = torch.linalg.norm(true - pred, dim=-1)
    r2 = _r2(true.reshape(-1, 2), pred.reshape(-1, 2))
    mse = float(torch.mean((true - pred) ** 2).item())

    xx_np = xx.cpu().numpy()
    yy_np = yy.cpu().numpy()
    true_np = true.cpu().numpy()
    pred_np = pred.cpu().numpy()
    error_np = error.cpu().numpy()
    centers = model.centers.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=int(args.dpi), constrained_layout=True)
    for ax in axes:
        ax.set_xlim(float(args.state_low), float(args.state_high))
        ax.set_ylim(float(args.state_low), float(args.state_high))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].streamplot(xx_np, yy_np, true_np[:, :, 0], true_np[:, :, 1], color="#6D737C", density=1.1, linewidth=0.55)
    axes[0].set_title("true asymmetric basin")

    axes[1].scatter(centers[:, 0], centers[:, 1], s=1.0, color="#CDD2D8", alpha=0.55, linewidths=0)
    axes[1].streamplot(xx_np, yy_np, pred_np[:, :, 0], pred_np[:, :, 1], color="#2F3A45", density=1.1, linewidth=0.55)
    axes[1].set_title(f"fitted RBF 41x41, ell={lengthscale:g}")

    im = axes[2].imshow(
        error_np,
        extent=(float(args.state_low), float(args.state_high), float(args.state_low), float(args.state_high)),
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    axes[2].set_title(f"L2 error, R2={r2:.4f}, mse={mse:.3g}")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"centers={model.centers.shape[0]}, active={model.max_active_centers}, ridge={ridge:.0e}",
        fontsize=9,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"plot={output}", flush=True)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate RBF reconstruction accuracy before online planning.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/rbf/reconstruction"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dynamics-type", default="asymmetric_basin")
    parser.add_argument("--dyn-params", default="-1.2,-0.8,0.5,1.1")
    parser.add_argument("--dynamics-alpha", type=float, default=1.0)
    parser.add_argument("--state-low", type=float, default=-5.0)
    parser.add_argument("--state-high", type=float, default=5.0)
    parser.add_argument("--train-grid-points", type=int, default=41)
    parser.add_argument("--eval-grid-points", type=int, default=61)
    parser.add_argument("--grid-points", default="21,31,41,51")
    parser.add_argument("--lengthscales", default="0.35,0.5,0.7,0.9,1.1")
    parser.add_argument("--ridges", default="1e-6,1e-4")
    parser.add_argument("--active-radius", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument("--plot-grid-points", type=int, default=81)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)

    grid_points_list = _ints(args.grid_points)
    lengthscale_list = _floats(args.lengthscales)
    ridge_list = _floats(args.ridges)

    rows = []
    for gp in grid_points_list:
        for ell in lengthscale_list:
            for ridge in ridge_list:
                row = evaluate(args, gp, ell, ridge)
                rows.append(row)
                print(
                    f"grid={gp:>2} centers={row['num_centers']:>4} ell={ell:.3g} "
                    f"active={row['max_active_centers']:>3} ridge={ridge:.0e} "
                    f"r2={row['r2']:.5f} mse={row['mse']:.5g}",
                    flush=True,
                )

    rows.sort(key=lambda r: (-float(r["r2"]), int(r["num_centers"]), float(r["lengthscale"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "asymmetric_basin_rbf_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    best_path = args.output_dir / "best.json"
    best_path.write_text(json.dumps(rows[0], indent=2, sort_keys=True) + "\n")
    print(f"best={json.dumps(rows[0], sort_keys=True)}", flush=True)
    print(f"csv={csv_path}", flush=True)
    print(f"best_json={best_path}", flush=True)
    if args.plot_output is not None:
        if len(grid_points_list) != 1 or len(lengthscale_list) != 1 or len(ridge_list) != 1:
            raise SystemExit("--plot-output expects one grid point, one lengthscale, and one ridge")
        plot_fit(args, grid_points_list[0], lengthscale_list[0], ridge_list[0], args.plot_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
