#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import torch

from actdyn.models.decoder import Decoder, LogLinearMapping, PoissonNoise
from actdyn.utils import save_load
from actdyn.utils.visualize import plot_vector_field


DEFAULT_EXP_IDS = ["active_short", "active_long", "RND", "random"]
DEFAULT_SEEDS = [0, 10, 20]


class _DuffingDynamics:
    """Torch-callable Duffing dynamics adapter for plot_vector_field."""

    def __init__(self, a: float, b: float, device: str = "cpu") -> None:
        self.a = float(a)
        self.b = float(b)
        self.device = torch.device(device)

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        x = state[..., 0]
        v = state[..., 1]
        dx = v
        dv = self.a * v - self.b * x - 0.1 * x**3
        return torch.stack((dx, dv), dim=-1)


def _parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str | None) -> list[int]:
    return [int(item) for item in _parse_csv_list(raw)]


def _resolve_trace_path(run_dir: Path, metadata: dict[str, Any], key: str, fallback_name: str) -> Path:
    trace_path = metadata.get(key)
    if isinstance(trace_path, str) and trace_path.strip():
        path = Path(trace_path)
        if not path.is_absolute():
            path = (run_dir / path).resolve()
        return path
    return run_dir / fallback_name


def _trace_index(trace_steps: np.ndarray, step: int) -> int:
    if trace_steps.size == 0:
        return 0
    idx = int(np.searchsorted(trace_steps, step, side="right") - 1)
    return int(np.clip(idx, 0, len(trace_steps) - 1))


def _load_decoder_from_model_state(results_path: Path, dt: float = 0.01) -> Decoder:
    model_path = results_path / "model" / "model_final.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint for exact info maps: {model_path}")

    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    if not isinstance(state, dict):
        raise ValueError(f"Unexpected model checkpoint format at {model_path}")

    weight = state.get("decoder.mapping.network.0.weight")
    bias = state.get("decoder.mapping.network.0.bias")
    if weight is None or bias is None:
        raise KeyError("decoder.mapping.network.0.{weight,bias} missing from model checkpoint")

    weight = torch.as_tensor(weight, dtype=torch.float32).detach().cpu()
    bias = torch.as_tensor(bias, dtype=torch.float32).detach().cpu()

    mapping = LogLinearMapping(latent_dim=2, obs_dim=weight.shape[0], dt=float(dt), device="cpu")
    noise = PoissonNoise(device="cpu")
    decoder = Decoder(mapping=mapping, noise=noise, device="cpu")
    decoder.mapping.set_weights(weight, requires_grad=False)
    decoder.mapping.set_bias(bias, requires_grad=False)
    return decoder


def _load_run_artifacts(
    run_dir: Path,
) -> tuple[
    dict[str, Any],
    Path,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing run metadata: {meta_path}")
    metadata = json.loads(meta_path.read_text())

    results_path = Path(metadata["results_path"])
    state_action_trace_path = _resolve_trace_path(
        run_dir,
        metadata,
        key="state_action_trace_path",
        fallback_name="state_action_trace.csv",
    )
    if state_action_trace_path.exists():
        sa_df = pd.read_csv(state_action_trace_path).sort_values("step")
        true_state = sa_df[["true_x", "true_v"]].to_numpy(dtype=float)
        model_state = sa_df[["model_x", "model_v"]].to_numpy(dtype=float)
        if {"action_x", "action_v"}.issubset(sa_df.columns):
            action = sa_df[["action_x", "action_v"]].to_numpy(dtype=float)
        else:
            action = np.zeros_like(model_state)
        if "policy_cost" in sa_df.columns:
            policy_cost = sa_df["policy_cost"].to_numpy(dtype=float)
        else:
            policy_cost = np.full((model_state.shape[0],), np.nan, dtype=float)
    else:
        rollout = save_load.load_and_concatenate_rollouts(str(results_path / "rollouts"))
        true_state = rollout["env_state"][0].detach().cpu().numpy()
        model_state = rollout["model_state"][0].detach().cpu().numpy()
        action_raw = rollout["action"][0].detach().cpu().numpy()
        action_flat = action_raw.reshape(action_raw.shape[0], -1)
        if action_flat.shape[1] < 2:
            pad = np.zeros((action_flat.shape[0], 2 - action_flat.shape[1]), dtype=action_flat.dtype)
            action_flat = np.concatenate([action_flat, pad], axis=1)
        action = action_flat[:, :2]
        policy_cost = np.full((model_state.shape[0],), np.nan, dtype=float)

    param_trace_path = _resolve_trace_path(
        run_dir,
        metadata,
        key="parameter_error_trace_path",
        fallback_name="parameter_error_trace.csv",
    )
    if param_trace_path.exists():
        param_df = pd.read_csv(param_trace_path).sort_values("step")
        param_steps = param_df["step"].to_numpy(dtype=int)
        param_err = param_df["parameter_error"].to_numpy(dtype=float)
    else:
        param_steps = np.arange(1, true_state.shape[0] + 1, dtype=int)
        fallback = float(metadata.get("embedding_error_final", 0.0))
        param_err = np.full(param_steps.shape, fallback, dtype=float)

    emb_trace_path = _resolve_trace_path(
        run_dir,
        metadata,
        key="embedding_estimate_trace_path",
        fallback_name="embedding_estimate_trace.csv",
    )
    if emb_trace_path.exists():
        emb_df = pd.read_csv(emb_trace_path).sort_values("step")
        emb_steps = emb_df["step"].to_numpy(dtype=int)
        e0 = emb_df["e0"].to_numpy(dtype=float)
        e1 = emb_df["e1"].to_numpy(dtype=float)
    else:
        emb_steps = param_steps.copy()
        theta_final = np.asarray(metadata.get("embedding_estimate", [1.0, 1.0]), dtype=float)
        e0 = np.full(emb_steps.shape, float(theta_final[0]), dtype=float)
        e1 = np.full(emb_steps.shape, float(theta_final[1]), dtype=float)

    info_trace_path = _resolve_trace_path(
        run_dir,
        metadata,
        key="information_trace_path",
        fallback_name="information_trace.csv",
    )
    if info_trace_path.exists():
        info_df = pd.read_csv(info_trace_path).sort_values("step")
        info_steps = info_df["step"].to_numpy(dtype=int)

        def _get_col(name: str) -> np.ndarray:
            if name in info_df.columns:
                return info_df[name].to_numpy(dtype=float)
            return np.zeros(info_steps.shape, dtype=float)

        pz00 = _get_col("Pz00")
        pz01 = _get_col("Pz01")
        pz11 = _get_col("Pz11")
    else:
        info_steps = param_steps.copy()
        pz00 = np.ones(info_steps.shape, dtype=float)
        pz01 = np.zeros(info_steps.shape, dtype=float)
        pz11 = np.ones(info_steps.shape, dtype=float)

    return (
        metadata,
        results_path,
        true_state,
        model_state,
        action,
        policy_cost,
        param_steps,
        param_err,
        emb_steps,
        e0,
        e1,
        info_steps,
        np.stack([pz00, pz01, pz11], axis=1),
    )


def _load_acquisition_map_trace(
    run_dir: Path,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    trace_path = _resolve_trace_path(
        run_dir,
        metadata,
        key="acquisition_map_trace_path",
        fallback_name="acquisition_map_trace.npz",
    )
    if not trace_path.exists():
        return None

    with np.load(trace_path, allow_pickle=True) as data:
        steps = np.asarray(data["steps"], dtype=int)
        axis = np.asarray(data["axis"], dtype=float)
        maps = np.asarray(data["maps"], dtype=float)

    if maps.ndim != 3 or steps.ndim != 1:
        raise ValueError(f"Invalid acquisition map trace format: {trace_path}")
    if maps.shape[0] != steps.shape[0]:
        raise ValueError(f"Acquisition map count mismatch in: {trace_path}")
    return steps, axis, maps


def _prepare_exact_info_grid(
    decoder: Decoder,
    *,
    grid_lim: float,
    n_grid: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, np.ndarray]:
    axis = np.linspace(-grid_lim, grid_lim, n_grid, dtype=np.float32)
    X, V = np.meshgrid(axis, axis, indexing="xy")
    z = torch.as_tensor(np.stack([X.ravel(), V.ravel()], axis=1), dtype=torch.float32)

    with torch.no_grad():
        H = decoder.jacobian(z)
        R_diag = torch.clamp(decoder.var(z), min=1e-8)
        invR_H = H / R_diag.unsqueeze(-1)
        I_z = torch.einsum("nyi,nyj->nij", H, invR_H)
        I_z = 0.5 * (I_z + I_z.transpose(-1, -2))

        S = torch.zeros(z.shape[0], 2, 2, dtype=torch.float32)
        S[:, 1, 0] = float(dt) * z[:, 1]
        S[:, 1, 1] = -float(dt) * z[:, 0]

    I_z_trace = (
        torch.diagonal(I_z, dim1=-2, dim2=-1).sum(dim=-1).detach().cpu().numpy().reshape(n_grid, n_grid)
    )
    I_z_trace = np.nan_to_num(I_z_trace, nan=1e-12, posinf=1e6, neginf=1e-12)
    I_z_trace = np.maximum(I_z_trace, 1e-12)
    return X, V, I_z, S, I_z_trace


def _compute_itheta_trace_map(
    I_z: torch.Tensor,
    S: torch.Tensor,
    pz00: float,
    pz01: float,
    pz11: float,
    n_grid: int,
) -> np.ndarray:
    pz = torch.tensor(
        [[float(pz00), float(pz01)], [float(pz01), float(pz11)]],
        dtype=torch.float32,
    )
    pz = 0.5 * (pz + pz.transpose(-1, -2))

    eye = torch.eye(2, dtype=torch.float32).unsqueeze(0).expand(I_z.shape[0], -1, -1)
    PIz = torch.einsum("ij,njk->nik", pz, I_z)
    atten = eye + PIz
    try:
        atten_Iz = torch.linalg.solve(atten, I_z)
    except RuntimeError:
        atten_Iz = torch.einsum("nij,njk->nik", torch.linalg.pinv(atten), I_z)

    I_theta = torch.einsum("nki,nkl,nlj->nij", S, atten_Iz, S)
    I_theta = 0.5 * (I_theta + I_theta.transpose(-1, -2))

    out = (
        torch.diagonal(I_theta, dim1=-2, dim2=-1)
        .sum(dim=-1)
        .detach()
        .cpu()
        .numpy()
        .reshape(n_grid, n_grid)
    )
    out = np.nan_to_num(out, nan=1e-12, posinf=1e6, neginf=1e-12)
    out = np.maximum(out, 1e-12)
    return out


def _make_lognorm(values: list[np.ndarray]) -> LogNorm:
    flat = np.concatenate([np.ravel(v) for v in values], axis=0)
    positive = flat[np.isfinite(flat) & (flat > 0)]
    if positive.size == 0:
        return LogNorm(vmin=1e-8, vmax=1.0)

    min_positive = float(np.min(positive))
    p1 = float(np.percentile(positive, 1.0))
    p99 = float(np.percentile(positive, 99.0))
    vmin = max(min_positive, p1)
    vmax = max(p99, vmin * 1.01)
    return LogNorm(vmin=vmin, vmax=vmax)


def _frame_indices(n_steps: int, stride: int) -> list[int]:
    idxs = list(range(0, n_steps, max(1, int(stride))))
    if not idxs:
        return [0]
    if idxs[-1] != n_steps - 1:
        idxs.append(n_steps - 1)
    return idxs


def make_info_maps_video(
    run_dir: Path,
    output_path: Path,
    *,
    stride: int,
    fps: int,
    grid_lim: float,
    ig_grid: int,
    ig_dt: float,
) -> Path:
    (
        metadata,
        results_path,
        true_state,
        _model_state,
        _action,
        _policy_cost,
        _param_steps,
        _param_err,
        emb_steps,
        emb_e0,
        emb_e1,
        info_steps,
        pz_cols,
    ) = _load_run_artifacts(run_dir)

    decoder = _load_decoder_from_model_state(results_path=results_path, dt=float(ig_dt))
    grid_n = max(25, int(ig_grid))
    _X, _V, I_z_flat, S_flat, I_z_map = _prepare_exact_info_grid(
        decoder,
        grid_lim=float(grid_lim),
        n_grid=grid_n,
        dt=float(ig_dt),
    )

    idxs = _frame_indices(true_state.shape[0], stride)
    I_theta_frames: list[np.ndarray] = []
    for i in idxs:
        step = int(i + 1)
        info_idx = _trace_index(info_steps, step)
        _emb_idx = _trace_index(emb_steps, step)
        _theta = np.array([float(emb_e0[_emb_idx]), float(emb_e1[_emb_idx])], dtype=float)
        pz00, pz01, pz11 = pz_cols[info_idx]
        I_theta_frames.append(
            _compute_itheta_trace_map(
                I_z=I_z_flat,
                S=S_flat,
                pz00=float(pz00),
                pz01=float(pz01),
                pz11=float(pz11),
                n_grid=grid_n,
            )
        )

    I_z_frames = [I_z_map for _ in idxs]
    norm_iz = _make_lognorm(I_z_frames)
    norm_itheta = _make_lognorm(I_theta_frames)

    extent = [-float(grid_lim), float(grid_lim), -float(grid_lim), float(grid_lim)]
    fig, (ax_iz, ax_itheta) = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=120)

    im_iz = ax_iz.imshow(
        I_z_frames[0],
        extent=extent,
        origin="lower",
        cmap="magma",
        norm=norm_iz,
        interpolation="nearest",
    )
    im_itheta = ax_itheta.imshow(
        I_theta_frames[0],
        extent=extent,
        origin="lower",
        cmap="viridis",
        norm=norm_itheta,
        interpolation="nearest",
    )

    for ax in (ax_iz, ax_itheta):
        ax.set_xlim(-grid_lim, grid_lim)
        ax.set_ylim(-grid_lim, grid_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("v")

    ax_iz.set_title(r"$I_{z,t}$ map (log scale)")
    ax_itheta.set_title(r"$I_{\theta,t}$ map (log scale)")

    cbar_iz = fig.colorbar(im_iz, ax=ax_iz, fraction=0.046, pad=0.02)
    cbar_iz.set_label(r"trace($I_z$)")
    cbar_itheta = fig.colorbar(im_itheta, ax=ax_itheta, fraction=0.046, pad=0.02)
    cbar_itheta.set_label(r"trace($I_{\theta}$)")

    step_text = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.12, top=0.90, wspace=0.24)

    frames: list[np.ndarray] = []
    for frame_k, i in enumerate(idxs):
        cur_step = int(i + 1)
        im_iz.set_data(I_z_frames[frame_k])
        im_itheta.set_data(I_theta_frames[frame_k])

        fig.suptitle(
            f"{metadata.get('exp_id', 'unknown')} seed={metadata.get('seed', 'n/a')} | step={cur_step}",
            fontsize=12,
            y=0.96,
        )
        step_text.set_text(
            f"I_z scale: [{norm_iz.vmin:.3e}, {norm_iz.vmax:.3e}] | "
            f"I_theta scale: [{norm_itheta.vmin:.3e}, {norm_itheta.vmax:.3e}]"
        )

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    plt.close(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=max(1, int(fps)))
    return output_path


def make_traj_vf_video(
    run_dir: Path,
    output_path: Path,
    *,
    stride: int,
    fps: int,
    grid_lim: float,
) -> Path:
    (
        metadata,
        _results_path,
        true_state,
        model_state,
        _action,
        _policy_cost,
        _param_steps,
        _param_err,
        emb_steps,
        emb_e0,
        emb_e1,
        _info_steps,
        _pz_cols,
    ) = _load_run_artifacts(run_dir)

    theta_true = np.asarray(metadata.get("embedding_true", [0.0, 0.0]), dtype=float)
    dyn_true = _DuffingDynamics(a=float(theta_true[0]), b=float(theta_true[1]))
    idxs = _frame_indices(true_state.shape[0], stride)

    fig = plt.figure(figsize=(14.0, 6.2), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    ax_true = fig.add_subplot(gs[0, 0])
    ax_est = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.09, top=0.90, wspace=0.22)

    def _draw_panel(ax, dynamics_obj, title: str, i: int) -> None:
        ax.clear()
        plot_vector_field(dynamics_obj, ax=ax, x_range=grid_lim, n_grid=26, is_residual=True, device="cpu")
        ax.plot(true_state[: i + 1, 0], true_state[: i + 1, 1], color="black", linewidth=1.8, label="true traj")
        ax.plot(
            model_state[: i + 1, 0],
            model_state[: i + 1, 1],
            color="tab:blue",
            linewidth=1.8,
            label="inferred traj",
        )
        ax.scatter([true_state[i, 0]], [true_state[i, 1]], color="black", s=28, zorder=5)
        ax.scatter([model_state[i, 0]], [model_state[i, 1]], color="tab:blue", s=28, zorder=5)
        ax.set_xlim(-grid_lim, grid_lim)
        ax.set_ylim(-grid_lim, grid_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_title(title)
        ax.legend(loc="upper right")

    frames: list[np.ndarray] = []
    for i in idxs:
        cur_step = int(i + 1)
        emb_idx = _trace_index(emb_steps, cur_step)
        theta_est = np.asarray([float(emb_e0[emb_idx]), float(emb_e1[emb_idx])], dtype=float)
        dyn_est = _DuffingDynamics(a=float(theta_est[0]), b=float(theta_est[1]))

        _draw_panel(
            ax_true,
            dyn_true,
            f"True VF [a,b]=[{float(theta_true[0]):.3f}, {float(theta_true[1]):.3f}]",
            i,
        )
        _draw_panel(
            ax_est,
            dyn_est,
            f"Inferred VF [a,b]=[{float(theta_est[0]):.3f}, {float(theta_est[1]):.3f}]",
            i,
        )

        fig.suptitle(
            f"{metadata.get('exp_id', 'unknown')} seed={metadata.get('seed', 'n/a')} | step={cur_step}",
            fontsize=12,
            y=0.97,
        )
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    plt.close(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=max(1, int(fps)))
    return output_path


def make_acq_action_video(
    run_dir: Path,
    output_path: Path,
    *,
    stride: int,
    fps: int,
    grid_lim: float,
) -> Path:
    (
        metadata,
        _results_path,
        true_state,
        model_state,
        action,
        policy_cost,
        _param_steps,
        _param_err,
        emb_steps,
        emb_e0,
        emb_e1,
        _info_steps,
        _pz_cols,
    ) = _load_run_artifacts(run_dir)

    acq_trace = _load_acquisition_map_trace(run_dir, metadata)
    if acq_trace is None:
        if str(metadata.get("exp_id", "")).lower() == "random":
            acq_steps = np.asarray([1], dtype=int)
            acq_axis = np.linspace(-float(grid_lim), float(grid_lim), 61, dtype=np.float32)
            acq_maps = np.full((1, 61, 61), 1e-8, dtype=np.float32)
        else:
            raise FileNotFoundError(
                f"Missing acquisition map trace for {run_dir}. "
                "Rerun tracks with --save-acq-map to render acquisition overlays."
            )
    else:
        acq_steps, acq_axis, acq_maps = acq_trace
    acq_maps = np.nan_to_num(acq_maps, nan=1e-12, posinf=1e6, neginf=1e-12)
    acq_maps = np.maximum(acq_maps, 1e-12)
    acq_norm = _make_lognorm([acq_maps[k] for k in range(acq_maps.shape[0])])

    theta_true = np.asarray(metadata.get("embedding_true", [0.0, 0.0]), dtype=float)
    dyn_true = _DuffingDynamics(a=float(theta_true[0]), b=float(theta_true[1]))
    idxs = _frame_indices(true_state.shape[0], stride)
    extent = [
        float(np.min(acq_axis)),
        float(np.max(acq_axis)),
        float(np.min(acq_axis)),
        float(np.max(acq_axis)),
    ]

    fig = plt.figure(figsize=(15.8, 6.2), dpi=120)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.05], wspace=0.18)
    ax_true = fig.add_subplot(gs[0, 0])
    ax_est = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    sm = plt.cm.ScalarMappable(norm=acq_norm, cmap="inferno")
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Acquisition Objective (log scale)")
    fig.subplots_adjust(left=0.05, right=0.93, bottom=0.09, top=0.90, wspace=0.18)

    def _decorate_axis(ax, *, title: str) -> None:
        ax.set_xlim(-grid_lim, grid_lim)
        ax.set_ylim(-grid_lim, grid_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_title(title)
        ax.legend(loc="upper right")

    def _draw_true_panel(i: int) -> None:
        ax_true.clear()
        plot_vector_field(dyn_true, ax=ax_true, x_range=grid_lim, n_grid=26, is_residual=True, device="cpu")
        ax_true.plot(
            true_state[: i + 1, 0],
            true_state[: i + 1, 1],
            color="black",
            linewidth=1.8,
            label="true traj",
        )
        ax_true.plot(
            model_state[: i + 1, 0],
            model_state[: i + 1, 1],
            color="tab:blue",
            linewidth=1.8,
            label="inferred traj",
        )
        ax_true.scatter([true_state[i, 0]], [true_state[i, 1]], color="black", s=28, zorder=5)
        ax_true.scatter([model_state[i, 0]], [model_state[i, 1]], color="tab:blue", s=28, zorder=5)
        _decorate_axis(
            ax_true,
            title=f"True VF [a,b]=[{float(theta_true[0]):.3f}, {float(theta_true[1]):.3f}]",
        )

    def _draw_acquisition_panel(i: int, acq_map: np.ndarray, theta_est: np.ndarray) -> None:
        ax_est.clear()
        ax_est.imshow(
            acq_map,
            extent=extent,
            origin="lower",
            cmap="inferno",
            norm=acq_norm,
            alpha=0.74,
            interpolation="nearest",
        )
        dyn_est = _DuffingDynamics(a=float(theta_est[0]), b=float(theta_est[1]))
        plot_vector_field(dyn_est, ax=ax_est, x_range=grid_lim, n_grid=26, is_residual=True, device="cpu")
        ax_est.plot(
            true_state[: i + 1, 0],
            true_state[: i + 1, 1],
            color="black",
            linewidth=1.8,
            label="true traj",
        )
        ax_est.plot(
            model_state[: i + 1, 0],
            model_state[: i + 1, 1],
            color="tab:blue",
            linewidth=1.8,
            label="inferred traj",
        )
        ax_est.scatter([true_state[i, 0]], [true_state[i, 1]], color="black", s=28, zorder=6)
        ax_est.scatter([model_state[i, 0]], [model_state[i, 1]], color="tab:blue", s=28, zorder=6)

        if i < action.shape[0]:
            act = np.asarray(action[i], dtype=float)
            if np.all(np.isfinite(act)):
                act_norm = float(np.linalg.norm(act))
                if act_norm > 1e-12:
                    display_len = min(2.5, 0.45 * act_norm)
                    direction = act / act_norm
                    dx = float(display_len * direction[0])
                    dy = float(display_len * direction[1])
                    ax_est.arrow(
                        float(model_state[i, 0]),
                        float(model_state[i, 1]),
                        dx,
                        dy,
                        color="white",
                        width=0.03,
                        head_width=0.28,
                        length_includes_head=True,
                        alpha=0.95,
                        zorder=7,
                    )
                ax_est.text(
                    0.02,
                    0.02,
                    f"u=({act[0]:.2f}, {act[1]:.2f})  |u|={act_norm:.2f}",
                    transform=ax_est.transAxes,
                    color="white",
                    fontsize=9,
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.45, edgecolor="none"),
                )

        _decorate_axis(
            ax_est,
            title=f"Acq + inferred VF [a,b]=[{float(theta_est[0]):.3f}, {float(theta_est[1]):.3f}]",
        )

    frames: list[np.ndarray] = []
    for i in idxs:
        cur_step = int(i + 1)
        acq_idx = _trace_index(acq_steps, cur_step)
        emb_idx = _trace_index(emb_steps, cur_step)
        acq_map = acq_maps[acq_idx]
        theta_est = np.asarray([float(emb_e0[emb_idx]), float(emb_e1[emb_idx])], dtype=float)

        _draw_true_panel(i)
        _draw_acquisition_panel(i, acq_map=acq_map, theta_est=theta_est)

        action_norm = float(np.linalg.norm(action[i])) if i < action.shape[0] else 0.0
        cost_now = float(policy_cost[i]) if i < policy_cost.shape[0] and np.isfinite(policy_cost[i]) else np.nan
        fig.suptitle(
            (
                f"{metadata.get('exp_id', 'unknown')} seed={metadata.get('seed', 'n/a')} | step={cur_step} | "
                f"|u|={action_norm:.3f} | policy_cost={cost_now:.4f}"
            ),
            fontsize=12,
            y=0.97,
        )
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    plt.close(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=max(1, int(fps)))
    return output_path


def _output_for_kind(base_output: Path, video_kind: str) -> Path:
    if video_kind == "info_maps":
        return base_output.with_name(f"{base_output.stem}_info_maps{base_output.suffix}")
    if video_kind == "traj_vf":
        return base_output.with_name(f"{base_output.stem}_traj_vf{base_output.suffix}")
    if video_kind == "acq_action":
        return base_output.with_name(f"{base_output.stem}_acq_action{base_output.suffix}")
    raise ValueError(f"Unsupported video kind: {video_kind}")


def _render_task(task: dict[str, Any]) -> list[str]:
    run_dir = Path(task["run_dir"])
    base_output = Path(task["output"])

    kinds: list[str]
    if task["video_kind"] == "all":
        kinds = ["info_maps", "traj_vf", "acq_action"]
    else:
        kinds = [str(task["video_kind"])]

    outputs: list[str] = []
    for kind in kinds:
        out = _output_for_kind(base_output, kind)
        if kind == "info_maps":
            saved = make_info_maps_video(
                run_dir=run_dir,
                output_path=out,
                stride=int(task["stride"]),
                fps=int(task["fps"]),
                grid_lim=float(task["grid_lim"]),
                ig_grid=int(task["ig_grid"]),
                ig_dt=float(task["ig_dt"]),
            )
        elif kind == "traj_vf":
            saved = make_traj_vf_video(
                run_dir=run_dir,
                output_path=out,
                stride=int(task["stride"]),
                fps=int(task["fps"]),
                grid_lim=float(task["grid_lim"]),
            )
        else:
            saved = make_acq_action_video(
                run_dir=run_dir,
                output_path=out,
                stride=int(task["stride"]),
                fps=int(task["fps"]),
                grid_lim=float(task["grid_lim"]),
            )
        outputs.append(str(saved))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate behavior video(s) for COSYNE runs")
    parser.add_argument("--base-dir", type=str, required=True)
    parser.add_argument("--model-tag", type=str, default="updated")

    # Single-run mode arguments.
    parser.add_argument("--exp-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repeat", type=str, default="repeat_01")
    parser.add_argument("--output", type=str, default=None)

    # Batch mode arguments.
    parser.add_argument("--exp-ids", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--jobs", type=int, default=1)

    parser.add_argument(
        "--video-kind",
        choices=["info_maps", "traj_vf", "acq_action", "all"],
        default="all",
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--grid-lim", type=float, default=10.0)
    parser.add_argument("--ig-grid", type=int, default=121)
    parser.add_argument("--ig-dt", type=float, default=0.01)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    use_batch = args.output_dir is not None or args.exp_ids is not None or args.seeds is not None

    if use_batch:
        exp_ids = _parse_csv_list(args.exp_ids) or list(DEFAULT_EXP_IDS)
        seeds = _parse_csv_ints(args.seeds) or list(DEFAULT_SEEDS)

        output_dir = (
            Path(args.output_dir)
            if args.output_dir is not None
            else (Path(args.base_dir) / "videos" / args.model_tag)
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        tasks: list[dict[str, Any]] = []
        for exp_id in exp_ids:
            for seed in seeds:
                run_dir = (
                    Path(args.base_dir)
                    / "tracks"
                    / args.model_tag
                    / exp_id
                    / f"seed_{seed}"
                    / args.repeat
                )
                out = output_dir / f"{exp_id}_seed_{seed}.mp4"
                tasks.append(
                    {
                        "run_dir": str(run_dir),
                        "output": str(out),
                        "video_kind": str(args.video_kind),
                        "stride": max(1, int(args.stride)),
                        "fps": max(1, int(args.fps)),
                        "grid_lim": float(args.grid_lim),
                        "ig_grid": max(25, int(args.ig_grid)),
                        "ig_dt": float(args.ig_dt),
                    }
                )

        jobs = max(1, int(args.jobs))
        failures = 0
        if jobs == 1:
            for task in tasks:
                try:
                    for path in _render_task(task):
                        print(path)
                except Exception as exc:
                    failures += 1
                    print(f"FAILED {task['run_dir']}: {type(exc).__name__}: {exc}")
        else:
            try:
                with ProcessPoolExecutor(max_workers=jobs) as executor:
                    futures = {executor.submit(_render_task, task): task for task in tasks}
                    for future in as_completed(futures):
                        task = futures[future]
                        try:
                            for path in future.result():
                                print(path)
                        except Exception as exc:
                            failures += 1
                            print(f"FAILED {task['run_dir']}: {type(exc).__name__}: {exc}")
            except Exception:
                for task in tasks:
                    try:
                        for path in _render_task(task):
                            print(path)
                    except Exception as exc:
                        failures += 1
                        print(f"FAILED {task['run_dir']}: {type(exc).__name__}: {exc}")

        return 1 if failures else 0

    if args.exp_id is None or args.seed is None or args.output is None:
        raise SystemExit("Single mode requires --exp-id, --seed, and --output.")

    run_dir = (
        Path(args.base_dir)
        / "tracks"
        / args.model_tag
        / args.exp_id
        / f"seed_{args.seed}"
        / args.repeat
    )
    base_output = Path(args.output)

    task = {
        "run_dir": str(run_dir),
        "output": str(base_output),
        "video_kind": str(args.video_kind),
        "stride": max(1, int(args.stride)),
        "fps": max(1, int(args.fps)),
        "grid_lim": float(args.grid_lim),
        "ig_grid": max(25, int(args.ig_grid)),
        "ig_dt": float(args.ig_dt),
    }
    for path in _render_task(task):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
