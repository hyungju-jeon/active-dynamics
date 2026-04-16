#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiment_common import (
        expected_loglinear_rate_hz,
        frame_indices,
        load_json,
        parse_csv_ints,
        reconstruct_loglinear_rate_model,
        resolve_artifact_path,
        resolve_session_root,
    )
    from experiment_specs import get_experiment_spec, list_experiment_ids
    from experiments.cosyne.planar_systems import get_planar_system_spec, residual_torch
else:
    from .experiment_common import (
        expected_loglinear_rate_hz,
        frame_indices,
        load_json,
        parse_csv_ints,
        reconstruct_loglinear_rate_model,
        resolve_artifact_path,
        resolve_session_root,
    )
    from .experiment_specs import get_experiment_spec, list_experiment_ids
    from .cosyne.planar_systems import get_planar_system_spec, residual_torch
from actdyn.utils.video import figure_to_rgb_array, write_video_frames
from actdyn.utils.visualize import (
    PlanarResidualDynamics,
    RbfVectorFieldDynamics,
    annotate_action_arrow,
    decorate_phase_space_axis,
    make_lognorm,
    overlay_planned_xy,
    planned_xy_for_step,
    plot_vector_field,
    trace_index,
)


PASSIVE_POLICIES = {"random", "off_policy", "baseline_prbs", "baseline_random"}


def _load_run_artifacts(
    run_dir: Path,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    metadata = load_json(run_dir / "run_metadata.json")
    sa_path = resolve_artifact_path(
        run_dir, metadata, key="state_action_trace_path", fallback_name="state_action_trace.csv"
    )
    emb_path = resolve_artifact_path(
        run_dir,
        metadata,
        key="embedding_estimate_trace_path",
        fallback_name="embedding_estimate_trace.csv",
    )
    info_path = resolve_artifact_path(
        run_dir, metadata, key="information_trace_path", fallback_name="information_trace.csv"
    )
    sa_df = pd.read_csv(sa_path).sort_values("step")
    emb_df = pd.read_csv(emb_path).sort_values("step")
    info_df = pd.read_csv(info_path).sort_values("step")
    true_state = sa_df[["true_x", "true_v"]].to_numpy(dtype=float)
    model_state = sa_df[["model_x", "model_v"]].to_numpy(dtype=float)
    action = sa_df[["action_x", "action_v"]].to_numpy(dtype=float)
    policy_cost = (
        sa_df["policy_cost"].to_numpy(dtype=float)
        if "policy_cost" in sa_df.columns
        else np.full((true_state.shape[0],), np.nan)
    )
    emb_steps = emb_df["step"].to_numpy(dtype=int)
    e0 = emb_df["e0"].to_numpy(dtype=float)
    e1 = emb_df["e1"].to_numpy(dtype=float)
    pz_cols = np.stack(
        [
            (
                info_df["Pz00"].to_numpy(dtype=float)
                if "Pz00" in info_df.columns
                else np.ones((len(info_df),), dtype=float)
            ),
            (
                info_df["Pz01"].to_numpy(dtype=float)
                if "Pz01" in info_df.columns
                else np.zeros((len(info_df),), dtype=float)
            ),
            (
                info_df["Pz11"].to_numpy(dtype=float)
                if "Pz11" in info_df.columns
                else np.ones((len(info_df),), dtype=float)
            ),
        ],
        axis=1,
    )
    return (
        metadata,
        true_state,
        model_state,
        action,
        policy_cost,
        emb_steps,
        np.stack([e0, e1], axis=1),
        pz_cols,
    )


def _load_npz_trace(
    run_dir: Path, metadata: dict[str, Any], key: str, fallback: str
) -> tuple[np.ndarray, ...] | None:
    path = resolve_artifact_path(run_dir, metadata, key=key, fallback_name=fallback)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        if key == "planned_trajectory_trace_path":
            return (
                np.asarray(data["steps"], dtype=int),
                np.asarray(data["paths"], dtype=float),
                np.asarray(data["lengths"], dtype=int),
            )
        return (
            np.asarray(data["steps"], dtype=int),
            np.asarray(data["axis"], dtype=float),
            np.asarray(data["maps"], dtype=float),
        )


def _load_run_artifacts_rbf(
    run_dir: Path,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    int,
]:
    metadata = load_json(run_dir / "run_metadata.json")
    sa_path = resolve_artifact_path(
        run_dir, metadata, key="state_action_trace_path", fallback_name="state_action_trace.csv"
    )
    sa_df = pd.read_csv(sa_path).sort_values("step")
    true_state = sa_df[["true_x", "true_v"]].to_numpy(dtype=float)
    model_state = sa_df[["model_x", "model_v"]].to_numpy(dtype=float)
    action = sa_df[["action_x", "action_v"]].to_numpy(dtype=float)
    policy_cost = (
        sa_df["policy_cost"].to_numpy(dtype=float)
        if "policy_cost" in sa_df.columns
        else np.full((true_state.shape[0],), np.nan)
    )
    rbf_path = resolve_artifact_path(
        run_dir, metadata, key="rbf_model_trace_path", fallback_name="rbf_model_trace.npz"
    )
    with np.load(rbf_path, allow_pickle=True) as data:
        steps = np.asarray(data["steps"], dtype=int)
        weights = np.asarray(data["weights"], dtype=float)
        centers = np.asarray(data["centers"], dtype=float)
        axis = np.asarray(data["axis"], dtype=float)
        width = float(np.asarray(data["width"]).reshape(-1)[0])
        support_radius = int(np.asarray(data["support_radius"]).reshape(-1)[0])
    return (
        metadata,
        true_state,
        model_state,
        action,
        policy_cost,
        steps,
        weights,
        centers,
        axis,
        width,
        support_radius,
    )


def _system_id_from_metadata(metadata: dict[str, Any]) -> str:
    raw = metadata.get("system_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if metadata.get("hard_setup"):
        return "bistable_attractor"
    return "single_attractor"


def _simulated_spike_and_rate_traces(
    metadata: dict[str, Any],
    true_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights, bias, dt = reconstruct_loglinear_rate_model(metadata)
    rate_hz = expected_loglinear_rate_hz(true_state, weights=weights, bias=bias)
    rate_per_bin = np.clip(rate_hz * dt, 1e-6, 1e6)
    seed = int(metadata.get("seed", 0))
    rng = np.random.default_rng(seed + 9173)
    spike_counts = rng.poisson(rate_per_bin).astype(np.int16)
    return spike_counts, rate_hz


def _load_true_state_trace(run_dir: Path) -> tuple[dict[str, Any], np.ndarray]:
    metadata = load_json(run_dir / "run_metadata.json")
    sa_path = resolve_artifact_path(
        run_dir,
        metadata,
        key="state_action_trace_path",
        fallback_name="state_action_trace.csv",
    )
    sa_df = pd.read_csv(sa_path).sort_values("step")
    true_state = sa_df[["true_x", "true_v"]].to_numpy(dtype=float)
    return metadata, true_state


def render_spike_rate_video(
    run_dir: Path,
    output_path: Path,
    *,
    stride: int,
    fps: int,
    history_window: int,
) -> Path:
    metadata, true_state = _load_true_state_trace(run_dir)
    spike_counts, rate_hz = _simulated_spike_and_rate_traces(metadata, true_state)
    idxs = frame_indices(true_state.shape[0], stride)
    history_window = max(2, int(history_window))
    rate_ylim = float(
        max(
            1.0,
            metadata.get("max_firing_rate_target", 0.0),
            np.nanpercentile(rate_hz, 99.5) * 1.05,
        )
    )

    fig = plt.figure(figsize=(12.0, 12.0), dpi=120)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.22)
    ax_spike = fig.add_subplot(gs[0, 0])
    ax_rate = fig.add_subplot(gs[1, 0], sharex=ax_spike)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.07, top=0.93, hspace=0.22)

    frames: list[np.ndarray] = []
    for i in idxs:
        cur_step = int(i + 1)
        start = max(0, cur_step - history_window)
        window_slice = slice(start, cur_step)
        rel_steps = np.arange(start, cur_step, dtype=int) - cur_step + 1
        spike_window = spike_counts[window_slice]
        rate_window = rate_hz[window_slice]

        ax_spike.clear()
        hit_rows, hit_cols = np.nonzero(spike_window > 0)
        if hit_rows.size > 0:
            ax_spike.scatter(
                rel_steps[hit_rows],
                hit_cols,
                s=14,
                color="black",
                alpha=0.9,
                linewidths=0.0,
            )
        ax_spike.set_xlim(-history_window + 1, 0)
        ax_spike.set_ylim(-0.5, spike_counts.shape[1] - 0.5)
        ax_spike.set_yticks([0, spike_counts.shape[1] // 2, spike_counts.shape[1] - 1])
        ax_spike.set_ylabel("Neuron")
        ax_spike.set_title("Spike Raster")
        ax_spike.grid(axis="x", alpha=0.20)
        ax_spike.grid(axis="y", alpha=0.06)
        ax_spike.axvline(0.0, color="tab:red", linewidth=1.4, alpha=0.9)

        ax_rate.clear()
        ax_rate.plot(rel_steps, rate_window, color="tab:blue", alpha=0.50, linewidth=0.9)
        mean_rate = np.mean(rate_window, axis=1)
        ax_rate.plot(rel_steps, mean_rate, color="black", linewidth=2.8, alpha=0.95)
        ax_rate.set_xlim(-history_window + 1, 0)
        ax_rate.set_ylim(0.0, rate_ylim)
        ax_rate.set_xticks([-history_window + 1, -(history_window // 2), 0])
        ax_rate.set_xlabel("Step Offset")
        ax_rate.set_ylabel("Rate (Hz)")
        ax_rate.set_title("Firing Rates")
        ax_rate.grid(alpha=0.22)
        ax_rate.axvline(0.0, color="tab:red", linewidth=1.4, alpha=0.9)

        mean_rate_now = float(mean_rate[-1]) if mean_rate.size > 0 else float("nan")
        fig.suptitle(
            (
                f"{metadata.get('policy_id', 'unknown')} seed={metadata.get('seed', 'n/a')} "
                f"| step={cur_step} | mean rate={mean_rate_now:.2f} Hz | window={history_window}"
            ),
            fontsize=12,
            y=0.975,
        )
        frames.append(figure_to_rgb_array(fig))

    plt.close(fig)
    write_video_frames(frames, output_path, fps=max(1, int(fps)))
    return output_path


def _build_vectorfield_figure(
    *,
    acq_trace: tuple[np.ndarray, ...] | None,
    show_acq: bool,
):
    if show_acq and acq_trace is not None:
        acq_steps, acq_axis, acq_maps = acq_trace
        acq_maps = np.nan_to_num(acq_maps, nan=1e-12, posinf=1e6, neginf=1e-12)
        acq_maps = np.maximum(acq_maps, 1e-12)
        acq_norm = make_lognorm([acq_maps[k] for k in range(acq_maps.shape[0])])
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
        return fig, ax_true, ax_est, np.asarray(acq_steps, dtype=int), acq_maps, acq_norm, extent

    fig = plt.figure(figsize=(14.2, 6.2), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.18)
    ax_true = fig.add_subplot(gs[0, 0])
    ax_est = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.09, top=0.90, wspace=0.18)
    return (
        fig,
        ax_true,
        ax_est,
        np.asarray([], dtype=int),
        np.zeros((0, 0, 0), dtype=float),
        None,
        None,
    )


def _draw_vectorfield_panel(
    ax: plt.Axes,
    *,
    dynamics,
    true_state: np.ndarray,
    model_state: np.ndarray,
    step_idx: int,
    planned_xy: np.ndarray | None,
    grid_lim: float,
    title: str,
    background: dict[str, Any] | None = None,
    action_xy: np.ndarray | None = None,
) -> None:
    ax.clear()
    if background is not None:
        ax.imshow(
            background["map"],
            extent=background["extent"],
            origin="lower",
            cmap="inferno",
            norm=background["norm"],
            alpha=0.74,
            interpolation="nearest",
        )
    plot_vector_field(dynamics, ax=ax, x_range=grid_lim, n_grid=26, is_residual=True, device="cpu")
    ax.plot(
        true_state[: step_idx + 1, 0],
        true_state[: step_idx + 1, 1],
        color="black",
        linewidth=1.8,
        label="true traj",
    )
    ax.plot(
        model_state[: step_idx + 1, 0],
        model_state[: step_idx + 1, 1],
        color="tab:blue",
        linewidth=1.8,
        label="inferred traj",
    )
    overlay_planned_xy(ax, planned_xy)
    ax.scatter([true_state[step_idx, 0]], [true_state[step_idx, 1]], color="black", s=28, zorder=6)
    ax.scatter(
        [model_state[step_idx, 0]],
        [model_state[step_idx, 1]],
        color="tab:blue",
        s=28,
        zorder=6,
    )
    if action_xy is not None:
        annotate_action_arrow(ax, origin=model_state[step_idx], action=action_xy)
    decorate_phase_space_axis(
        ax,
        xlim=(-grid_lim, grid_lim),
        ylim=(-grid_lim, grid_lim),
        title=title,
    )


def _render_vectorfield_comparison_video(
    *,
    metadata: dict[str, Any],
    output_path: Path,
    stride: int,
    fps: int,
    grid_lim: float,
    true_state: np.ndarray,
    model_state: np.ndarray,
    action: np.ndarray,
    policy_cost: np.ndarray,
    planned_trace: tuple[np.ndarray, ...] | None,
    acq_trace: tuple[np.ndarray, ...] | None,
    dyn_true,
    true_title: str,
    estimate_at_step,
) -> Path:
    show_acq = metadata.get("policy_id") not in PASSIVE_POLICIES and acq_trace is not None
    (
        fig,
        ax_true,
        ax_est,
        acq_steps,
        acq_maps,
        acq_norm,
        extent,
    ) = _build_vectorfield_figure(acq_trace=acq_trace, show_acq=show_acq)
    idxs = frame_indices(true_state.shape[0], stride)
    frames: list[np.ndarray] = []

    for i in idxs:
        cur_step = int(i + 1)
        planned_xy = planned_xy_for_step(planned_trace, cur_step)
        dyn_est, est_title = estimate_at_step(cur_step)
        background = None
        if extent is not None and acq_norm is not None:
            background = {
                "map": acq_maps[trace_index(acq_steps, cur_step)],
                "extent": extent,
                "norm": acq_norm,
            }
        _draw_vectorfield_panel(
            ax_true,
            dynamics=dyn_true,
            true_state=true_state,
            model_state=model_state,
            step_idx=i,
            planned_xy=planned_xy,
            grid_lim=grid_lim,
            title=true_title,
        )
        _draw_vectorfield_panel(
            ax_est,
            dynamics=dyn_est,
            true_state=true_state,
            model_state=model_state,
            step_idx=i,
            planned_xy=planned_xy,
            grid_lim=grid_lim,
            title=est_title,
            background=background,
            action_xy=np.asarray(action[i], dtype=float),
        )
        action_norm = float(np.linalg.norm(action[i])) if i < action.shape[0] else 0.0
        cost_now = (
            float(policy_cost[i])
            if i < policy_cost.shape[0] and np.isfinite(policy_cost[i])
            else np.nan
        )
        fig.suptitle(
            (
                f"{metadata.get('policy_id', 'unknown')} seed={metadata.get('seed', 'n/a')} "
                f"| step={cur_step} | |u|={action_norm:.3f} | policy_cost={cost_now:.4f}"
            ),
            fontsize=12,
            y=0.97,
        )
        frames.append(figure_to_rgb_array(fig))

    plt.close(fig)
    write_video_frames(frames, output_path, fps=max(1, int(fps)))
    return output_path


def _render_rbf_vectorfield_video(
    run_dir: Path,
    output_path: Path,
    *,
    stride: int,
    fps: int,
    grid_lim: float,
) -> Path:
    (
        metadata,
        true_state,
        model_state,
        action,
        policy_cost,
        rbf_steps,
        weight_trace,
        centers,
        axis,
        width,
        support_radius,
    ) = _load_run_artifacts_rbf(run_dir)
    planned_trace = _load_npz_trace(
        run_dir,
        metadata,
        key="planned_trajectory_trace_path",
        fallback="planned_trajectory_trace.npz",
    )
    acq_trace = _load_npz_trace(
        run_dir,
        metadata,
        key="acquisition_map_trace_path",
        fallback="acquisition_map_trace.npz",
    )
    system_id = _system_id_from_metadata(metadata)
    system_spec = get_planar_system_spec(system_id)
    dynamics_alpha = float(metadata.get("dynamics_alpha", 1.0))
    theta_true = np.asarray(metadata.get("embedding_true", [0.0, 0.0]), dtype=float)
    dyn_true = PlanarResidualDynamics(
        system_id=system_spec.system_id,
        embedding=theta_true,
        residual_fn=residual_torch,
        dynamics_alpha=dynamics_alpha,
    )

    def _estimate_at_step(cur_step: int):
        weight_idx = trace_index(rbf_steps, cur_step)
        return (
            RbfVectorFieldDynamics(
                centers=centers,
                axis=axis,
                weights=np.asarray(weight_trace[weight_idx], dtype=float),
                width=float(width),
                support_radius=int(support_radius),
                device="cpu",
            ),
            "Inferred RBF Vector Field",
        )

    return _render_vectorfield_comparison_video(
        metadata=metadata,
        output_path=output_path,
        stride=stride,
        fps=fps,
        grid_lim=grid_lim,
        true_state=true_state,
        model_state=model_state,
        action=action,
        policy_cost=policy_cost,
        planned_trace=planned_trace,
        acq_trace=acq_trace,
        dyn_true=dyn_true,
        true_title=(
            f"True {system_spec.label} VF "
            f"[t0,t1]=[{float(theta_true[0]):.3f}, {float(theta_true[1]):.3f}]"
        ),
        estimate_at_step=_estimate_at_step,
    )


def render_acq_vectorfield_video(
    run_dir: Path,
    output_path: Path,
    *,
    stride: int,
    fps: int,
    grid_lim: float,
) -> Path:
    metadata = load_json(run_dir / "run_metadata.json")
    if metadata.get("exp_id") == "exp03" or metadata.get("rbf_model_trace_path"):
        return _render_rbf_vectorfield_video(
            run_dir,
            output_path,
            stride=stride,
            fps=fps,
            grid_lim=grid_lim,
        )
    metadata, true_state, model_state, action, policy_cost, emb_steps, emb_trace, _pz_cols = (
        _load_run_artifacts(run_dir)
    )
    planned_trace = _load_npz_trace(
        run_dir,
        metadata,
        key="planned_trajectory_trace_path",
        fallback="planned_trajectory_trace.npz",
    )
    acq_trace = _load_npz_trace(
        run_dir,
        metadata,
        key="acquisition_map_trace_path",
        fallback="acquisition_map_trace.npz",
    )
    system_id = _system_id_from_metadata(metadata)
    system_spec = get_planar_system_spec(system_id)
    dynamics_alpha = float(metadata.get("dynamics_alpha", 1.0))
    theta_true = np.asarray(metadata.get("embedding_true", [0.0, 0.0]), dtype=float)
    dyn_true = PlanarResidualDynamics(
        system_id=system_spec.system_id,
        embedding=theta_true,
        residual_fn=residual_torch,
        dynamics_alpha=dynamics_alpha,
    )

    def _estimate_at_step(cur_step: int):
        emb_idx = trace_index(emb_steps, cur_step)
        theta_est = np.asarray(
            [float(emb_trace[emb_idx, 0]), float(emb_trace[emb_idx, 1])],
            dtype=float,
        )
        return (
            PlanarResidualDynamics(
                system_id=system_spec.system_id,
                embedding=theta_est,
                residual_fn=residual_torch,
                dynamics_alpha=dynamics_alpha,
            ),
            (
                f"Inferred {system_spec.label} VF "
                f"[t0,t1]=[{float(theta_est[0]):.3f}, {float(theta_est[1]):.3f}]"
            ),
        )

    return _render_vectorfield_comparison_video(
        metadata=metadata,
        output_path=output_path,
        stride=stride,
        fps=fps,
        grid_lim=grid_lim,
        true_state=true_state,
        model_state=model_state,
        action=action,
        policy_cost=policy_cost,
        planned_trace=planned_trace,
        acq_trace=acq_trace,
        dyn_true=dyn_true,
        true_title=(
            f"True {system_spec.label} VF "
            f"[t0,t1]=[{float(theta_true[0]):.3f}, {float(theta_true[1]):.3f}]"
        ),
        estimate_at_step=_estimate_at_step,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render COSYNE v2 videos")
    parser.add_argument("--base-dir", type=str, default="results/cosyne")
    parser.add_argument("--exp-id", choices=list_experiment_ids(), required=True)
    parser.add_argument("--seeds", type=str, default="0,10,20,30")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--grid-lim", type=float, default=10.0)
    parser.add_argument("--history-window", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    exp_spec = get_experiment_spec(str(args.exp_id))
    if str(exp_spec.experiment_kind) == "realdata":
        print(f"Skipping video rendering for {exp_spec.exp_id}: planar-only renderer")
        return 0
    base_dir = resolve_session_root(Path(args.base_dir), create=False, exp_ids=[exp_spec.exp_id])
    seeds = parse_csv_ints(args.seeds) or [0, 10, 20, 30]
    out_dir = base_dir / exp_spec.exp_id / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    for policy_id in exp_spec.policy_ids:
        for seed in seeds:
            run_root = base_dir / exp_spec.exp_id / "track" / policy_id / f"seed_{seed}"
            if not run_root.exists():
                continue
            for run_dir in sorted(run_root.glob("repeat_*")):
                run_meta = run_dir / "run_metadata.json"
                if not run_meta.exists():
                    continue
                output_path = run_dir / "video" / "acq_vectorfield.mp4"
                saved = render_acq_vectorfield_video(
                    run_dir,
                    output_path,
                    stride=int(args.stride),
                    fps=int(args.fps),
                    grid_lim=float(args.grid_lim),
                )
                shutil.copy2(
                    saved, out_dir / f"{policy_id}_seed_{seed}_{run_dir.name}_acq_vectorfield.mp4"
                )
                spike_rate_path = run_dir / "video" / "spike_rate.mp4"
                spike_saved = render_spike_rate_video(
                    run_dir,
                    spike_rate_path,
                    stride=int(args.stride),
                    fps=int(args.fps),
                    history_window=int(args.history_window),
                )
                shutil.copy2(
                    spike_saved, out_dir / f"{policy_id}_seed_{seed}_{run_dir.name}_spike_rate.mp4"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
