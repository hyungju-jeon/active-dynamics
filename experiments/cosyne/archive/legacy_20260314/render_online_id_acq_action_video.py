from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from cosyne_common import frame_indices
from render_online_id_trajectory_video import (
    align_embedding_trace,
    build_field_grid,
    compute_axis_limits,
    extract_rollout_arrays,
    figure_to_frame,
    precompute_vectorfields,
    resolve_meta_dynamics_device,
    resolve_record,
    resolve_rollout,
    resolve_spec_and_bundle,
    resolve_step_offset,
    write_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an acquisition-map plus action video from a saved online-ID session."
    )
    parser.add_argument(
        "--summary",
        default="/home/hyungju/Desktop/active-dynamics/results/cosyne/metadynamics_online_id/summary.json",
        help="Summary file used to select a record when --record-path/--session-dir are omitted.",
    )
    parser.add_argument("--record-path", default=None)
    parser.add_argument("--session-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--system-bank", default=None)
    parser.add_argument("--system", default=None)
    parser.add_argument(
        "--policy",
        choices=["active_long", "active_short", "active_chunk", "async_windowed_update", "random", "no_policy"],
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--select-best-policy",
        choices=["active_long", "active_short", "active_chunk", "async_windowed_update", "random", "no_policy"],
        default="active_short",
    )
    parser.add_argument("--grid-n", type=int, default=25)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--video-filename", default="acq_action.mp4")
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def extract_action_array(rollout) -> np.ndarray:
    return rollout["action"][0].detach().cpu().numpy()


def load_acquisition_map_trace(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trace_path = record.get("acquisition_map_trace_path")
    if trace_path:
        path = Path(str(trace_path))
    else:
        path = Path(str(record["session_dir"])) / "acquisition_map_trace.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing acquisition map trace for {record.get('session_dir', path.parent)}. "
            "Re-run online ID with acquisition-map saving enabled."
        )
    with np.load(path, allow_pickle=True) as data:
        steps = np.asarray(data["steps"], dtype=int)
        axis = np.asarray(data["axis"], dtype=float)
        maps = np.asarray(data["maps"], dtype=float)
    if maps.ndim != 3 or steps.ndim != 1:
        raise ValueError(f"Invalid acquisition map trace format: {path}")
    if maps.shape[0] != steps.shape[0]:
        raise ValueError(f"Acquisition map count mismatch in: {path}")
    maps = np.nan_to_num(maps, nan=1e-12, posinf=1e6, neginf=1e-12)
    maps = np.maximum(maps, 1e-12)
    return steps, axis, maps


def load_planned_trajectory_trace(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    trace_path = record.get("planned_trajectory_trace_path")
    if trace_path:
        path = Path(str(trace_path))
    else:
        path = Path(str(record["session_dir"])) / "planned_trajectory_trace.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        steps = np.asarray(data["steps"], dtype=int)
        paths = np.asarray(data["paths"], dtype=float)
        lengths = np.asarray(data["lengths"], dtype=int)
    if paths.ndim != 3 or paths.shape[-1] != 2 or steps.ndim != 1 or lengths.ndim != 1:
        raise ValueError(f"Invalid planned trajectory trace format: {path}")
    if paths.shape[0] != steps.shape[0] or lengths.shape[0] != steps.shape[0]:
        raise ValueError(f"Planned trajectory count mismatch in: {path}")
    return steps, paths, lengths


def trace_index(steps: np.ndarray, step: int) -> int:
    idx = int(np.searchsorted(steps, int(step), side="right") - 1)
    return max(0, min(idx, int(steps.shape[0]) - 1))


def planned_xy_for_step(
    trace: tuple[np.ndarray, np.ndarray, np.ndarray] | None, step: int
) -> np.ndarray | None:
    if trace is None:
        return None
    steps, paths, lengths = trace
    idx = trace_index(steps, step)
    n_points = int(lengths[idx])
    if n_points < 2:
        return None
    return np.asarray(paths[idx, :n_points, :], dtype=float)


def overlay_planned_xy(ax, planned_xy: np.ndarray | None) -> None:
    if planned_xy is None or planned_xy.shape[0] < 2:
        return
    ax.plot(
        planned_xy[:, 0],
        planned_xy[:, 1],
        color="#22d3ee",
        lw=2.0,
        linestyle=":",
        alpha=0.95,
        label="planned traj",
        zorder=5,
    )


def make_frame_figure(
    *,
    record: dict,
    step_idx: int,
    actual_step: int,
    action_idx: int,
    env_path: np.ndarray,
    model_path: np.ndarray,
    actions: np.ndarray,
    x_np: np.ndarray,
    y_np: np.ndarray,
    true_field: np.ndarray,
    inferred_field: np.ndarray,
    acq_steps: np.ndarray,
    acq_axis: np.ndarray,
    acq_maps: np.ndarray,
    planned_xy: np.ndarray | None,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14.4, 7.2), dpi=100)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.05], wspace=0.15)
    ax_true = fig.add_subplot(gs[0, 0])
    ax_est = fig.add_subplot(gs[0, 1], sharex=ax_true, sharey=ax_true)
    cax = fig.add_subplot(gs[0, 2])

    cur_step = int(step_idx + 1)
    env_hist = env_path[: cur_step + 1]
    model_hist = model_path[: cur_step + 1]
    acq_idx = trace_index(acq_steps, actual_step)
    acq_map = acq_maps[acq_idx]
    extent = [
        float(np.min(acq_axis)),
        float(np.max(acq_axis)),
        float(np.min(acq_axis)),
        float(np.max(acq_axis)),
    ]
    acq_norm = colors.LogNorm(vmin=float(np.min(acq_maps)), vmax=float(np.max(acq_maps)))

    ax_true.streamplot(
        x_np,
        y_np,
        true_field[..., 0],
        true_field[..., 1],
        color="#6b7280",
        density=1.15,
        linewidth=1.0,
        arrowsize=1.0,
    )
    ax_est.imshow(
        acq_map,
        extent=extent,
        origin="lower",
        cmap="inferno",
        norm=acq_norm,
        alpha=0.74,
        interpolation="nearest",
        zorder=0,
    )
    ax_est.streamplot(
        x_np,
        y_np,
        inferred_field[..., 0],
        inferred_field[..., 1],
        color="white",
        density=1.1,
        linewidth=0.9,
        arrowsize=1.0,
    )

    for ax, title in (
        (ax_true, "True vector field"),
        (ax_est, "Acquisition map + inferred vector field"),
    ):
        overlay_planned_xy(ax, planned_xy)
        ax.plot(env_hist[:, 0], env_hist[:, 1], color="white", lw=3.6, alpha=0.95)
        ax.plot(env_hist[:, 0], env_hist[:, 1], color="black", lw=1.8, label="true trajectory")
        ax.plot(model_hist[:, 0], model_hist[:, 1], color="white", lw=3.0, alpha=0.95)
        ax.plot(
            model_hist[:, 0],
            model_hist[:, 1],
            color="tab:blue",
            lw=1.8,
            linestyle="--",
            label="inferred trajectory",
        )
        ax.scatter(env_hist[-1, 0], env_hist[-1, 1], color="black", s=34, zorder=6)
        ax.scatter(model_hist[-1, 0], model_hist[-1, 1], color="tab:blue", s=34, zorder=6)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_title(title)

    if action_idx < actions.shape[0]:
        act = np.asarray(actions[action_idx], dtype=float)
        act_norm = float(np.linalg.norm(act))
        if np.all(np.isfinite(act)) and act_norm > 1e-12:
            display_len = min(1.8, 0.7 * act_norm)
            direction = act / act_norm
            ax_est.arrow(
                float(model_hist[-1, 0]),
                float(model_hist[-1, 1]),
                float(display_len * direction[0]),
                float(display_len * direction[1]),
                color="white",
                width=0.03,
                head_width=0.24,
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
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="black",
                    alpha=0.45,
                    edgecolor="none",
                ),
            )

    sm = plt.cm.ScalarMappable(norm=acq_norm, cmap="inferno")
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Acquisition objective (log scale)")

    handles, labels = ax_est.get_legend_handles_labels()
    if handles:
        ax_est.legend(handles, labels, loc="upper right")

    fig.suptitle(
        f"{record['system']} | {record['policy']} | seed={int(record['seed'])} | step={actual_step}",
        fontsize=12,
        y=0.98,
    )
    fig.canvas.draw()
    frame = figure_to_frame(fig)
    plt.close(fig)
    return frame


def render_video(record: dict, args: argparse.Namespace) -> Path:
    rollout = resolve_rollout(record)
    env_path, model_path = extract_rollout_arrays(rollout)
    actions = extract_action_array(rollout)
    num_steps = env_path.shape[0] - 1
    step_offset = resolve_step_offset(record, num_steps)
    embedding_trace = align_embedding_trace(record, num_steps, step_offset=step_offset)
    spec, bundle, checkpoint_path = resolve_spec_and_bundle(record, args)
    xlim, ylim = compute_axis_limits(env_path, model_path)
    field_device = resolve_meta_dynamics_device(bundle)
    x_np, y_np, z_grid = build_field_grid(xlim, ylim, args.grid_n, device=field_device)
    frame_indices_list = frame_indices(num_steps, int(args.stride))
    true_field, inferred_fields, _speed_max = precompute_vectorfields(
        spec=spec,
        bundle=bundle,
        embedding_trace=embedding_trace,
        selected_indices=frame_indices_list,
        z_grid=z_grid,
        grid_n=int(args.grid_n),
        dynamics_scale=float(record.get("dynamics_scale", 1.0)),
    )
    acq_steps, acq_axis, acq_maps = load_acquisition_map_trace(record)
    planned_trace = load_planned_trajectory_trace(record)

    if args.output_path is not None:
        out_path = Path(args.output_path)
    else:
        session_dir = Path(record["session_dir"])
        out_path = session_dir / "video" / args.video_filename

    frames: list[np.ndarray] = []
    for frame_idx in frame_indices_list:
        actual_step = int(step_offset + frame_idx + 1)
        planned_step = min(actual_step + 1, int(record.get("n_steps", actual_step)))
        action_idx = min(frame_idx + 1, int(actions.shape[0] - 1))
        frames.append(
            make_frame_figure(
                record=record,
                step_idx=frame_idx,
                actual_step=actual_step,
                action_idx=action_idx,
                env_path=env_path,
                model_path=model_path,
                actions=actions,
                x_np=x_np,
                y_np=y_np,
                true_field=true_field,
                inferred_field=inferred_fields[int(frame_idx)],
                acq_steps=acq_steps,
                acq_axis=acq_axis,
                acq_maps=acq_maps,
                planned_xy=planned_xy_for_step(planned_trace, planned_step),
                xlim=xlim,
                ylim=ylim,
            )
        )

    write_video(frames, out_path, fps=int(args.fps))
    record["checkpoint"] = checkpoint_path
    return out_path


def main() -> None:
    args = parse_args()
    record = resolve_record(args)
    out_path = render_video(record, args)
    print(
        json.dumps(
            {
                "system": record["system"],
                "policy": record["policy"],
                "seed": int(record["seed"]),
                "rollout_path": record.get("rollout_path"),
                "record_path": record.get("record_path"),
                "acquisition_map_trace_path": record.get("acquisition_map_trace_path"),
                "video_path": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
