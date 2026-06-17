#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from actdyn.environment.vectorfield import ResidualDynamicsCallable
from actdyn.utils.video import figure_to_rgb_array, write_video_frames
from actdyn.utils.plotting import trace_index
from experiments.experiment_io import (
    expected_loglinear_rate_hz,
    get_environment_preset_from_metadata,
    load_json,
    reconstruct_loglinear_rate_model,
    resolve_artifact_path,
)
from experiments.tbme.run_tbme_experiments import configure_tbme_catalogs

configure_tbme_catalogs()

DEFAULT_ASSET_DIR = REPO_ROOT / "results/tbme/assets"
# Reference commands for the 60 fps vector-field error videos:
# ./.venv/bin/python experiments/tbme/render_exp02_asymmetric_basin_behavior.py \
#     --mode video --policy active_planning_u20_r20_h40 --seed 0 --fps 60 \
#     --stride 1 --inferred-panel error --error-vmax 4.275478363037109 \
#     --output results/tbme/assets/exp02_hard_asymmetric_basin_session4_seed0_active_planning_behavior_vf_error_every_step_60fps.mp4
# ./.venv/bin/python experiments/tbme/render_exp02_asymmetric_basin_behavior.py \
#     --mode video --policy random --seed 0 --fps 60 --stride 1 --planned off \
#     --inferred-panel error --error-vmax 4.275478363037109 \
#     --output results/tbme/assets/exp02_hard_asymmetric_basin_session4_seed0_random_behavior_vf_error_every_step_60fps.mp4
VECTOR_FIELD_GRID_SIZE = 50
ERROR_CMAP = plt.get_cmap("hot_r")
ERROR_CMAP_MAX = 1.0
ERROR_ALPHA_MAX = 0.8


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir.resolve()
    return (
        REPO_ROOT
        / "results/tbme"
        / args.experiment
        / f"session_{int(args.session)}"
        / args.environment
        / args.task
        / args.policy
        / f"seed_{int(args.seed)}"
        / args.repeat
    ).resolve()


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))


def _run_label(run_dir: Path) -> str:
    try:
        seed = run_dir.parents[0].name
        policy = run_dir.parents[1].name
        environment = run_dir.parents[3].name
        session = run_dir.parents[4].name
        return _safe_slug(f"{environment}_{session}_{seed}_{policy}")
    except IndexError:
        return _safe_slug(run_dir.name)


def _fps_label(fps: float) -> str:
    value = float(fps)
    if value.is_integer():
        return f"{int(value)}fps"
    return f"{value:g}fps"


def _default_output(args: argparse.Namespace, run_dir: Path) -> Path:
    label = _run_label(run_dir)
    panel_label = "" if args.inferred_panel == "field" else "_vf_error"
    if args.mode == "frame":
        return (
            DEFAULT_ASSET_DIR / f"{label}_behavior{panel_label}_frame_step{int(args.step):04d}.png"
        )
    stride = int(args.stride)
    stride_label = "every_step" if stride == 1 else f"stride{stride}"
    return (
        DEFAULT_ASSET_DIR
        / f"{label}_behavior{panel_label}_{stride_label}_{_fps_label(args.fps)}.mp4"
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_state_action(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = sorted(_read_csv_rows(path), key=lambda row: int(float(row["step"])))
    steps = np.asarray([int(float(row["step"])) for row in rows], dtype=int)
    true_state = np.asarray(
        [[float(row["true_x"]), float(row["true_v"])] for row in rows], dtype=np.float32
    )
    model_state = np.asarray(
        [[float(row["model_x"]), float(row["model_v"])] for row in rows], dtype=np.float32
    )
    action = np.asarray(
        [[float(row["action_x"]), float(row["action_v"])] for row in rows], dtype=np.float32
    )
    return steps, true_state, model_state, action


def _read_embedding_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(_read_csv_rows(path), key=lambda row: int(float(row["step"])))
    e_cols = sorted(
        (key for key in rows[0] if key.startswith("e") and key[1:].isdigit()),
        key=lambda key: int(key[1:]),
    )
    steps = np.asarray([int(float(row["step"])) for row in rows], dtype=int)
    theta = np.asarray(
        [[float(row[col]) for col in e_cols] for row in rows],
        dtype=np.float32,
    )
    return steps, theta


def _load_planned_trace(run_dir: Path, metadata: dict[str, Any]) -> tuple[np.ndarray, ...] | None:
    planned_path = metadata.get("planned_trajectory_trace_path")
    if planned_path is None and not (run_dir / "planned_trajectory_trace.npz").exists():
        return None
    path = resolve_artifact_path(
        run_dir,
        metadata,
        key="planned_trajectory_trace_path",
        fallback_name="planned_trajectory_trace.npz",
    )
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return (
            np.asarray(data["steps"], dtype=int),
            np.asarray(data["paths"], dtype=np.float32),
            np.asarray(data["lengths"], dtype=int),
        )


def _planned_xy_cycle_for_step(
    trace: tuple[np.ndarray, ...] | None, step: int
) -> np.ndarray | None:
    """Return the full plan cycle active at step from the saved plan trace."""
    if trace is None:
        return None
    steps, paths, lengths = trace
    steps = np.asarray(steps, dtype=int)
    if steps.size == 0:
        return None
    idx = int(np.searchsorted(steps, int(step), side="right") - 1)
    idx = int(np.clip(idx, 0, steps.size - 1))
    while (
        idx > 0
        and steps[idx - 1] == steps[idx] - 1
        and int(lengths[idx - 1]) == int(lengths[idx]) + 1
    ):
        idx -= 1
    n_points = int(lengths[idx])
    if n_points < 2:
        return None
    path = np.asarray(paths[idx, :n_points, :2], dtype=float)
    path = path[np.all(np.isfinite(path), axis=1)]
    return path if path.shape[0] >= 2 else None


def _simulate_spikes(metadata: dict[str, Any], true_state: np.ndarray) -> np.ndarray:
    """Generate reproducible Poisson observations y from the saved latent path.

    The exp02 result stores dense observation means for these runs. When raw
    integer counts are absent, this reconstructs log-linear rates from metadata
    and samples counts with the saved run seed.
    """
    weights, bias, dt = reconstruct_loglinear_rate_model(metadata)
    rate_hz = expected_loglinear_rate_hz(true_state, weights=weights, bias=bias)
    mean_counts = np.clip(rate_hz * float(dt), 1e-6, 1e6)
    rng = np.random.default_rng(int(metadata.get("seed", 0)) + 9173)
    return rng.poisson(mean_counts).astype(np.int16)


def _load_saved_spikes(run_dir: Path, *, seed: int) -> np.ndarray | None:
    """Return spike-count observations from saved rollout observations."""
    rollouts_dirs = [run_dir / "rollouts"]
    rollouts_dirs += sorted(path for path in run_dir.glob("*/rollouts") if path.is_dir())
    rollouts_dirs = [path for path in rollouts_dirs if path.is_dir()]
    if not rollouts_dirs:
        return None
    from actdyn.utils.persistence import load_and_concatenate_rollouts

    rollout = load_and_concatenate_rollouts(str(rollouts_dirs[0]), device="cpu")
    obs = rollout._data.get("next_obs")
    if obs is None:
        return None
    arr = obs.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    if np.allclose(arr, np.rint(arr), atol=1e-5) and float(np.mean(arr == 0.0)) >= 0.02:
        return arr.astype(np.float32, copy=False)
    if np.all(np.isfinite(arr)) and float(np.min(arr)) >= 0.0:
        rng = np.random.default_rng(int(seed) + 9173)
        return rng.poisson(np.clip(arr, 1e-6, 1e6)).astype(np.float32, copy=False)
    return None


def _sort_spikes_by_tuning(metadata: dict[str, Any], spikes: np.ndarray) -> np.ndarray:
    """Sort observation columns by each neuron preferred latent direction.

    The log-linear observation weights have shape (n_neurons, 2). Sorting by
    atan2(weight_v, weight_x) groups neurons with similar directional tuning in
    the spike raster without changing the rollout data used elsewhere.
    """
    weights, _, _ = reconstruct_loglinear_rate_model(metadata)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != spikes.shape[1] or weights.shape[1] < 2:
        return spikes
    order = np.argsort(np.arctan2(weights[:, 1], weights[:, 0]))
    return spikes[:, order]


def _dynamics_from_theta(metadata: dict[str, Any], theta: np.ndarray, *, estimator: bool):
    env_preset = get_environment_preset_from_metadata(metadata)
    return ResidualDynamicsCallable(
        dynamics_type=env_preset.resolved_dynamics_type(estimator=estimator),
        dyn_params=env_preset.params_from_embedding(theta, estimator=estimator),
        dynamics_alpha=float(metadata.get("dynamics_alpha", 1.0)),
        device="cpu",
    )


def _load_run_arrays(run_dir: Path):
    metadata = load_json(run_dir / "run_metadata.json")
    state_action_path = resolve_artifact_path(
        run_dir,
        metadata,
        key="state_action_trace_path",
        fallback_name="state_action_trace.csv",
    )
    embedding_path = resolve_artifact_path(
        run_dir,
        metadata,
        key="embedding_estimate_trace_path",
        fallback_name="embedding_estimate_trace.csv",
    )
    steps, true_state, model_state, action = _read_state_action(state_action_path)
    embedding_steps, embedding = _read_embedding_trace(embedding_path)
    planned_trace = _load_planned_trace(run_dir, metadata)
    spikes = _load_saved_spikes(run_dir, seed=int(metadata.get("seed", 0)))
    if spikes is None:
        spikes = _simulate_spikes(metadata, true_state)
    spikes = _sort_spikes_by_tuning(metadata, spikes)
    return (
        metadata,
        steps,
        true_state,
        model_state,
        action,
        embedding_steps,
        embedding,
        planned_trace,
        spikes,
    )


def _grid_limit(
    metadata: dict[str, Any],
    true_state: np.ndarray,
    model_state: np.ndarray,
    grid_lim: float | None,
) -> float:
    if grid_lim is not None:
        return float(grid_lim)
    state_low = np.asarray(metadata.get("state_low", [-5.0, -5.0]), dtype=np.float32)
    state_high = np.asarray(metadata.get("state_high", [5.0, 5.0]), dtype=np.float32)
    limit = float(max(np.max(np.abs(state_low)), np.max(np.abs(state_high))))
    finite = np.concatenate([true_state[:, :2], model_state[:, :2]], axis=0)
    finite = finite[np.all(np.isfinite(finite), axis=1)]
    if finite.size:
        limit = max(limit, 1.05 * float(np.max(np.abs(finite))))
    return limit


def _vector_grid(dynamics: Any, grid_points: np.ndarray, shape: tuple[int, int]):
    pts = torch.as_tensor(grid_points, dtype=torch.float32)
    with torch.no_grad():
        vel = dynamics(pts).detach().cpu().numpy().reshape(shape[0], shape[1], 2)
    return vel[:, :, 0], vel[:, :, 1]


def _draw_vector_field(
    ax: plt.Axes, xx: np.ndarray, yy: np.ndarray, u: np.ndarray, v: np.ndarray
) -> None:
    ax.streamplot(
        xx,
        yy,
        u,
        v,
        color="#8A8F98",
        density=1.10,
        linewidth=0.43,
        arrowsize=0.55,
        zorder=1,
    )


def _vector_field_error(
    true_u: np.ndarray,
    true_v: np.ndarray,
    inferred_u: np.ndarray,
    inferred_v: np.ndarray,
) -> np.ndarray:
    """Return pointwise L2 vector-field error on the plotting grid."""
    return np.sqrt(
        (np.asarray(inferred_u) - np.asarray(true_u)) ** 2
        + (np.asarray(inferred_v) - np.asarray(true_v)) ** 2
    )


def _draw_vector_error(
    ax: plt.Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    error: np.ndarray,
    *,
    vmax: float,
) -> None:
    error = np.asarray(error, dtype=float)
    if not np.any(np.isfinite(error) & (error > 0.0)):
        return
    vmax = max(float(vmax), 1e-12)
    scaled_error = np.clip(error / vmax, 0.0, 1.0)
    scaled_error[~np.isfinite(error)] = 0.0
    alpha = ERROR_ALPHA_MAX * scaled_error
    rgba = ERROR_CMAP(np.clip(ERROR_CMAP_MAX * scaled_error, 0.0, ERROR_CMAP_MAX))

    rgba[..., 3] = np.clip(alpha, 0.0, ERROR_ALPHA_MAX)
    ax.imshow(
        rgba,
        extent=(float(np.min(xx)), float(np.max(xx)), float(np.min(yy)), float(np.max(yy))),
        origin="lower",
        interpolation="nearest",
        zorder=2,
    )


def _draw_fading_trajectory(
    ax: plt.Axes,
    points: np.ndarray,
    *,
    current_idx: int,
    color: str,
    linewidth: float,
    label: str | None = None,
) -> None:
    """Draw history with alpha 1.0 at the current position and 0.35 after 200 steps."""
    history = np.asarray(points[: current_idx + 1, :2], dtype=np.float32)
    valid = np.all(np.isfinite(history), axis=1)
    history = history[valid]
    if history.shape[0] < 2:
        if history.shape[0] == 1:
            ax.scatter(history[:, 0], history[:, 1], color=color, s=14, zorder=6)
        return
    segments = np.stack([history[:-1], history[1:]], axis=1)
    segment_end = np.arange(history.shape[0] - len(segments), history.shape[0])
    age = history.shape[0] - 1 - segment_end
    alpha = np.maximum(0.35, 1.0 - 0.65 * age / 200.0)
    colors = [to_rgba(color, float(a)) for a in alpha]
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=linewidth, zorder=4))
    if label is not None:
        ax.plot([], [], color=color, linewidth=linewidth, label=label)
    ax.scatter(history[-1:, 0], history[-1:, 1], color=color, s=18, zorder=6)


def _draw_spike_raster(
    ax: plt.Axes,
    spikes: np.ndarray,
    *,
    current_idx: int,
    dt: float,
    window_sec: float,
) -> None:
    half_window = float(window_sec) / 2.0
    half_steps = max(1, int(round(half_window / float(dt))))
    start = max(0, current_idx - half_steps)
    stop = min(spikes.shape[0] - 1, current_idx + half_steps)
    window = spikes[start : stop + 1]
    rel_t = (np.arange(start, stop + 1) - current_idx) * float(dt)
    rows, cols = np.nonzero(window > 0)
    if rows.size:
        counts = window[rows, cols]
        ax.scatter(
            rel_t[rows],
            cols,
            s=1.4 + 0.9 * np.minimum(counts, 3),
            color="black",
            alpha=0.9,
            linewidths=0,
        )
    ax.axvline(0.0, color="#C92828", linewidth=0.75, alpha=0.9)
    ax.set_xlim(-half_window, half_window)
    ax.set_ylim(-0.5, spikes.shape[1] - 0.5)
    ax.set_ylabel("neurons")
    ax.set_xticks([])
    n_neurons = spikes.shape[1]
    mid_neuron = min(n_neurons - 1, max(0, n_neurons // 2 - 1))
    yticks = [0, mid_neuron, n_neurons - 1]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(tick + 1) for tick in yticks])
    ax.tick_params(axis="x", bottom=False, labelbottom=False, length=0)
    ax.tick_params(axis="y", labelsize=6, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_action_arrow(ax: plt.Axes, origin: np.ndarray, action: np.ndarray) -> None:
    origin = np.asarray(origin, dtype=float).reshape(-1)
    action = np.asarray(action, dtype=float).reshape(-1)
    if origin.size < 2 or action.size < 2:
        return
    norm = float(np.linalg.norm(action[:2]))
    if norm <= 1e-12:
        return
    direction = action[:2] / norm
    length = min(0.75, 0.55 * norm)
    ax.arrow(
        float(origin[0]),
        float(origin[1]),
        float(length * direction[0]),
        float(length * direction[1]),
        color="#2CA02C",
        width=0.018,
        head_width=0.16,
        length_includes_head=True,
        alpha=0.95,
        zorder=7,
    )
    ax.plot([], [], color="#2CA02C", linewidth=1.4, label="u")


def _style_phase_axis(ax: plt.Axes, *, title: str, grid_lim: float, show_ylabel: bool) -> None:
    ax.set_xlim(-grid_lim, grid_lim)
    ax.set_ylim(-grid_lim, grid_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.grid(False)
    ax.text(
        0.5,
        -0.055,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def _setup_figure(dpi: int):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(6.0, 4.0), dpi=int(dpi))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.0], hspace=0.08)
    ax_spikes = fig.add_subplot(outer[0, 0])
    bottom = outer[1, 0].subgridspec(1, 2, wspace=0.08)
    ax_true = fig.add_subplot(bottom[0, 0])
    ax_inferred = fig.add_subplot(bottom[0, 1])
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.075, top=0.955)
    return fig, (ax_spikes, ax_true, ax_inferred)


def _draw_planned(planned_trace: tuple[np.ndarray, ...] | None, planned_mode: str) -> bool:
    if planned_mode == "off":
        return False
    if planned_mode == "on" and planned_trace is None:
        print(
            "planned trace requested but no planned_trajectory_trace was found; skipping",
            file=sys.stderr,
        )
    return planned_trace is not None


def _draw_frame(
    axes,
    *,
    metadata: dict[str, Any],
    steps: np.ndarray,
    true_state: np.ndarray,
    model_state: np.ndarray,
    action: np.ndarray,
    planned_trace: tuple[np.ndarray, ...] | None,
    spikes: np.ndarray,
    current_idx: int,
    window_sec: float,
    grid_lim: float,
    xx: np.ndarray,
    yy: np.ndarray,
    true_u: np.ndarray,
    true_v: np.ndarray,
    inferred_u: np.ndarray,
    inferred_v: np.ndarray,
    draw_planned: bool,
    inferred_panel: str,
    error_vmax: float | None,
) -> None:
    ax_spikes, ax_true, ax_inferred = axes
    for ax in axes:
        ax.clear()

    _draw_spike_raster(
        ax_spikes,
        spikes,
        current_idx=current_idx,
        dt=float(metadata.get("dt", 0.01)),
        window_sec=float(window_sec),
    )

    _draw_vector_field(ax_true, xx, yy, true_u, true_v)
    _draw_fading_trajectory(
        ax_true,
        true_state,
        current_idx=current_idx,
        color="black",
        linewidth=1.25,
        label="true z",
    )
    _draw_fading_trajectory(
        ax_true,
        model_state,
        current_idx=current_idx,
        color="#1F77B4",
        linewidth=1.1,
        label="inferred z",
    )
    _style_phase_axis(
        ax_true, title="true vector field", grid_lim=float(grid_lim), show_ylabel=True
    )

    _draw_vector_field(ax_inferred, xx, yy, inferred_u, inferred_v)
    inferred_title = "inferred vector field"
    if inferred_panel == "error":
        error = _vector_field_error(true_u, true_v, inferred_u, inferred_v)
        vmax = float(error_vmax) if error_vmax is not None else float(np.nanmax(error))
        _draw_vector_error(ax_inferred, xx, yy, error, vmax=max(vmax, 1e-12))
        inferred_title = "vector field error"

    _draw_fading_trajectory(
        ax_inferred,
        model_state,
        current_idx=current_idx,
        color="#1F77B4",
        linewidth=1.1,
        label="inferred z",
    )
    if draw_planned:
        planned_xy = _planned_xy_cycle_for_step(planned_trace, int(steps[current_idx]))
        if planned_xy is not None:
            ax_inferred.plot(
                planned_xy[:, 0],
                planned_xy[:, 1],
                color="#D62728",
                linewidth=0.85,
                alpha=0.9,
                label="planned",
                zorder=9,
            )
    _draw_action_arrow(ax_inferred, model_state[current_idx], action[current_idx])
    _style_phase_axis(
        ax_inferred, title=inferred_title, grid_lim=float(grid_lim), show_ylabel=False
    )

    for ax in (ax_true, ax_inferred):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper right",
                fontsize=5.5,
                frameon=False,
                handlelength=1.4,
                borderaxespad=0.2,
            )


def _grid_for_run(
    metadata: dict[str, Any],
    true_state: np.ndarray,
    model_state: np.ndarray,
    grid_lim: float | None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    limit = _grid_limit(metadata, true_state, model_state, grid_lim)
    axis = np.linspace(-limit, limit, VECTOR_FIELD_GRID_SIZE, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    grid_points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    return limit, xx, yy, grid_points


def _true_vector_field(metadata: dict[str, Any], grid_points: np.ndarray, shape: tuple[int, int]):
    theta_true = np.asarray(metadata.get("embedding_true", []), dtype=np.float32)
    if theta_true.size == 0:
        theta_true = np.asarray(metadata.get("true_params_full", []), dtype=np.float32)
    true_dynamics = _dynamics_from_theta(metadata, theta_true, estimator=False)
    return _vector_grid(true_dynamics, grid_points, shape)


def _frame_indices(n_steps: int, stride: int) -> list[int]:
    if int(stride) < 1:
        raise ValueError("stride must be >= 1")
    indices = list(range(0, int(n_steps), int(stride)))
    if indices[-1] != int(n_steps) - 1:
        indices.append(int(n_steps) - 1)
    return indices


def _validate_error_vmax(error_vmax: float | None) -> float | None:
    if error_vmax is None:
        return None
    value = float(error_vmax)
    if value <= 0.0:
        raise ValueError("error_vmax must be positive")
    return value


def _validate_error_vmax_percentile(percentile: float) -> float:
    value = float(percentile)
    if not (0.0 < value <= 100.0):
        raise ValueError("error_vmax_percentile must be in (0, 100]")
    return value


def _error_vmax_from_values(error: np.ndarray, percentile: float) -> float:
    values = np.asarray(error, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-12
    if float(percentile) >= 100.0:
        vmax = float(np.nanmax(values))
    else:
        vmax = float(np.nanpercentile(values, float(percentile)))
    return max(vmax, 1e-12)


def _error_scale_mask(
    xx: np.ndarray, yy: np.ndarray, error_scale_radius: float | None
) -> np.ndarray | None:
    if error_scale_radius is None or float(error_scale_radius) <= 0.0:
        return None
    radius = float(error_scale_radius)
    mask = (np.abs(xx) <= radius) & (np.abs(yy) <= radius)
    return mask if np.any(mask) else None


def _error_vmax_for_indices(
    indices: list[int],
    steps: np.ndarray,
    true_u: np.ndarray,
    true_v: np.ndarray,
    inferred_grid,
    percentile: float,
    scale_mask: np.ndarray | None,
) -> float:
    errors = []
    for idx in indices:
        inferred_u, inferred_v = inferred_grid(int(steps[idx]))
        error = _vector_field_error(true_u, true_v, inferred_u, inferred_v)
        if scale_mask is not None:
            error = error[scale_mask]
        values = error[np.isfinite(error)]
        if values.size:
            errors.append(values.reshape(-1))
    if not errors:
        return 1e-12
    return _error_vmax_from_values(np.concatenate(errors), percentile)


def render_frame(
    run_dir: Path,
    output_path: Path,
    *,
    step: int,
    window_sec: float,
    grid_lim: float | None,
    dpi: int,
    planned_mode: str,
    inferred_panel: str,
    error_vmax: float | None,
    error_vmax_percentile: float,
    error_scale_radius: float | None,
) -> Path:
    """Render one behavior frame with the same layout used by the video CLI."""
    (
        metadata,
        steps,
        true_state,
        model_state,
        action,
        embedding_steps,
        embedding,
        planned_trace,
        spikes,
    ) = _load_run_arrays(run_dir)

    current_idx = int(np.searchsorted(steps, int(step), side="left"))
    current_idx = int(np.clip(current_idx, 0, len(steps) - 1))
    limit, xx, yy, grid_points = _grid_for_run(metadata, true_state, model_state, grid_lim)
    true_u, true_v = _true_vector_field(metadata, grid_points, xx.shape)

    emb_idx = trace_index(embedding_steps, int(steps[current_idx]))
    theta_est = np.asarray(embedding[emb_idx], dtype=np.float32)
    inferred_dynamics = _dynamics_from_theta(metadata, theta_est, estimator=True)
    inferred_u, inferred_v = _vector_grid(inferred_dynamics, grid_points, xx.shape)
    panel_error_vmax = _validate_error_vmax(error_vmax)
    panel_error_vmax_percentile = _validate_error_vmax_percentile(error_vmax_percentile)
    error_scale = _error_scale_mask(xx, yy, error_scale_radius)
    if inferred_panel == "error" and panel_error_vmax is None:
        error = _vector_field_error(true_u, true_v, inferred_u, inferred_v)
        if error_scale is not None:
            error = error[error_scale]
        panel_error_vmax = _error_vmax_from_values(error, panel_error_vmax_percentile)

    fig, axes = _setup_figure(dpi)
    _draw_frame(
        axes,
        metadata=metadata,
        steps=steps,
        true_state=true_state,
        model_state=model_state,
        action=action,
        planned_trace=planned_trace,
        spikes=spikes,
        current_idx=current_idx,
        window_sec=float(window_sec),
        grid_lim=limit,
        xx=xx,
        yy=yy,
        true_u=true_u,
        true_v=true_v,
        inferred_u=inferred_u,
        inferred_v=inferred_v,
        draw_planned=_draw_planned(planned_trace, planned_mode),
        inferred_panel=inferred_panel,
        error_vmax=panel_error_vmax,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def render_video(
    run_dir: Path,
    output_path: Path,
    *,
    fps: float,
    stride: int,
    window_sec: float,
    grid_lim: float | None,
    dpi: int,
    planned_mode: str,
    gif_output: Path | None,
    inferred_panel: str,
    error_vmax: float | None,
    error_vmax_percentile: float,
    error_scale_radius: float | None,
) -> Path:
    """Render one TBME behavior run as an MP4 and optional GIF.

    Inputs are read from run_dir. With stride=1 the animation includes
    every saved state/action sample. Planned trajectories are drawn only when
    the run has a saved plan trace and planned_mode is not off.
    """
    (
        metadata,
        steps,
        true_state,
        model_state,
        action,
        embedding_steps,
        embedding,
        planned_trace,
        spikes,
    ) = _load_run_arrays(run_dir)

    limit, xx, yy, grid_points = _grid_for_run(metadata, true_state, model_state, grid_lim)
    true_u, true_v = _true_vector_field(metadata, grid_points, xx.shape)
    inferred_cache: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}

    def inferred_grid(step: int):
        emb_idx = trace_index(embedding_steps, int(step))
        theta_est = np.asarray(embedding[emb_idx], dtype=np.float32)
        key = tuple(np.round(np.asarray(theta_est, dtype=float), decimals=8))
        if key not in inferred_cache:
            dynamics = _dynamics_from_theta(metadata, theta_est, estimator=True)
            inferred_cache[key] = _vector_grid(dynamics, grid_points, xx.shape)
        return inferred_cache[key]

    fig, axes = _setup_figure(dpi)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indices = _frame_indices(len(steps), int(stride))
    draw_planned = _draw_planned(planned_trace, planned_mode)
    panel_error_vmax = _validate_error_vmax(error_vmax)
    panel_error_vmax_percentile = _validate_error_vmax_percentile(error_vmax_percentile)
    error_scale = _error_scale_mask(xx, yy, error_scale_radius)
    if inferred_panel == "error" and panel_error_vmax is None:
        scale_label = "full grid"
        if error_scale is not None:
            scale_label = f"central radius {float(error_scale_radius):g}"
        print(
            "scanning inferred vector-field error color scale "
            f"(p{panel_error_vmax_percentile:g}, {scale_label})",
            flush=True,
        )
        panel_error_vmax = _error_vmax_for_indices(
            indices,
            steps,
            true_u,
            true_v,
            inferred_grid,
            panel_error_vmax_percentile,
            error_scale,
        )
        print(f"error_vmax={panel_error_vmax:.6g}", flush=True)

    print(
        f"rendering {len(indices)} frames, steps {int(steps[indices[0]])}-"
        f"{int(steps[indices[-1]])}, fps={float(fps):g}",
        flush=True,
    )

    def frames():
        for n, idx in enumerate(indices, start=1):
            inferred_u, inferred_v = inferred_grid(int(steps[idx]))
            _draw_frame(
                axes,
                metadata=metadata,
                steps=steps,
                true_state=true_state,
                model_state=model_state,
                action=action,
                planned_trace=planned_trace,
                spikes=spikes,
                current_idx=idx,
                window_sec=float(window_sec),
                grid_lim=limit,
                xx=xx,
                yy=yy,
                true_u=true_u,
                true_v=true_v,
                inferred_u=inferred_u,
                inferred_v=inferred_v,
                draw_planned=draw_planned,
                inferred_panel=inferred_panel,
                error_vmax=panel_error_vmax,
            )
            yield figure_to_rgb_array(fig)
            if n % 100 == 0 or n == len(indices):
                print(f"frame {n}/{len(indices)} step={int(steps[idx])}", flush=True)

    try:
        write_video_frames(frames(), output_path, fps=float(fps))
    finally:
        plt.close(fig)

    print(f"mp4={output_path}", flush=True)
    print(f"cached inferred vector fields={len(inferred_cache)}", flush=True)

    if gif_output is not None:
        _write_gif_from_mp4(output_path, gif_output, fps=float(fps))
    return output_path


def _write_gif_from_mp4(mp4_path: Path, gif_path: Path, *, fps: float) -> Path:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg = get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"
    palette_filter = (
        f"fps={float(fps):g},split[s0][s1];"
        "[s0]palettegen=max_colors=192[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=5"
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(mp4_path),
            "-vf",
            palette_filter,
            "-loop",
            "0",
            str(gif_path),
        ],
        check=True,
    )
    print(f"gif={gif_path}", flush=True)
    return gif_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render exp02 hard asymmetric-basin behavior frames or videos."
    )
    parser.add_argument("--mode", choices=["frame", "video"], default="frame")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--experiment", default="exp02_hard")
    parser.add_argument("--session", type=int, default=4)
    parser.add_argument("--environment", default="exp02_hard_asymmetric_basin")
    parser.add_argument("--task", default="track")
    parser.add_argument("--policy", default="active_planning_u20_r20_h40")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", default="repeat_01")
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--grid-lim", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--planned", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--inferred-panel", choices=["field", "error"], default="field")
    parser.add_argument("--error-vmax", type=float, default=None)
    parser.add_argument(
        "--error-vmax-percentile",
        type=float,
        default=99.0,
        help="percentile color scale used in error mode when --error-vmax is not set",
    )
    parser.add_argument(
        "--error-scale-radius",
        type=float,
        default=2.0,
        help="central |x|,|v| radius used for automatic error color scaling; <=0 uses the full grid",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gif", action="store_true", help="also write a GIF")
    parser.add_argument("--gif-output", type=Path, default=None, help="GIF path used with --gif")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = _resolve_run_dir(args)
    output = args.output if args.output is not None else _default_output(args, run_dir)
    output = output.resolve()

    if args.mode == "frame":
        render_frame(
            run_dir,
            output,
            step=int(args.step),
            window_sec=float(args.window_sec),
            grid_lim=None if args.grid_lim is None else float(args.grid_lim),
            dpi=300 if args.dpi is None else int(args.dpi),
            planned_mode=str(args.planned),
            inferred_panel=str(args.inferred_panel),
            error_vmax=args.error_vmax,
            error_vmax_percentile=float(args.error_vmax_percentile),
            error_scale_radius=float(args.error_scale_radius),
        )
        print(output)
        return 0

    if args.gif_output is not None and not args.gif:
        raise SystemExit("--gif-output requires --gif")
    gif_output = args.gif_output
    if args.gif and gif_output is None:
        gif_output = output.with_suffix(".gif")
    render_video(
        run_dir,
        output,
        fps=float(args.fps),
        stride=int(args.stride),
        window_sec=float(args.window_sec),
        grid_lim=None if args.grid_lim is None else float(args.grid_lim),
        dpi=150 if args.dpi is None else int(args.dpi),
        planned_mode=str(args.planned),
        gif_output=None if gif_output is None else gif_output.resolve(),
        inferred_panel=str(args.inferred_panel),
        error_vmax=args.error_vmax,
        error_vmax_percentile=float(args.error_vmax_percentile),
        error_scale_radius=float(args.error_scale_radius),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
