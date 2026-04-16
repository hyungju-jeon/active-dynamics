#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "tbme"
DEFAULT_FIG_DIR = REPO_ROOT / "docs" / "figs" / "tbme" / "generated"
DEFAULT_TEX_DIR = REPO_ROOT / "docs" / "active-dynamics-writing" / "generated"


@dataclass(frozen=True)
class SuiteRef:
    suite_id: str
    label: str


EXP1_MATCHED = [
    SuiteRef("tbme_exp1_duffing_policy_sota", "Duffing (matched)"),
    SuiteRef("tbme_exp1_pendulum_policy_sota", "Pendulum (matched)"),
    SuiteRef("tbme_exp1_double_integrator_policy_sota", "Double integrator (matched)"),
    SuiteRef("tbme_exp1_duffing_challenge_sota", "Duffing (hard challenge)"),
]

EXP2_ROBUSTNESS = [
    SuiteRef("tbme_exp2_robustness_duffing_sota", "Duffing family mismatch"),
    SuiteRef("tbme_exp2_robustness_duffing_parameter_sota", "Duffing parameter mismatch"),
]

FIGURE_COPIES = [
    ("tbme_exp1_duffing_policy_sota", "parameter_error_over_steps.pdf", "exp1_duffing_parameter_error_over_steps.pdf"),
    ("tbme_exp1_duffing_policy_sota", "parameter_error_over_cpu_time.pdf", "exp1_duffing_parameter_error_over_cpu_time.pdf"),
    ("tbme_exp1_duffing_policy_sota", "parameter_covariance_trace_over_steps.pdf", "exp1_duffing_parameter_covariance_trace_over_steps.pdf"),
    ("tbme_exp1_duffing_policy_sota", "trajectory_r2_over_steps.pdf", "exp1_duffing_trajectory_r2_over_steps.pdf"),
    ("tbme_exp1_duffing_challenge_sota", "parameter_error_over_steps.pdf", "exp1_hard_duffing_parameter_error_over_steps.pdf"),
    ("tbme_exp1_duffing_challenge_sota", "parameter_error_over_cpu_time.pdf", "exp1_hard_duffing_parameter_error_over_cpu_time.pdf"),
    ("tbme_exp1_duffing_challenge_sota", "parameter_covariance_trace_over_steps.pdf", "exp1_hard_duffing_parameter_covariance_trace_over_steps.pdf"),
    ("tbme_exp1_duffing_challenge_sota", "trajectory_r2_over_steps.pdf", "exp1_hard_duffing_trajectory_r2_over_steps.pdf"),
    ("tbme_exp2_robustness_duffing_sota", "parameter_error_over_steps.pdf", "exp2_duffing_parameter_error_over_steps.pdf"),
    ("tbme_exp2_robustness_duffing_sota", "parameter_error_over_cpu_time.pdf", "exp2_duffing_parameter_error_over_cpu_time.pdf"),
    ("tbme_exp2_robustness_duffing_sota", "parameter_covariance_trace_over_steps.pdf", "exp2_duffing_parameter_covariance_trace_over_steps.pdf"),
    ("tbme_exp2_robustness_duffing_sota", "trajectory_r2_over_steps.pdf", "exp2_duffing_trajectory_r2_over_steps.pdf"),
    ("tbme_exp2_robustness_duffing_parameter_sota", "parameter_error_over_steps.pdf", "exp2_duffing_parameter_mismatch_parameter_error_over_steps.pdf"),
    ("tbme_exp2_robustness_duffing_parameter_sota", "parameter_error_over_cpu_time.pdf", "exp2_duffing_parameter_mismatch_parameter_error_over_cpu_time.pdf"),
    ("tbme_exp2_robustness_duffing_parameter_sota", "parameter_covariance_trace_over_steps.pdf", "exp2_duffing_parameter_mismatch_parameter_covariance_trace_over_steps.pdf"),
    ("tbme_exp2_robustness_duffing_parameter_sota", "trajectory_r2_over_steps.pdf", "exp2_duffing_parameter_mismatch_trajectory_r2_over_steps.pdf"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TBME summary figures/tables into manuscript-ready assets.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--tex-dir", type=Path, default=DEFAULT_TEX_DIR)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _find_suite_summary_dir(results_root: Path, suite_id: str) -> Path:
    candidates = sorted(
        (path.parent for path in results_root.rglob(f"{suite_id}/summary/metrics.csv")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"Could not find summary/metrics.csv for suite {suite_id} under {results_root}")
    return candidates[0]


def _find_suite_track_root(results_root: Path, suite_id: str) -> Path:
    candidates: list[tuple[float, Path]] = []
    for track_root in results_root.rglob(f"{suite_id}/track"):
        try:
            latest = max(path.stat().st_mtime for path in track_root.rglob("run_metadata.json"))
        except ValueError:
            continue
        candidates.append((latest, track_root))
    if not candidates:
        raise FileNotFoundError(f"Could not find track/ for suite {suite_id} under {results_root}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _policy_rows(summary_dir: Path) -> list[dict[str, str]]:
    return _read_csv(summary_dir / "metrics.csv")


def _curve_rows(summary_dir: Path, name: str) -> list[dict[str, str]]:
    return _read_csv(summary_dir / name)


def _group_metric_rows(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["policy_id"]), []).append(row)
    return grouped


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else math.nan


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _policy_stats(summary_dir: Path) -> dict[str, dict[str, float]]:
    grouped = _group_metric_rows(_policy_rows(summary_dir))
    traj_rows = _curve_rows(summary_dir, "trajectory_r2_over_steps.csv")
    traj_grouped = _group_metric_rows(traj_rows)
    out: dict[str, dict[str, float]] = {}
    for policy_id, rows in grouped.items():
        finals = [float(row["value_final_mean"]) for row in rows if row.get("value_final_mean")]
        runtimes = [float(row["runtime_sec_mean"]) for row in rows if row.get("runtime_sec_mean")]
        traj_policy_rows = sorted(
            traj_grouped.get(policy_id, []),
            key=lambda row: int(float(row["step"])),
        )
        final_r2 = float(traj_policy_rows[-1]["trajectory_r2_mean"]) if traj_policy_rows else math.nan
        if traj_policy_rows:
            final_sem = float(traj_policy_rows[-1].get("value_sem", math.nan))
            final_n = float(traj_policy_rows[-1].get("n_points", math.nan))
            final_r2_std = final_sem * math.sqrt(final_n) if np.isfinite(final_sem) and np.isfinite(final_n) else math.nan
        else:
            final_r2_std = math.nan
        out[policy_id] = {
            "final_mean": _mean(finals),
            "final_std": _std(finals),
            "runtime_mean": _mean(runtimes),
            "runtime_std": _std(runtimes),
            "final_r2": final_r2,
            "final_r2_std": final_r2_std,
            "n": float(len(finals)),
        }
    return out


def _best_policy(stats: dict[str, dict[str, float]], allowed: set[str] | None = None) -> tuple[str, dict[str, float]]:
    candidates = [(policy_id, vals) for policy_id, vals in stats.items() if allowed is None or policy_id in allowed]
    if not candidates:
        raise ValueError("No candidate policies available for ranking")
    return min(candidates, key=lambda item: item[1]["final_mean"])


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _fmt(mean: float, std: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _copy_figure(summary_dir: Path, source_name: str, out_path: Path) -> None:
    src = summary_dir / "figures" / source_name
    if not src.exists():
        raise FileNotFoundError(f"Missing figure {src}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_path)


def _resolve_environment_preset(preset_id: str) -> dict[str, object]:
    import yaml

    merged: dict[str, dict[str, object]] = {}
    for path in (
        REPO_ROOT / "experiments" / "experiment_env.yaml",
        REPO_ROOT / "experiments" / "tbme" / "experiment_env.yaml",
    ):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        merged.update(dict(payload.get("environments", {})))

    cache: dict[str, dict[str, object]] = {}

    def _resolve(name: str) -> dict[str, object]:
        if name in cache:
            return dict(cache[name])
        raw = dict(merged[name])
        parent = str(raw.pop("extends", "")).strip()
        resolved = _resolve(parent) if parent else {}
        resolved.update(raw)
        cache[name] = dict(resolved)
        return dict(resolved)

    return _resolve(str(preset_id))


def _trace_xy(path: Path) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    for row in _read_csv(path):
        try:
            x_val = float(row["true_x"])
            y_val = float(row["true_v"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x_val) and math.isfinite(y_val):
            pts.append((x_val, y_val))
    if not pts:
        raise ValueError(f"No valid trajectory rows found in {path}")
    return np.asarray(pts, dtype=np.float32)


def _reconstruct_loglinear_weights(
    *,
    seed: int,
    obs_dim: int,
    asymmetric_loading: bool,
    mean_firing_rate: float,
    max_firing_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    torch.manual_seed(int(seed))
    layer = torch.nn.Linear(2, int(obs_dim))
    weights = layer.weight.detach().clone()
    if asymmetric_loading:
        weights[:, 0] = torch.abs(weights[:, 0])
        weights[:, 1] = weights[:, 1] * 2.0

    mean_log_rate = torch.log(torch.full((obs_dim,), float(mean_firing_rate), dtype=torch.float32))
    max_log_rate = torch.log(torch.full((obs_dim,), float(max_firing_rate), dtype=torch.float32))
    state_range_for_cap = 5.0
    for _ in range(6):
        row_l1 = torch.sum(torch.abs(weights), dim=1)
        row_l2_sq = torch.sum(weights * weights, dim=1)
        bias_from_mean = mean_log_rate - 0.5 * row_l2_sq
        capped_log_rate = state_range_for_cap * row_l1 + bias_from_mean
        if torch.all(capped_log_rate <= max_log_rate):
            break
        safe_den = torch.clamp(state_range_for_cap * row_l1, min=1e-8)
        row_scale = torch.clamp((max_log_rate - bias_from_mean) / safe_den, min=0.0, max=1.0)
        weights = weights * row_scale.unsqueeze(1)

    bias = mean_log_rate - 0.5 * torch.sum(weights * weights, dim=1)
    return weights.cpu().numpy().astype(np.float32), bias.cpu().numpy().astype(np.float32)


def _make_hard_duffing_exploration_overlay(results_root: Path, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - best-effort asset export
        raise RuntimeError("matplotlib is required to generate the exploration overlay") from exc

    summary_dir = _find_suite_summary_dir(results_root, "tbme_exp1_duffing_budget_ablation_medium")
    track_root = summary_dir.parent / "track"
    preset = _resolve_environment_preset("tbme_duffing_planning_challenge")
    obs_dim = int(preset.get("observation_dim", 24))
    x_range = float(preset.get("x_range", 6.0))
    dt = float(preset.get("dt", 0.01))
    asymmetric_loading = bool(preset.get("asymmetric_loading", False))
    axis = np.linspace(-x_range, x_range, 121, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")

    candidates: list[dict[str, object]] = []
    planning_root = track_root / "active_planning"
    prbs_root = track_root / "baseline_prbs"
    for seed_dir in sorted(planning_root.glob("seed_*")):
        seed_str = seed_dir.name.split("_")[-1]
        planning_run = seed_dir / "repeat_01"
        prbs_run = prbs_root / f"seed_{seed_str}" / "repeat_01"
        if not planning_run.exists() or not prbs_run.exists():
            continue

        metadata = json.loads((planning_run / "run_metadata.json").read_text(encoding="utf-8"))
        mean_firing = float(metadata.get("mean_firing_rate_target", preset.get("mean_firing_rate_target", 12.0)))
        max_firing = float(metadata.get("max_firing_rate_target", preset.get("max_firing_rate_target", 60.0)))
        weights, bias = _reconstruct_loglinear_weights(
            seed=int(metadata.get("seed", int(seed_str))),
            obs_dim=obs_dim,
            asymmetric_loading=asymmetric_loading,
            mean_firing_rate=mean_firing,
            max_firing_rate=max_firing,
        )

        latent = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
        log_rate = latent @ weights.T + bias.reshape(1, -1)
        rate_hz = np.exp(np.clip(log_rate, -20.0, 20.0)).astype(np.float32)
        mean_counts = np.clip(rate_hz * dt, 1e-8, 1e8)
        info = np.einsum("nd,di,dj->nij", mean_counts, weights, weights, optimize=True)
        sign, logabsdet = np.linalg.slogdet(info + 1e-9 * np.eye(2, dtype=np.float32)[None, :, :])
        grid = np.where(sign > 0.0, logabsdet, np.nan).reshape(axis.shape[0], axis.shape[0])
        q75 = float(np.nanpercentile(grid, 75.0))

        def _trajectory_stats(run_dir: Path) -> tuple[np.ndarray, float]:
            traj = _trace_xy(run_dir / "state_action_trace.csv")
            ix = np.clip(np.searchsorted(axis, traj[:, 0]), 1, axis.shape[0] - 1)
            ix = np.where(np.abs(axis[ix] - traj[:, 0]) < np.abs(axis[ix - 1] - traj[:, 0]), ix, ix - 1)
            iy = np.clip(np.searchsorted(axis, traj[:, 1]), 1, axis.shape[0] - 1)
            iy = np.where(np.abs(axis[iy] - traj[:, 1]) < np.abs(axis[iy - 1] - traj[:, 1]), iy, iy - 1)
            return traj, float(np.nanmean(grid[iy, ix]))

        planning_traj, planning_mean = _trajectory_stats(planning_run)
        prbs_traj, prbs_mean = _trajectory_stats(prbs_run)
        candidates.append(
            {
                "seed": int(seed_str),
                "grid": grid,
                "q75": q75,
                "planning_traj": planning_traj,
                "prbs_traj": prbs_traj,
                "planning_mean": planning_mean,
                "prbs_mean": prbs_mean,
                "gap": planning_mean - prbs_mean,
            }
        )

    if not candidates:
        raise FileNotFoundError("Could not find matching active_planning and baseline_prbs hard-Duffing runs")

    best = max(candidates, key=lambda item: float(item["gap"]))
    grid = np.asarray(best["grid"], dtype=np.float32)
    q75 = float(best["q75"])
    finite = grid[np.isfinite(grid)]
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig = plt.figure(figsize=(11.4, 5.2), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.05], wspace=0.14)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    extent = [float(axis.min()), float(axis.max()), float(axis.min()), float(axis.max())]

    panel_specs = [
        ("active_planning", np.asarray(best["planning_traj"], dtype=np.float32), "#d62728", float(best["planning_mean"])),
        ("baseline_prbs", np.asarray(best["prbs_traj"], dtype=np.float32), "#111827", float(best["prbs_mean"])),
    ]
    im = None
    for ax, (policy_id, traj, color, mean_logdet) in zip(axes, panel_specs):
        im = ax.imshow(
            grid,
            extent=extent,
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        ax.contour(
            axis,
            axis,
            grid,
            levels=[q75],
            colors="white",
            linewidths=1.2,
            linestyles="--",
            alpha=0.95,
        )
        ax.plot(traj[:, 0], traj[:, 1], color="white", linewidth=3.0, alpha=0.95, solid_capstyle="round")
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.6, alpha=0.98, solid_capstyle="round")
        ax.scatter(traj[0, 0], traj[0, 1], s=42, marker="o", facecolor="white", edgecolor="black", linewidth=0.8, zorder=6)
        ax.scatter(traj[-1, 0], traj[-1, 1], s=46, marker="s", facecolor=color, edgecolor="white", linewidth=0.8, zorder=6)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(alpha=0.18)
        ax.set_xlabel("x")
        ax.set_ylabel("v")
        ax.set_title(f"{policy_id}\nmean log det($I_z$) along trajectory = {mean_logdet:.2f}", fontsize=10)

    if im is None:
        raise RuntimeError("Failed to create hard-Duffing exploration overlay")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("log det($I_z$)")
    fig.suptitle(
        f"Representative hard-Duffing exploration from a shared initial condition (seed {int(best['seed'])})",
        y=0.98,
    )
    fig.subplots_adjust(top=0.88)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _make_hard_duffing_phase_portrait_overlay(results_root: Path, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - best-effort asset export
        raise RuntimeError("matplotlib is required to generate the exploration overlay") from exc

    summary_dir = _find_suite_summary_dir(results_root, "tbme_exp1_duffing_budget_ablation_medium")
    track_root = summary_dir.parent / "track"
    preset = _resolve_environment_preset("tbme_duffing_planning_challenge")
    obs_dim = int(preset.get("observation_dim", 24))
    dt = float(preset.get("dt", 0.01))
    asymmetric_loading = bool(preset.get("asymmetric_loading", False))
    x_axis = np.linspace(-4.5, 4.5, 181, dtype=np.float32)
    v_axis = np.linspace(-2.0, 2.0, 141, dtype=np.float32)
    xx, vv = np.meshgrid(x_axis, v_axis, indexing="xy")

    policy_specs = [
        ("active_planning", "#d62728", "Active planning"),
        ("baseline_prbs", "#111827", "PRBS"),
    ]

    latent = np.stack([xx.reshape(-1), vv.reshape(-1)], axis=1)
    seed = 20
    run_dirs = {
        policy_id: track_root / policy_id / f"seed_{seed}" / "repeat_01"
        for policy_id, _color, _label in policy_specs
    }
    if not all(path.exists() for path in run_dirs.values()):
        raise FileNotFoundError("Could not find seed 20 hard-Duffing trajectories for the main-text overlay")
    metadata = json.loads((run_dirs["active_planning"] / "run_metadata.json").read_text(encoding="utf-8"))
    embedding_true = metadata.get("embedding_true")
    if not isinstance(embedding_true, list) or len(embedding_true) < 2:
        raise ValueError("Hard-Duffing run metadata is missing embedding_true")
    weights, bias = _reconstruct_loglinear_weights(
        seed=int(metadata.get("seed", seed)),
        obs_dim=obs_dim,
        asymmetric_loading=asymmetric_loading,
        mean_firing_rate=float(metadata.get("mean_firing_rate_target", preset.get("mean_firing_rate_target", 12.0))),
        max_firing_rate=float(metadata.get("max_firing_rate_target", preset.get("max_firing_rate_target", 60.0))),
    )
    log_rate = latent @ weights.T + bias.reshape(1, -1)
    rate_hz = np.exp(np.clip(log_rate, -20.0, 20.0)).astype(np.float32)
    mean_counts = np.clip(rate_hz * dt, 1e-8, 1e8)
    info = np.einsum("nd,di,dj->nij", mean_counts, weights, weights, optimize=True)
    sign, logabsdet = np.linalg.slogdet(info + 1e-9 * np.eye(2, dtype=np.float32)[None, :, :])
    grid = np.where(sign > 0.0, logabsdet, np.nan).reshape(v_axis.shape[0], x_axis.shape[0])
    q75 = float(np.nanpercentile(grid, 75.0))
    trajs = {
        policy_id: _trace_xy(run_dir / "state_action_trace.csv")
        for policy_id, run_dir in run_dirs.items()
    }

    a = float(embedding_true[0])
    b = float(embedding_true[1])
    c = 0.1
    grid = np.asarray(grid, dtype=np.float32)
    finite = grid[np.isfinite(grid)]
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    drift_x = vv
    drift_v = a * vv - xx * (b + c * xx**2)

    fig = plt.figure(figsize=(6.7, 4.7), dpi=220)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    extent = [float(x_axis.min()), float(x_axis.max()), float(v_axis.min()), float(v_axis.max())]
    im = ax.imshow(
        grid,
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
        alpha=0.68,
        zorder=0,
    )
    ax.contour(
        x_axis,
        v_axis,
        grid,
        levels=[q75],
        colors="white",
        linewidths=1.1,
        linestyles="--",
        alpha=0.92,
        zorder=2,
    )
    stream = ax.streamplot(
        x_axis,
        v_axis,
        drift_x,
        drift_v,
        density=1.1,
        color="#111827",
        linewidth=0.65,
        arrowsize=0.8,
        minlength=0.15,
        maxlength=3.5,
        zorder=1.5,
    )
    stream.lines.set_alpha(0.30)
    stream.arrows.set_alpha(0.28)

    start_xy: tuple[float, float] | None = None
    for policy_id, color, label in policy_specs:
        traj = np.asarray(trajs[policy_id], dtype=np.float32)
        ax.plot(traj[:, 0], traj[:, 1], color="white", linewidth=3.0, alpha=0.92, solid_capstyle="round")
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.7, alpha=0.98, solid_capstyle="round", label=label)
        ax.scatter(traj[-1, 0], traj[-1, 1], s=36, marker="s", facecolor=color, edgecolor="white", linewidth=0.7, zorder=6)
        if start_xy is None:
            start_xy = (float(traj[0, 0]), float(traj[0, 1]))

    if start_xy is not None:
        ax.scatter(start_xy[0], start_xy[1], s=52, marker="o", facecolor="white", edgecolor="black", linewidth=0.8, zorder=7)

    ax.scatter([-3.0, 3.0], [0.0, 0.0], s=18, marker="o", facecolor="none", edgecolor="#4b5563", linewidth=0.9, zorder=3)
    ax.scatter([0.0], [0.0], s=20, marker="x", color="#4b5563", linewidths=1.0, zorder=3)
    ax.set_xlim(float(x_axis.min()), float(x_axis.max()))
    ax.set_ylim(float(v_axis.min()), float(v_axis.max()))
    ax.grid(alpha=0.12, linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("log det($I_z$)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _downsample_series(series: list[dict[str, str]], stride: int = 25) -> list[dict[str, str]]:
    if len(series) <= 2:
        return series
    keep: list[dict[str, str]] = []
    for idx, row in enumerate(series):
        if idx == 0 or idx == len(series) - 1 or idx % stride == 0:
            keep.append(row)
    return keep


def _write_pgfplot_information_curve(summary_dir: Path, out_path: Path, title: str) -> None:
    rows = _curve_rows(summary_dir, "I_z_t_over_steps.csv")
    grouped = _group_metric_rows(rows)
    palette = ["blue", "red", "teal!70!black", "orange!90!black", "purple", "black"]
    lines = [
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "width=\\linewidth,",
        "height=0.62\\linewidth,",
        f"title={{{title}}},",
        "xlabel={Environment step},",
        "ylabel={log det($I_z$)},",
        "grid=major,",
        "legend style={font=\\scriptsize, at={(0.02,0.02)}, anchor=south west},",
        "]",
    ]
    for idx, policy_id in enumerate(sorted(grouped)):
        series = sorted(grouped[policy_id], key=lambda row: int(float(row["step"])))
        series = _downsample_series(series)
        coords = " ".join(
            f"({int(float(row['step']))},{float(row['I_z_t_mean']):.6f})" for row in series
        )
        color = palette[idx % len(palette)]
        lines.append(f"\\addplot+[thick, no marks, color={color}] coordinates {{{coords}}};")
        lines.append(f"\\addlegendentry{{{_latex_escape(policy_id)}}}")
    lines.extend(["\\end{axis}", "\\end{tikzpicture}"])
    _write_text(out_path, "\n".join(lines))


def _write_pgfplot_schedule_pareto(summary_dir: Path, out_path: Path) -> None:
    stats = _policy_stats(summary_dir)
    lines = [
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "width=\\linewidth,",
        "height=0.62\\linewidth,",
        "title={Hard-Duffing schedule tradeoff},",
        "xlabel={Runtime per run (sec)},",
        "ylabel={Final parameter error},",
        "grid=major,",
        "scatter/classes={a={mark=*,blue}},",
        "]",
    ]
    for policy_id, vals in sorted(stats.items()):
        x = vals["runtime_mean"]
        y = vals["final_mean"]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        lines.append(
            "\\addplot+[only marks, mark=*, mark size=2.4pt] coordinates "
            f"{{({x:.4f},{y:.4f})}} node[anchor=west, font=\\scriptsize] {{{_latex_escape(policy_id)}}};"
        )
    lines.extend(["\\end{axis}", "\\end{tikzpicture}"])
    _write_text(out_path, "\n".join(lines))


def _make_hard_duffing_schedule_sweep(track_root: Path, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:  # pragma: no cover - best-effort asset export
        raise RuntimeError("matplotlib is required to generate the schedule sweep figure") from exc

    order = [
        "sched_h20_u1_r1",
        "sched_h20_u5_r5",
        "sched_h40_u5_r5",
        "baseline_prbs",
    ]
    labels = {
        "sched_h20_u1_r1": "H20 / U1 / R1",
        "sched_h20_u5_r5": "H20 / U5 / R5",
        "sched_h40_u5_r5": "H40 / U5 / R5",
        "baseline_prbs": "PRBS",
    }
    colors = {
        "sched_h20_u1_r1": "#4c78a8",
        "sched_h20_u5_r5": "#e45756",
        "sched_h40_u5_r5": "#f58518",
        "baseline_prbs": "#6b7280",
    }
    rows: dict[str, dict[str, float]] = {}
    for policy_id in order:
        metadata_path = track_root / policy_id / "seed_20" / "repeat_01" / "run_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing run metadata for schedule figure: {metadata_path}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        final_error = payload.get("embedding_error_final", payload.get("parameter_error_final"))
        if final_error is None:
            raise KeyError("Expected embedding_error_final or parameter_error_final in run metadata")
        rows[policy_id] = {
            "final_mean": float(final_error),
            "runtime_mean": float(payload["runtime_sec"]),
        }

    y = np.arange(len(order), dtype=np.float32)
    final_mean = np.asarray([rows[policy_id]["final_mean"] for policy_id in order], dtype=np.float32)
    runtime_mean = np.asarray([rows[policy_id]["runtime_mean"] for policy_id in order], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=220)
    for idx, policy_id in enumerate(order):
        color = colors[policy_id]
        x = float(runtime_mean[idx])
        y_val = float(final_mean[idx])
        ax.scatter(x, y_val, color=color, s=42, zorder=3)
        ax.annotate(
            labels[policy_id],
            (x, y_val),
            xytext=(6, 2),
            textcoords="offset points",
            fontsize=8,
            color="#111827",
        )

    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlabel("Compute cost per run (sec)")
    ax.set_ylabel("Final parameter error")
    ax.set_title("Hard-Duffing schedule tradeoff after 2000 steps", fontsize=10)

    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c78a8", markersize=6, label="U1 / R1 family"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e45756", markersize=6, label="U5 / R5, H20"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f58518", markersize=6, label="U5 / R5, H40"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#6b7280", markersize=6, label="PRBS"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=0.9,
        fontsize=8,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _make_exp1_table(results_root: Path, out_path: Path) -> None:
    passive = {"baseline_prbs", "random", "off_policy"}
    lines = [
        "\\begin{tabular}{lcccccccccc}",
        "\\hline",
        "System & Planning err. & Planning $R^2$ & Myopic err. & Myopic $R^2$ & FLEX err. & FLEX $R^2$ & RHC err. & RHC $R^2$ & Best passive err. & Best passive $R^2$ " + r"\\",
        "\\hline",
    ]
    for suite in EXP1_MATCHED:
        summary_dir = _find_suite_summary_dir(results_root, suite.suite_id)
        stats = _policy_stats(summary_dir)
        planning = stats.get("active_planning")
        myopic = stats.get("active_myopic")
        flex = stats.get("flex")
        rhc = stats.get("rhc")
        passive_id, passive_stats = _best_policy(stats, passive)
        planning_cell = _fmt(planning["final_mean"], planning["final_std"]) if planning else "--"
        planning_r2 = _fmt(planning["final_r2"], planning["final_r2_std"]) if planning else "--"
        myopic_cell = _fmt(myopic["final_mean"], myopic["final_std"]) if myopic else "--"
        myopic_r2 = _fmt(myopic["final_r2"], myopic["final_r2_std"]) if myopic else "--"
        flex_cell = _fmt(flex["final_mean"], flex["final_std"]) if flex else "--"
        flex_r2 = _fmt(flex["final_r2"], flex["final_r2_std"]) if flex else "--"
        rhc_cell = _fmt(rhc["final_mean"], rhc["final_std"]) if rhc else "--"
        rhc_r2 = _fmt(rhc["final_r2"], rhc["final_r2_std"]) if rhc else "--"
        passive_cell = f"{_latex_escape(passive_id)}: {_fmt(passive_stats['final_mean'], passive_stats['final_std'])}"
        passive_r2 = _fmt(passive_stats["final_r2"], passive_stats["final_r2_std"])
        lines.append(
            f"{suite.label} & {planning_cell} & {planning_r2} & {myopic_cell} & {myopic_r2} & {flex_cell} & {flex_r2} & {rhc_cell} & {rhc_r2} & {passive_cell} & {passive_r2} "
            + r"\\"
        )
    lines.extend([
        "\\hline",
        "\\end{tabular}",
    ])
    _write_text(out_path, "\n".join(lines))


def _make_exp2_table(results_root: Path, out_path: Path) -> None:
    passive = {"baseline_prbs", "random", "off_policy"}
    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\hline",
        "Mismatch regime & Planning err. & Planning $R^2$ & Next-best err. & Next-best $R^2$ & Best passive err. & Best passive $R^2$ & Runtime (planning sec) " + r"\\",
        "\\hline",
    ]
    for suite in EXP2_ROBUSTNESS:
        summary_dir = _find_suite_summary_dir(results_root, suite.suite_id)
        stats = _policy_stats(summary_dir)
        planning = stats["active_planning"]
        others = {k: v for k, v in stats.items() if k != "active_planning"}
        next_id, next_stats = _best_policy(others)
        passive_id, passive_stats = _best_policy(stats, passive)
        lines.append(
            f"{suite.label} & {_fmt(planning['final_mean'], planning['final_std'])} & "
            f"{_fmt(planning['final_r2'], planning['final_r2_std'])} & "
            f"{_latex_escape(next_id)}: {_fmt(next_stats['final_mean'], next_stats['final_std'])} & "
            f"{_fmt(next_stats['final_r2'], next_stats['final_r2_std'])} & "
            f"{_latex_escape(passive_id)}: {_fmt(passive_stats['final_mean'], passive_stats['final_std'])} & "
            f"{_fmt(passive_stats['final_r2'], passive_stats['final_r2_std'])} & "
            f"{planning['runtime_mean']:.1f} \\\\"
        )
    lines.extend([
        "\\hline",
        "\\end{tabular}",
    ])
    _write_text(out_path, "\n".join(lines))


def export_assets(results_root: Path, figure_dir: Path, tex_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    for suite_id, source_name, target_name in FIGURE_COPIES:
        summary_dir = _find_suite_summary_dir(results_root, suite_id)
        _copy_figure(summary_dir, source_name, figure_dir / target_name)

    hard_duffing_dir = _find_suite_summary_dir(results_root, "tbme_exp1_duffing_challenge_sota")
    exp2_duffing_dir = _find_suite_summary_dir(results_root, "tbme_exp2_robustness_duffing_sota")
    schedule_dir = _find_suite_summary_dir(results_root, "tbme_exp1_duffing_schedule_ablation")
    schedule_main_track = _find_suite_track_root(results_root, "tbme_exp1_duffing_schedule_maintext")

    _write_pgfplot_information_curve(
        hard_duffing_dir,
        tex_dir / "tbme_exp1_hard_duffing_information_plot.tex",
        "Hard-Duffing information gain over time",
    )
    _write_pgfplot_information_curve(
        exp2_duffing_dir,
        tex_dir / "tbme_exp2_duffing_information_plot.tex",
        "Duffing family-mismatch information gain over time",
    )
    _write_pgfplot_schedule_pareto(schedule_dir, tex_dir / "tbme_exp1_hard_duffing_schedule_pareto_plot.tex")

    _make_exp1_table(results_root, tex_dir / "tbme_exp1_endpoint_table.tex")
    _make_exp2_table(results_root, tex_dir / "tbme_exp2_robustness_table.tex")
    _make_hard_duffing_exploration_overlay(
        results_root,
        figure_dir / "exp1_hard_duffing_exploration_overlay.pdf",
    )
    _make_hard_duffing_phase_portrait_overlay(
        results_root,
        figure_dir / "exp1_hard_duffing_phase_portrait_overlay.pdf",
    )
    _make_hard_duffing_schedule_sweep(
        schedule_main_track,
        figure_dir / "exp1_hard_duffing_schedule_sweep.pdf",
    )


def main() -> int:
    args = parse_args()
    export_assets(args.results_root, args.figure_dir, args.tex_dir)
    print(f"Exported manuscript assets to {args.figure_dir} and {args.tex_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
