#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from actdyn.environment.vectorfield import (
    ResidualDynamicsCallable,
    jacobian_state_torch,
    step_np,
)
from actdyn.utils.experiment_runtime import (
    compute_loglinear_loading_fisher_snr_db,
    shared_loglinear_loading,
)
from actdyn.utils.figure_io import load_plotting, parse_figure_formats, save_figure_formats
from actdyn.utils.plotting import apply_manuscript_figure_style, plot_vector_field
from experiments.experiment_definitions import EnvironmentPreset, get_environment_preset
from experiments.experiment_io import experiment_env_slug, parse_csv_list
from experiments.tbme.run_tbme_experiments import (
    configure_tbme_catalogs,
    shared_tbme_experiment_suites,
)


DEFAULT_ENV_IDS = ("all",)
ENV_ALIASES = {
    "duffing": "tbme_duffing",
    "damped_pendulum": "tbme_damped_pendulum",
    "asymmetric_basin": "tbme_asymmetric_basin",
    "gated_duffing": "tbme_asymmetric_basin",
    "tbme_gated_duffing": "tbme_asymmetric_basin",
}


def _resolve_env_id(raw: str) -> str:
    value = str(raw).strip()
    if not value:
        raise ValueError("empty environment id")
    if value in ENV_ALIASES:
        return ENV_ALIASES[value]
    if value.startswith("gated_duffing_"):
        return f"tbme_asymmetric_basin_{value.removeprefix('gated_duffing_')}"
    if value.startswith("tbme_gated_duffing_"):
        return f"tbme_asymmetric_basin_{value.removeprefix('tbme_gated_duffing_')}"
    if value.startswith("tbme_"):
        return value
    return f"tbme_{value}"


def _all_shared_env_ids() -> list[str]:
    env_ids: list[str] = []
    seen: set[str] = set()
    for suite in shared_tbme_experiment_suites().values():
        env_id = str(suite["env_preset_id"])
        if env_id in seen:
            continue
        seen.add(env_id)
        env_ids.append(env_id)
    return env_ids


def _true_dynamics(env_preset: EnvironmentPreset) -> ResidualDynamicsCallable:
    theta_true = env_preset.true_embedding_vector()
    return ResidualDynamicsCallable(
        dynamics_type=env_preset.resolved_dynamics_type(),
        dyn_params=env_preset.params_from_embedding(theta_true),
        dynamics_alpha=float(env_preset.dynamics_alpha),
        device="cpu",
    )


def _vectorfield_grid(
    dynamics: ResidualDynamicsCallable,
    *,
    plot_lim: float,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-plot_lim, plot_lim, int(n_grid), dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    states = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    with torch.no_grad():
        drift = dynamics(torch.as_tensor(states, dtype=torch.float32)).detach().cpu().numpy()
    u = drift[:, 0].reshape(xx.shape)
    v = drift[:, 1].reshape(yy.shape)
    speed = np.hypot(u, v)
    return xx, yy, u, v, speed


def _simulate_trajectories(
    env_preset: EnvironmentPreset,
    *,
    n_trajectories: int,
    steps: int,
    seed: int,
) -> list[np.ndarray]:
    trajectories: list[np.ndarray] = []
    dyn_params = np.asarray(env_preset.resolved_true_params(), dtype=np.float64)
    for idx in range(int(n_trajectories)):
        rng = np.random.default_rng(int(seed) + idx)
        current = env_preset.sample_initial_state(int(seed) + idx).astype(np.float64)
        states = np.empty((int(steps), current.shape[0]), dtype=np.float64)
        states[0] = current
        action = np.zeros_like(current)
        noise_std = float(np.sqrt(max(0.0, float(env_preset.state_noise)) * float(env_preset.dt)))
        for step in range(1, int(steps)):
            current = step_np(
                env_preset.resolved_dynamics_type(),
                current,
                action,
                dyn_params=dyn_params,
                dt=float(env_preset.dt),
                dynamics_alpha=float(env_preset.dynamics_alpha),
                clip_limit=float(env_preset.resolved_plot_limit()),
            )
            if noise_std > 0.0:
                current = current + rng.normal(scale=noise_std, size=current.shape)
                current = np.clip(
                    current,
                    -float(env_preset.resolved_plot_limit()),
                    float(env_preset.resolved_plot_limit()),
                )
            states[step] = current
        trajectories.append(states)
    return trajectories


def _local_lyapunov_grid(
    env_preset: EnvironmentPreset,
    *,
    plot_lim: float,
    n_grid: int,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute the local one-sided growth coefficient over a 2D state grid."""
    axis = np.linspace(-plot_lim, plot_lim, int(n_grid), dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    states = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    dyn_params = torch.as_tensor(env_preset.resolved_true_params(), dtype=torch.float32)
    values: list[np.ndarray] = []
    for start in range(0, states.shape[0], int(chunk_size)):
        batch = torch.as_tensor(states[start : start + int(chunk_size)], dtype=torch.float32)
        jac = jacobian_state_torch(
            env_preset.resolved_dynamics_type(),
            batch,
            dyn_params,
            dynamics_alpha=float(env_preset.dynamics_alpha),
        )
        jac_np = jac.detach().cpu().numpy()
        sym = 0.5 * (jac_np + np.swapaxes(jac_np, -1, -2))
        values.append(np.linalg.eigvalsh(sym)[:, -1])
    local = np.concatenate(values, axis=0).reshape(xx.shape)
    finite = local[np.isfinite(local)]
    coefficient = float(np.max(finite)) if finite.size else float("nan")
    return xx, yy, local, coefficient


def _finite_time_lyapunov_exponent(
    env_preset: EnvironmentPreset,
    trajectories: Sequence[np.ndarray],
) -> float:
    """Estimate the largest finite-time Lyapunov exponent along trajectories."""
    dt = float(env_preset.dt)
    if dt <= 0.0:
        return float("nan")
    dyn_params = torch.as_tensor(env_preset.resolved_true_params(), dtype=torch.float32)
    exponents: list[float] = []
    for trajectory in trajectories:
        states = np.asarray(trajectory[:-1], dtype=np.float32)
        if states.shape[0] == 0:
            continue
        jac = jacobian_state_torch(
            env_preset.resolved_dynamics_type(),
            torch.as_tensor(states, dtype=torch.float32),
            dyn_params,
            dynamics_alpha=float(env_preset.dynamics_alpha),
        )
        q = np.eye(jac.shape[-1], dtype=np.float64)
        log_diag = np.zeros(jac.shape[-1], dtype=np.float64)
        eye = np.eye(jac.shape[-1], dtype=np.float64)
        for jac_step in jac.detach().cpu().numpy():
            q, r = np.linalg.qr((eye + dt * jac_step.astype(np.float64)) @ q)
            diag = np.clip(np.abs(np.diag(r)), 1e-12, None)
            log_diag += np.log(diag)
        exponents.append(float(np.max(log_diag / (states.shape[0] * dt))))
    return float(np.mean(exponents)) if exponents else float("nan")


def _loading_model(
    env_preset: EnvironmentPreset,
    *,
    snr_trajectories: int,
    snr_trajectory_length: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    weights, bias = shared_loglinear_loading(
        env_preset,
        device="cpu",
        snr_num_trajectories=int(snr_trajectories),
        snr_trajectory_length=int(snr_trajectory_length),
    )
    return weights.cpu().numpy(), bias.cpu().numpy(), float(env_preset.dt)


def _rate_hz(latent: np.ndarray, *, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    latent = np.asarray(latent, dtype=np.float64)
    if latent.ndim == 1:
        latent = latent.reshape(1, -1)
    log_rate = latent @ np.asarray(weights, dtype=np.float64).T + np.asarray(
        bias, dtype=np.float64
    ).reshape(1, -1)
    return np.exp(np.clip(log_rate, -20.0, 20.0))


def _state_information_grid(
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    dt: float,
    plot_lim: float,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute log-det Poisson state Fisher information on a 2D state grid."""
    axis = np.linspace(-plot_lim, plot_lim, int(n_grid), dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    states = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    mean_counts = np.clip(
        _rate_hz(states, weights=weights, bias=bias) * float(dt),
        1e-12,
        1e12,
    )
    info = np.einsum("no,od,oe->nde", mean_counts, weights, weights, optimize=True)
    info = 0.5 * (info + np.swapaxes(info, -1, -2))
    info = info + 1e-9 * np.eye(weights.shape[1], dtype=np.float64)[None, :, :]
    sign, logabsdet = np.linalg.slogdet(info)
    logdet = np.where(sign > 0.0, logabsdet, np.nan).reshape(xx.shape)
    return xx, yy, logdet


def _snr_label(
    env_preset: EnvironmentPreset,
    *,
    snr_trajectories: int,
    snr_trajectory_length: int,
) -> str:
    target = getattr(env_preset, "loading_target_snr_db", None)
    try:
        snr = compute_loglinear_loading_fisher_snr_db(
            env_preset,
            num_trajectories=int(snr_trajectories),
            trajectory_length=int(snr_trajectory_length),
        )
    except Exception as exc:
        if target is None:
            return f"SNR unavailable ({type(exc).__name__})"
        return f"target SNR {float(target):.1f} dB"
    if target is None:
        return f"SNR {snr:.1f} dB"
    return f"SNR {snr:.1f} dB, target {float(target):.1f} dB"


def _finite_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def _plot_env_diagnostics(
    env_preset: EnvironmentPreset,
    *,
    output_dir: Path,
    figure_formats: Sequence[str],
    steps: int,
    n_trajectories: int,
    seed: int,
    n_grid: int,
    snr_trajectories: int,
    snr_trajectory_length: int,
) -> list[Path]:
    slug = experiment_env_slug(env_preset.preset_id)
    output_stem = output_dir / f"{slug}_diagnostics"
    plt_module = load_plotting(output_stem.with_suffix(".pdf"), path_is_file=True)
    if plt_module is None:
        raise RuntimeError("Matplotlib is unavailable")
    apply_manuscript_figure_style(plt_module, font_size=7.4)

    plot_lim = float(env_preset.resolved_plot_limit())
    dynamics = _true_dynamics(env_preset)
    xx, yy, u, v, speed = _vectorfield_grid(dynamics, plot_lim=plot_lim, n_grid=n_grid)
    trajectories = _simulate_trajectories(
        env_preset,
        n_trajectories=n_trajectories,
        steps=steps,
        seed=seed,
    )
    _lx, _ly, lyap, lyap_coeff = _local_lyapunov_grid(
        env_preset,
        plot_lim=plot_lim,
        n_grid=n_grid,
    )
    lyap_exponent = _finite_time_lyapunov_exponent(env_preset, trajectories)
    weights, bias, dt = _loading_model(
        env_preset,
        snr_trajectories=snr_trajectories,
        snr_trajectory_length=snr_trajectory_length,
    )
    _ix, _iy, info = _state_information_grid(
        weights,
        bias,
        dt=dt,
        plot_lim=plot_lim,
        n_grid=n_grid,
    )
    first_traj = trajectories[0]
    mean_counts = np.clip(
        _rate_hz(first_traj, weights=weights, bias=bias) * dt,
        1e-8,
        1e8,
    )
    observations = np.random.default_rng(int(seed)).poisson(mean_counts).astype(np.float32)
    snr_text = _snr_label(
        env_preset,
        snr_trajectories=snr_trajectories,
        snr_trajectory_length=snr_trajectory_length,
    )

    fig = plt_module.figure(figsize=(9.4, 5.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_vector = fig.add_subplot(gs[0, 0])
    ax_lyapunov = fig.add_subplot(gs[0, 1])
    ax_loading = fig.add_subplot(gs[0, 2])
    ax_info = fig.add_subplot(gs[1, 0])
    obs_gs = gs[1, 1:].subgridspec(2, 1, hspace=0.0, height_ratios=[1, 2])
    ax_rate = fig.add_subplot(obs_gs[0, 0])
    ax_spikes = fig.add_subplot(obs_gs[1, 0], sharex=ax_rate)
    axes = [ax_vector, ax_lyapunov, ax_loading, ax_info, ax_rate, ax_spikes]
    title = env_preset.system_label or slug.replace("_", " ").title()
    fig.suptitle(title, y=1.02, fontsize=9.5)

    ax = ax_vector
    plot_vector_field(
        dynamics,
        ax=ax,
        x_range=plot_lim,
        n_grid=n_grid,
        is_residual=True,
        device="cpu",
    )
    for idx, traj in enumerate(trajectories):
        ax.plot(traj[:, 0], traj[:, 1], linewidth=0.8, alpha=0.85, label=f"traj {idx}")
        ax.scatter(traj[0, 0], traj[0, 1], s=8, zorder=3)
    ax.set_title(f"vector field + noisy trajectories (mean speed {float(np.nanmean(speed)):.2f})")
    ax.set_xlim(-plot_lim, plot_lim)
    ax.set_ylim(-plot_lim, plot_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("v")

    ax = ax_lyapunov
    vmin, vmax = _finite_limits(lyap)
    im = ax.imshow(
        lyap,
        origin="lower",
        extent=[-plot_lim, plot_lim, -plot_lim, plot_lim],
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title(f"Lyapunov coeff. {lyap_coeff:.2f}; exponent {lyap_exponent:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("v")

    ax = ax_loading
    ax.axhline(0.0, color="#B0B0B0", linewidth=0.45)
    ax.axvline(0.0, color="#B0B0B0", linewidth=0.45)
    colors = np.linspace(0.15, 0.95, weights.shape[0])
    ax.scatter(weights[:, 0], weights[:, 1], c=colors, cmap="magma", s=14, alpha=0.85)
    for row in weights:
        ax.plot([0.0, row[0]], [0.0, row[1]], color="#606060", alpha=0.25, linewidth=0.35)
    h_norm = float(np.linalg.norm(weights[:, 0]))
    v_norm = float(np.linalg.norm(weights[:, 1]))
    ax.set_title(f"loading vectors (|C_v|/|C_x|={v_norm / max(h_norm, 1e-12):.1f})")
    ax.set_xlabel("horizontal loading")
    ax.set_ylabel("vertical loading")
    ax.set_aspect("equal", adjustable="datalim")

    ax = ax_info
    info_vmin, info_vmax = _finite_limits(info)
    im = ax.imshow(
        info,
        origin="lower",
        extent=[-plot_lim, plot_lim, -plot_lim, plot_lim],
        cmap="plasma",
        vmin=info_vmin,
        vmax=info_vmax,
        interpolation="nearest",
    )
    ax.plot(first_traj[:, 0], first_traj[:, 1], color="white", linewidth=0.7, alpha=0.9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title("Poisson state information")
    ax.set_xlabel("x")
    ax.set_ylabel("v")

    ax = ax_rate
    obs_display = observations[:, : min(observations.shape[1], 24)]
    rate_display = _rate_hz(first_traj, weights=weights, bias=bias)[:, : obs_display.shape[1]]
    time = np.arange(rate_display.shape[0])
    for neuron_idx in range(rate_display.shape[1]):
        ax.plot(time, rate_display[:, neuron_idx], linewidth=0.45, alpha=0.65)
    ax.set_title(f"firing rate, {snr_text}")
    ax.set_ylabel("Hz")
    ax.tick_params(labelbottom=False)

    ax = ax_spikes
    spike_steps, spike_neurons = np.nonzero(obs_display > 0)
    if spike_steps.size:
        bar_steps: list[float] = []
        bar_neurons: list[float] = []
        for step_idx, neuron_idx, count in zip(
            spike_steps,
            spike_neurons,
            obs_display[spike_steps, spike_neurons].astype(int),
        ):
            offsets = [0.0] if count <= 1 else np.linspace(-0.32, 0.32, count)
            bar_steps.extend(float(step_idx) + float(offset) for offset in offsets)
            bar_neurons.extend([float(neuron_idx)] * len(offsets))
        bar_steps_np = np.asarray(bar_steps, dtype=np.float32)
        bar_neurons_np = np.asarray(bar_neurons, dtype=np.float32)
        ax.vlines(
            bar_steps_np,
            bar_neurons_np - 0.36,
            bar_neurons_np + 0.36,
            color="#222222",
            linewidth=0.35,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("neuron")
    ax.set_ylim(-0.5, obs_display.shape[1] - 0.5)

    for ax in axes:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)
    return save_figure_formats(fig, output_stem, figure_formats, plt_module=plt_module)


def generate_diagnostics(
    *,
    env_ids: Sequence[str],
    output_dir: Path,
    figure_formats: Sequence[str],
    steps: int,
    n_trajectories: int,
    seed: int,
    n_grid: int,
    snr_trajectories: int,
    snr_trajectory_length: int,
) -> list[Path]:
    """Generate one diagnostics figure per requested TBME environment."""
    configure_tbme_catalogs(suite_entries={})
    expanded_env_ids = (
        _all_shared_env_ids()
        if any(str(env_id).strip().lower() == "all" for env_id in env_ids)
        else list(env_ids)
    )
    paths: list[Path] = []
    for raw_env_id in expanded_env_ids:
        env_preset = get_environment_preset(_resolve_env_id(raw_env_id))
        paths.extend(
            _plot_env_diagnostics(
                env_preset,
                output_dir=output_dir,
                figure_formats=figure_formats,
                steps=steps,
                n_trajectories=n_trajectories,
                seed=seed,
                n_grid=n_grid,
                snr_trajectories=snr_trajectories,
                snr_trajectory_length=snr_trajectory_length,
            )
        )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate TBME dynamics and observation diagnostics.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--env-ids",
        type=str,
        default=",".join(DEFAULT_ENV_IDS),
        help="Comma-separated environment ids, clean slugs, or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tbme/diagnostics"),
        help="Directory for diagnostic figures.",
    )
    parser.add_argument("--figure-formats", type=str, default=".pdf")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", type=int, default=51)
    parser.add_argument("--snr-trajectories", type=int, default=100)
    parser.add_argument("--snr-trajectory-length", type=int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = generate_diagnostics(
        env_ids=parse_csv_list(args.env_ids),
        output_dir=Path(args.output_dir),
        figure_formats=parse_figure_formats(args.figure_formats),
        steps=int(args.steps),
        n_trajectories=int(args.trajectories),
        seed=int(args.seed),
        n_grid=int(args.grid),
        snr_trajectories=int(args.snr_trajectories),
        snr_trajectory_length=int(args.snr_trajectory_length),
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
