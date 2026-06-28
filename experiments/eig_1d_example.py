"""One-dimensional EIG planning example for a nonlinear Poisson model.

The scalar dynamics and observation model are

    z_{k+1} = z_k + sin(z_k * theta)
    y_{k+1} ~ Poisson(exp(c * z_{k+1} + b)).

The candidate planning variable is the initial probe state z_0.  This keeps
the example focused on the information geometry: a real controller would add a
separate map from actions to reachable probe states.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from actdyn.utils.figure_io import load_plotting, save_figure
from actdyn.utils.plotting import apply_manuscript_figure_style, style_manuscript_axis


_C_STROKE = "#3A3A3A"
_C_GRID = "#DDD7CE"
_C_STEP = ("#5DADE2", "#45B8AC", "#F1948A")
_C_EIG = "#DC2626"


def _apply_eig_style(plt_module) -> None:
    apply_manuscript_figure_style(plt_module, font_size=7.4, stroke_color=_C_STROKE)


def compute_eig_curve(
    z_probe: np.ndarray,
    *,
    theta_mean: float,
    theta_var: float,
    c: float,
    b: float,
    state_var: float,
    horizon: int = 1,
    state_noise: float = 0.02,
) -> dict[str, np.ndarray]:
    """Return finite-horizon EIG terms for candidate scalar initial states.

    Args:
        z_probe: Candidate initial states with shape ``(n,)``.
        theta_mean: Current posterior mean of the scalar dynamics parameter.
        theta_var: Current posterior variance of ``theta``.
        c: Scalar log-linear observation loading.
        b: Scalar log-rate bias.
        state_var: Scalar variance of the initial latent state.
        horizon: Number of future observations to score.
        state_noise: Scalar process variance added after each dynamics step.

    Returns:
        Dictionary containing predicted states, sensitivities, and state
        prior variances with shape ``(horizon + 1, n)``, state information with
        shape ``(horizon, n)``, and objective arrays with shape ``(n,)``.
        The EIG uses the 1D specialization of
        ``0.5 log det(I + Sigma_theta I_theta)``.  Sensitivity follows the scalar
        recursion ``S_{k+1}=(1+f_z)S_k + f_theta`` for residual dynamics.  The
        stored covariance path contains the prior variance before each future
        observation; the posterior variance is carried into the next step.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if state_var < 0.0:
        raise ValueError("state_var must be nonnegative.")
    if state_noise < 0.0:
        raise ValueError("state_noise must be nonnegative.")

    z_probe = np.asarray(z_probe, dtype=np.float64)
    z = z_probe.copy()
    sensitivity = np.zeros_like(z)
    state_posterior_variance = np.full_like(z, float(state_var))

    z_path = [z.copy()]
    sensitivity_path = [sensitivity.copy()]
    state_variance_path = [state_posterior_variance.copy()]
    state_information_steps = []
    theta_information_steps = []

    theta_fisher = np.zeros_like(z)
    for step in range(horizon):
        residual_z = theta_mean * np.cos(z * theta_mean)
        residual_theta = z * np.cos(z * theta_mean)
        transition_z = 1.0 + residual_z
        sensitivity = transition_z * sensitivity + residual_theta
        state_prior_variance = (
            transition_z * transition_z * state_posterior_variance + state_noise
        )
        z = z + np.sin(z * theta_mean)

        rate = np.exp(c * z + b)
        state_information = c * c * rate
        attenuated_state_fisher = state_information / (
            1.0 + state_prior_variance * state_information
        )
        step_theta_fisher = sensitivity * sensitivity * attenuated_state_fisher
        theta_fisher += step_theta_fisher
        state_posterior_variance = state_prior_variance / (
            1.0 + state_prior_variance * state_information
        )

        z_path.append(z.copy())
        sensitivity_path.append(sensitivity.copy())
        state_variance_path.append(state_prior_variance.copy())
        state_information_steps.append(state_information)
        theta_information_steps.append(step_theta_fisher)

    eig = 0.5 * np.log1p(theta_var * theta_fisher)
    return {
        "z_probe": z_probe,
        "z_path": np.stack(z_path),
        "sensitivity_path": np.stack(sensitivity_path),
        "state_variance_path": np.stack(state_variance_path),
        "state_information_steps": np.stack(state_information_steps),
        "theta_information_steps": np.stack(theta_information_steps),
        "theta_fisher": theta_fisher,
        "eig": eig,
    }


def build_figure(
    curve: dict[str, np.ndarray],
    *,
    theta_mean: float,
    c: float,
    b: float,
    state_noise: float,
    plt,
):
    """Build the 1D planning figure from arrays returned by ``compute_eig_curve``."""
    z_probe = curve["z_probe"]
    best = int(np.argmax(curve["eig"]))
    best_z = float(z_probe[best])
    horizon = curve["theta_information_steps"].shape[0]

    fig, axes_grid = plt.subplots(2, 3, figsize=(7.25, 4.75), sharex=True)
    axes = axes_grid.ravel()
    colors = [_C_STEP[idx % len(_C_STEP)] for idx in range(horizon)]
    for step in range(horizon):
        linewidth = 2.2 if step == horizon - 1 else 1.2
        alpha = 0.95 if step == horizon - 1 else 0.68
        axes[0].plot(
            z_probe,
            curve["z_path"][step + 1],
            color=colors[step],
            linewidth=linewidth,
            alpha=alpha,
            label=fr"$z_{{{step + 1}}}$",
        )
    axes[0].set_ylabel(r"state $z_k$")
    axes[0].set_title(r"A. rollout $z_{k+1}=z_k+\sin(z_k\hat\theta)$")
    axes[0].legend(loc="upper left", frameon=False, ncol=horizon, fontsize=6.3)

    for step in range(horizon):
        linewidth = 2.2 if step == horizon - 1 else 1.2
        alpha = 0.95 if step == horizon - 1 else 0.68
        axes[1].plot(
            z_probe,
            curve["sensitivity_path"][step + 1],
            color=colors[step],
            linewidth=linewidth,
            alpha=alpha,
        )
    axes[1].axhline(0.0, color=_C_STROKE, linewidth=0.65, alpha=0.48)
    axes[1].set_ylabel(r"sensitivity $S_k$")
    axes[1].set_title(r"B. propagated parameter sensitivity")
    axes[1].set_yscale("symlog", linthresh=1.0)

    for step in range(horizon):
        linewidth = 2.2 if step == horizon - 1 else 1.2
        alpha = 0.95 if step == horizon - 1 else 0.68
        axes[2].plot(
            z_probe,
            curve["state_information_steps"][step],
            color=colors[step],
            linewidth=linewidth,
            alpha=alpha,
        )
    axes[2].set_ylabel(r"state info $I_{z,k}$")
    axes[2].set_title(r"C. Poisson state information")
    axes[2].set_yscale("log")

    for step in range(1, horizon + 1):
        linewidth = 2.2 if step == horizon else 1.2
        alpha = 0.95 if step == horizon else 0.68
        axes[3].plot(
            z_probe,
            np.clip(curve["state_variance_path"][step], 1e-24, None),
            color=colors[step - 1],
            linewidth=linewidth,
            alpha=alpha,
        )
    axes[3].set_ylabel(r"latent prior $P_k^-$")
    axes[3].set_title(r"D. latent uncertainty before observation")
    axes[3].set_yscale("log")

    info_floor = 1e-3
    for step in range(horizon):
        axes[4].plot(
            z_probe,
            np.clip(curve["theta_information_steps"][step], info_floor, None),
            color=colors[step],
            linewidth=1.2,
            alpha=0.68,
        )
    axes[4].plot(
        z_probe,
        np.clip(curve["theta_fisher"], info_floor, None),
        color=_C_STROKE,
        linewidth=2.2,
        label="sum",
    )
    axes[4].set_ylabel(r"param. info")
    axes[4].set_title(r"E. per-step $I_{\theta,k}$ and sum")
    axes[4].set_yscale("log")
    axes[4].set_ylim(bottom=info_floor)
    axes[4].legend(loc="upper left", frameon=False, fontsize=6.3)

    axes[5].plot(z_probe, curve["eig"], color=_C_EIG, linewidth=2.0)
    axes[5].scatter([best_z], [curve["eig"][best]], color=_C_STROKE, s=16, zorder=3)
    axes[5].set_ylabel("multi-step EIG")
    axes[5].set_title(fr"F. 3-step objective, best $z_0={best_z:.2f}$")

    for ax in axes:
        ax.axvline(best_z, color=_C_STROKE, linewidth=0.8, alpha=0.38)
        ax.title.set_fontsize(9.4)
        style_manuscript_axis(ax, grid_color=_C_GRID, grid_alpha=0.35)
    for ax in axes[3:]:
        ax.set_xlabel(r"candidate initial state $z_0$")
    axes[0].text(
        0.02,
        0.05,
        fr"$\hat\theta={theta_mean:g}$, $c={c:g}$, $b={b:g}$, $Q={state_noise:g}$",
        transform=axes[0].transAxes,
        fontsize=6.5,
        color=_C_STROKE,
    )
    fig.tight_layout(w_pad=0.95, h_pad=1.0)
    return fig


def build_detailed_figure(
    curve: dict[str, np.ndarray],
    *,
    theta_mean: float,
    c: float,
    b: float,
    state_noise: float,
    plt,
):
    """Build a detailed two-step diagnostic figure for the scalar EIG example."""
    z_probe = curve["z_probe"]
    best = int(np.argmax(curve["eig"]))
    best_z = float(z_probe[best])
    horizon = curve["theta_information_steps"].shape[0]
    colors = [_C_STEP[idx % len(_C_STEP)] for idx in range(horizon)]

    fig, axes_grid = plt.subplots(2, 4, figsize=(9.5, 4.9), sharex=True)
    axes = axes_grid.ravel()

    axes[0].plot(
        z_probe,
        np.sin(z_probe * theta_mean),
        color=colors[0],
        linewidth=2.0,
    )
    axes[1].plot(
        z_probe,
        np.exp(c * z_probe + b),
        color=colors[0],
        linewidth=2.0,
    )
    for step in range(horizon):
        z_next = curve["z_path"][step + 1]
        linewidth = 2.0 if step == horizon - 1 else 1.25
        axes[2].plot(
            z_probe,
            z_next,
            color=colors[step],
            linewidth=linewidth,
            label=fr"$z_{{{step + 1}}}$",
        )
        axes[3].plot(
            z_probe,
            curve["sensitivity_path"][step + 1],
            color=colors[step],
            linewidth=linewidth,
        )
        axes[4].plot(
            z_probe,
            curve["state_information_steps"][step],
            color=colors[step],
            linewidth=linewidth,
        )
        axes[5].plot(
            z_probe,
            np.clip(curve["state_variance_path"][step + 1], 1e-24, None),
            color=colors[step],
            linewidth=linewidth,
        )
        axes[6].plot(
            z_probe,
            np.clip(curve["theta_information_steps"][step], 1e-3, None),
            color=colors[step],
            linewidth=linewidth,
        )

    axes[0].set_title(r"A. initial residual $f(z_0,\theta)$")
    axes[0].set_ylabel(r"residual $f$")

    axes[1].set_title(r"B. initial observation rate")
    axes[1].set_ylabel(r"$\lambda_0=\exp(cz_0+b)$")

    axes[2].set_title(r"C. two-step state rollout")
    axes[2].set_ylabel(r"state $z_k$")
    axes[2].legend(loc="upper left", frameon=False, fontsize=6.3)

    axes[3].axhline(0.0, color=_C_STROKE, linewidth=0.65, alpha=0.48)
    axes[3].set_title(r"D. sensitivity propagation")
    axes[3].set_ylabel(r"$S_k=\partial z_k/\partial\theta$")
    axes[3].set_yscale("symlog", linthresh=1.0)

    axes[4].set_title(r"E. state Fisher information")
    axes[4].set_ylabel(r"$I_{z,k}=c^2\lambda_k$")
    axes[4].set_yscale("log")

    axes[5].set_title(r"F. latent prior covariance")
    axes[5].set_ylabel(r"$P_k^-$")
    axes[5].set_yscale("log")

    axes[6].plot(
        z_probe,
        np.clip(curve["theta_fisher"], 1e-3, None),
        color=_C_STROKE,
        linewidth=2.2,
        label="sum",
    )
    axes[6].set_title(r"G. parameter information")
    axes[6].set_ylabel(r"$I_{\theta,k}$")
    axes[6].set_yscale("log")
    axes[6].legend(loc="upper left", frameon=False, fontsize=6.3)

    axes[7].plot(z_probe, curve["eig"], color=_C_EIG, linewidth=2.0)
    axes[7].scatter([best_z], [curve["eig"][best]], color=_C_STROKE, s=16, zorder=3)
    axes[7].set_title(fr"H. two-step EIG, best $z_0={best_z:.2f}$")
    axes[7].set_ylabel("EIG")

    for ax in axes:
        ax.axvline(best_z, color=_C_STROKE, linewidth=0.8, alpha=0.38)
        ax.title.set_fontsize(9.3)
        style_manuscript_axis(ax, grid_color=_C_GRID, grid_alpha=0.35)
    for ax in axes[4:]:
        ax.set_xlabel(r"candidate initial state $z_0$")
    axes[0].text(
        0.02,
        0.05,
        fr"$\hat\theta={theta_mean:g}$, $c={c:g}$, $b={b:g}$, $Q={state_noise:g}$",
        transform=axes[0].transAxes,
        fontsize=6.5,
        color=_C_STROKE,
    )
    fig.tight_layout(w_pad=0.95, h_pad=1.0)
    return fig


def main(argv: list[str] | None = None) -> Path:
    """Write the 1D EIG planning figure and return its path."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path("results/eig_1d_example/eig_1d_example.png")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Write the two-step diagnostic plot.",
    )
    parser.add_argument("--theta-mean", type=float, default=2.0)
    parser.add_argument("--theta-var", type=float, default=0.35)
    parser.add_argument("--c", type=float, default=1.4)
    parser.add_argument("--b", type=float, default=-0.4)
    parser.add_argument("--state-var", type=float, default=0.05)
    parser.add_argument("--state-noise", type=float, default=0.02)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--z-min", type=float, default=-2.25)
    parser.add_argument("--z-max", type=float, default=2.25)
    parser.add_argument("--num-points", type=int, default=401)
    args = parser.parse_args(argv)

    if args.detailed:
        args.horizon = 2
        if args.output == default_output:
            args.output = Path("results/eig_1d_example/eig_1d_example_detailed.png")

    z_probe = np.linspace(args.z_min, args.z_max, args.num_points)
    curve = compute_eig_curve(
        z_probe,
        theta_mean=args.theta_mean,
        theta_var=args.theta_var,
        c=args.c,
        b=args.b,
        state_var=args.state_var,
        horizon=args.horizon,
        state_noise=args.state_noise,
    )
    plt = load_plotting(
        args.output,
        apply_style=_apply_eig_style,
        path_is_file=True,
        use_agg=True,
    )
    if plt is None:
        raise RuntimeError("Matplotlib is required to build the EIG example figure.")
    figure_builder = build_detailed_figure if args.detailed else build_figure
    fig = figure_builder(
        curve,
        theta_mean=args.theta_mean,
        c=args.c,
        b=args.b,
        state_noise=args.state_noise,
        plt=plt,
    )
    return save_figure(fig, args.output, plt_module=plt, dpi=300)


if __name__ == "__main__":
    print(main())
