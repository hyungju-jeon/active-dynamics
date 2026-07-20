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
_C_NEUTRAL_LIGHT = "#C8C1B8"
# Candidate colors are drawn from the shared TBME policy palette so this figure
# reads as part of the same asset set; they are inlined to keep the example
# importable without the experiment result machinery.
_C_STEP = ("#5DADE2", "#45B8AC", "#F1948A", "#8E63CE", "#B7791F")
_C_EIG = "#DC2626"
# Yellow reads against the dark end of the afmhot_r ramp where the optimum sits.
_C_EIG_STAR = "#FFD400"
_CANDIDATE_Z = (0.2, 1.5, 3, 4.5)

# Shared manuscript asset font rule: Helvetica, bold 10 pt panel letters,
# 8 pt titles/axis labels, 6 pt tick values.
_ASSET_FONT_STACK = ("Helvetica", "Nimbus Sans", "TeX Gyre Heros", "Arial", "DejaVu Sans")
_ASSET_PANEL_LABEL_SIZE = 10.0
_ASSET_TITLE_SIZE = 8.0
_ASSET_LABEL_SIZE = 8.0
_ASSET_TICK_SIZE = 6.0


def _apply_eig_style(plt_module) -> None:
    apply_manuscript_figure_style(plt_module, font_size=7.4, stroke_color=_C_STROKE)
    plt_module.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(_ASSET_FONT_STACK),
            "mathtext.fontset": "dejavusans",
            "axes.titlesize": _ASSET_TITLE_SIZE,
            "axes.labelsize": _ASSET_LABEL_SIZE,
            "xtick.labelsize": _ASSET_TICK_SIZE,
            "ytick.labelsize": _ASSET_TICK_SIZE,
            "legend.fontsize": _ASSET_TICK_SIZE,
        }
    )


def _panel_label(ax, letter: str) -> None:
    """Set the bold panel letter used by every asset figure."""
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=_ASSET_PANEL_LABEL_SIZE, pad=3.0)


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
    dt: float = 0.1,
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
        state_noise: Scalar process noise power ``Q``; the per-step variance is
            ``Q * dt`` under Euler-Maruyama discretization.
        dt: Sampling interval. The drift is integrated as
            ``z + dt * sin(z * theta)`` and each observation is a Poisson count
            over a bin of width ``dt``.

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
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

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
        residual_z = dt * theta_mean * np.cos(z * theta_mean)
        residual_theta = dt * z * np.cos(z * theta_mean)
        transition_z = 1.0 + residual_z
        sensitivity = transition_z * sensitivity + residual_theta
        state_prior_variance = (
            transition_z * transition_z * state_posterior_variance + state_noise * dt
        )
        z = z + dt * np.sin(z * theta_mean)

        rate = np.exp(c * z + b)
        state_information = c * c * rate * dt
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


def _add_candidate_legend(fig, colors: list[str], *, single_column: bool) -> None:
    """Add the shared candidate-color and line-style key above the panels.

    The per-step versus cumulative distinction is carried here rather than in the
    C and D axis labels, which is where it used to live.
    """
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=color, linewidth=1.2) for color in colors[: len(_CANDIDATE_Z)]
    ]
    labels = [rf"$z_0={candidate_z:g}$" for candidate_z in _CANDIDATE_Z]
    handles += [
        Line2D([0], [0], color=_C_STROKE, linewidth=1.2),
        Line2D([0], [0], color=_C_STROKE, linewidth=0.8, linestyle=":"),
    ]
    labels += ["cumulative", "per-step"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        # One row does not fit across a 3.5 in column.
        ncol=3 if single_column else len(handles),
        fontsize=_ASSET_TICK_SIZE,
        columnspacing=0.9,
        handlelength=1.5,
    )


def build_figure(
    curve: dict[str, np.ndarray],
    *,
    theta_mean: float,
    theta_var: float,
    c: float,
    b: float,
    dt: float,
    plt,
    single_column: bool = False,
):
    """Build the 1D planning figure from arrays returned by ``compute_eig_curve``.

    Args:
        single_column: Lay the panels out for a 3.5 in column instead of the
            7.25 in double-column width used by the other manuscript assets.
    """
    z_probe = curve["z_probe"]
    horizon = curve["theta_information_steps"].shape[0]
    candidate_indices = [
        int(np.argmin(np.abs(z_probe - candidate_z))) for candidate_z in _CANDIDATE_Z
    ]
    candidate_paths = curve["z_path"][:, candidate_indices]
    colors = [_C_STEP[idx % len(_C_STEP)] for idx in range(len(_CANDIDATE_Z))]
    time_info = np.arange(1, horizon + 1)
    eig_by_horizon = 0.5 * np.log1p(
        theta_var * np.cumsum(curve["theta_information_steps"], axis=0),
    )

    # Both widths give the EIG landscape (E) a wide, short cell on the closing
    # row so the panels stay the same shape across the two versions.
    if single_column:
        fig = plt.figure(figsize=(3.5, 4.6))
        grid = fig.add_gridspec(
            3,
            2,
            height_ratios=[1.0, 1.0, 1.15],
            left=0.135,
            right=0.895,
            top=0.9,
            bottom=0.085,
            wspace=0.4,
            hspace=0.55,
        )
        slots = [grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1], grid[2, :]]
    else:
        fig = plt.figure(figsize=(7.25, 3.7))
        grid = fig.add_gridspec(
            2,
            3,
            left=0.055,
            right=0.95,
            top=0.87,
            bottom=0.105,
            wspace=0.3,
            hspace=0.5,
        )
        slots = [grid[0, 0], grid[0, 1], grid[0, 2], grid[1, 0], grid[1, 1:3]]
    # B splits into a rate panel stacked directly on a state panel, sharing one
    # time axis with no gap between them.
    b_grid = slots[1].subgridspec(2, 1, hspace=0.0)
    axis_rate = fig.add_subplot(b_grid[0])
    axes = [
        fig.add_subplot(slots[0]),
        fig.add_subplot(b_grid[1], sharex=axis_rate),
        fig.add_subplot(slots[2]),
        fig.add_subplot(slots[3]),
        fig.add_subplot(slots[4]),
    ]

    panel_a_min = -0.25
    panel_a_max = 2 * np.pi
    z_map = np.linspace(panel_a_min, panel_a_max, z_probe.size)
    dynamics_next = z_map + dt * np.sin(z_map * theta_mean)
    axes[0].plot(z_map, dynamics_next, color=_C_STROKE, linewidth=1.1)
    axes[0].plot(
        [panel_a_min, panel_a_max],
        [panel_a_min, panel_a_max],
        color=_C_NEUTRAL_LIGHT,
        linewidth=0.65,
        linestyle="--",
    )
    for color, label_z, path in zip(
        colors,
        _CANDIDATE_Z,
        candidate_paths.T,
        strict=True,
    ):
        for step in range(horizon):
            z_t = float(path[step])
            z_next = float(path[step + 1])
            axes[0].plot(
                [z_t, z_t, z_next],
                [z_t, z_next, z_next],
                color=color,
                linewidth=0.7,
                linestyle="dashed",
                alpha=0.9,
            )
        axes[0].scatter(
            [path[0]],
            [path[0]],
            color=color,
            s=11,
            edgecolor="white",
            linewidth=0.3,
            zorder=4,
        )
    axes[0].set_xlim(panel_a_min, panel_a_max)
    axes[0].set_ylim(panel_a_min, panel_a_max)
    axes[0].set_xticks([0, 2, 4, 6])
    axes[0].set_yticks([0, 2, 4, 6])
    axes[0].set_xlabel(r"$z_t$")
    axes[0].set_ylabel(r"$z_{t+1}$")
    _panel_label(axes[0], "A")

    # Panel B unrolls each candidate in time: the rate it induces above the state
    # that drives it, on a time base shared with C and D.
    from matplotlib.ticker import MaxNLocator

    time_full = np.arange(horizon + 1)
    for color, path in zip(colors, candidate_paths.T, strict=True):
        axis_rate.plot(time_full, np.exp(c * path + b), color=color, linewidth=0.95)
        axes[1].plot(time_full, path, color=color, linewidth=0.95)
    axis_rate.set_ylim(bottom=0.0)
    axis_rate.set_ylabel(r"Rate $\lambda_t$")
    axis_rate.tick_params(axis="x", labelbottom=False)
    # Drop the bottom rate tick: with no gap it would sit on the state panel's
    # top tick label.
    axis_rate.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="lower"))
    _panel_label(axis_rate, "B")

    axes[1].set_xlim(0.0, float(horizon))
    axes[1].set_xticks(time_full)
    axes[1].set_xlabel(r"Future step $t$")
    axes[1].set_ylabel(r"State $z_t$")

    info_floor = 1e-8
    state_info = curve["state_information_steps"][:, candidate_indices]
    theta_info = curve["theta_information_steps"][:, candidate_indices]

    state_prior = curve["state_variance_path"][1:, candidate_indices]
    state_objective_steps = 0.5 * np.log1p(state_prior * state_info)
    state_objective_cumulative = np.cumsum(state_objective_steps, axis=0)
    for color, step_objective, cumulative_objective in zip(
        colors,
        state_objective_steps.T,
        state_objective_cumulative.T,
        strict=True,
    ):
        axes[2].plot(
            time_info,
            np.clip(step_objective, info_floor, None),
            color=color,
            linewidth=0.65,
            linestyle=":",
        )
        axes[2].plot(
            time_info,
            np.clip(cumulative_objective, info_floor, None),
            color=color,
            linewidth=0.95,
        )
    axes[2].set_xlabel(r"Future step $t$")
    # The expression rides as a title so C and D keep no left margin for a label.
    axes[2].set_title(r"$\frac{1}{2}\log(1+P_t^-I_{z,t})$", loc="center", pad=3.0)
    axes[2].set_yscale("log")
    _panel_label(axes[2], "C")

    eig_steps = 0.5 * np.log1p(theta_var * theta_info)
    eig_cumulative = eig_by_horizon[:, candidate_indices]
    eig_winner = np.argmax(eig_cumulative, axis=1)
    # Shade the horizon by the leading candidate: the argmax switching mid-horizon
    # is the point of the panel.
    for step, winner in enumerate(eig_winner, start=1):
        axes[3].axvspan(
            step - 0.5,
            step + 0.5,
            color=colors[int(winner)],
            alpha=0.10,
            linewidth=0.0,
        )
    for color, step_eig, cumulative_eig in zip(
        colors,
        eig_steps.T,
        eig_cumulative.T,
        strict=True,
    ):
        axes[3].plot(
            time_info,
            np.clip(step_eig, info_floor, None),
            color=color,
            linewidth=0.65,
            linestyle=":",
        )
        axes[3].plot(
            time_info,
            np.clip(cumulative_eig, info_floor, None),
            color=color,
            linewidth=0.95,
        )
    axes[3].scatter(
        time_info,
        eig_cumulative[np.arange(horizon), eig_winner],
        color=[colors[int(winner)] for winner in eig_winner],
        edgecolor="white",
        linewidth=0.3,
        s=11,
        zorder=4,
    )
    axes[3].set_xlabel(r"Future step $t$")
    axes[3].set_title(r"$\frac{1}{2}\log(1+\sigma_\theta^2 I_{\theta,t})$", loc="center", pad=3.0)
    axes[3].set_yscale("log")
    _panel_label(axes[3], "D")

    eig_panel_horizon = min(5, horizon)
    eig_cmap = plt.get_cmap("afmhot_r")

    eig_panel_values = eig_by_horizon[:eig_panel_horizon]
    panel_e_min = float(np.min(eig_panel_values))
    panel_e_max = float(np.max(eig_panel_values))
    steps_axis = time_info[:eig_panel_horizon]
    heatmap = axes[4].pcolormesh(
        steps_axis,
        z_probe,
        eig_panel_values.T,
        cmap=eig_cmap,
        shading="gouraud",
        vmin=panel_e_min,
        vmax=panel_e_max,
        rasterized=True,
    )
    best_z = z_probe[np.argmax(eig_panel_values, axis=1)]
    # Markers only: the optimum jumps between disjoint basins, so a connecting
    # line would imply a continuous path that does not exist.
    axes[4].plot(
        steps_axis,
        best_z,
        linestyle="none",
        marker="*",
        markersize=7.0,
        markerfacecolor=_C_EIG_STAR,
        markeredgecolor="black",
        markeredgewidth=0.45,
        # The k=1 and k=5 stars sit on the axis edge; the mesh ends there, so
        # let them overhang rather than padding white strips into the heatmap.
        clip_on=False,
        zorder=6,
        label="EIG-optimal $z_0$",
    )
    # Anchor the candidates from A, C and D onto the landscape. These sit just
    # inside the left edge so they never collide with the y tick labels.
    for color, candidate_z in zip(colors, _CANDIDATE_Z, strict=True):
        axes[4].hlines(
            candidate_z,
            steps_axis[0],
            steps_axis[0] + 0.13,
            color=color,
            linewidth=1.3,
            zorder=5,
        )
    colorbar = fig.colorbar(heatmap, ax=axes[4], pad=0.025, fraction=0.045)
    colorbar.set_label("EIG (nats)", fontsize=_ASSET_TITLE_SIZE)
    colorbar.ax.tick_params(labelsize=_ASSET_TICK_SIZE, width=0.4, length=2.0)
    colorbar.outline.set_linewidth(0.4)
    axes[4].set_ylim(float(np.min(z_probe)), float(np.max(z_probe)))
    axes[4].set_xlim(float(steps_axis[0]), float(steps_axis[-1]))
    axes[4].set_xlabel(r"Planning horizon $k$")
    axes[4].set_ylabel(r"Candidate initial state $z_0$")
    _panel_label(axes[4], "E")
    axes[4].set_xticks(steps_axis)
    # Framed: an unframed star floating on the heatmap reads as another optimum.
    axes[4].legend(
        loc="upper right",
        fontsize=_ASSET_TICK_SIZE,
        frameon=True,
        facecolor="white",
        edgecolor=_C_NEUTRAL_LIGHT,
        framealpha=0.9,
        borderpad=0.3,
        handletextpad=0.4,
    )

    for ax in (*axes, axis_rate):
        style_manuscript_axis(ax, grid_color=_C_GRID, grid_alpha=0.35)
    for ax in axes[2:4]:
        ax.set_xticks(time_info)
    # Panel E is a heatmap; a grid over it only adds noise.
    axes[4].grid(False)

    _add_candidate_legend(fig, colors, single_column=single_column)
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
            label=rf"$z_{{{step + 1}}}$",
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
    axes[7].set_title(rf"H. two-step EIG, best $z_0={best_z:.2f}$")
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
        rf"$\hat\theta={theta_mean:g}$, $c={c:g}$, $b={b:g}$, $Q={state_noise:g}$",
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
    parser.add_argument(
        "--column",
        choices=("double", "single"),
        default="double",
        help="Panel layout width: 7.25 in double column or 3.5 in single column.",
    )
    parser.add_argument("--theta-mean", type=float, default=1)
    parser.add_argument("--theta-var", type=float, default=2)
    parser.add_argument("--c", type=float, default=-1.6)
    parser.add_argument("--b", type=float, default=0)
    parser.add_argument("--state-var", type=float, default=0.5)
    parser.add_argument("--state-noise", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=1)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--z-min", type=float, default=0)
    parser.add_argument("--z-max", type=float, default=3 / 2 * np.pi)
    parser.add_argument("--num-points", type=int, default=401)
    args = parser.parse_args(argv)

    if args.detailed:
        args.horizon = 2
        if args.output == default_output:
            args.output = Path("results/eig_1d_example/eig_1d_example_detailed.png")

    z_probe = np.sort(
        np.unique(
            np.concatenate(
                [np.linspace(args.z_min, args.z_max, args.num_points), _CANDIDATE_Z],
            ),
        ),
    )
    curve = compute_eig_curve(
        z_probe,
        theta_mean=args.theta_mean,
        theta_var=args.theta_var,
        c=args.c,
        b=args.b,
        state_var=args.state_var,
        horizon=args.horizon,
        state_noise=args.state_noise,
        dt=args.dt,
    )
    plt = load_plotting(
        args.output,
        apply_style=_apply_eig_style,
        path_is_file=True,
        use_agg=True,
    )
    if plt is None:
        raise RuntimeError("Matplotlib is required to build the EIG example figure.")
    if args.detailed:
        fig = build_detailed_figure(
            curve,
            theta_mean=args.theta_mean,
            c=args.c,
            b=args.b,
            state_noise=args.state_noise,
            plt=plt,
        )
    else:
        fig = build_figure(
            curve,
            theta_mean=args.theta_mean,
            theta_var=args.theta_var,
            c=args.c,
            b=args.b,
            dt=args.dt,
            plt=plt,
            single_column=args.column == "single",
        )
    return save_figure(fig, args.output, plt_module=plt, dpi=300)


if __name__ == "__main__":
    print(main())
