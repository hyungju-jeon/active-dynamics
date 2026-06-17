import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.colors import to_rgba

from actdyn.utils.torch_utils import to_np


def set_matplotlib_style():
    plt.rcParams.update(
        {
            "font.family": "helvetica",
            "font.size": 14.0,
            "lines.linewidth": 2,
            "lines.antialiased": True,
            "axes.prop_cycle": cycler(color=sns.color_palette("husl", 8)),
            "axes.facecolor": "fdfdfd",
            "axes.edgecolor": "777777",
            "axes.linewidth": 1,
            "axes.titlesize": "medium",
            "axes.labelsize": "medium",
            "axes.axisbelow": True,
            "xtick.major.size": 0,  # major tick size in points
            "xtick.minor.size": 0,  # minor tick size in points
            "xtick.major.pad": 6,  # distance to major tick label in points
            "xtick.minor.pad": 6,  # distance to the minor tick label in points
            "xtick.color": "333333",  # color of the tick labels
            "xtick.labelsize": "medium",  # fontsize of the tick labels
            "xtick.direction": "in",  # direction: in or out
            "ytick.major.size": 0,  # major tick size in points
            "ytick.minor.size": 0,  # minor tick size in points
            "ytick.major.pad": 6,  # distance to major tick label in points
            "ytick.minor.pad": 6,  # distance to the minor tick label in points
            "ytick.color": "333333",  # color of the tick labels
            "ytick.labelsize": "medium",  # fontsize of the tick labels
            "ytick.direction": "in",  # direction: in or out
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 1,
            "legend.fancybox": True,
            "legend.fontsize": "Small",
            "figure.facecolor": "1.0",
            "figure.edgecolor": "0.5",
            "hatch.linewidth": 0.1,
            "text.usetex": True,
        }
    )



def apply_manuscript_figure_style(
    plt_module=plt,
    *,
    font_size: float = 7.8,
    stroke_color: str = "#3A3A3A",
) -> None:
    """Apply compact manuscript figure defaults used by experiment summaries."""
    plt_module.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": float(font_size),
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": stroke_color,
            "axes.linewidth": 0.55,
            "axes.labelcolor": stroke_color,
            "xtick.color": stroke_color,
            "ytick.color": stroke_color,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def style_manuscript_axis(
    ax,
    *,
    grid_axis: str | None = None,
    grid_color: str = "#C8C1B8",
    grid_alpha: float = 0.42,
    stroke_color: str = "#3A3A3A",
    grid_linewidth: float = 0.35,
    spine_linewidth: float = 0.55,
) -> None:
    """Apply compact axis styling for manuscript summary figures."""
    if grid_axis is None:
        ax.grid(color=grid_color, linewidth=grid_linewidth, alpha=grid_alpha)
    else:
        ax.grid(axis=grid_axis, color=grid_color, linewidth=grid_linewidth, alpha=grid_alpha)
    for spine in ax.spines.values():
        spine.set_color(stroke_color)
        spine.set_linewidth(spine_linewidth)
    ax.tick_params(width=0.45, length=2.0, colors=stroke_color)


def create_grid(x_range=2, n_grid=50, device="cpu"):
    """Create a grid of points in the specified range."""
    x = torch.linspace(-x_range, x_range, n_grid, device=device)
    y = torch.linspace(-x_range, x_range, n_grid, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="xy")  # [H, W]
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return grid, xx, yy


@torch.no_grad()
def compute_vector_field(
    dynamics, x_range=2.5, n_grid=50, tform=(None, None), is_residual=True, device="cpu"
):
    """
    Produces a vector field for a given dynamical system
    :param queries: N by dx torch tensor of query points where each row is a query
    :param dynamics: function handle for dynamics
    """
    xy, X, Y = create_grid(x_range=x_range, n_grid=n_grid, device=device)
    if hasattr(dynamics, "device"):
        xy = xy.to(dynamics.device)
    else:
        xy = xy.to(device)
    if tform[0] is not None:
        xy = (tform[0] @ xy.T).T + tform[1]

    vel = torch.zeros(xy.shape, device=device)
    with torch.no_grad():
        for n in range(xy.shape[0]):
            vel[n, :] = dynamics(xy[[n]])
            if not is_residual:
                vel[n, :] = vel[n, :] - xy[[n]].to(device)

    U = vel[:, 0].reshape(X.shape[0], X.shape[1])
    V = vel[:, 1].reshape(Y.shape[0], Y.shape[1])
    return X, Y, U, V


def plot_vector_field(dynamics, ax=None, title=None, **kwargs):
    X, Y, U, V = compute_vector_field(dynamics, **kwargs)
    X, Y, U, V = X.cpu().numpy(), Y.cpu().numpy(), U.cpu().numpy(), V.cpu().numpy()
    speed = np.sqrt(U**2 + V**2)

    if ax is not None:
        plt.sca(ax)
    else:
        plt.figure(figsize=(8, 8))
    plt.streamplot(
        X,
        Y,
        U,
        V,
        color=speed,
        linewidth=0.5,
        density=2,
        cmap="viridis",
    )
    title = "Vector Field of Latent Dynamics" if title is None else title
    if ax is None:
        # plt.colorbar(label="Speed", aspect=20)
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.title(title)
        # plt.axis("off")
        plt.axis("equal")
        plt.tight_layout()
        plt.colorbar(label="Speed", aspect=20)


@torch.no_grad()
def evaluate_vector_field_grid(
    dynamics,
    grid_points: np.ndarray,
    shape: tuple[int, int],
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a 2D vector field on flattened grid points.

    grid_points has shape (H * W, 2). Returned components each have shape
    (H, W), matching shape.
    """
    pts = torch.as_tensor(grid_points, dtype=torch.float32, device=device)
    if hasattr(dynamics, "device"):
        pts = pts.to(dynamics.device)
    vel = dynamics(pts).detach().cpu().numpy().reshape(shape[0], shape[1], 2)
    return vel[:, :, 0], vel[:, :, 1]


def vector_field_l2_error(
    true_u: np.ndarray,
    true_v: np.ndarray,
    inferred_u: np.ndarray,
    inferred_v: np.ndarray,
) -> np.ndarray:
    """Return pointwise L2 vector-field error on a shared plotting grid."""
    return np.sqrt(
        (np.asarray(inferred_u) - np.asarray(true_u)) ** 2
        + (np.asarray(inferred_v) - np.asarray(true_v)) ** 2
    )


class RbfVectorFieldDynamics:
    """Evaluate a sparse local RBF vector field on arbitrary query points."""

    def __init__(
        self,
        *,
        centers,
        axis,
        weights,
        width: float,
        support_radius: int,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.centers = torch.as_tensor(centers, dtype=torch.float32, device=self.device)
        self.axis = torch.as_tensor(axis, dtype=torch.float32, device=self.device)
        self.weights = torch.nan_to_num(
            torch.as_tensor(weights, dtype=torch.float32, device=self.device),
            nan=0.0,
            posinf=1e3,
            neginf=-1e3,
        ).clamp(-1e3, 1e3)
        self.width = float(max(width, 1e-6))
        self.support_radius = int(support_radius)
        n_axis = int(self.axis.numel())
        grid_i, grid_j = torch.meshgrid(
            torch.arange(n_axis, device=self.device),
            torch.arange(n_axis, device=self.device),
            indexing="ij",
        )
        self.center_i = grid_i.reshape(-1)
        self.center_j = grid_j.reshape(-1)

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        flat = state.reshape(-1, state.shape[-1])
        dx = torch.abs(flat[:, 0:1] - self.axis.view(1, -1))
        dy = torch.abs(flat[:, 1:2] - self.axis.view(1, -1))
        x_idx = torch.argmin(dx, dim=1)
        y_idx = torch.argmin(dy, dim=1)
        mask = (
            torch.abs(self.center_i.view(1, -1) - x_idx.view(-1, 1))
            + torch.abs(self.center_j.view(1, -1) - y_idx.view(-1, 1))
        ) <= self.support_radius
        scaled = (flat.unsqueeze(1) - self.centers.unsqueeze(0)) / self.width
        phi = torch.exp(-0.5 * torch.sum(scaled * scaled, dim=-1)) * mask.to(torch.float32)
        out = torch.nan_to_num(
            phi @ self.weights,
            nan=0.0,
            posinf=1e3,
            neginf=-1e3,
        ).clamp(-1e3, 1e3)
        return out.reshape(*state.shape[:-1], self.weights.shape[-1])


def trace_index(trace_steps: np.ndarray, step: int) -> int:
    if trace_steps.size == 0:
        return 0
    idx = int(np.searchsorted(trace_steps, step, side="right") - 1)
    return int(np.clip(idx, 0, len(trace_steps) - 1))


def planned_xy_for_step(trace: tuple[np.ndarray, ...] | None, step: int) -> np.ndarray | None:
    if trace is None:
        return None
    steps, paths, lengths = trace
    idx = trace_index(np.asarray(steps, dtype=int), int(step))
    n_points = int(lengths[idx])
    if n_points < 2:
        return None
    path = np.asarray(paths[idx, :n_points, :2], dtype=float)
    valid = np.all(np.isfinite(path), axis=1)
    path = path[valid]
    return path if path.shape[0] >= 2 else None


def overlay_planned_xy(
    ax,
    planned_xy: np.ndarray | None,
    *,
    color: str = "tab:orange",
    linewidth: float = 2.2,
    linestyle: str = "--",
    alpha: float = 0.32,
    label: str = "planned traj",
    zorder: int = 4,
) -> None:
    if planned_xy is None or planned_xy.shape[0] < 2:
        return
    ax.plot(
        planned_xy[:, 0],
        planned_xy[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )


def make_lognorm(values: list[np.ndarray]) -> LogNorm:
    flat = np.concatenate([np.ravel(np.asarray(value, dtype=float)) for value in values], axis=0)
    positive = flat[np.isfinite(flat) & (flat > 0)]
    if positive.size == 0:
        return LogNorm(vmin=1e-8, vmax=1.0)
    min_positive = float(np.min(positive))
    p1 = float(np.percentile(positive, 1.0))
    p99 = float(np.percentile(positive, 99.0))
    vmin = max(min_positive, p1)
    vmax = max(p99, vmin * 1.01)
    return LogNorm(vmin=vmin, vmax=vmax)


def decorate_phase_space_axis(
    ax,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
    xlabel: str = "x",
    ylabel: str = "v",
    legend_loc: str = "upper right",
    grid_alpha: float = 0.25,
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=grid_alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc=legend_loc)


def annotate_action_arrow(
    ax,
    *,
    origin,
    action,
    max_display_len: float = 2.5,
    scale: float = 0.45,
    color: str = "white",
    width: float = 0.03,
    head_width: float = 0.28,
    zorder: int = 7,
) -> float:
    origin_xy = np.asarray(origin, dtype=float).reshape(-1)
    action_xy = np.asarray(action, dtype=float).reshape(-1)
    if origin_xy.size < 2 or action_xy.size < 2 or not np.all(np.isfinite(action_xy[:2])):
        return float("nan")
    action_xy = action_xy[:2]
    act_norm = float(np.linalg.norm(action_xy))
    if act_norm > 1e-12:
        display_len = min(float(max_display_len), float(scale) * act_norm)
        direction = action_xy / act_norm
        ax.arrow(
            float(origin_xy[0]),
            float(origin_xy[1]),
            float(display_len * direction[0]),
            float(display_len * direction[1]),
            color=color,
            width=width,
            head_width=head_width,
            length_includes_head=True,
            alpha=0.95,
            zorder=zorder,
        )
    ax.text(
        0.02,
        0.02,
        f"u=({action_xy[0]:.2f}, {action_xy[1]:.2f})  |u|={act_norm:.2f}",
        transform=ax.transAxes,
        color=color,
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
    return act_norm


@torch.no_grad()
def compute_fisher_map(
    fisher,
    x_range=2.5,
    n_grid=50,
    show_plot=False,
    ax=None,
    device="cpu",
):
    """Create a Fisher information map by computing FIM on sampled points in the grid."""
    if ax is not None:
        plt.sca(ax)
    else:
        plt.figure(figsize=(10, 8))

    xy, X, Y = create_grid(x_range=x_range, n_grid=n_grid, device=device)
    xy = xy.to(device)

    grid_dict = {"model_state": xy.unsqueeze(1)}
    fisher_map = fisher.compute(grid_dict)
    fisher_map = fisher_map.reshape(len(X), len(Y))

    if show_plot:
        plt.contourf(X.cpu(), Y.cpu(), fisher_map.cpu(), levels=10, cmap="plasma")
        plt.colorbar(label="Fisher Information")
        plt.title("Fisher Information Map")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.grid(True)
        plt.tight_layout()

    return fisher_map, X.cpu(), Y.cpu()


def plot_per_dimension(x, ax=None, title=None, **kwargs):
    """Plot each dimension of a 2D tensor x over time."""
    fig, axs = create_subplot(x)

    for i in range(x.shape[-1]):
        axs[i].plot(to_np(x[:, i]), **kwargs)
        axs[i].set_title(f"Dimension {i+1}")
        axs[i].set_xlabel("Time Step")
        axs[i].set_ylabel("Value")
        axs[i].grid(True)

    if title is not None:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()


def create_subplot(x):
    """Create a grid of subplots based on the dimension of x."""
    d = x.shape[-1]
    if d % 2 == 0:
        if d % 3 == 0:
            n_cols = 3
        else:
            n_cols = 2
    else:
        n_cols = min(3, d)
    n_rows = (d + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = axs.flatten() if d > 1 else [axs]

    return fig, axs


def plot_spike_train(z, y, dt, fname=None):
    if isinstance(z, torch.Tensor):
        z = to_np(z.squeeze())
    if isinstance(y, torch.Tensor):
        y = to_np(y.squeeze())

    tr = np.arange(0, z.shape[0]) * dt

    dy = y.shape[1]
    spike_times = [np.where(y[:, k] > 0)[0] * dt for k in range(dy)]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]}
    )

    for i, st in enumerate(spike_times):
        ax1.eventplot(st, colors="black", lineoffsets=i, linelengths=0.5)

    ax1.set_ylim(-1, dy)
    ax1.set_ylabel("Neurons")
    ax1.xaxis.set_ticklabels([])

    ax2.plot(tr, z)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Latents")
    ax2.set_xlim(tr[0], tr[-1])

    plt.subplots_adjust(hspace=0.05)
    if fname is not None:
        plt.savefig(f"../figs/{fname}.pdf")


def create_gradient_line(
    ax,
    data,
    base_color,
    label=None,
    alpha_start=0.2,
    alpha_end=0.95,
    linewidth=1.5,
):
    """Plot a 2D trajectory with a fading alpha gradient."""
    if isinstance(data, torch.Tensor):
        data = to_np(data)
    data = np.asarray(data)

    if data.ndim == 3:
        if data.shape[0] == 1:
            data = data[0]
        elif data.shape[1] == 1:
            data = data[:, 0, :]
        else:
            data = data.reshape(-1, data.shape[-1])
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        return None

    points = data[:, :2]
    segments = np.stack([points[:-1], points[1:]], axis=1)
    alphas = np.linspace(alpha_start, alpha_end, len(segments))
    colors = [to_rgba(base_color, alpha) for alpha in alphas]
    line = LineCollection(segments, colors=colors, linewidths=linewidth, zorder=3)
    ax.add_collection(line)
    if label is not None:
        ax.plot([], [], color=base_color, linewidth=linewidth, label=label)
    return line


def plot_embedding_error_comparison(
    unknown_results,
    known_results,
    methods=("active (k=5)", "step", "active chunk(k=20)"),
    max_steps=500,
    ax=None,
):
    """Plot mean/std embedding error for unknown vs known observation settings."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    colorset = sns.color_palette("Set1", n_colors=max(len(methods), 1))
    color_idx = 0

    for method in methods:
        if method not in unknown_results or method not in known_results:
            continue

        unknown = np.asarray(unknown_results[method])
        known = np.asarray(known_results[method])
        if unknown.size == 0 or known.size == 0:
            continue

        unknown_mean = unknown.mean(axis=0)
        unknown_std = unknown.std(axis=0)
        known_mean = known.mean(axis=0)
        known_std = known.std(axis=0)

        color = colorset[color_idx % len(colorset)]
        color_idx += 1

        ax.plot(unknown_mean, label=f"{method} (unknown obs.)", linestyle="--", color=color)
        ax.fill_between(
            np.arange(len(unknown_mean)),
            unknown_mean - unknown_std,
            unknown_mean + unknown_std,
            alpha=0.1,
            color=color,
        )

        ax.plot(known_mean, label=f"{method} (known obs.)", linestyle="-", color=color)
        ax.fill_between(
            np.arange(len(known_mean)),
            known_mean - known_std,
            known_mean + known_std,
            alpha=0.1,
            color=color,
        )

    if max_steps is not None:
        ax.set_xlim(0, max_steps)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Embedding Error Norm")
    ax.set_title("Embedding Error Norm over Environment Steps")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_current_state(
    env,
    model,
    delta_f=None,
    x=None,
    z=None,
    title=None,
):
    def plot_trajectory(x, ax):
        num_bold = min(20, x.shape[1] // 10)
        ax.plot(
            x[0, :-num_bold, 0],
            x[0, :-num_bold, 1],
            color="red",
            alpha=0.5,
            lw=1,
        )
        ax.plot(
            x[0, -num_bold:, 0],
            x[0, -num_bold:, 1],
            color="red",
            alpha=0.7,
            marker=".",
            lw=1,
        )

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs = axs.flatten()

    plot_vector_field(env.dynamics, ax=axs[0], x_range=5)
    axs[0].set_xlim(-5, 5)
    axs[0].set_ylim(-5, 5)
    axs[0].set_title("True Vector Field")
    plot_trajectory(x, axs[0])

    plot_vector_field(model.dynamics, ax=axs[1], x_range=5)
    axs[1].set_xlim(-5, 5)
    axs[1].set_ylim(-5, 5)
    axs[1].set_title("Learned Vector Field")
    plot_trajectory(z, axs[1])

    axs[2].plot(
        delta_f,
        color="red",
    )
    axs[2].set_title(r"norm($f - \hat{f}$) over time")

    if title is not None:
        fig.suptitle(title)

    return fig, axs


def plot_rollout_latent_comparison(
    env_state,
    model_state,
    ax=None,
    title="Latent Trajectory Comparison",
    labels=("true", "model"),
):
    """Overlay true and model latent trajectories in 2D."""
    env_xy = to_np(env_state)
    model_xy = to_np(model_state)
    if env_xy.ndim == 3:
        env_xy = env_xy[0]
    if model_xy.ndim == 3:
        model_xy = model_xy[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    ax.plot(env_xy[:, 0], env_xy[:, 1], alpha=0.7, label=labels[0])
    ax.plot(model_xy[:, 0], model_xy[:, 1], alpha=0.7, label=labels[1])
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig, ax


def plot_observation_channels(obs, n_channels=5, ax=None, title="Observation Channels"):
    """Plot the first ``n_channels`` observation channels over time."""
    obs_np = to_np(obs)
    if obs_np.ndim == 3:
        obs_np = obs_np[0]
    n_channels = min(n_channels, obs_np.shape[-1])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    ax.plot(obs_np[:, :n_channels])
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Observation")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax
