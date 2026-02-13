import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from actdyn.utils.helper import to_np


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
