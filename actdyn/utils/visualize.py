import numpy as np
import torch
import matplotlib.pyplot as plt

from actdyn.utils.torch_helper import to_np


def set_matplotlib_style():
    plt.rcParams.update(
        {
            "font.family": "times",
            "font.size": 14.0,
            "lines.linewidth": 2,
            "lines.antialiased": True,
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
    dynamics, x_range=2.5, n_grid=50, tform=(None, None), is_residual=False, device="cpu"
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
        plt.figure(figsize=(10, 8))
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
        plt.colorbar(label="Speed", aspect=20)
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
