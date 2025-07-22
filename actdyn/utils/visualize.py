import numpy as np
import scipy as sp
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def compute_vector_field(dynamics, x_range=2.5, n_grid=50, device="cpu"):
    """
    Produces a vector field for a given dynamical system
    :param queries: N by dx torch tensor of query points where each row is a query
    :param dynamics: function handle for dynamics
    """

    x = torch.linspace(-x_range, x_range, n_grid, device=device)
    y = torch.linspace(-x_range, x_range, n_grid, device=device)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
    if hasattr(dynamics, "device"):
        xy = xy.to(dynamics.device)
    else:
        xy = xy.to(device)

    vel = torch.zeros(xy.shape, device=device)
    with torch.no_grad():
        for n in range(xy.shape[0]):
            vel[n, :] = (dynamics(xy[[n]])).to("cpu")

    U = vel[:, 0].reshape(X.shape[0], X.shape[1])
    V = vel[:, 1].reshape(Y.shape[0], Y.shape[1])
    return X, Y, U, V


def plot_vector_field(dynamics, ax=None, **kwargs):
    # if hasattr(dynamics, "X"):
    #     X, Y, U, V = dynamics.X, dynamics.Y, dynamics.U, dynamics.V
    # else:
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
    if ax is None:
        plt.colorbar(label="Speed", aspect=20)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Vector Field of Latent Dynamics")
    # plt.axis("off")
    plt.axis("equal")
