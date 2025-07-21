import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.functional import softplus
from .base import BaseDynamics


# Small constant to prevent numerical instability
eps = 1e-6


class LinearDynamics(BaseDynamics):
    """
    Linear dynamics model using nn.Linear.
    """

    def __init__(self, state_dim, device="cpu", **kwargs):
        super().__init__(state_dim, device=device)
        self.network = nn.Linear(state_dim, state_dim)


class MLPDynamics(BaseDynamics):
    """
    MLP-based dynamics model.
    """

    def __init__(self, state_dim, hidden_dim=16, device="cpu", **kwargs):
        super().__init__(state_dim, device=device)
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )


class RBFDynamics(BaseDynamics):
    """
    RBF-based dynamics model.
    """

    class RBFNetwork(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent

        def forward(self, state):
            rbf = self.parent.rbf(state)
            return torch.matmul(rbf, self.parent.weights)

    def __init__(
        self,
        state_dim,
        num_centers,
        alpha=0.1,
        gamma=1.0,
        centers=None,
        device="cpu",
        **kwargs
    ):
        super().__init__(state_dim=state_dim, device=device)
        self.has_center = False
        self.centers = centers.to(self.device) if centers is not None else None
        self.alpha = alpha
        self.gamma = gamma
        self.weights = nn.Parameter(
            torch.randn(num_centers * state_dim, device=self.device),
            requires_grad=True,
        )
        self.network = self.RBFNetwork(self)

    def set_centers(self, centers):
        """Set the centers for the RBF."""
        self.centers = centers.to(self.device)
        self.has_center = True

    def rbf(self, state):
        if not self.has_center:
            raise ValueError("Centers must be set before calling rbf.")
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        return self.alpha * torch.exp(
            -torch.cdist(state, self.centers, p=2) ** 2 * self.gamma
        )
