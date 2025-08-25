import torch
import torch.nn as nn
from .base import BaseDynamics, BaseDynamicsEnsemble


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

    def __init__(
        self,
        state_dim,
        alpha=0.1,
        gamma=1.0,
        centers=None,
        range=5.0,
        num_grid_pts=25,
        device="cpu",
        **kwargs
    ):
        super().__init__(state_dim=state_dim, dt=kwargs.get("dt", 1), device=device)
        self.alpha = alpha
        self.gamma = gamma
        self.range = range
        self.num_grid_pts = num_grid_pts
        self.has_center = False

        # Initialize centers if provided
        if centers is not None:
            self.set_centers(centers)
        else:
            rbf_grid_x = torch.linspace(-self.range, self.range, self.num_grid_pts)
            rbf_grid_y = torch.linspace(-self.range, self.range, self.num_grid_pts)
            rbf_xx, rbf_yy = torch.meshgrid(rbf_grid_x, rbf_grid_y, indexing="ij")  # [H, W]
            self.centers = torch.stack([rbf_xx.flatten(), rbf_yy.flatten()], dim=1)
            self.has_center = True

        # Initialize weights with proper shape
        self.weights = nn.Parameter(
            torch.randn(self.centers.shape[0], state_dim, device=self.device),
            requires_grad=True,
        )

        # Use a simple lambda function instead of a nested class to avoid circular references
        self.network = lambda state: torch.matmul(self.rbf(state), self.weights)

    def set_centers(self, centers):
        """Set the centers for the RBF."""
        self.centers = centers.to(self.device)
        self.has_center = True
        # Reinitialize weights with proper shape when centers are set

    def rbf(self, state):
        if not self.has_center:
            raise ValueError("Centers must be set before calling rbf.")
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        return self.alpha * torch.exp(-torch.cdist(state, self.centers, p=2) ** 2 * self.gamma)


class RBFDynamicsEnsemble(BaseDynamicsEnsemble):
    """
    Ensemble of RBF-based dynamics models.
    """

    def __init__(self, state_dim, n_models=5, **kwargs):
        super().__init__(
            dynamics_cls=RBFDynamics,
            state_dim=state_dim,
            n_models=n_models,
            dynamics_kwargs=kwargs,
        )


class MLPDynamicsEnsemble(BaseDynamicsEnsemble):
    """
    Ensemble of MLP-based dynamics models.
    """

    def __init__(self, state_dim, n_models=5, **kwargs):
        super().__init__(
            dynamics_class=MLPDynamics,
            state_dim=state_dim,
            n_models=n_models,
            dynamics_kwargs=kwargs,
        )
