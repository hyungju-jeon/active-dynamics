import torch
import torch.nn as nn
from .base import BaseDynamics, BaseDynamicsEnsemble
from actdyn.utils.torch_helper import activation_from_str


# Small constant to prevent numerical instability
eps = 1e-6


class _RBFNetwork(nn.Module):
    def __init__(self, rbf_fn, weights):
        super().__init__()
        self.rbf = rbf_fn
        self.weights = weights

    def forward(self, state):
        return torch.matmul(self.rbf(state), self.weights)


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

    def __init__(
        self, state_dim, hidden_dims: int | list = [16], activation="relu", device="cpu", **kwargs
    ):
        super().__init__(
            state_dim,
            dt=kwargs.get("dt", 1),
            is_residual=kwargs.get("is_residual", False),
            device=device,
        )
        self.activation = activation_from_str(activation)

        # Build encoder layers
        layers = []
        prev_dim = state_dim

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), self.activation])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, state_dim))
        self.network = nn.Sequential(*layers)


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
        z_max=5.0,
        num_grid_pts=25,
        device="cpu",
        **kwargs
    ):
        super().__init__(
            state_dim=state_dim,
            dt=kwargs.get("dt", 1),
            is_residual=kwargs.get("is_residual", False),
            device=device,
        )
        self.alpha = alpha
        self.gamma = gamma
        self.z_max = z_max
        self.num_grid_pts = num_grid_pts
        self.has_center = False

        # Initialize centers if provided
        if centers is not None:
            self.set_centers(centers)
        else:
            grid_coords = [
                torch.linspace(-self.z_max, self.z_max, self.num_grid_pts)
                for _ in range(self.state_dim)
            ]
            mesh = torch.meshgrid(*grid_coords, indexing="ij")
            self.centers = torch.stack([m.flatten() for m in mesh], dim=1)
            self.has_center = True

        # Initialize weights with proper shape
        self.weights = nn.Parameter(
            torch.randn(self.centers.shape[0], state_dim, device=self.device),
            requires_grad=True,
        )

        self.network = _RBFNetwork(self.rbf, self.weights)

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
            dynamics_config=kwargs,
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
            dynamics_config=kwargs,
        )
