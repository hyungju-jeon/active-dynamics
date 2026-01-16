from functools import partial
from os import read
from einops import rearrange
import torch
import torch.nn as nn
from .base import BaseDynamics, BaseDynamicsEnsemble
from actdyn.utils.helper import activation_from_str, eps
from typing import Dict


# Small constant to prevent numerical instability


class _RBFNetwork(nn.Module):
    def __init__(self, rbf_fn, n_centers, state_dim):
        super().__init__()
        self.rbf = rbf_fn
        self.weights = nn.Parameter(torch.randn(n_centers, state_dim) * eps)

    def forward(self, state):
        return self.rbf(state) @ self.weights


class LinearDynamics(BaseDynamics):
    """
    Linear dynamics model using nn.Linear.

    DESIGN NOTE: Missing dt parameter in constructor
    =================================================

    Issue: BaseDynamics.__init__ requires 'dt' parameter but LinearDynamics doesn't accept it
    Impact: 5 test errors when trying to instantiate LinearDynamics(state_dim=4, device="cpu")

    Problem:
    - BaseDynamics.__init__(state_dim, dt, is_residual, device) signature
    - LinearDynamics only passes state_dim and device
    - Parent __init__ fails with "missing required positional argument: 'dt'"

    Better design:
    Match FunctionDynamics pattern - accept **kwargs and extract dt/is_residual:

    def __init__(self, state_dim, device="cpu", **kwargs):
        super().__init__(
            state_dim,
            dt=kwargs.get("dt", 0.1),  # default dt
            is_residual=kwargs.get("is_residual", False),
            device=device,
        )
        self.network = nn.Linear(state_dim, state_dim).to(device)

    This allows flexible instantiation:
    - LinearDynamics(state_dim=4) -> uses defaults
    - LinearDynamics(state_dim=4, dt=0.05) -> custom dt
    - LinearDynamics(state_dim=4, is_residual=True) -> residual mode

    Usage: Used in experiments for simple linear system identification
    """

    def __init__(self, state_dim, device="cpu", **kwargs):
        super().__init__(
            state_dim,
            dt=kwargs.get("dt", 1),
            is_residual=kwargs.get("is_residual", False),
            device=device,
        )
        self.network = nn.Linear(state_dim, state_dim).to(device)


class FunctionDynamics(BaseDynamics):
    """
    Dynamics model using a user-defined function.
    """

    def __init__(self, state_dim, dynamics_fn, device="cpu", **kwargs):
        super().__init__(
            state_dim,
            dt=kwargs.get("dt", 1),
            is_residual=kwargs.get("is_residual", True),
            device=device,
        )
        self.dynamics_fn = dynamics_fn
        self.network = dynamics_fn
        self.dyn_param = None

    def set_params(self, dyn_param: torch.Tensor | list[float] | Dict[str, float]):
        """Set dynamics parameters.

        DESIGN NOTE: Parameter type handling
        ====================================

        Issue: Line 100 uses dyn_param.mT which requires tensor, but list input gets converted later
        Impact: AttributeError when passing list: 'list' object has no attribute 'mT'

        Current flow:
        1. Convert list/dict to tensor
        2. Try to use .mT on potentially unconverted data

        Better design:
        Ensure tensor conversion happens before any tensor operations

        def set_params(self, dyn_param):
            # Convert to tensor first, regardless of input type
            if isinstance(dyn_param, dict):
                param_tensor = torch.tensor(
                    list(dyn_param.values()), device=self.device, dtype=torch.float32
                ).unsqueeze(0)
            elif isinstance(dyn_param, list):
                param_tensor = torch.tensor(
                    dyn_param, device=self.device, dtype=torch.float32
                ).unsqueeze(0)
            elif isinstance(dyn_param, torch.Tensor):
                param_tensor = dyn_param
                if param_tensor.ndim == 1:
                    param_tensor = param_tensor.unsqueeze(0)
            else:
                raise TypeError(f"Unsupported type for dyn_param: {type(dyn_param)}")

            self.dyn_param = param_tensor

            # Now safe to use tensor operations
            if hasattr(self.dynamics_fn, "set_params"):
                self.dynamics_fn.set_params(*param_tensor.mT.to(self.device))

        Usage: experiments/vectorfield uses this to update parameters during learning
        """
        if isinstance(dyn_param, dict):
            self.dyn_param = torch.tensor(
                [v for k, v in dyn_param.items()], device=self.device, dtype=torch.float32
            ).unsqueeze(0)
        elif isinstance(dyn_param, list):
            self.dyn_param = torch.tensor(
                dyn_param, device=self.device, dtype=torch.float32
            ).unsqueeze(0)
        else:
            if dyn_param.ndim == 1:
                dyn_param = dyn_param.unsqueeze(0)
            elif dyn_param.ndim == 3:
                dyn_param = rearrange(dyn_param, "b T d -> (b T) d")
            self.dyn_param = dyn_param

        # DESIGN FIX: Ensure dyn_param is tensor before using .mT
        # Current: May fail if list/dict conversion above doesn't happen first
        if hasattr(self.dynamics_fn, "set_params"):
            self.dynamics_fn.set_params(*self.dyn_param.mT.to(self.device))
        self.network = self.dynamics_fn

    def compute_param(self, state, e=None):
        if e is not None:
            if "e" in self.dynamics_fn.__code__.co_varnames:
                self.network = partial(self.dynamics_fn, e=e)
            else:
                self.network = self.dynamics_fn
        return super().compute_param(state)

    def sample_forward(self, e=None, **kwargs):
        if e is not None:
            if "e" in self.dynamics_fn.__code__.co_varnames:
                self.network = partial(self.dynamics_fn, e=e)
            else:
                self.network = self.dynamics_fn
        return super().sample_forward(**kwargs)


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
            layers.extend([nn.Linear(prev_dim, hidden_dim, device=device), self.activation])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, state_dim))
        self.network = nn.Sequential(*layers).to(device)


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
            is_residual=kwargs.get("is_residual", True),
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
            self.centers = torch.stack([m.flatten() for m in mesh], dim=1).to(self.device)
            self.has_center = True

        # Initialize weights with proper shape
        self.weights = nn.Parameter(
            torch.randn(self.centers.shape[0], state_dim, device=self.device),
            requires_grad=True,
        )

        self.network = _RBFNetwork(self.rbf, self.centers.shape[0], state_dim).to(self.device)

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
