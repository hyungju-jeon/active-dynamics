"""Vector field environment implementation."""

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from actdyn.utils.vectorfield_definition import (
    LimitCycle,
    MultiAttractor,
    DoubleLimitCycle,
    VanDerPol,
    Duffing,
    SnowMan,
)
from typing import Optional, Tuple, Dict, Any, Sequence
from actdyn.utils.visualize import plot_vector_field


vf_from_string = {
    "limit_cycle": LimitCycle,
    "double_limit_cycle": DoubleLimitCycle,
    "multi_attractor": MultiAttractor,
    "van_der_pol": VanDerPol,
    "duffing": Duffing,
    "snowman": SnowMan,
}


class VectorFieldEnv(gym.Env):
    """Unified environment for latent dynamics simulation."""

    def __init__(
        self,
        dynamics_type: str = "limit_cycle",
        state_dim: int = 2,
        Q: float = 0.1,
        dt: float = 0.1,
        device: str = "cpu",
        dyn_params: Optional[list[float]] | torch.Tensor = None,  #
        render_mode: Optional[str] = None,
        action_bounds: Sequence[float] = (-1.0, 1.0),
        state_bounds: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.Q = Q
        self.dt = dt
        self.device = torch.device(device)
        self.render_mode = render_mode

        # Initialize spaces with configurable bounds
        self.action_space = self._set_space_bounds(action_bounds, state_dim)

        if state_bounds is None:
            state_bounds = (-np.inf, np.inf)
        self.observation_space = self._set_space_bounds(state_bounds, state_dim)

        # Initialize dynamics
        if dynamics_type not in vf_from_string:
            raise ValueError(f"Unknown dynamics type: {dynamics_type}")
        self.dynamics = vf_from_string[dynamics_type](
            dyn_param=dyn_params, device=self.device, **kwargs
        )
        self.set_params(dyn_params)

        # Initialize state
        self.state = torch.tensor(
            self.observation_space.sample(), device=self.device, dtype=torch.float16
        )

    def _set_space_bounds(self, bounds: Sequence[float], dim: int) -> spaces.Box:
        """Set space bounds for action and observation spaces."""
        if not (isinstance(bounds, (tuple, list)) and len(bounds) == 2):
            raise ValueError(f"bounds must be a tuple or list of (low, high), got {bounds}")
        low = np.full((dim,), bounds[0], dtype=np.float16)
        high = np.full((dim,), bounds[1], dtype=np.float16)
        return spaces.Box(low=low, high=high, dtype=np.float16)

    def get_params(self) -> torch.Tensor:
        return self.dynamics.dyn_params

    def set_params(self, dyn_params: torch.Tensor | list[float] | Dict[str, float]):
        """Set dynamics parameters."""
        if isinstance(dyn_params, dict):
            _dyn_params = torch.tensor(
                [v for k, v in dyn_params.items()], device=self.device, dtype=torch.float16
            )
        elif isinstance(dyn_params, list):
            _dyn_params = torch.tensor(dyn_params, device=self.device, dtype=torch.float16)
        else:
            _dyn_params = dyn_params.to(self.device)

        if _dyn_params.ndim == 1:
            _dyn_params = _dyn_params.unsqueeze(0)

        if hasattr(self.dynamics, "set_params"):
            self.dynamics.set_params(*_dyn_params.mT)

    def compute_dynamics(self, state: torch.Tensor) -> torch.Tensor:
        """Compute vector field at given state."""
        return self.dynamics(state)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        self.state = torch.tensor(
            self.observation_space.sample(), device=self.device, dtype=torch.float16
        )
        return self.state, {}

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """Step the environment."""
        # Compute dynamics
        dynamics = self.compute_dynamics(self.state)

        # Update state
        self.state = self.state + (dynamics + action) * self.dt

        # Add noise
        self.state += torch.randn_like(self.state) * torch.sqrt(torch.tensor(self.Q) * self.dt)

        # Compute reward
        reward = 0

        return self.state, reward, False, False, {}

    @property
    def logvar(self) -> torch.Tensor:
        return torch.log(self.var)

    @property
    def var(self) -> torch.Tensor:
        return torch.tensor(self.Q)

    def render(self, ax=None, x_range=1):
        if self.render_mode == "rgb_array":
            pass
        elif self.render_mode == "human":
            plot_vector_field(self.dynamics, x_range=x_range, ax=ax)

    def close(self):
        pass
