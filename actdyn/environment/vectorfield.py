"""Vector field environment implementation."""

from math import sqrt
import torch
from actdyn.utils.vectorfield_definition import (
    LimitCycle,
    MultiAttractor,
    DoubleLimitCycle,
    VanDerPol,
    Duffing,
)
from typing import Optional, Tuple, Dict, Any, Sequence
from actdyn.utils.visualize import plot_vector_field

from .base import BaseDynamicsEnv

vf_from_string = {
    "limit_cycle": LimitCycle,
    "double_limit_cycle": DoubleLimitCycle,
    "multi_attractor": MultiAttractor,
    "van_der_pol": VanDerPol,
    "duffing": Duffing,
}


class VectorFieldEnv(BaseDynamicsEnv):
    """Unified environment for latent dynamics simulation."""

    def __init__(
        self,
        dynamics_type: str = "limit_cycle",
        state_dim: int = 2,
        noise_scale: float = 0.1,
        dt: float = 0.1,
        device: str = "cpu",
        dyn_param: Optional[list[float]] | torch.Tensor = None,  #
        render_mode: Optional[str] = None,
        action_bounds: Sequence[float] = (-1.0, 1.0),
        state_bounds: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ):
        # Initialize base environment
        super().__init__(
            state_dim=state_dim,
            noise_scale=noise_scale,
            dt=dt,
            device=device,
            render_mode=render_mode,
            action_bounds=action_bounds,
            state_bounds=state_bounds,
        )
        self.dyn_param = dyn_param if dyn_param is not None else {}
        self.dynamics_type = dynamics_type

        # Initialize dynamics
        if dynamics_type not in vf_from_string:
            raise ValueError(f"Unknown dynamics type: {dynamics_type}")

        # Determine x_range: use from kwargs if available, otherwise from state_bounds
        if "x_range" in kwargs:
            x_range = kwargs.pop("x_range")  # Remove from kwargs to avoid duplication
        elif state_bounds is not None:
            x_range = state_bounds[-1]
        else:
            raise ValueError(
                "Either 'x_range' must be provided in kwargs or 'state_bounds' must be set"
            )

        self.dynamics = vf_from_string[dynamics_type](
            dyn_param=dyn_param, device=self.device, x_range=x_range, **kwargs
        )

    def _rk4_step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Perform a single RK4 integration step."""
        dt = self.dt
        k1 = self._get_dynamics(state) + action
        k2 = self._get_dynamics(state + dt / 2 * k1) + action
        k3 = self._get_dynamics(state + dt / 2 * k2) + action
        k4 = self._get_dynamics(state + dt * k3) + action
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _get_dynamics(self, state: torch.Tensor) -> torch.Tensor:
        """Compute vector field at given state."""
        return self.dynamics(state)

    @torch.no_grad()
    def generate_trajectory(self, x0, n_steps, action=None):
        if x0.ndim == 2:
            if x0.shape[0] == 1:
                x0 = x0.unsqueeze(0)
            else:
                x0 = x0.unsqueeze(1)
        B, T, D = x0.shape
        if action is None:
            action = torch.zeros(B, n_steps, D, device=self.device)
        traj = [x0]
        for i in range(n_steps):
            current_state = traj[i]
            current_action = action[:, i].unsqueeze(1)  # Shape: [B, 1, D]
            next_state = self._rk4_step(current_state, current_action)
            next_state += torch.randn_like(next_state) * torch.sqrt(
                torch.tensor(self.noise_scale * self.dt)
            )
            traj.append(next_state)
        return torch.cat(traj, dim=1)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        self.state = torch.rand(self.state_dim, device=self.device) * 2 - 1
        return self.state.to(self.device), {}

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """Step the environment."""
        # Update state with RK4
        self.state = self._rk4_step(self.state, action)

        # Add noise
        self.state += torch.randn_like(self.state) * torch.sqrt(
            torch.tensor(self.noise_scale * self.dt)
        )

        # Compute reward
        reward = 0

        return self.state, reward, False, False, {}

    def render(self, ax=None, x_range=1):
        if self.render_mode == "rgb_array":
            pass
        elif self.render_mode == "human":
            plot_vector_field(self.dynamics, x_range=x_range, ax=ax)

    def close(self):
        pass
