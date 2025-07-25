"""Vector field environment implementation."""

import torch
from actdyn.utils.vectorfield_definition import (
    LimitCycle,
    MultiAttractor,
    DoubleLimitCycle,
)
from typing import Optional, Tuple, Dict, Any, Sequence
from actdyn.utils.visualize import plot_vector_field

from .base import BaseDynamicsEnv

vf_from_string = {
    "limit_cycle": LimitCycle,
    "double_limit_cycle": DoubleLimitCycle,
    "multi_attractor": MultiAttractor,
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
        render_mode: Optional[str] = None,
        action_bounds: Sequence[float] = (-1.0, 1.0),
        state_bounds: Optional[Sequence[float]] = None,
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
        self.dynamics_type = dynamics_type

        # Initialize dynamics
        if dynamics_type not in vf_from_string:
            raise ValueError(f"Unknown dynamics type: {dynamics_type}")
        self.dynamics = vf_from_string[dynamics_type]()

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
            traj.append(
                traj[i]
                + (
                    self._get_dynamics(traj[i])
                    + action[:, i].unsqueeze(1)
                    + torch.randn_like(traj[i]) * self.noise_scale
                )
                * self.dt
            )
        return torch.cat(traj, dim=-2)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        self.state = torch.rand(self.state_dim, device=self.device) * 2 - 1
        return self.state.to(self.device), {}

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """Step the environment."""
        # Compute dynamics
        dynamics = self._get_dynamics(self.state)

        # Update state
        self.state = self.state + (dynamics + action) * self.dt

        # Add noise
        self.state += torch.randn_like(self.state) * self.noise_scale * self.dt

        # Compute reward
        reward = 0

        return self.state, reward, False, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            pass
        elif self.render_mode == "human":
            plot_vector_field(self.dynamics)

    def close(self):
        pass
