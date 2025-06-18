"""A 2D “wind” field based on vector-field dynamics."""

import torch
import numpy as np
from actdyn.environment.vectorfield import VectorFieldEnv
from actdyn.utils.vectorfield_definition import (
    LimitCycle,
    DoubleLimitCycle,
    MultiAttractor,
)

vf_from_string = {
    "limit_cycle":     LimitCycle,
    "double_limit":    DoubleLimitCycle,
    "multi_attractor": MultiAttractor,
}

class WindField(VectorFieldEnv):
    """
    A 2D “wind” field based on vector-field dynamics.
    Inherits all the VF machinery, but only exposes a get_wind(x,y) -> np.array([wx,wy]).
    """
    def __init__(
        self,
        dynamics_type: str = "multi_attractor",
        noise_scale: float = 0.0,
        dt: float = 1.0,
        device: str = "cpu",
        action_bounds=(0.0, 0.0),
    ):
        # 2D state space, no actions in a “wind field”
        super().__init__(
            dynamics_type=dynamics_type,
            state_dim=2,
            noise_scale=noise_scale,
            dt=dt,
            device=device,
            render_mode=None,
            action_bounds=action_bounds,
            state_bounds=None,
        )

        # instantiate dynamics
        dyn_cls = vf_from_string[dynamics_type]
        self.dynamics = dyn_cls(device=device)

    def get_wind(self, x: float, y: float) -> np.ndarray:
        """
        Query the wind vector at world-coords (x, y).
        Returns a NumPy array shape (2,) = [wx, wy].
        """
        # make a batch of size‐1, shape (1,2)
        pt = torch.tensor([[x, y]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            w = self.dynamics(pt)
        return w[0].cpu().numpy()
