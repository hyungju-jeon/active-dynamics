from __future__ import annotations

"""Vector field environment implementation."""

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from actdyn.utils.vectorfield_definition import (
    BistableLimitCycle,
    DampedPendulum,
    LimitCycle,
    MultiAttractor,
    DoubleLimitCycle,
    DoubleIntegrator,
    VanDerPol,
    Duffing,
    FitzHughNagumo,
    Hopf,
    MultiStable,
    SnowMan,
    AsymmetricBasin,
)
from typing import Optional, Tuple, Dict, Any, Sequence
from actdyn.utils.visualize import plot_vector_field


vf_from_string = {
    "limit_cycle": LimitCycle,
    "double_limit_cycle": DoubleLimitCycle,
    "bistable_limit_cycle": BistableLimitCycle,
    "multi_attractor": MultiAttractor,
    "van_der_pol": VanDerPol,
    "duffing": Duffing,
    "asymmetric_basin": AsymmetricBasin,
    "multi_stable": MultiStable,
    "damped_pendulum": DampedPendulum,
    "double_integrator": DoubleIntegrator,
    "fitzhugh_nagumo": FitzHughNagumo,
    "hopf": Hopf,
    "snowman": SnowMan,
}


def build_vectorfield(
    dynamics_type: str,
    dyn_params: torch.Tensor | list[float] | Dict[str, float] | None = None,
    *,
    dynamics_alpha: float = 1.0,
    device: torch.device | str = "cpu",
):
    if dynamics_type not in vf_from_string:
        raise ValueError(f"Unknown dynamics type: {dynamics_type}")
    vf_device = torch.device(device)
    vf = vf_from_string[dynamics_type](device=str(vf_device), alpha=float(dynamics_alpha))
    if dyn_params is not None:
        value: torch.Tensor | Dict[str, float]
        if isinstance(dyn_params, dict):
            value = dyn_params
        else:
            value = torch.as_tensor(dyn_params, device=vf_device, dtype=torch.float32)
        vf.set_params(value)
    return vf


def _state_device(state: torch.Tensor | np.ndarray, dyn_params: Any) -> torch.device:
    if torch.is_tensor(state):
        return state.device
    if torch.is_tensor(dyn_params):
        return dyn_params.device
    return torch.device("cpu")


def _align_dyn_params_torch(
    state: torch.Tensor, dyn_params: torch.Tensor | list[float] | np.ndarray
) -> torch.Tensor:
    params = torch.as_tensor(dyn_params, dtype=torch.float32, device=state.device)
    while params.ndim < state.ndim:
        params = params.unsqueeze(-2)
    return params


def _align_dyn_params_np(
    state: np.ndarray, dyn_params: np.ndarray | list[float] | torch.Tensor
) -> np.ndarray:
    params = np.asarray(dyn_params, dtype=np.float64)
    while params.ndim < state.ndim:
        params = np.expand_dims(params, axis=-2)
    return params


def _batched_jacobian(output: torch.Tensor, wrt: torch.Tensor) -> torch.Tensor:
    with torch.enable_grad():
        flat_output = output.reshape(-1, output.shape[-1])
        rows = []
        for out_idx in range(flat_output.shape[-1]):
            grads = torch.autograd.grad(
                flat_output[:, out_idx].sum(),
                wrt,
                retain_graph=out_idx < flat_output.shape[-1] - 1,
                create_graph=True,
                allow_unused=False,
            )[0]
            rows.append(grads)
        jac = torch.stack(rows, dim=-2)
    return jac.reshape(*wrt.shape[:-1], output.shape[-1], wrt.shape[-1])


def residual_torch(
    dynamics_type: str,
    state: torch.Tensor,
    dyn_params: torch.Tensor | list[float] | Dict[str, float],
    *,
    dynamics_alpha: float,
) -> torch.Tensor:
    device = _state_device(state, dyn_params)
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
    params_t: torch.Tensor | Dict[str, float]
    if isinstance(dyn_params, dict):
        params_t = dyn_params
    else:
        params_t = _align_dyn_params_torch(state_t, dyn_params)
    flat_state = state_t.reshape(-1, state_t.shape[-1])
    if isinstance(params_t, dict):
        params_value: torch.Tensor | Dict[str, float] = params_t
    else:
        params_value = params_t.reshape(-1, params_t.shape[-1])
    vf = build_vectorfield(
        dynamics_type,
        params_value,
        dynamics_alpha=float(dynamics_alpha),
        device=state_t.device,
    )
    drift = vf.compute(flat_state)
    return drift.reshape(state_t.shape)


def jacobian_state_torch(
    dynamics_type: str,
    state: torch.Tensor,
    dyn_params: torch.Tensor | list[float] | Dict[str, float],
    *,
    dynamics_alpha: float,
) -> torch.Tensor:
    with torch.enable_grad():
        device = _state_device(state, dyn_params)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).detach().clone()
        state_t.requires_grad_(True)
        drift = residual_torch(
            dynamics_type,
            state_t,
            dyn_params,
            dynamics_alpha=float(dynamics_alpha),
        )
        return _batched_jacobian(drift, state_t)


def jacobian_param_torch(
    dynamics_type: str,
    state: torch.Tensor,
    dyn_params: torch.Tensor | list[float] | np.ndarray,
    *,
    dynamics_alpha: float,
) -> torch.Tensor:
    if isinstance(dyn_params, dict):
        raise TypeError("jacobian_param_torch does not support dict-valued dyn_params")
    with torch.enable_grad():
        device = _state_device(state, dyn_params)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        params_t = _align_dyn_params_torch(state_t, dyn_params).detach().clone()
        params_t.requires_grad_(True)
        drift = residual_torch(
            dynamics_type,
            state_t,
            params_t,
            dynamics_alpha=float(dynamics_alpha),
        )
        return _batched_jacobian(drift, params_t)


def residual_np(
    dynamics_type: str,
    state: np.ndarray,
    dyn_params: np.ndarray | list[float] | Dict[str, float],
    *,
    dynamics_alpha: float,
) -> np.ndarray:
    state_np = np.asarray(state, dtype=np.float32)
    if isinstance(dyn_params, dict):
        params_t: torch.Tensor | Dict[str, float] = dyn_params
    else:
        params_t = torch.as_tensor(
            _align_dyn_params_np(state_np, dyn_params), dtype=torch.float32
        )
    state_t = torch.as_tensor(state_np, dtype=torch.float32)
    with torch.no_grad():
        drift = residual_torch(
            dynamics_type,
            state_t,
            params_t,
            dynamics_alpha=float(dynamics_alpha),
        )
    return drift.cpu().numpy().astype(np.float64, copy=False)



def _pad_embedding_to_params(
    embedding: torch.Tensor,
    *,
    full_params: torch.Tensor | np.ndarray | Sequence[float],
    min_embedding_dim: int,
) -> torch.Tensor:
    if embedding.shape[-1] < int(min_embedding_dim):
        raise ValueError(
            f"Embedding must have at least {min_embedding_dim} coordinates, got shape {tuple(embedding.shape)}."
        )
    full = torch.as_tensor(full_params, dtype=embedding.dtype, device=embedding.device)
    if embedding.shape[-1] >= full.shape[0]:
        return embedding[..., : full.shape[0]]
    tail = full[embedding.shape[-1]:]
    if embedding.ndim == 1:
        return torch.cat((embedding, tail), dim=0)
    tail = tail.reshape(*([1] * (embedding.ndim - 1)), -1).expand(*embedding.shape[:-1], -1)
    return torch.cat((embedding, tail), dim=-1)


def build_system_jacobians(
    *,
    dynamics_type: str,
    full_params: torch.Tensor | np.ndarray | Sequence[float],
    min_embedding_dim: int = 1,
    dynamics_alpha: float,
):
    def _fe(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        z_t = torch.as_tensor(z, dtype=torch.float32, device=z.device).detach()
        e_t = torch.as_tensor(e, dtype=torch.float32, device=z_t.device).detach().clone()
        e_t.requires_grad_(True)
        drift = residual_torch(
            dynamics_type,
            z_t,
            _pad_embedding_to_params(
                e_t,
                full_params=full_params,
                min_embedding_dim=int(min_embedding_dim),
            ),
            dynamics_alpha=float(dynamics_alpha),
        )
        return _batched_jacobian(drift, e_t)

    def _fz(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return jacobian_state_torch(
            dynamics_type,
            z,
            _pad_embedding_to_params(
                torch.as_tensor(e, dtype=torch.float32, device=z.device),
                full_params=full_params,
                min_embedding_dim=int(min_embedding_dim),
            ),
            dynamics_alpha=float(dynamics_alpha),
        )

    return _fe, _fz


def rollout_no_input(
    z0: torch.Tensor,
    e: torch.Tensor,
    *,
    dynamics_type: str,
    full_params: torch.Tensor | np.ndarray | Sequence[float],
    min_embedding_dim: int = 1,
    horizon: int,
    dt: float,
    dynamics_alpha: float,
) -> torch.Tensor:
    z = z0.clone()
    dyn_params = _pad_embedding_to_params(
        torch.as_tensor(e, dtype=torch.float32, device=z.device),
        full_params=full_params,
        min_embedding_dim=int(min_embedding_dim),
    )
    traj = [z]
    for _ in range(int(horizon)):
        drift = residual_torch(
            dynamics_type,
            z,
            dyn_params,
            dynamics_alpha=float(dynamics_alpha),
        )
        z = z + float(dt) * drift
        traj.append(z)
    return torch.stack(traj, dim=1)

def step_np(
    dynamics_type: str,
    state: np.ndarray,
    action: np.ndarray,
    *,
    dyn_params: np.ndarray | list[float] | Dict[str, float],
    dt: float,
    dynamics_alpha: float,
    clip_limit: float,
) -> np.ndarray:
    next_state = np.asarray(state, dtype=np.float64) + float(dt) * (
        residual_np(dynamics_type, state, dyn_params, dynamics_alpha=float(dynamics_alpha))
        + np.asarray(action, dtype=np.float64)
    )
    return np.clip(next_state, -float(clip_limit), float(clip_limit))


class VectorFieldEnv(gym.Env):
    """Unified environment for latent dynamics simulation."""

    def __init__(
        self,
        dynamics_type: str = "limit_cycle",
        d_state: int = 2,
        Q: float = 0.1,
        dt: float = 0.1,
        device: str = "cpu",
        dyn_params: Optional[list[float]] | torch.Tensor = None,  #
        render_mode: Optional[str] = None,
        action_bounds: Sequence[float] = (-1.0, 1.0),
        state_bounds: Optional[Sequence[float]] = None,
        initial_state: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.d_state = d_state
        self.Q = Q
        self.dt = dt
        self.device = torch.device(device)
        self.render_mode = render_mode
        self.initial_state = None
        if initial_state is not None:
            init = torch.as_tensor(initial_state, dtype=torch.float16, device=self.device).reshape(-1)
            if init.numel() != d_state:
                raise ValueError(
                    f"initial_state must have {d_state} values, got {init.numel()}"
                )
            self.initial_state = init

        # Initialize spaces with configurable bounds
        self.action_space = self._set_space_bounds(action_bounds, d_state)

        if state_bounds is None:
            state_bounds = (-np.inf, np.inf)
        self.observation_space = self._set_space_bounds(state_bounds, d_state)

        # Initialize dynamics
        if dynamics_type not in vf_from_string:
            raise ValueError(f"Unknown dynamics type: {dynamics_type}")
        self.dynamics = vf_from_string[dynamics_type](
            dyn_param=dyn_params, device=self.device, **kwargs
        )
        self.set_params(dyn_params)

        # Initialize state
        if self.initial_state is not None:
            self.state = self.initial_state.clone()
        else:
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

    def set_params(self, dyn_params: torch.Tensor | list[float] | Dict[str, float] | None):
        """Set dynamics parameters."""
        if dyn_params is None:
            return  # Do nothing if no params provided
        if hasattr(self.dynamics, "set_params"):
            value: torch.Tensor | Dict[str, float]
            if isinstance(dyn_params, dict):
                value = dyn_params
            else:
                value = torch.as_tensor(dyn_params, device=self.device, dtype=torch.float32)
            self.dynamics.set_params(value)

    def compute_dynamics(self, state: torch.Tensor) -> torch.Tensor:
        """Compute vector field at given state."""
        return self.dynamics(state)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        reset_state = None
        if options is not None and "initial_state" in options:
            reset_state = torch.as_tensor(
                options["initial_state"], dtype=torch.float16, device=self.device
            ).reshape(-1)
            if reset_state.numel() != self.d_state:
                raise ValueError(
                    f"options['initial_state'] must have {self.d_state} values, got {reset_state.numel()}"
                )
        elif self.initial_state is not None:
            reset_state = self.initial_state

        if reset_state is not None:
            self.state = reset_state.clone()
        else:
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
