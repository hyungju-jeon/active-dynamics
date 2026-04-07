from __future__ import annotations

from typing import Dict, Tuple, Optional
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class VectorField:
    """A class for generating and manipulating 2D vector fields."""

    def __init__(
        self,
        alpha: float = 1.0,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize the VectorField instance.

        Args:
            model: Type of vector field model.
            x_range: Range of coordinates (-x_range to x_range).
            n_grid: Number of grid points in each dimension.
        """
        self.device = torch.device(device)
        self.alpha = alpha
        self.xy = None

    @torch.no_grad()
    def compute(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("compute method must be implemented in subclasses.")

    def set_params(self, *dyn_params: torch.Tensor | list[float] | Dict[str, float]):
        """Set model parameters from a tensor, list, dict, or expanded arguments."""

        def _format_param_args(params_tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
            param_args = []
            for param in params_tensor.mT:
                if params_tensor.shape[0] > 1 and param.ndim == 1:
                    param = param.unsqueeze(-1)
                param_args.append(param)
            return tuple(param_args)

        if len(dyn_params) == 1 and isinstance(dyn_params[0], dict):
            self._set_params(**dyn_params[0])
            return

        if len(dyn_params) == 1:
            raw_params = dyn_params[0]
            if isinstance(raw_params, list):
                params_tensor = torch.tensor(raw_params, device=self.device, dtype=torch.float32)
            else:
                params_tensor = raw_params.to(self.device, dtype=torch.float32)

            if params_tensor.ndim == 1:
                params_tensor = params_tensor.unsqueeze(0)
            self.dyn_params = params_tensor
            self._set_params(*_format_param_args(params_tensor))
            return

        params_list = []
        for param in dyn_params:
            if isinstance(param, torch.Tensor):
                param_tensor = param.to(self.device, dtype=torch.float32)
            else:
                param_tensor = torch.as_tensor(param, device=self.device, dtype=torch.float32)
            params_list.append(param_tensor)

        params_tensor = torch.stack(params_list, dim=-1)
        if params_tensor.ndim == 1:
            params_tensor = params_tensor.unsqueeze(0)
        self.dyn_params = params_tensor
        self._set_params(*_format_param_args(params_tensor))

    def _set_params(self, *args, **kwargs):
        raise NotImplementedError("_set_params method must be implemented in subclasses.")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Handle single vector input
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            result = self.compute(x)
            return result.squeeze(0)
        return self.compute(x)


class LimitCycle(VectorField):
    def __init__(
        self,
        dyn_params: torch.Tensor | list[float] | Dict[str, float] = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        if dyn_params is None:
            self.w = 1
            self.d = 1
        else:
            self.set_params(dyn_params)

    def _set_params(self, w=1.0, d=1.0):
        self.w = w
        self.d = d

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        U = x[..., 0] * (self.d - r**2) - self.w * x[..., 1]
        V = x[..., 1] * (self.d - r**2) + self.w * x[..., 0]

        U = self.alpha * U
        V = self.alpha * V

        return torch.stack([U, V], dim=-1)


class DoubleLimitCycle(VectorField):
    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        if dyn_param is None:
            self.w = 1
            self.d = 1
        else:
            self.set_params(dyn_param)

    def _set_params(self, w=1.0, d=1.0):
        self.w = w
        self.d = d

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        U = x[..., 0] * (self.d - r) - self.w * x[..., 1] * (2 * self.d - r)
        V = x[..., 1] * (self.d - r) + self.w * x[..., 0] * (2 * self.d - r)

        U = self.alpha * U
        V = self.alpha * V
        return torch.stack([U, V], dim=-1)


class BistableLimitCycle(VectorField):
    """Smoothly blends two local limit cycles into a left/right bistable system."""

    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.center_offset = float(kwargs.get("center_offset", 1.6))
        self.gate_sharpness = float(kwargs.get("gate_sharpness", 3.0))
        if dyn_param is None:
            self.omega = 1.0
            self.radius = 0.9
        else:
            self.set_params(dyn_param)

    def _set_params(self, omega=1.0, radius=0.9):
        self.omega = omega
        self.radius = radius

    def _local_cycle(self, x: torch.Tensor, center_x: float) -> torch.Tensor:
        px = x[..., 0] - float(center_x)
        py = x[..., 1]
        r = torch.sqrt(px**2 + py**2).clamp_min(1e-6)
        k = self.radius - r
        u = px * k - self.omega * py
        v = py * k + self.omega * px
        return torch.stack([u, v], dim=-1)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_sharpness * x[..., 0])
        left = self._local_cycle(x, center_x=-self.center_offset)
        right = self._local_cycle(x, center_x=self.center_offset)
        field = (1.0 - gate).unsqueeze(-1) * left + gate.unsqueeze(-1) * right
        return self.alpha * field


# TODO: Fix the code to match the new structure
class MultiAttractor(VectorField):

    def __init__(
        self,
        x_range: float = 2,
        n_grid: int = 40,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        alpha: float = 0.25,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(x_range=x_range, n_grid=n_grid, device=device, **kwargs)
        if dyn_param is None:
            self.w_attractor = 1.0
            self.length_scale = 0.5
        else:
            self.set_params(*dyn_param)

        self.alpha = alpha
        self.U, self.V = self.generate_vector_field()

    def set_params(self, w_attractor=1.0, length_scale=0.5):
        self.w_attractor = w_attractor
        self.length_scale = length_scale

    @torch.no_grad()
    def generate_vector_field(self) -> Tuple[torch.Tensor, torch.Tensor]:
        base_kernel = RBFKernel(ard_num_dims=2)
        base_kernel.lengthscale = self.length_scale
        kernel = ScaleKernel(base_kernel)
        kernel.outputscale = 0.5

        with torch.no_grad(), gpytorch.settings.fast_computations(True):
            kernel.eval()
            K = kernel(self.xy).evaluate()

        eigenvalues, eigenvectors = torch.linalg.eigh(K)
        eigenvalues = eigenvalues.clamp(min=1e-4)
        eps = torch.randn(2, K.shape[0], device=self.xy.device)
        samples = torch.matmul(eps * torch.sqrt(eigenvalues), eigenvectors.T)

        # Reshape and normalize
        grid_size = int(torch.sqrt(torch.tensor(self.xy.shape[0])))
        U = samples[0].reshape(grid_size, grid_size)
        V = samples[1].reshape(grid_size, grid_size)

        magnitude = torch.hypot(U, V).clamp(min=1e-8)
        U = self.alpha * U / magnitude
        V = self.alpha * V / magnitude

        if self.w_attractor > 0:
            U_attract = -self.xy[:, 0] * torch.sqrt(torch.sum(self.xy**2, 1)) * self.w_attractor
            V_attract = -self.xy[:, 1] * torch.sqrt(torch.sum(self.xy**2, 1)) * self.w_attractor
            U += U_attract.reshape(grid_size, grid_size)
            V += V_attract.reshape(grid_size, grid_size)

        self.U, self.V = U, V
        return U, V

    def compute(self, state: torch.Tensor) -> torch.Tensor:
        """Compute vector field at given state points using interpolation."""
        # Generate the vector field if not already generated
        if self.U is None or self.V is None:
            self.generate_vector_field()

        # Create interpolator for U and V components based on X, Y grid
        x = self.X[0].cpu().numpy()
        y = self.Y[:, 0].cpu().numpy()
        U_interp = RegularGridInterpolator(
            (x, y),
            self.U.cpu().numpy(),
            bounds_error=False,
            fill_value=None,
        )
        V_interp = RegularGridInterpolator(
            (x, y),
            self.V.cpu().numpy(),
            bounds_error=False,
            fill_value=None,
        )

        # Interpolate at given state points
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy()
            device = state.device
        else:
            state_np = np.array(state)
            device = torch.device("cpu")

        u_vals = U_interp(state_np)
        v_vals = V_interp(state_np)

        # Stack and return as tensor
        result = np.stack([v_vals, u_vals], axis=-1)
        return torch.tensor(result, device=device, dtype=torch.float32)


class VanDerPol(VectorField):
    """Van der Pol oscillator"""

    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        if dyn_param is None:
            self.mu = 1.0
            self.w = 1.0
        else:
            self.set_params(dyn_param)

    def _set_params(self, mu=1.0, w=1.0):
        self.mu = mu
        self.w = w

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        U = x[..., 1]
        V = self.mu * (1 - x[..., 0] ** 2) * x[..., 1] - self.w * x[..., 0]

        U = self.alpha * U
        V = self.alpha * V

        return torch.stack([U, V], dim=-1)


class Duffing(VectorField):
    """Duffing oscillator"""

    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        if dyn_param is None:
            self.a = 0.1
            self.b = -0.1
            self.c = 0.1
        else:
            self.set_params(dyn_param)

    def _set_params(self, a=0.1, b=-0.1, c=0.1):
        self.a = a
        self.b = b
        self.c = c

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        U = x[..., 1]
        V = self.a * x[..., 1] - x[..., 0] * (self.b + self.c * x[..., 0] ** 2)
        U = self.alpha * U
        V = self.alpha * V

        return torch.stack([U, V], dim=-1)


class DampedPendulum(VectorField):
    """Damped pendulum with learnable damping and gravity scale."""

    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        if dyn_param is None:
            self.damping = -0.35
            self.gravity = 1.2
        else:
            self.set_params(dyn_param)

    def _set_params(self, damping=-0.35, gravity=1.2):
        self.damping = damping
        self.gravity = gravity

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        U = x[..., 1]
        V = self.damping * x[..., 1] - self.gravity * torch.sin(x[..., 0])
        U = self.alpha * U
        V = self.alpha * V
        return torch.stack([U, V], dim=-1)


class DoubleIntegrator(VectorField):
    """Controlled double-integrator with unknown drift bias and damping."""

    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        if dyn_param is None:
            self.bias = 0.15
            self.damping = 0.55
        else:
            self.set_params(dyn_param)

    def _set_params(self, bias=0.15, damping=0.55):
        self.bias = bias
        self.damping = damping

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        U = x[..., 1]
        V = self.bias - self.damping * x[..., 1]
        U = self.alpha * U
        V = self.alpha * V
        return torch.stack([U, V], dim=-1)


class FitzHughNagumo(VectorField):
    """FitzHugh-Nagumo excitable dynamics."""

    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        if dyn_param is None:
            self.set_params([0.7, 0.8, 12.5, 0.5])
        else:
            self.set_params(dyn_param)

    def _set_params(self, a=0.7, b=0.8, tau=12.5, i_ext=0.5):
        self.a = a
        self.b = b
        self.tau = tau
        self.i_ext = i_ext

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Compute FitzHugh-Nagumo velocity for state shape (..., 2)."""

        U = x[..., 0] - (x[..., 0] ** 3) / 3.0 - x[..., 1] + self.i_ext
        V = (x[..., 0] + self.a - self.b * x[..., 1]) / self.tau
        U = self.alpha * U
        V = self.alpha * V
        return torch.stack([U, V], dim=-1)


class Hopf(VectorField):
    """Generalized Hopf normal form with optional nonlinear frequency shift."""

    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        if dyn_param is None:
            self.set_params([1.0, 1.0, 0.0])
        else:
            self.set_params(dyn_param)

    def _set_params(self, mu=1.0, omega=1.0, beta=0.0):
        self.mu = mu
        self.omega = omega
        self.beta = beta

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Hopf velocity for state shape (..., 2)."""

        radius_sq = x[..., 0] ** 2 + x[..., 1] ** 2
        omega_eff = self.omega + self.beta * radius_sq
        U = (self.mu - radius_sq) * x[..., 0] - omega_eff * x[..., 1]
        V = omega_eff * x[..., 0] + (self.mu - radius_sq) * x[..., 1]
        U = self.alpha * U
        V = self.alpha * V
        return torch.stack([U, V], dim=-1)


class SnowMan(VectorField):
    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        if dyn_param is None:
            self.w = 1
            self.d = 1
        else:
            self.set_params(dyn_param)

    def _set_params(self, w=1.0, d=1.0, beta=10.0):
        self.w = w
        self.d = d
        self.beta = beta

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        d = self.d
        r = torch.sqrt((x[..., 0] - d) ** 2 + x[..., 1] ** 2)
        U1 = (x[..., 0] - d) * (d**2 - r**2) - self.w * x[..., 1]
        V1 = x[..., 1] * (d**2 - r**2) + self.w * (x[..., 0] - d)

        r = torch.sqrt((x[..., 0] + d) ** 2 + x[..., 1] ** 2)
        U2 = (x[..., 0] + d) * (d**2 - r**2) + self.w * x[..., 1]
        V2 = x[..., 1] * (d**2 - r**2) - self.w * (x[..., 0] + d)

        U = self.alpha * (
            torch.sigmoid(self.beta * x[..., 0]) * U1 + torch.sigmoid(-self.beta * x[..., 0]) * U2
        )
        V = self.alpha * (
            torch.sigmoid(self.beta * x[..., 0]) * V1 + torch.sigmoid(-self.beta * x[..., 0]) * V2
        )

        return torch.stack([U, V], dim=-1)


if __name__ == "__main__":
    # Example usage with smaller grid size
    vf = VanDerPol(x_range=2.5, n_grid=50)
