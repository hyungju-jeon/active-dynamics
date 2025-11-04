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

    def set_params(self, dyn_params: torch.Tensor | list[float] | Dict[str, float]):
        if isinstance(dyn_params, dict):
            self._set_params(**dyn_params)
        else:
            if isinstance(dyn_params, list):
                _dyn_params = torch.tensor(dyn_params, device=self.device, dtype=torch.float16)
            else:
                _dyn_params = dyn_params.to(self.device)

            if _dyn_params.ndim == 1:
                _dyn_params = _dyn_params.unsqueeze(0)

            self._set_params(*_dyn_params.mT)

    def _set_params(self, *args, **kwargs):
        raise NotImplementedError("_set_params method must be implemented in subclasses.")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Handle single vector input
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            result = self.compute(x)
            return result.view_as(x)  # Remove batch dimension
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
