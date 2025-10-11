from typing import Any, Dict, Tuple, Optional, Literal
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from abc import abstractmethod
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Type aliases
ArrayType = torch.Tensor
ModelType = Literal["multi", "ring", "limitcycle", "single"]


class VectorField:
    """A class for generating and manipulating 2D vector fields.

    Args:
        model: Type of vector field model. Defaults to "multi".
        x_range: Range of coordinates (-x_range to x_range). Defaults to 2.
        n_grid: Number of grid points in each dimension. Defaults to 40.

    Attributes:
        x_range: Range of x and y coordinates (-x_range to x_range).
        n_grid: Number of grid points in each dimension.
        model: Type of vector field model to generate.
        X: X coordinates of the grid points.
        Y: Y coordinates of the grid points.
        xy: Combined XY coordinates.
        U: X components of the vector field.
        V: Y components of the vector field.
    """

    X: torch.Tensor
    Y: torch.Tensor
    xy: torch.Tensor
    U: torch.Tensor
    V: torch.Tensor

    def __init__(
        self,
        x_range: float = 2,
        n_grid: int = 40,
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
        self.x_range = x_range
        self.n_grid = n_grid
        self.device = torch.device(device)

        self.create_grid(self.x_range, self.n_grid)

    @torch.no_grad()
    def create_grid(self, x_range: float, n_grid: int) -> None:
        """Create a 2D grid for the vector field using PyTorch."""
        x = torch.linspace(-x_range, x_range, n_grid, device=self.device)
        y = torch.linspace(-x_range, x_range, n_grid, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        xy = torch.stack([X.flatten(), Y.flatten()], dim=1).to(self.device)

        self.X, self.Y, self.xy = X, Y, xy

    @abstractmethod
    @torch.no_grad()
    def generate_vector_field(self, **kwargs) -> None:
        """Generate the vector field components U and V and store them as attributes."""
        pass

    @abstractmethod
    @torch.no_grad()
    def compute(self, x: ArrayType) -> ArrayType:
        raise NotImplementedError("compute method must be implemented in subclasses.")

    def __call__(self, x: ArrayType) -> ArrayType:
        """Callable interface for interpolation.

        Args:
            x: Points to interpolate at
        Returns:
            Array containing interpolated vector field values
        """
        # Handle single vector input
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            result = self.compute(x)
            return result.view_as(x)  # Remove batch dimension
        return self.compute(x)


class LimitCycle(VectorField):
    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        x_range: float = 2,
        n_grid: int = 40,
        w: float = 1,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(x_range=x_range, n_grid=n_grid, device=device)
        if dyn_param is None:
            self.w = 1
            self.d = x_range / 2
        else:
            self.set_params(dyn_param)

        self.alpha = 1
        self.scaling = self.get_scaling()
        self.alpha = 2 / self.scaling

    def set_params(self, dyn_param):
        if isinstance(dyn_param, list):
            dyn_param = torch.tensor(dyn_param, device=self.device, dtype=torch.float32)
        self.w = dyn_param[..., 0]
        self.d = dyn_param[..., 1]

    def get_scaling(self):
        if self.xy is None:
            self.create_grid(self.x_range, self.n_grid)
        U, V = self.generate_vector_field()
        speed = torch.sqrt(U**2 + V**2)
        speed = speed.max()
        return speed

    def compute(self, x: ArrayType) -> ArrayType:
        r = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        U = x[..., 0] * (self.d - r**2) - self.w * x[..., 1]
        V = x[..., 1] * (self.d - r**2) + self.w * x[..., 0]

        U = self.alpha * U
        V = self.alpha * V

        return torch.stack([U, V], dim=-1)

    def generate_vector_field(self) -> Tuple[ArrayType, ArrayType]:
        if self.xy is None:
            self.create_grid(self.x_range, self.n_grid)
        grid_size = int(torch.sqrt(torch.tensor(self.xy.shape[0])))
        UV = self.compute(self.xy)
        return UV[:, 0].reshape(grid_size, grid_size), UV[:, 1].reshape(grid_size, grid_size)


class DoubleLimitCycle(VectorField):
    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        x_range: float = 2,
        n_grid: int = 40,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(x_range=x_range, n_grid=n_grid, device=device, **kwargs)
        if dyn_param is None:
            self.w = 1
            self.d = x_range / 2
        else:
            self.set_params(dyn_param)

        self.alpha = 1
        self.scaling = self.get_scaling()
        self.alpha = 1 / self.scaling

    def set_params(self, dyn_param):
        if isinstance(dyn_param, list):
            dyn_param = torch.tensor(dyn_param, device=self.device, dtype=torch.float32)
        self.w = dyn_param[..., 0]
        self.d = dyn_param[..., 1]

    def get_scaling(self):
        if self.xy is None:
            self.create_grid(self.x_range, self.n_grid)
        U, V = self.generate_vector_field()
        speed = torch.sqrt(U**2 + V**2)
        speed = speed.max()
        return speed

    def compute(self, x: ArrayType) -> ArrayType:
        r = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        U = x[..., 0] * (self.d - r) - self.w * x[..., 1] * (2 * self.d - r)
        V = x[..., 1] * (self.d - r) + self.w * x[..., 0] * (2 * self.d - r)

        U = self.alpha * U
        V = self.alpha * V
        return torch.stack([U, V], dim=-1)

    def generate_vector_field(self) -> Tuple[ArrayType, ArrayType]:
        if self.xy is None:
            self.create_grid(self.x_range, self.n_grid)
        grid_size = int(torch.sqrt(torch.tensor(self.xy.shape[0])))
        UV = self.compute(self.xy)
        return UV[:, 0].reshape(grid_size, grid_size), UV[:, 1].reshape(grid_size, grid_size)


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
            self.set_params(dyn_param)

        self.alpha = alpha
        self.U, self.V = self.generate_vector_field()

    def set_params(self, dyn_param):
        if isinstance(dyn_param, list):
            dyn_param = torch.tensor(dyn_param, device=self.device, dtype=torch.float32)
        self.w_attractor = dyn_param[..., 0]
        self.length_scale = dyn_param[..., 1]

    @torch.no_grad()
    def generate_vector_field(self) -> Tuple[ArrayType, ArrayType]:
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

    def compute(self, state: ArrayType) -> ArrayType:
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
        x_range: float = 2,
        n_grid: int = 40,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(x_range=x_range, n_grid=n_grid, device=device, **kwargs)
        if dyn_param is None:
            self.mu = 1.0
            self.w = 1.0
        else:
            self.set_params(dyn_param)

        self.alpha = 1
        if kwargs.get("alpha") is not None:
            self.alpha = kwargs.get("alpha")
        else:
            self.scaling = self.get_scaling()
            self.alpha = 2 / self.scaling

    def set_params(self, dyn_param):
        if isinstance(dyn_param, list):
            dyn_param = torch.tensor(dyn_param, device=self.device, dtype=torch.float32)
        self.mu = dyn_param[..., 0]
        self.w = dyn_param[..., 1]

    def get_scaling(self):
        if self.xy is None:
            self.create_grid(self.x_range, self.n_grid)
        U, V = self.generate_vector_field()
        speed = torch.sqrt(U**2 + V**2)
        speed = speed.max()
        return speed

    def compute(self, x: ArrayType) -> ArrayType:
        U = x[..., 1]
        V = self.mu * (1 - x[..., 0] ** 2) * x[..., 1] - self.w * x[..., 0]

        U = self.alpha * U
        V = self.alpha * V

        return torch.stack([U, V], dim=-1)

    def generate_vector_field(self) -> Tuple[ArrayType, ArrayType]:
        if self.xy is None:
            self.create_grid(self.x_range, self.n_grid)
        grid_size = int(torch.sqrt(torch.tensor(self.xy.shape[0])))
        UV = self.compute(self.xy)
        return UV[:, 0].reshape(grid_size, grid_size), UV[:, 1].reshape(grid_size, grid_size)


class Duffing(VectorField):
    """Duffing oscillator"""

    def __init__(
        self,
        x_range: float = 2,
        n_grid: int = 40,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(x_range=x_range, n_grid=n_grid, device=device, **kwargs)
        if dyn_param is None:
            self.a = 0.1
            self.b = -0.1
            self.c = 0.1
        else:
            self.set_params(dyn_param)

    def set_params(self, dyn_param):
        if isinstance(dyn_param, list):
            dyn_param = torch.tensor(dyn_param, device=self.device, dtype=torch.float32)
        self.a = dyn_param[..., 0]
        self.b = dyn_param[..., 1]
        self.c = dyn_param[..., 2]

    def get_scaling(self):
        if self.xy is None:
            self.create_grid(self.x_range, self.n_grid)
        U, V = self.generate_vector_field()
        speed = torch.sqrt(U**2 + V**2)
        speed = speed.max()
        return speed

    def compute(self, x: ArrayType) -> ArrayType:
        U = x[..., 1]
        V = self.a * x[..., 1] - x[..., 0] * (self.b + self.c * x[..., 0] ** 2)

        return torch.stack([U, V], dim=-1)

    def generate_vector_field(self) -> Tuple[ArrayType, ArrayType]:
        if self.xy is None:
            self.create_grid(self.x_range, self.n_grid)
        grid_size = int(torch.sqrt(torch.tensor(self.xy.shape[0])))
        UV = self.compute(self.xy)
        return UV[:, 0].reshape(grid_size, grid_size), UV[:, 1].reshape(grid_size, grid_size)


if __name__ == "__main__":
    # Example usage with smaller grid size
    vf = VanDerPol(x_range=2.5, n_grid=50)
    vf.generate_vector_field()
