from typing import Tuple, Optional, Literal
import numpy as np  # kept only for matplotlib compatibility
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel

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

    def __init__(
        self,
        x_range: float = 2,
        n_grid: int = 40,
        device: str = "cpu",
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
        self.X: Optional[ArrayType] = None
        self.Y: Optional[ArrayType] = None
        self.xy: Optional[ArrayType] = None
        self.device = torch.device(device)

        self.create_grid(self.x_range, self.n_grid)

    @torch.no_grad()
    def create_grid(self, x_range: float, n_grid: int) -> None:
        """Create a 2D grid for the vector field using PyTorch."""
        x = torch.linspace(-x_range, x_range, n_grid, device=self.device)
        y = torch.linspace(-x_range, x_range, n_grid, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        xy = torch.stack([X.flatten(), Y.flatten()], dim=1)

        self.X, self.Y, self.xy = X, Y, xy

    @torch.no_grad()
    def generate_vector_field(self, **kwargs) -> None:
        raise NotImplementedError(
            "initialize_vector_field method must be implemented in subclasses."
        )

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
        x_range: float = 2,
        n_grid: int = 40,
        w: float = 1,
        d: float = 1.0,
        **kwargs
    ):
        super().__init__(x_range=x_range, n_grid=n_grid)
        self.w = w
        self.d = d
        self.alpha = 1
        self.scaling = self.get_scaling()
        self.alpha = 1 / self.scaling

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
        return UV[:, 0].reshape(grid_size, grid_size), UV[:, 1].reshape(
            grid_size, grid_size
        )


class DoubleLimitCycle(VectorField):
    def __init__(
        self,
        x_range: float = 2,
        n_grid: int = 40,
        w: float = 1,
        d: float = 1.0,
        **kwargs
    ):
        super().__init__(x_range=x_range, n_grid=n_grid)
        self.w = w
        self.d = d
        self.alpha = 1
        self.scaling = self.get_scaling()
        self.alpha = 1 / self.scaling

    def get_scaling(self):
        if self.xy is None:
            self.create_grid(self.x_range, self.n_grid)
        U, V = self.generate_vector_field()
        speed = torch.sqrt(U**2 + V**2)
        speed = speed.max()
        return speed

    def compute(self, x: ArrayType) -> ArrayType:
        r = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        U = x[..., 0] * (self.d - r**2) - self.w * x[..., 1] * (2 * self.d - r**2)
        V = x[..., 1] * (self.d - r**2) + self.w * x[..., 0] * (2 * self.d - r**2)

        U = self.alpha * U
        V = self.alpha * V
        return torch.stack([U, V], dim=-1)

    def generate_vector_field(self) -> Tuple[ArrayType, ArrayType]:
        if self.xy is None:
            self.create_grid(self.x_range, self.n_grid)
        grid_size = int(torch.sqrt(torch.tensor(self.xy.shape[0])))
        UV = self.compute(self.xy)
        return UV[:, 0].reshape(grid_size, grid_size), UV[:, 1].reshape(
            grid_size, grid_size
        )


class MultiAttractor(VectorField):

    def __init__(
        self,
        x_range: float = 2,
        n_grid: int = 40,
        w_attractor: float = 0.1,
        length_scale: float = 0.5,
        alpha: float = 0.1,
        **kwargs
    ):
        super().__init__(x_range=x_range, n_grid=n_grid)
        self.w_attractor = w_attractor
        self.length_scale = length_scale
        self.alpha = alpha

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
            U_attract = (
                -self.xy[:, 0] * torch.sqrt(torch.sum(self.xy**2, 1)) * self.w_attractor
            )
            V_attract = (
                -self.xy[:, 1] * torch.sqrt(torch.sum(self.xy**2, 1)) * self.w_attractor
            )
            U += U_attract.reshape(grid_size, grid_size)
            V += V_attract.reshape(grid_size, grid_size)

        return U, V


if __name__ == "__main__":
    # Example usage with smaller grid size
    vf = VectorField(x_range=2.5, n_grid=50)
    vf.generate_vector_field(random_seed=10)
