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
            return tuple(
                params_tensor[..., idx] for idx in range(params_tensor.shape[-1])
            )

        if len(dyn_params) == 1 and isinstance(dyn_params[0], dict):
            self._set_params(**dyn_params[0])
            return

        if len(dyn_params) == 1:
            raw_params = dyn_params[0]
            if isinstance(raw_params, list):
                params_tensor = torch.tensor(
                    raw_params, device=self.device, dtype=torch.float32
                )
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
                param_tensor = torch.as_tensor(
                    param, device=self.device, dtype=torch.float32
                )
            params_list.append(param_tensor)

        params_tensor = torch.stack(params_list, dim=-1)
        if params_tensor.ndim == 1:
            params_tensor = params_tensor.unsqueeze(0)
        self.dyn_params = params_tensor
        self._set_params(*_format_param_args(params_tensor))

    def _set_params(self, *args, **kwargs):
        raise NotImplementedError(
            "_set_params method must be implemented in subclasses."
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Handle single vector input
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            result = self.compute(x)
            return result.squeeze(0)
        return self.compute(x)

    def _broadcast_param(
        self, param: torch.Tensor | float, x: torch.Tensor
    ) -> torch.Tensor:
        value = torch.as_tensor(param, dtype=x.dtype, device=x.device)
        target = x[..., 0]
        while value.ndim < target.ndim:
            value = value.unsqueeze(-1)
        return value


class LimitCycle(VectorField):
    """Stable radial limit cycle.

    Equation:
        dx/dt = x (d - r^2) - w y
        dy/dt = y (d - r^2) + w x
        r^2 = x^2 + y^2
    """

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
    """Double-ring limit-cycle field.

    Equation:
        dx/dt = x (d - r) - w y (2 d - r)
        dy/dt = y (d - r) + w x (2 d - r)
        r = sqrt(x^2 + y^2)
    """

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
    """Smoothly blends two local limit cycles into a left/right bistable system.

    Equation:
        f(x, y) = (1 - g(x)) f_L(x, y) + g(x) f_R(x, y)
        g(x) = sigmoid(k x)
        f_c(p, q) = [p (r0 - rho) - omega q, q (r0 - rho) + omega p]
        rho = sqrt(p^2 + q^2), p = x - c, q = y
    """

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
    """Random smooth multi-attractor field sampled on a grid.

    Equation:
        f(x) = alpha * normalize(f_GP(x)) - w_att ||x|| x
        f_GP sampled from a zero-mean RBF Gaussian-process prior on the grid
    """

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
            U_attract = (
                -self.xy[:, 0] * torch.sqrt(torch.sum(self.xy**2, 1)) * self.w_attractor
            )
            V_attract = (
                -self.xy[:, 1] * torch.sqrt(torch.sum(self.xy**2, 1)) * self.w_attractor
            )
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
    """Van der Pol oscillator.

    Equation:
        dx/dt = y
        dy/dt = mu (1 - x^2) y - w x
    """

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
        mu = self._broadcast_param(self.mu, x)
        w = self._broadcast_param(self.w, x)
        U = x[..., 1]
        V = mu * (1 - x[..., 0] ** 2) * x[..., 1] - w * x[..., 0]

        U = self.alpha * U
        V = self.alpha * V

        return torch.stack([U, V], dim=-1)


class Duffing(VectorField):
    """Duffing oscillator.

    Equation:
        dx/dt = y
        dy/dt = a y - b x - c x^3
    """

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
        a = self._broadcast_param(self.a, x)
        b = self._broadcast_param(self.b, x)
        c = self._broadcast_param(self.c, x)
        U = x[..., 1]
        V = a * x[..., 1] - x[..., 0] * (b + c * x[..., 0] ** 2)
        U = self.alpha * U
        V = self.alpha * V

        return torch.stack([U, V], dim=-1)


class AsymmetricBasin(VectorField):
    """Duffing-style asymmetric two-basin system with gated parameters.

    Equation:
        dx/dt = y
        dy/dt = a_eff(x) y - b_eff(x) x - c x^3
        a_eff(x) = (1 - g(x)) a_left + g(x) a_right
        b_eff(x) = (1 - g(x)) b_left + g(x) b_right
        g(x) = sigmoid(k x)
    """

    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.gate_sharpness = float(kwargs.get("gate_sharpness", 3.0))
        self.c = float(kwargs.get("cubic", 0.1))
        if dyn_param is None:
            self.a_left = -1.2
            self.b_left = -0.8
            self.a_right = 0.55
            self.b_right = 0.1
        else:
            self.set_params(dyn_param)

    def _set_params(self, a_left=-1.2, b_left=-0.8, a_right=0.55, b_right=0.1):
        self.a_left = a_left
        self.b_left = b_left
        self.a_right = a_right
        self.b_right = b_right

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        a_left = self._broadcast_param(self.a_left, x)
        b_left = self._broadcast_param(self.b_left, x)
        a_right = self._broadcast_param(self.a_right, x)
        b_right = self._broadcast_param(self.b_right, x)
        gate = torch.sigmoid(self.gate_sharpness * x[..., 0])
        a_eff = (1.0 - gate) * a_left + gate * a_right
        b_eff = (1.0 - gate) * b_left + gate * b_right
        U = x[..., 1]
        V = a_eff * x[..., 1] - b_eff * x[..., 0] - self.c * x[..., 0] ** 3
        U = self.alpha * U
        V = self.alpha * V
        return torch.stack([U, V], dim=-1)


class ConfoundedGate(VectorField):
    r"""Three-state gate with parameter/nuisance confounding.

    For state ``z=(r,s,h)`` and learned scalar ``theta``, define
    ``g_c(r)=exp(-0.5((r-c)/w)^2)`` and

    .. math::

        \dot r &= -(r-c_A), \\
        \dot s &= -2s + 10g_I(r)\theta + 20g_A(r)(\theta+h), \\
        \dot h &= 0.

    The initial gate ``A`` has larger raw parameter sensitivity, but ``theta``
    and the persistent nuisance state ``h`` produce the same response.  At gate
    ``I``, ``theta`` acts without ``h``.  Actions control ``r`` and ``s`` only;
    ``h`` remains an uncontrolled latent nuisance.

    Inputs and outputs have shape ``(..., 3)`` and inherit the input dtype.
    This implements the confounded-gate stress design in the TBME README.
    """

    AMBIGUITY_CENTER = -0.5
    INFORMATIVE_CENTER = -0.32
    GATE_WIDTH = 0.04
    SELECTOR_CONTRACTION = 1.0
    RESPONSE_CONTRACTION = 2.0
    AMBIGUITY_SCALE = 20.0
    INFORMATIVE_SCALE = 10.0

    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        if dyn_param is None:
            self.set_params([0.5])
        else:
            self.set_params(dyn_param)

    def _set_params(self, theta=0.5):
        self.theta = theta

    @classmethod
    def _gate(cls, selector: torch.Tensor, center: float) -> torch.Tensor:
        return torch.exp(-0.5 * ((selector - float(center)) / cls.GATE_WIDTH) ** 2)

    def compute(self, state: torch.Tensor) -> torch.Tensor:
        selector = state[..., 0]
        response = state[..., 1]
        nuisance = state[..., 2]
        theta = self._broadcast_param(self.theta, state)
        ambiguity_gate = self._gate(selector, self.AMBIGUITY_CENTER)
        informative_gate = self._gate(selector, self.INFORMATIVE_CENTER)

        d_selector = -self.SELECTOR_CONTRACTION * (selector - self.AMBIGUITY_CENTER)
        d_response = -self.RESPONSE_CONTRACTION * response
        d_response = d_response + self.INFORMATIVE_SCALE * informative_gate * theta
        d_response = d_response + self.AMBIGUITY_SCALE * ambiguity_gate * (
            theta + nuisance
        )
        d_nuisance = torch.zeros_like(nuisance)
        return self.alpha * torch.stack((d_selector, d_response, d_nuisance), dim=-1)


class RankImbalancedGate(VectorField):
    r"""Four-state gate with rank-imbalanced parameter information.

    .. math::

        \dot r &= -(r-c_B), \\
        \dot s_j &= -2s_j + [g_M(r)a_{Mj} + g_B(r)a_{Bj}]\theta_j.

    At the main gate, ``a_M = (4, 2, 0.2)`` gives the diagonal
    information profile ``(16, 4, 0.04)``. At the balanced gate,
    ``a_B = (0.4, 0.4, 0.4)`` gives ``(0.16, 0.16, 0.16)``. A log-det
    objective therefore prefers the main gate, whereas E-optimality prefers
    the balanced gate's larger minimum eigenvalue.

    State and output tensors have shape ``(..., 4)``; the learned parameter
    has shape ``(..., 3)``. Actions control only the selector coordinate.
    """

    BALANCED_CENTER = -0.5
    MAIN_CENTER = 0.0
    GATE_WIDTH = 0.12
    SELECTOR_CONTRACTION = 1.0
    RESPONSE_CONTRACTION = 2.0
    MAIN_SENSITIVITY = (4.0, 2.0, 0.2)
    BALANCED_SENSITIVITY = (0.4, 0.4, 0.4)

    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.set_params([1.0, 0.5, 0.0] if dyn_param is None else dyn_param)

    def _set_params(self, theta_1=1.0, theta_2=0.5, theta_3=0.0):
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.theta_3 = theta_3

    @classmethod
    def _gate(cls, selector: torch.Tensor, center: float) -> torch.Tensor:
        return torch.exp(-0.5 * ((selector - float(center)) / cls.GATE_WIDTH) ** 2)

    def compute(self, state: torch.Tensor) -> torch.Tensor:
        selector = state[..., 0]
        response = state[..., 1:]
        main_gate = self._gate(selector, self.MAIN_CENTER).unsqueeze(-1)
        balanced_gate = self._gate(selector, self.BALANCED_CENTER).unsqueeze(-1)
        main_scale = state.new_tensor(self.MAIN_SENSITIVITY)
        balanced_scale = state.new_tensor(self.BALANCED_SENSITIVITY)
        theta = torch.stack(
            tuple(
                self._broadcast_param(param, state)
                for param in (self.theta_1, self.theta_2, self.theta_3)
            ),
            dim=-1,
        )

        d_selector = -self.SELECTOR_CONTRACTION * (selector - self.BALANCED_CENTER)
        sensitivity = main_gate * main_scale + balanced_gate * balanced_scale
        d_response = -self.RESPONSE_CONTRACTION * response + sensitivity * theta
        return self.alpha * torch.cat((d_selector.unsqueeze(-1), d_response), dim=-1)


class CompoundTriGate(VectorField):
    r"""Five-state selector system combining three objective-ablation traps.

    For state ``z=(r,s_1,s_2,s_3,h)`` and learned parameter
    ``theta=(theta_1,theta_2,theta_3)``, let
    ``g_c(r)=exp(-0.5((r-c)/w)^2)`` and define

    .. math::

        \dot r &= -(r-c_A), \\
        \dot s &= -4s
          + g_A(r)(20\theta_1+200h,0,0)^\top \\
          &\quad + g_B(r)\operatorname{diag}(0.02,0.02,0.02)\theta \\
          &\quad + g_M(r)\operatorname{diag}(4,2,0)\theta, \\
        \dot h &= 0.

    Gate ``A`` has the largest scalar parameter sensitivity but aliases
    ``theta_1`` with a strongly amplified nuisance state ``h``. Gate ``B`` is
    a balanced but uniformly weak E-optimal decoy. Gate ``M`` has the largest
    initial log-determinant but no third-parameter sensitivity. PALDI should
    therefore prioritize ``M`` and use ``B`` only as the third posterior
    direction becomes limiting, while individual ablations are attracted to
    one of the decoy gates.

    States and outputs have shape ``(..., 5)`` and inherit the input dtype.
    The learned parameter has shape ``(..., 3)``. The single action enters the
    selector coordinate through ``PaddedIdentityActionEncoder``.
    """

    AMBIGUITY_CENTER = -0.5
    BALANCED_CENTER = -0.32
    MAIN_CENTER = 0.0
    GATE_WIDTH = 0.04
    SELECTOR_CONTRACTION = 1.0
    RESPONSE_CONTRACTION = 4.0
    NUISANCE_CONTRACTION = 0.0
    AMBIGUITY_SCALE = 20.0
    AMBIGUITY_NUISANCE_SCALE = 200.0
    BALANCED_SENSITIVITY = (0.02, 0.02, 0.02)
    MAIN_SENSITIVITY = (4.0, 2.0, 0.0)

    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.set_params([1.0, 1.0, 0.0] if dyn_param is None else dyn_param)

    def _set_params(self, theta_1=1.0, theta_2=1.0, theta_3=0.0):
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.theta_3 = theta_3

    @classmethod
    def _gate(cls, selector: torch.Tensor, center: float) -> torch.Tensor:
        return torch.exp(-0.5 * ((selector - float(center)) / cls.GATE_WIDTH) ** 2)

    def compute(self, state: torch.Tensor) -> torch.Tensor:
        selector = state[..., 0]
        response = state[..., 1:4]
        nuisance = state[..., 4]
        ambiguity_gate = self._gate(selector, self.AMBIGUITY_CENTER)
        balanced_gate = self._gate(selector, self.BALANCED_CENTER).unsqueeze(-1)
        main_gate = self._gate(selector, self.MAIN_CENTER).unsqueeze(-1)
        theta = torch.stack(
            tuple(
                self._broadcast_param(param, state)
                for param in (self.theta_1, self.theta_2, self.theta_3)
            ),
            dim=-1,
        )

        d_selector = -self.SELECTOR_CONTRACTION * (selector - self.AMBIGUITY_CENTER)
        balanced_scale = state.new_tensor(self.BALANCED_SENSITIVITY)
        main_scale = state.new_tensor(self.MAIN_SENSITIVITY)
        d_response = -self.RESPONSE_CONTRACTION * response
        d_response = d_response + balanced_gate * balanced_scale * theta
        d_response = d_response + main_gate * main_scale * theta
        d_response_first = d_response[..., 0] + ambiguity_gate * (
            self.AMBIGUITY_SCALE * theta[..., 0]
            + self.AMBIGUITY_NUISANCE_SCALE * nuisance
        )
        d_response = torch.cat(
            (d_response_first.unsqueeze(-1), d_response[..., 1:]), dim=-1
        )
        d_nuisance = -self.NUISANCE_CONTRACTION * nuisance
        return self.alpha * torch.cat(
            (d_selector.unsqueeze(-1), d_response, d_nuisance.unsqueeze(-1)), dim=-1
        )


class SimpleTriGate(VectorField):
    r"""Five-state, three-parameter system exposing three acquisition traps.

    For ``z=(r,s_1,s_2,s_3,h)``, ``theta=(theta_1,theta_2,theta_3)``, and Gaussian gates
    ``g_c(r)=exp(-0.5((r-c)/0.1)^2)``, this implements

    .. math::

        \dot r &= -(r+1), \\
        \dot s &= -4s + 20g_A(r)(\theta_1+5h,0,0)^\top \\
        &\quad + g_B(r)\operatorname{diag}(1,1,1)\theta
          + g_M(r)\operatorname{diag}(5,5,0.75)\theta, \\
        \dot h &= 0,

    with gate centers ``c_A=-0.5``, ``c_B=-0.1``, and ``c_M=0.3``. The single
    action is added to the selector equation by ``PaddedIdentityActionEncoder``.

    The passive equilibrium ``r=-1`` lies outside all gates. Gate ``A`` is a
    high-sensitivity but state-confounded decoy. Gate ``B`` has the largest
    weakest parameter direction and is therefore the E-optimal decoy. Gate
    ``M`` is full rank and has nonzero sensitivity to every unknown parameter;
    in particular, ``theta_3`` must be learned because its truth is one and its
    initial estimate is zero.

    States and outputs have shape ``(..., 5)`` and inherit the input dtype.
    The learned parameter has shape ``(..., 3)``.
    """

    REST_CENTER = -1.0
    AMBIGUITY_CENTER = -0.5
    BALANCED_CENTER = -0.1
    MAIN_CENTER = 0.3
    GATE_WIDTH = 0.1
    SELECTOR_CONTRACTION = 1.0
    RESPONSE_CONTRACTION = 4.0
    AMBIGUITY_SCALE = 20.0
    AMBIGUITY_NUISANCE_RATIO = 5.0
    BALANCED_SENSITIVITY = (1.0, 1.0, 1.0)
    MAIN_SENSITIVITY = (5.0, 5.0, 0.75)

    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.set_params([1.0, 1.0, 1.0] if dyn_param is None else dyn_param)

    def _set_params(self, theta_1=1.0, theta_2=1.0, theta_3=1.0):
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.theta_3 = theta_3

    @classmethod
    def _gate(cls, selector: torch.Tensor, center: float) -> torch.Tensor:
        return torch.exp(-0.5 * ((selector - float(center)) / cls.GATE_WIDTH) ** 2)

    def compute(self, state: torch.Tensor) -> torch.Tensor:
        selector = state[..., 0]
        response = state[..., 1:4]
        nuisance = state[..., 4]
        ambiguity_gate = self._gate(selector, self.AMBIGUITY_CENTER)
        balanced_gate = self._gate(selector, self.BALANCED_CENTER).unsqueeze(-1)
        main_gate = self._gate(selector, self.MAIN_CENTER).unsqueeze(-1)
        theta = torch.stack(
            tuple(
                self._broadcast_param(param, state)
                for param in (self.theta_1, self.theta_2, self.theta_3)
            ),
            dim=-1,
        )

        d_selector = -self.SELECTOR_CONTRACTION * (selector - self.REST_CENTER)
        d_response = -self.RESPONSE_CONTRACTION * response
        d_response = (
            d_response
            + balanced_gate * state.new_tensor(self.BALANCED_SENSITIVITY) * theta
            + main_gate * state.new_tensor(self.MAIN_SENSITIVITY) * theta
        )
        d_response_first = d_response[..., 0] + ambiguity_gate * (
            self.AMBIGUITY_SCALE * theta[..., 0]
            + self.AMBIGUITY_SCALE * self.AMBIGUITY_NUISANCE_RATIO * nuisance
        )
        d_response = torch.cat(
            (d_response_first.unsqueeze(-1), d_response[..., 1:]), dim=-1
        )
        return self.alpha * torch.cat(
            (
                d_selector.unsqueeze(-1),
                d_response,
                torch.zeros_like(nuisance).unsqueeze(-1),
            ),
            dim=-1,
        )


class MultiStable(VectorField):
    """Gaussian-well multistable dynamics with local contraction and swirl.

    Equation:
        f(x) = sum_i exp(-||x-c_i||^2 / (2 sigma^2))
               * ( -a_i (x-c_i) + w_i R_90 (x-c_i) )
        R_90 [u, v] = [-v, u]

    The Gaussian envelope makes both the drift and its state/parameter Jacobians decay
    rapidly away from the attractor region. The learned embedding remains backward
    compatible with the original four amplitude parameters; the swirl strengths use
    fixed defaults unless explicitly provided.
    """

    def __init__(
        self,
        dyn_param: Optional[list[float]] | torch.Tensor = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.sigma = float(kwargs.get("sigma", 0.9))
        center_scale = float(kwargs.get("center_scale", 2))
        self._centers = torch.tensor(
            [
                [-center_scale, center_scale],
                [center_scale, center_scale],
                [-center_scale, -center_scale],
                [center_scale, -center_scale],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        if dyn_param is None:
            self.set_params([1.15, -0.1, -0.2, 1.5])
        else:
            self.set_params(dyn_param)

    def _set_params(
        self,
        a_nw=1.15,
        a_ne=0.95,
        a_sw=1.05,
        a_se=0.85,
        w_nw=1.55,
        w_ne=-0.25,
        w_sw=-0.40,
        w_se=-2.0,
    ):
        self.a_nw = a_nw
        self.a_ne = a_ne
        self.a_sw = a_sw
        self.a_se = a_se
        self.w_nw = w_nw
        self.w_ne = w_ne
        self.w_sw = w_sw
        self.w_se = w_se

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        centers = self._centers.to(device=x.device, dtype=x.dtype)
        disp = x.unsqueeze(-2) - centers
        r2 = torch.sum(disp * disp, dim=-1)
        sigma2 = max(self.sigma**2, 1e-6)
        envelope = torch.exp(-0.5 * r2 / sigma2)
        amplitudes = torch.stack(
            [
                self._broadcast_param(self.a_nw, x),
                self._broadcast_param(self.a_ne, x),
                self._broadcast_param(self.a_sw, x),
                self._broadcast_param(self.a_se, x),
            ],
            dim=-1,
        )
        rotations = torch.stack(
            [
                self._broadcast_param(self.w_nw, x),
                self._broadcast_param(self.w_ne, x),
                self._broadcast_param(self.w_sw, x),
                self._broadcast_param(self.w_se, x),
            ],
            dim=-1,
        )
        tangent = torch.stack((-disp[..., 1], disp[..., 0]), dim=-1)
        local_field = (
            -amplitudes.unsqueeze(-1) * disp + rotations.unsqueeze(-1) * tangent
        )
        field = torch.sum(envelope.unsqueeze(-1) * local_field, dim=-2)
        return self.alpha * field


class DampedPendulum(VectorField):
    """Damped pendulum with learnable damping and gravity scale.

    Equation:
        dtheta/dt = omega
        domega/dt = damping * omega - gravity * sin(theta)
    """

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
        damping = self._broadcast_param(self.damping, x)
        gravity = self._broadcast_param(self.gravity, x)
        U = x[..., 1]
        V = damping * x[..., 1] - gravity * torch.sin(x[..., 0])
        U = self.alpha * U
        V = self.alpha * V
        return torch.stack([U, V], dim=-1)


class DoubleIntegrator(VectorField):
    """Controlled double-integrator with unknown drift bias and damping.

    Equation:
        dx/dt = v
        dv/dt = bias - damping * v
    """

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
        bias = self._broadcast_param(self.bias, x)
        damping = self._broadcast_param(self.damping, x)
        U = x[..., 1]
        V = bias - damping * x[..., 1]
        U = self.alpha * U
        V = self.alpha * V
        return torch.stack([U, V], dim=-1)


class FitzHughNagumo(VectorField):
    """FitzHugh-Nagumo excitable dynamics.

    Equation:
        du/dt = u - u^3 / 3 - v + I_ext
        dv/dt = (u + a - b v) / tau
    """

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

        a = self._broadcast_param(self.a, x)
        b = self._broadcast_param(self.b, x)
        tau = self._broadcast_param(self.tau, x)
        i_ext = self._broadcast_param(self.i_ext, x)
        U = x[..., 0] - (x[..., 0] ** 3) / 3.0 - x[..., 1] + i_ext
        V = (x[..., 0] + a - b * x[..., 1]) / tau
        U = self.alpha * U
        V = self.alpha * V
        return torch.stack([U, V], dim=-1)


class Hopf(VectorField):
    """Generalized Hopf normal form with optional nonlinear frequency shift.

    Equation:
        dx/dt = (mu - r^2) x - (omega + beta r^2) y
        dy/dt = (omega + beta r^2) x + (mu - r^2) y
        r^2 = x^2 + y^2
    """

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

        mu = self._broadcast_param(self.mu, x)
        omega = self._broadcast_param(self.omega, x)
        beta = self._broadcast_param(self.beta, x)
        radius_sq = x[..., 0] ** 2 + x[..., 1] ** 2
        omega_eff = omega + beta * radius_sq
        U = (mu - radius_sq) * x[..., 0] - omega_eff * x[..., 1]
        V = omega_eff * x[..., 0] + (mu - radius_sq) * x[..., 1]
        U = self.alpha * U
        V = self.alpha * V
        return torch.stack([U, V], dim=-1)


class SnowMan(VectorField):
    """Two coupled limit cycles blended by a horizontal sigmoid gate.

    Equation:
        f(x, y) = s(x) f_R(x-d, y) + (1 - s(x)) f_L(x+d, y)
        s(x) = sigmoid(beta x)
        f_R, f_L are mirrored single-cycle fields around centers +/- d
    """

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
        w = self._broadcast_param(self.w, x)
        d = self._broadcast_param(self.d, x)
        beta = self._broadcast_param(self.beta, x)
        r = torch.sqrt((x[..., 0] - d) ** 2 + x[..., 1] ** 2)
        U1 = (x[..., 0] - d) * (d**2 - r**2) - w * x[..., 1]
        V1 = x[..., 1] * (d**2 - r**2) + w * (x[..., 0] - d)

        r = torch.sqrt((x[..., 0] + d) ** 2 + x[..., 1] ** 2)
        U2 = (x[..., 0] + d) * (d**2 - r**2) + w * x[..., 1]
        V2 = x[..., 1] * (d**2 - r**2) - w * (x[..., 0] + d)

        U = self.alpha * (
            torch.sigmoid(beta * x[..., 0]) * U1 + torch.sigmoid(-beta * x[..., 0]) * U2
        )
        V = self.alpha * (
            torch.sigmoid(beta * x[..., 0]) * V1 + torch.sigmoid(-beta * x[..., 0]) * V2
        )

        return torch.stack([U, V], dim=-1)


if __name__ == "__main__":
    # Example usage with smaller grid size
    vf = VanDerPol(x_range=2.5, n_grid=50)
