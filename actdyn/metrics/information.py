import torch
import sys
from typing import Union, Dict
from actdyn.models import BaseDynamics, Decoder
from actdyn.utils.rollout import Rollout, RolloutBuffer
from .base import BaseMetric


def compute_jacobian_state(function, state, **kwargs):
    """Compute Jacobian of a function with respect to state."""
    state = state.clone().detach().requires_grad_(True)
    J_f = torch.autograd.functional.jacobian(function, state, **kwargs)
    return J_f


def compute_jacobian_params(function, state, **kwargs):
    """Compute Jacobian of a function with respect to parameters efficiently.
    Uses torch.autograd.functional.jacobian and stateless.functional_call if available (PyTorch 2.0+).
    """
    params = tuple(function.parameters())
    # PyTorch 2.0+ provides torch.func.functional_call for efficient parameter substitution
    if hasattr(torch, "func") and hasattr(torch.func, "functional_call"):
        from torch.func import functional_call

        param_names = [n for n, _ in function.named_parameters()]

        def wrapped_params(*params):
            param_dict = dict(zip(param_names, params))
            return functional_call(function, param_dict, (state,))

        return torch.autograd.functional.jacobian(wrapped_params, params, **kwargs)
    else:
        # Fallback for older PyTorch: original (inefficient) implementation
        f_val = function(state)
        out_dim = f_val.shape[0]
        J_list = []
        for i in range(out_dim):
            grads = torch.autograd.grad(
                f_val[i], params, retain_graph=True, allow_unused=True, **kwargs
            )
            grad_vec = torch.cat(
                [
                    torch.zeros_like(p).view(-1) if g is None else g.view(-1)
                    for p, g in zip(params, grads)
                ]
            )
            J_list.append(grad_vec.unsqueeze(0))
        return torch.cat(J_list, dim=0)


class FisherInformationMetric(BaseMetric):
    """Metric that computes Fisher Information Matrix using provided models."""

    def __init__(
        self,
        dynamics: BaseDynamics,
        decoder: Decoder,
        use_diag: bool = True,
        device: str = "cuda",
    ):
        super().__init__(device)
        self.dynamics = dynamics
        self.decoder = decoder
        self.use_diag = use_diag

    def compute_rbf_fim(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        pass

    def compute_fim(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        pass

    def compute_point(
        self, trajectory: Dict[str, torch.Tensor], t: int
    ) -> torch.Tensor:
        z = trajectory["state"][t : t + 2]  # Need current and next state
        if len(z) < 2:
            return torch.tensor(0.0, device=self.device)

        C = self.decoder.decoder[0].weight
        dh_dz = self.decoder(z[1]).unsqueeze(-1) * C
        phi = self.dynamics.rbf(z[0]).unsqueeze(-1)

        if self.use_diag:
            fim_diag = torch.einsum("d,k->dk", (dh_dz**2).sum(-2), phi.squeeze(-1) ** 2)
            return fim_diag.sum()
        else:
            J = torch.einsum(
                "nd,kd->nk",
                dh_dz,
                torch.kron(torch.eye(z.shape[-1], device=self.device), phi),
            )
            fim = J.mT @ J
            return torch.trace(fim)


class AOptimality(FisherInformationMetric):
    """Metric that computes A-optimality."""

    def __init__(
        self,
        dynamics: BaseDynamics,
        decoder: Decoder,
        use_diag: bool = False,
        device: str = "cuda",
    ):
        super().__init__(
            dynamics=dynamics, decoder=decoder, use_diag=use_diag, device=device
        )

    def compute(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        pass


class DOptimality(FisherInformationMetric):
    """Metric that computes D-optimality."""

    def __init__(
        self,
        dynamics: BaseDynamics,
        decoder: Decoder,
        use_diag: bool = False,
        device: str = "cuda",
    ):
        super().__init__(
            dynamics=dynamics, decoder=decoder, use_diag=use_diag, device=device
        )

    def compute(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        pass
