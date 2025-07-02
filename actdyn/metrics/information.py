import torch
from typing import Union, Dict
from actdyn.models import BaseDynamics, Decoder
from actdyn.models.decoder import LinearMapping, LogLinearMapping
from actdyn.models.dynamics import RBFDynamics
from actdyn.utils.rollout import Rollout, RolloutBuffer
from .base import BaseMetric


def compute_jacobian_state(function, state, **kwargs):
    """Compute Jacobian of a function with respect to state."""
    state = state.clone().detach().requires_grad_(True)
    J_f = torch.autograd.functional.jacobian(function, state, **kwargs)
    return J_f


def compute_jacobian_params(function, state, **kwargs):
    """Compute Jacobian of a function with respect to parameters."""
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
        self.I = None

    def compute_dh_dz(self, z):
        if isinstance(self.decoder, LogLinearMapping):
            C = self.decoder.mapping.network.weight.data.clone()
            dh_dz = C.view(1, 1, C.shape[0], C.shape[1]).expand(
                z.shape[0], z.shape[1], C.shape[0], C.shape[1]
            )

        elif isinstance(self.decoder, LinearMapping):
            C = self.decoder.mapping.network.weight.data.clone()
            dh_dz = torch.einsum(
                "btd,dn->btdn",
                self.decoder(z),
                C,
            )
        else:
            # TODO: add support for other decoder types, use compute_jacobian_state
            raise ValueError(f"Decoder type {type(self.decoder)} not supported")

        return dh_dz

    def compute_df_dtheta(self, z):
        if isinstance(self.dynamics, RBFDynamics):
            df_dtheta = self.dynamics._rbf(z)
        else:
            df_dtheta = compute_jacobian_params(self.dynamics, z)
        return df_dtheta

    def compute_rbf_fim(
        self, rollout: Union[Rollout, RolloutBuffer], use_diag=True
    ) -> torch.Tensor:
        z = rollout["model_state"]
        assert len(z.shape) == 3, "z must be a tensor of shape (batch, T, d_latent)"
        batch, T, d_latent = z.shape
        d_param = self.dynamics.weights.numel()

        dh_dz = self.compute_dh_dz(z)
        df_dtheta = self.compute_df_dtheta(z).unsqueeze(-1)

        # TODO temporally discounted I
        if use_diag:
            I_new = (
                torch.einsum(
                    "btd, btk->btdk", (dh_dz**2).sum(-2), df_dtheta.squeeze(-1) ** 2
                )
                .reshape(batch, T, d_param)
                .sum(dim=1)
            ).unsqueeze(1)
        else:
            J = torch.einsum(
                "...nd,...kd->...nk", dh_dz, torch.kron(torch.eye(d_latent), df_dtheta)
            )  # (batch, T, d_obs, d_param)
            I_new = (J.mT @ J).sum(dim=1)  # (batch, d_param, d_param)

        return I_new

    def compute_nn_fim(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        pass

    def compute_fim(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        if isinstance(self.dynamics, RBFDynamics):
            return self.compute_rbf_fim(rollout)
        else:
            return self.compute_nn_fim(rollout)

    def update_fim(self, rollout: Union[Rollout, RolloutBuffer], discount_factor=0.99):
        I_new = self.compute_fim(rollout)
        if self.I is None:
            self.I = I_new
        else:
            self.I = self.I * discount_factor + I_new


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
        self.update_fim(rollout)

        if self.use_diag:
            # return reciprocal sum of fim that are greater than 1e-3
            return torch.reciprocal(self.I).sum(dim=-1)
        else:
            # TODO: implement non-diagonal A-optimality
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
