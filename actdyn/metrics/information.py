from actdyn.models.base import BaseDynamicsEnsemble
import torch
from typing import Dict, Union
from actdyn.models import BaseDynamics, Decoder
from actdyn.models.decoder import LinearMapping, LogLinearMapping
from actdyn.models.dynamics import RBFDynamics
from actdyn.utils.rollout import Rollout, RolloutBuffer
from .base import BaseMetric

eps = 1e-12


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
        model,
        compute_type: str = "sum",
        use_diag: bool = True,
        discount_factor: float = 0.99,
        device: str = "cuda",
        covariance: str = "invariant",
        sensitivity: bool = True,
        **kwargs,
    ):
        super().__init__(compute_type, device)
        self.dynamics = model.dynamics
        if isinstance(self.dynamics, BaseDynamicsEnsemble):
            self.dynamics = self.dynamics.ensemble[0]  # Use the first model for FIM
        else:
            self.dynamics = self.dynamics
        self.decoder = model.decoder
        self.encoder = model.encoder
        self.use_diag = use_diag
        self.discount_factor = discount_factor
        self.covariance = covariance
        self.sensitivity = sensitivity
        self.I = None

    def compute_dh_dz(self, z):
        if isinstance(self.decoder.mapping, LinearMapping):
            C = self.decoder.mapping.network.weight.data.clone()
            dh_dz = C.view(1, 1, C.shape[0], C.shape[1]).expand(
                z.shape[0], z.shape[1], C.shape[0], C.shape[1]
            )

        elif isinstance(self.decoder.mapping, LogLinearMapping):
            C = self.decoder.mapping.network[0].weight.data.clone()
            dh_dz = torch.einsum(
                "btd,dn->btdn",
                self.decoder(z),
                C,
            )
        else:
            # TODO: add support for other decoder types, use compute_jacobian_state
            raise ValueError(f"Decoder type {type(self.decoder)} not supported")

        return dh_dz

    @torch.no_grad()
    def compute_dz_dtheta(self, z):
        if isinstance(self.dynamics, RBFDynamics):
            if self.sensitivity:
                e_z = self.dynamics.centers - z.unsqueeze(-2)  # (batch, T, num_centers, d_latent)
                batch, T, num_centers, d_latent = e_z.shape
                J = torch.einsum(
                    "...tc,...tcd->...tcd",
                    self.dynamics.rbf(z),
                    e_z,
                )
                df_dz = 1 + (J.mT @ self.dynamics.weights)  # (batch, T, num_centers, d_latent)
                # Compute sensitivity of RBF centers
                df_dtheta = self.dynamics.rbf(z)  # (batch, T, num_centers)

                # S_{t+1} + S_{t} * df_dz[t] + I x df_dtheta[t]
                dz_dtheta = torch.zeros(batch, T, d_latent, d_latent * num_centers).to(self.device)
                # kronecker product of df_dz and eye(d_latent)
                dz_dtheta[:, :1, :, :] = (
                    torch.kron(torch.eye(d_latent, device=self.device), df_dtheta[:, 0])
                    .view(d_latent, batch, 1, d_latent * num_centers)
                    .movedim(0, 2)
                )

                for t in range(1, T):
                    dz_dtheta[:, t, :, :] = (
                        torch.kron(
                            torch.eye(d_latent, device=self.device),
                            df_dtheta[:, t - 1 : t],
                        )
                        + df_dz[:, t - 1, :, :] @ dz_dtheta[:, t - 1, :, :]
                    )

            else:
                dz_dtheta = self.dynamics.rbf(z)
        else:
            dz_dtheta = compute_jacobian_params(self.dynamics, z)

        return dz_dtheta

    def compute_rbf_fim(
        self, rollout: Union[Rollout, RolloutBuffer, Dict], use_diag=True
    ) -> torch.Tensor:
        z = rollout["model_state"]
        if len(z.shape) != 3:
            z = z.unsqueeze(0)  # Ensure z is (batch, T, d_latent)
        assert len(z.shape) == 3, "z must be a tensor of shape (batch, T, d_latent)"
        batch, T, d_latent = z.shape
        d_param = self.dynamics.weights.numel()

        dh_dz = self.compute_dh_dz(z)
        dz_dtheta = self.compute_dz_dtheta(z).detach()  # (B, T, d, p)
        if self.covariance == "invariant":
            invCC = torch.linalg.pinv(dh_dz @ dh_dz.mT)  # (B, T, d, d)
            Ht_H = dh_dz.mT @ invCC @ dh_dz  # (B, T, d, d)
        else:
            Ht_H = torch.einsum("btnd,btnf->btdf", dh_dz, dh_dz)  # (B, T, d, d)

        # TODO temporally discounted I
        if use_diag:
            if self.sensitivity:
                I_new = (
                    torch.einsum(
                        "...dp,...dp,...dd->...p",
                        dz_dtheta,
                        dz_dtheta,
                        Ht_H,
                    )
                    .sum(dim=1)
                    .unsqueeze(1)
                )  # (batch, d_param, d_param)
            else:
                I_new = (
                    torch.einsum("btd, btk->btdk", (dh_dz**2).sum(-2), dz_dtheta.squeeze(-1) ** 2)
                    .reshape(batch, T, d_param)
                    .sum(dim=1)
                ).unsqueeze(1)
            # # compare difference between I_new and I_new2
            # if torch.allclose(I_new, I_new2, atol=1e-6):
            #     print("I_new and I_new2 are close enough, using I_new2 for efficiency.")

        else:
            J = torch.einsum(
                "...nd,...kd->...nk", dh_dz, torch.kron(torch.eye(d_latent), dz_dtheta)
            )  # (batch, T, d_obs, d_param)
            I_new = (J.mT @ J).sum(dim=1)  # (batch, d_param, d_param)

        if self.I is not None:
            I_new += self.I * self.discount_factor

        return I_new

    def compute_nn_fim(self, rollout: Union[Rollout, RolloutBuffer, Dict]) -> torch.Tensor:
        pass

    def compute_fim(self, rollout: Union[Rollout, RolloutBuffer, Dict]) -> torch.Tensor:
        if isinstance(self.dynamics, RBFDynamics):
            return self.compute_rbf_fim(rollout)
        else:
            return self.compute_nn_fim(rollout)

    def update_fim(self, rollout: Union[Rollout, RolloutBuffer]):
        self.I = self.compute_fim(rollout)


class AOptimality(FisherInformationMetric):
    """Metric that computes A-optimality."""

    def compute(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        fim_traj = self.compute_fim(rollout)  # (batch, 1, d_param)
        if self.use_diag:
            # reciprocal of element greater than 1e-3
            # return shape (batch, 1)
            fim_traj[fim_traj < eps] = eps
            return torch.reciprocal(fim_traj).sum(dim=-1)

        else:
            # TODO: implement non-diagonal A-optimality
            pass


class DOptimality(FisherInformationMetric):
    """Metric that computes D-optimality."""

    def compute(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        fim_traj = self.compute_fim(rollout)  # (batch, 1, d_param)
        if self.use_diag:
            # reciprocal of element greater than 1e-3
            # return shape (batch, 1)
            fim_traj[fim_traj < eps] = eps
            return -torch.log1p(fim_traj).sum(dim=-1)

        else:
            # TODO: implement non-diagonal A-optimality
            pass
