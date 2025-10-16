from zmq import FD
import actdyn
import actdyn.models
from actdyn.models.base import BaseDynamicsEnsemble
import torch
from typing import Dict, Union
from actdyn.models import BaseDynamics, Decoder
from actdyn.models.decoder import LinearMapping, LogLinearMapping
from actdyn.models.dynamics import RBFDynamics
from actdyn.utils.rollout import Rollout, RolloutBuffer
from .base import BaseMetric
from torch.nn.functional import softplus

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


class EmbeddingFisherMetric(BaseMetric):
    """Metric that computes information gain in the embedding space."""

    def __init__(
        self,
        compute_type: str = "sum",
        device: str = "cuda",
        Fe_net: callable = None,
        Fz_net: callable = None,
        decoder: actdyn.models.Decoder = None,
        **kwargs,
    ):
        super().__init__(compute_type, device)
        self.I = None
        self.Fe_net = Fe_net
        self.Fz_net = Fz_net
        self.decoder = decoder

    def compute(
        self, rollout: Union[Rollout, RolloutBuffer, Dict], e_bel, z_bel, Q
    ) -> torch.Tensor:
        z = rollout["model_state"].to(self.device)

        # Throw error if Fe_net, Fz_net, or decoder is not set

        if len(z.shape) != 3:
            z = z.unsqueeze(0)  # Ensure z is (batch, T, d_latent)
        assert len(z.shape) == 3, "z must be a tensor of shape (batch, T, d_latent)"
        batch, T, d_latent = z.shape
        d_embedding = e_bel["m"].shape[-1]

        # Move beliefs/covariances to device to avoid implicit copies
        P = z_bel["P"].to(self.device).squeeze(0)  # (batch, d_latent, d_latent)
        Q = Q.to(self.device).unsqueeze(0)  # (batch, d_latent, d_latent)
        R = softplus(self.decoder.logvar).diag_embed().to(self.device)  # (batch, d_obs, d_obs)

        # Ensure the embedding mean is on the same device and expand appropriately
        e_m = e_bel["m"].to(self.device)

        # Compute Fe and Fz without tracking gradients (already in no_grad scope)
        Fe = self.Fe_net(z, e_m.repeat(batch, T, 1)).detach()  # (batch, T, d_latent, d_emb)
        Fz = self.Fz_net(z, e_m.repeat(batch, T, 1)).detach()  # (batch, T, d_latent, d_latent)
        # Preallocate accumulators on the correct device
        J = torch.zeros(batch, d_embedding, d_embedding, device=self.device)
        Gt = torch.zeros(batch, d_latent, d_latent, device=self.device)
        Gt = self.Fe_net(z_bel["m"], e_m).detach()

        # Cache decoder jacobian on device (broadcasting handles batch dim)
        Ht = (
            self.decoder.jacobian.unsqueeze(0).unsqueeze(0).to(self.device)
        )  # (batch, d_obs, d_latent)
        for i in range(T):
            dyde = Ht @ Gt  # (batch, time, d_obs, d_emb)
            S = Ht @ P @ Ht.mT + R
            # cholesky on symmetric positive-definite S
            chol_S = torch.linalg.cholesky(S)
            # solve S x = dyde -> x has same shape as dyde
            sol = torch.cholesky_solve(dyde, chol_S)
            J += (dyde.mT @ sol).squeeze(0)  # (batch, d_emb, d_emb)
            Gt = Fz[:, i] @ Gt + Fe[:, i]
            P = Fz[:, i] @ P @ Fz[:, i].mT + Q

        # Use slogdet for numerical stability and to avoid accidental graph retention
        mat = e_bel["P"].to(self.device) @ J + torch.eye(d_embedding, device=self.device)
        sign, logabsdet = torch.linalg.slogdet(mat)
        # if sign <= 0, logabsdet may be -inf or NaN; keep current behaviour but avoid crash
        EIG = logabsdet

        # Explicitly delete large temporaries (helps long-running processes)
        del Fe, Fz, sol, mat, chol_S, Ht, dyde, S
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return -EIG  # (batch, )

        # with torch.no_grad():
        #     z = rollout["next_model_state"].to(self.device)
        #     if len(z.shape) != 3:
        #         z = z.unsqueeze(0)  # Ensure z is (batch, T, d_latent)
        #     assert len(z.shape) == 3, "z must be a tensor of shape (batch, T, d_latent)"
        #     batch, T, d_latent = z.shape
        #     d_embedding = e_bel["m"].shape[-1]

        #     e_m = e_bel["m"].to(self.device)
        #     z_m = z_bel["m"].to(self.device)
        #     if z_m.shape[0] == 1 and batch > 1:
        #         z_m = z_m.expand(batch, -1, -1)

        #     P = z_bel["P"].to(self.device)
        #     # If P has an extra leading singleton, broadcast to batch
        #     if P.dim() == 4 and P.shape[0] == 1:
        #         P = P.expand(batch, -1, -1, -1).squeeze(1)
        #     P = P.squeeze(0) if P.dim() == 3 and P.shape[0] == 1 else P
        #     # Ensure P is (batch, d_latent, d_latent)
        #     if P.dim() == 2:
        #         P = P.unsqueeze(0).expand(batch, -1, -1)

        #     Qb = Q.to(self.device)
        #     if Qb.dim() == 2:
        #         Qb = Qb.unsqueeze(0).expand(batch, -1, -1)
        #     elif Qb.dim() == 3 and Qb.shape[0] == 1:
        #         Qb = Qb.expand(batch, -1, -1)

        #     R = softplus(self.decoder.logvar).diag_embed().to(self.device)
        #     if R.dim() == 3 and R.shape[0] == 1:
        #         R = R.expand(batch, -1, -1)
        #     # Ensure the embedding mean is on the same device and expand appropriately
        #     e_rep = e_m.unsqueeze(0).expand(batch, -1) if e_m.dim() == 1 else e_m
        #     e_rep = e_rep.unsqueeze(1).expand(batch, T, -1)

        #     # Compute Fe and Fz without tracking gradients (already in no_grad scope)
        #     Fe = self.Fe_net(z, e_rep).detach()  # (batch, T, d_latent, d_emb)
        #     Fz = self.Fz_net(z, e_rep).detach()  # (batch, T, d_latent, d_latent)

        #     # Compute initial sensitivity Gt at belief mean for t=0: shape (batch, d_latent, d_emb)
        #     Fe_bel = self.Fe_net(z_m, e_rep[:, :1, :]).detach()
        #     Gt = Fe_bel[:, 0]  # (batch, d_latent, d_emb)

        #     # Cache decoder jacobian on device (broadcasting handles batch dim)
        #     H = self.decoder.jacobian.to(self.device)
        #     H = H.unsqueeze(0).expand(batch, -1, -1)

        #     # Accumulator for embedding-space Fisher information (batch, d_emb, d_emb)
        #     J = torch.zeros(batch, d_embedding, d_embedding, device=self.device)

        #     for i in range(T):
        #         # dy/de = H @ Gt  -> (batch, d_obs, d_emb)
        #         dyde = torch.einsum("bod,bdp->bop", H, Gt)

        #         # Innovation covariance S = H P H^T + R  -> (batch, d_obs, d_obs)
        #         S = H @ P @ H.mT + R

        #         # Cholesky and solve: solve S x = dyde for x
        #         chol_S = torch.linalg.cholesky(S)
        #         sol = torch.cholesky_solve(dyde, chol_S)

        #         # Update J: sum over observations -> (batch, d_emb, d_emb)
        #         # dyde^T @ sol  => (batch, d_emb, d_emb)
        #         J += torch.einsum("bop,boq->bpq", dyde, sol)

        #         # Propagate Gt and P to next step
        #         Fi = Fz[:, i]
        #         Gi = Fe[:, i]
        #         # Gt <- Fz_i @ Gt + Fe_i
        #         Gt = Fi @ Gt + Gi

        #         # P <- Fz_i @ P @ Fz_i^T + Q
        #         P = Fi @ P @ Fi.mT + Qb

        #     # Use slogdet for numerical stability and to avoid accidental graph retention
        #     mat = e_bel["P"].to(self.device) @ J + torch.eye(d_embedding, device=self.device)
        #     sign, logabsdet = torch.linalg.slogdet(mat)
        #     # if sign <= 0, logabsdet may be -inf or NaN; keep current behaviour but avoid crash
        #     EIG = logabsdet

        #     # Explicitly delete large temporaries (helps long-running processes)
        #     del Fe, Fz, Fe_bel, sol, mat, chol_S, H, dyde, S
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()

        #     return -EIG  # (batch, )

        # with torch.no_grad():
        #     # Basic validation
        #     if self.Fe_net is None or self.Fz_net is None or self.decoder is None:
        #         raise ValueError(
        #             "EmbeddingFisherMetric requires Fe_net, Fz_net, and decoder to be set"
        #         )

        #     # Accept numpy/sequence or tensor; as_tensor will handle both and place on device
        #     z = torch.as_tensor(rollout["model_state"], device=self.device)
        #     if len(z.shape) != 3:
        #         z = z.unsqueeze(0)  # Ensure z is (batch, T, d_latent)
        #     assert len(z.shape) == 3, "z must be a tensor of shape (batch, T, d_latent)"
        #     batch, T, d_latent = z.shape
        #     d_embedding = e_bel["m"].shape[-1]

        #     # Move beliefs/covariances to device once
        #     e_m = e_bel["m"].to(self.device)
        #     # Ensure z_bel['m'] and z_bel['P'] have batch dim consistent with z
        #     z_bel_m = z_bel["m"].to(self.device)
        #     if z_bel_m.shape[0] == 1 and batch > 1:
        #         z_bel_m = z_bel_m.expand(batch, -1, -1)

        #     P = z_bel["P"].to(self.device)
        #     # If P has an extra leading singleton, broadcast to batch
        #     if P.dim() == 4 and P.shape[0] == 1:
        #         P = P.expand(batch, -1, -1, -1).squeeze(1)
        #     P = P.squeeze(0) if P.dim() == 3 and P.shape[0] == 1 else P
        #     # Ensure P is (batch, d_latent, d_latent)
        #     if P.dim() == 2:
        #         P = P.unsqueeze(0).expand(batch, -1, -1)

        #     Qb = Q.to(self.device)
        #     if Qb.dim() == 2:
        #         Qb = Qb.unsqueeze(0).expand(batch, -1, -1)
        #     elif Qb.dim() == 3 and Qb.shape[0] == 1:
        #         Qb = Qb.expand(batch, -1, -1)

        #     R = softplus(self.decoder.logvar).diag_embed().to(self.device)
        #     if R.dim() == 3 and R.shape[0] == 1:
        #         R = R.expand(batch, -1, -1)

        #     # Prepare repeated embedding means for Fe/Fz calls
        #     e_rep = e_m.unsqueeze(0).expand(batch, -1) if e_m.dim() == 1 else e_m
        #     e_rep = e_rep.unsqueeze(1).expand(batch, T, -1)

        #     # Compute Fe and Fz for trajectory once (shapes: Fe (batch,T,d_latent,d_emb),
        #     # Fz (batch,T,d_latent,d_latent)) and detach for safety
        #     Fe = self.Fe_net(z, e_rep).detach()
        #     Fz = self.Fz_net(z, e_rep).detach()

        #     # Compute initial sensitivity Gt at belief mean for t=0: shape (batch, d_latent, d_emb)
        #     Fe_bel = self.Fe_net(z_bel_m, e_rep[:, :1, :]).detach()
        #     Gt = Fe_bel[:, 0]  # (batch, d_latent, d_emb)

        #     # Cache decoder jacobian and make batch-aware: H (batch, d_obs, d_latent)
        #     H = self.decoder.jacobian.to(self.device)
        #     H = H.unsqueeze(0).expand(batch, -1, -1)

        #     # Accumulator for embedding-space Fisher information (batch, d_emb, d_emb)
        #     J = torch.zeros(batch, d_embedding, d_embedding, device=self.device)

        #     # Time loop (sequential because P depends on previous step)
        #     for i in range(T):
        #         # dy/de = H @ Gt  -> (batch, d_obs, d_emb)
        #         dyde = torch.einsum("bod,bdp->bop", H, Gt)

        #         # Innovation covariance S = H P H^T + R  -> (batch, d_obs, d_obs)
        #         S = H @ P @ H.mT + R

        #         # Cholesky and solve: solve S x = dyde for x
        #         chol_S = torch.linalg.cholesky(S)
        #         sol = torch.cholesky_solve(dyde, chol_S)

        #         # Update J: sum over observations -> (batch, d_emb, d_emb)
        #         # dyde^T @ sol  => (batch, d_emb, d_emb)
        #         J += torch.einsum("bop,boq->bpq", dyde, sol)

        #         # Propagate Gt and P to next step
        #         Fi = Fz[:, i]
        #         Gi = Fe[:, i]
        #         # Gt <- Fz_i @ Gt + Fe_i
        #         Gt = Fi @ Gt + Gi

        #         # P <- Fz_i @ P @ Fz_i^T + Q
        #         P = Fi @ P @ Fi.mT + Qb

        #     # Compute log-determinant term safely
        #     mat = e_bel["P"].to(self.device) @ J + torch.eye(d_embedding, device=self.device)
        #     sign, logabsdet = torch.linalg.slogdet(mat)
        #     EIG = logabsdet

        #     # Cleanup temporaries
        #     del Fe, Fz, Fe_bel, sol, mat, chol_S, H, dyde, S
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()

        #     return -EIG
