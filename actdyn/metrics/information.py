import actdyn
import actdyn.models
from actdyn.models.base import BaseDynamicsEnsemble
import torch
from typing import Dict, Union, Callable
from actdyn.models import BaseDynamics, Decoder
from actdyn.models.decoder import LinearMapping, LogLinearMapping
from actdyn.models.dynamics import RBFDynamics
from actdyn.models.model import FilteringEmbedding
from actdyn.utils.rollout import Rollout, RolloutBuffer
from .base import BaseMetric
from torch.nn.functional import softplus
from actdyn.utils.helper import safe_cholesky, symmetrize

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
        sensitivity: bool = False,
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
            rates = torch.nan_to_num(
                self.decoder(z),
                nan=eps,
                posinf=1e3,
                neginf=eps,
            ).clamp(min=eps, max=1e3)
            dh_dz = torch.einsum(
                "btd,dn->btdn",
                rates,
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
            # Using cholesky decomposition for solving the linear system is faster than pinv
            CC = dh_dz @ dh_dz.mT  # (B, T, d_obs, d_obs)
            try:
                # Add a small epsilon for numerical stability before cholesky
                L = torch.linalg.cholesky(CC + eps * torch.eye(CC.shape[-1], device=CC.device))
                # Solve (CC)X = dh_dz -> X = inv(CC) @ dh_dz
                invCC_dh_dz = torch.cholesky_solve(dh_dz, L)
                Ht_H = dh_dz.mT @ invCC_dh_dz  # (B, T, d_latent, d_latent)
            except torch.linalg.LinAlgError:
                # Fallback to pinverse if cholesky fails
                invCC = torch.linalg.pinv(CC)
                Ht_H = dh_dz.mT @ invCC @ dh_dz
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
        z = rollout["model_state"]
        if len(z.shape) != 3:
            z = z.unsqueeze(0)
        assert len(z.shape) == 3, "z must be a tensor of shape (batch, T, d_latent)"
        batch, T, d_latent = z.shape

        dh_dz = self.compute_dh_dz(z)
        dz_dtheta_raw = compute_jacobian_params(self.dynamics, z)
        if isinstance(dz_dtheta_raw, tuple):
            parts = []
            for part in dz_dtheta_raw:
                if part is None:
                    continue
                parts.append(part.reshape(batch, T, d_latent, -1))
            if not parts:
                raise ValueError("Dynamics parameter Jacobian is empty for neural dynamics FIM computation")
            dz_dtheta = torch.cat(parts, dim=-1)
        else:
            dz_dtheta = dz_dtheta_raw.reshape(batch, T, d_latent, -1)

        if self.covariance == "invariant":
            cc = dh_dz @ dh_dz.mT
            cc = symmetrize(torch.nan_to_num(cc, nan=0.0, posinf=1e6, neginf=-1e6))
            eye_obs = torch.eye(cc.shape[-1], device=cc.device).view(1, 1, cc.shape[-1], cc.shape[-1])
            chol = None
            for jitter in (eps, 1e-8, 1e-6, 1e-4, 1e-2):
                try:
                    chol = torch.linalg.cholesky(cc + float(jitter) * eye_obs)
                    break
                except torch.linalg.LinAlgError:
                    continue
            if chol is not None:
                invcc_dh = torch.cholesky_solve(dh_dz, chol)
                ht_h = dh_dz.mT @ invcc_dh
            else:
                # Final fallback: diagonal inverse approximation avoids SVD failures on
                # severely ill-conditioned observation covariance batches.
                diag = torch.diagonal(cc, dim1=-2, dim2=-1).clamp_min(eps)
                weighted_dh = dh_dz / diag.unsqueeze(-1)
                ht_h = dh_dz.mT @ weighted_dh
        else:
            ht_h = torch.einsum("btnd,btnf->btdf", dh_dz, dh_dz)

        if self.use_diag:
            i_new = (
                torch.einsum("btdp,btdf,btfp->btp", dz_dtheta, ht_h, dz_dtheta)
                .sum(dim=1)
                .unsqueeze(1)
            )
        else:
            j = torch.einsum("btod,btdp->btop", dh_dz, dz_dtheta)
            i_new = (j.transpose(-1, -2) @ j).sum(dim=1)

        if self.I is not None:
            i_new += self.I * self.discount_factor

        return i_new

    def compute_fim(self, rollout: Union[Rollout, RolloutBuffer, Dict]) -> torch.Tensor:
        if isinstance(self.dynamics, RBFDynamics):
            return self.compute_rbf_fim(rollout)
        else:
            return self.compute_nn_fim(rollout)

    def update_fim(self, rollout: Union[Rollout, RolloutBuffer]):
        self.I = self.compute_fim(rollout)

    def update(self, rollout: Union[Rollout, RolloutBuffer]) -> None:
        self.update_fim(rollout)


class AOptimality(FisherInformationMetric):
    """Metric that computes A-optimality."""

    def compute_stepwise(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        fim_traj = self.compute_fim(rollout)  # (batch, 1, d_param)
        if self.use_diag:
            # reciprocal of element greater than 1e-3
            # return shape (batch, 1)
            fim_traj[fim_traj < eps] = eps
            self.current_cost = torch.reciprocal(fim_traj).sum(dim=-1)
            return self.current_cost

        else:
            # TODO: implement non-diagonal A-optimality
            pass


class DOptimality(FisherInformationMetric):
    """Metric that computes D-optimality."""

    def compute_stepwise(self, rollout: Union[Rollout, RolloutBuffer]) -> torch.Tensor:
        fim_traj = self.compute_fim(rollout)  # (batch, 1, d_param)
        if self.use_diag:
            # reciprocal of element greater than 1e-3
            # return shape (batch, 1)
            fim_traj[fim_traj < eps] = eps
            self.current_cost = -torch.log1p(fim_traj).sum(dim=-1)
            return self.current_cost

        else:
            # TODO: implement non-diagonal A-optimality
            pass


class EmbeddingFisherMetric(BaseMetric):
    """Metric that computes information gain in the embedding space."""

    def __init__(
        self,
        model: FilteringEmbedding,
        compute_type: str = "sum",
        device: str = "cuda",
        Fe_net: Callable = None,
        Fz_net: Callable = None,
        gamma: float | None = None,
        freeze_covariance: bool = False,
        no_sensitivity_propagation: bool = False,
        fully_observed: bool = False,
        **kwargs,
    ):
        super().__init__(compute_type, device)
        self.I = None
        self.Fe_net = Fe_net
        self.Fz_net = Fz_net
        self.model = model
        # Backward-compatible alias from existing config fields.
        legacy_gamma = kwargs.get("met_discount_factor")
        if gamma is None:
            gamma = legacy_gamma if legacy_gamma is not None else 1.0
        self.gamma = float(gamma)
        self.freeze_covariance = bool(freeze_covariance)
        self.no_sensitivity_propagation = bool(no_sensitivity_propagation)
        self.fully_observed = bool(fully_observed)

    def compute_stepwise(self, rollout: Union[Rollout, RolloutBuffer, Dict]) -> torch.Tensor:
        e_bel = self.model.e
        z_bel = self.model.z

        z = rollout["model_state"].to(self.device).float()

        if len(z.shape) != 3:
            z = z.unsqueeze(0)  # Ensure z is (batch, T, d_latent)
        assert len(z.shape) == 3, "z must be a tensor of shape (batch, T, d_latent)"
        batch, T, d_latent = z.shape
        d_embedding = e_bel["m"].shape[-1]
        dt = float(getattr(self.model, "dt", 1.0))
        eye_latent = torch.eye(d_latent, device=self.device).unsqueeze(0).expand(batch, -1, -1)

        def _to_batch_latent_cov(P_in: torch.Tensor) -> torch.Tensor:
            if P_in.dim() == 4:  # (B,1,d,d)
                P_in = P_in.squeeze(1)
            elif P_in.dim() == 2:  # (d,d)
                P_in = P_in.unsqueeze(0)
            if P_in.shape[0] == 1 and batch > 1:
                P_in = P_in.expand(batch, -1, -1)
            return symmetrize(P_in)

        def _to_batch_matrix(M_in: torch.Tensor) -> torch.Tensor:
            if M_in.dim() == 4:  # (B,1,r,c)
                M_in = M_in.squeeze(1)
            elif M_in.dim() == 2:  # (r,c)
                M_in = M_in.unsqueeze(0)
            if M_in.shape[0] == 1 and batch > 1:
                M_in = M_in.expand(batch, M_in.shape[-2], M_in.shape[-1])
            return M_in

        def _to_batch_cov_from_diag(R_diag: torch.Tensor) -> torch.Tensor:
            if R_diag.dim() == 4:
                R = R_diag.squeeze(1)
            else:
                if R_diag.dim() == 2:
                    R_diag = R_diag.unsqueeze(0).unsqueeze(0)
                elif R_diag.dim() == 3 and R_diag.shape[1] != 1:
                    R_diag = R_diag.unsqueeze(1)
                if R_diag.shape[0] == 1 and batch > 1:
                    R_diag = R_diag.expand(batch, -1, -1)
                R = R_diag.diag_embed().squeeze(1)
            return symmetrize(R)

        def _spd_factor(M: torch.Tensor, min_eig: float = 1e-9) -> tuple[torch.Tensor, torch.Tensor]:
            M = torch.nan_to_num(M.float(), nan=0.0, posinf=1e6, neginf=-1e6)
            M = symmetrize(M)
            eye = (
                torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
                .unsqueeze(0)
                .expand(M.shape[0], -1, -1)
            )
            M_spd = symmetrize(M + max(float(min_eig), 1e-8) * eye)
            chol = safe_cholesky(M_spd)
            return M_spd, chol

        # Predicted-state sensitivity recursion S_{k+1}=Fz_k S_k + Fe_k.
        e_m = e_bel["m"].to(self.device)
        if e_m.dim() == 1:
            e_m = e_m.unsqueeze(0)
        if e_m.shape[0] == 1 and batch > 1:
            e_rep = e_m.expand(batch, -1)
        else:
            e_rep = e_m
        e_rep_time = e_rep.unsqueeze(1).expand(batch, T, -1)

        Fe = self.Fe_net(z, e_rep_time).detach()
        Fz = self.Fz_net(z, e_rep_time).detach()
        S_sens = torch.zeros(batch, d_latent, d_embedding, device=self.device, dtype=z.dtype)

        P_pred = _to_batch_latent_cov(z_bel["P"].to(self.device))
        P_pred_initial = P_pred.clone()
        Q = softplus(self.model.dynamics.logvar).diag_embed().to(self.device) * dt
        if Q.dim() == 2:
            Q = Q.unsqueeze(0)
        if Q.shape[0] == 1 and batch > 1:
            Q = Q.expand(batch, -1, -1)
        Q = symmetrize(Q)

        # Discounted accumulation of predicted parameter information.
        J = torch.zeros(batch, d_embedding, d_embedding, device=self.device, dtype=z.dtype)
        for i in range(T):
            dfdz = eye_latent + Fz[:, i] * dt
            dfde = Fe[:, i] * dt
            if self.no_sensitivity_propagation:
                S_sens = dfde
            else:
                S_sens = dfdz @ S_sens + dfde

            z_i = z[:, i : i + 1]
            H_i = self.model.decoder.jacobian(z_i).to(self.device)
            H_i = _to_batch_matrix(H_i)

            if isinstance(self.model.decoder.noise, actdyn.models.decoder.PoissonNoise):
                R_diag = self.model.decoder(z_i).to(self.device)
            else:
                R_diag = self.model.decoder.var(z_i).to(self.device)
            R_i = _to_batch_cov_from_diag(R_diag)
            R_i, chol_R = _spd_factor(R_i)

            # I_z = H^T R^{-1} H (Fisher approximation in state space).
            invR_H = torch.cholesky_solve(H_i, chol_R)
            I_z = symmetrize(H_i.transpose(-1, -2) @ invR_H)

            # DeltaLambda = S^T (I + P^- I_z)^{-1} I_z S.
            if self.fully_observed:
                atten_Iz = I_z
            else:
                P_for_gain = P_pred_initial if self.freeze_covariance else P_pred
                atten_base, chol_atten = _spd_factor(eye_latent + P_for_gain @ I_z)
                del atten_base
                atten_Iz = torch.cholesky_solve(I_z, chol_atten)
            info_step = symmetrize(S_sens.transpose(-1, -2) @ atten_Iz @ S_sens)
            J += (self.gamma ** i) * info_step

            P_pred = symmetrize(dfdz @ P_pred @ dfdz.transpose(-1, -2) + Q)

        P_theta = e_bel["P"].to(self.device)
        if P_theta.dim() == 2:
            P_theta = P_theta.unsqueeze(0)
        if P_theta.shape[0] == 1 and batch > 1:
            P_theta = P_theta.expand(batch, -1, -1)

        eye = torch.eye(d_embedding, device=self.device).unsqueeze(0).expand(batch, -1, -1)
        mat, chol_mat = _spd_factor(eye + P_theta @ J)
        chol_diag = torch.diagonal(chol_mat, dim1=-2, dim2=-1).clamp_min(eps)
        logabsdet = 2.0 * torch.log(chol_diag).sum(dim=-1)
        EIG = 0.5 * logabsdet

        # Explicitly delete large temporaries (helps long-running processes)
        del mat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.current_cost = (-EIG).unsqueeze(-1)
        return self.current_cost
