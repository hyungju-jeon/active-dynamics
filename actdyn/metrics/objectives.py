"""Objective factory helpers for catalog-driven active-learning policies."""

from __future__ import annotations

from typing import Callable

import torch
from torch.nn.functional import softplus

from actdyn.metrics.base import BaseMetric
from actdyn.metrics.information import (
    AmbiguityAwareEmbeddingFisherMetric,
    EmbeddingFisherMetric,
)
from actdyn.models.model import FilteringEmbedding
from actdyn.utils.torch_utils import (
    attenuated_state_information,
    safe_cholesky,
    symmetrize,
)

eps = 1e-12


def parameter_eig(
    *,
    model: FilteringEmbedding,
    Fe_net: Callable,
    Fz_net: Callable,
    gamma: float,
    device: str,
    freeze_covariance: bool = False,
    diagonal_covariance: bool = False,
) -> EmbeddingFisherMetric:
    return EmbeddingFisherMetric(
        model=model,
        Fe_net=Fe_net,
        Fz_net=Fz_net,
        gamma=gamma,
        freeze_covariance=freeze_covariance,
        diagonal_covariance=diagonal_covariance,
        device=device,
    )


def shrinkage_parameter_eig(
    *,
    model: FilteringEmbedding,
    Fe_net: Callable,
    Fz_net: Callable,
    gamma: float,
    device: str,
) -> EmbeddingFisherMetric:
    return parameter_eig(model=model, Fe_net=Fe_net, Fz_net=Fz_net, gamma=gamma, device=device)


def _scaled_sensitivity_network(network: Callable, scale: float) -> Callable:
    def _wrapped(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return float(scale) * network(z, e)

    return _wrapped


def ambiguity_aware_parameter_eig(
    *,
    model: FilteringEmbedding,
    Fe_net: Callable,
    Fz_net: Callable,
    gamma: float,
    device: str,
    ambiguity_temperature: float = 1.0,
    ensemble_kind: str | None = None,
) -> AmbiguityAwareEmbeddingFisherMetric:
    if ensemble_kind in {None, "sensitivity_gain"}:
        scale_factors = (0.75, 1.0, 1.25)
    elif ensemble_kind == "wide_sensitivity_gain":
        scale_factors = (0.5, 1.0, 1.5)
    else:
        raise ValueError(f"Unsupported ensemble_kind={ensemble_kind!r}")

    ensemble_members = [
        {
            "Fe_net": _scaled_sensitivity_network(Fe_net, scale),
            "Fz_net": _scaled_sensitivity_network(Fz_net, scale),
            "decoder": model.decoder,
            "weight": 1.0 / len(scale_factors),
        }
        for scale in scale_factors
    ]
    return AmbiguityAwareEmbeddingFisherMetric(
        model=model,
        ensemble_members=ensemble_members,
        ambiguity_temperature=float(ambiguity_temperature),
        gamma=gamma,
        device=device,
    )


def fully_observable_parameter_eig(
    *,
    model: FilteringEmbedding,
    Fe_net: Callable,
    Fz_net: Callable,
    gamma: float,
    device: str,
) -> EmbeddingFisherMetric:
    return EmbeddingFisherMetric(
        model=model,
        Fe_net=Fe_net,
        Fz_net=Fz_net,
        gamma=gamma,
        fully_observed=True,
        device=device,
    )


class EOptimalityMetric(BaseMetric):
    def __init__(
        self,
        *,
        model: FilteringEmbedding,
        Fe_net: Callable,
        Fz_net: Callable,
        gamma: float,
        device: str,
    ) -> None:
        super().__init__(compute_type="sum", device=device)
        self.model = model
        self.Fe_net = Fe_net
        self.Fz_net = Fz_net
        self.gamma = float(gamma)

    def compute_stepwise(self, rollout: dict) -> torch.Tensor:
        z = rollout["model_state"].to(self.device).float()
        if z.ndim != 3:
            z = z.unsqueeze(0)
        batch, steps, d_latent = z.shape
        e_bel = self.model.e
        z_bel = self.model.z
        d_embedding = int(e_bel["m"].shape[-1])

        e_m = e_bel["m"].to(self.device)
        if e_m.ndim == 1:
            e_m = e_m.unsqueeze(0)
        if e_m.shape[0] == 1 and batch > 1:
            e_rep = e_m.expand(batch, -1)
        else:
            e_rep = e_m
        e_rep_time = e_rep.unsqueeze(1).expand(batch, steps, -1)
        Fe = self.Fe_net(z, e_rep_time).detach()
        Fz = self.Fz_net(z, e_rep_time).detach()

        p_pred = z_bel["P"].to(self.device)
        if p_pred.ndim == 4:
            p_pred = p_pred.squeeze(1)
        elif p_pred.ndim == 2:
            p_pred = p_pred.unsqueeze(0)
        if p_pred.shape[0] == 1 and batch > 1:
            p_pred = p_pred.expand(batch, -1, -1)
        p_pred = symmetrize(p_pred)

        dt = float(getattr(self.model, "dt", 1.0))
        q = softplus(self.model.dynamics.logvar).diag_embed().to(self.device) * dt
        if q.ndim == 2:
            q = q.unsqueeze(0)
        if q.shape[0] == 1 and batch > 1:
            q = q.expand(batch, -1, -1)
        q = symmetrize(q)

        eye_latent = torch.eye(d_latent, device=self.device).unsqueeze(0).expand(batch, -1, -1)
        eye_embed = torch.eye(d_embedding, device=self.device).unsqueeze(0).expand(batch, -1, -1)
        s_sens = torch.zeros(batch, d_latent, d_embedding, device=self.device, dtype=z.dtype)
        j_total = torch.zeros(batch, d_embedding, d_embedding, device=self.device, dtype=z.dtype)

        for i in range(steps):
            dfdz = eye_latent + Fz[:, i] * dt
            dfde = Fe[:, i] * dt
            s_sens = dfdz @ s_sens + dfde

            z_i = z[:, i : i + 1]
            H_i = self.model.decoder.jacobian(z_i).to(self.device)
            if H_i.ndim == 4:
                H_i = H_i.squeeze(1)
            elif H_i.ndim == 2:
                H_i = H_i.unsqueeze(0)
            if H_i.shape[0] == 1 and batch > 1:
                H_i = H_i.expand(batch, -1, -1)

            if hasattr(self.model.decoder.noise, "__class__") and self.model.decoder.noise.__class__.__name__ == "PoissonNoise":
                r_diag = self.model.decoder(z_i).to(self.device)
            else:
                r_diag = self.model.decoder.var(z_i).to(self.device)
            if r_diag.ndim == 4:
                r = r_diag.squeeze(1)
            else:
                if r_diag.ndim == 2:
                    r_diag = r_diag.unsqueeze(0).unsqueeze(0)
                elif r_diag.ndim == 3 and r_diag.shape[1] != 1:
                    r_diag = r_diag.unsqueeze(1)
                if r_diag.shape[0] == 1 and batch > 1:
                    r_diag = r_diag.expand(batch, -1, -1)
                r = r_diag.diag_embed().squeeze(1)
            r = symmetrize(r)
            eye_obs = torch.eye(r.shape[-1], device=self.device).unsqueeze(0).expand(batch, -1, -1)
            chol_r = safe_cholesky(r + 1e-8 * eye_obs)
            invr_h = torch.cholesky_solve(H_i, chol_r)
            i_z = symmetrize(H_i.transpose(-1, -2) @ invr_h)

            atten_i_z = attenuated_state_information(p_pred, i_z)
            info_step = symmetrize(s_sens.transpose(-1, -2) @ atten_i_z @ s_sens)
            j_total = j_total + (self.gamma**i) * info_step

            p_pred = symmetrize(dfdz @ p_pred @ dfdz.transpose(-1, -2) + q)

        p_theta = e_bel["P"].to(self.device)
        if p_theta.ndim == 2:
            p_theta = p_theta.unsqueeze(0)
        if p_theta.shape[0] == 1 and batch > 1:
            p_theta = p_theta.expand(batch, -1, -1)
        chol_theta = safe_cholesky(symmetrize(p_theta) + 1e-8 * eye_embed)
        scaled_info = symmetrize(chol_theta.transpose(-1, -2) @ j_total @ chol_theta)
        eigvals = torch.linalg.eigvalsh(scaled_info + 1e-8 * eye_embed)
        e_opt = eigvals[..., 0]
        self.current_cost = (-e_opt).unsqueeze(-1)
        return self.current_cost


def e_optimality(
    *,
    model: FilteringEmbedding,
    Fe_net: Callable,
    Fz_net: Callable,
    gamma: float,
    device: str,
) -> EOptimalityMetric:
    return EOptimalityMetric(
        model=model,
        Fe_net=Fe_net,
        Fz_net=Fz_net,
        gamma=gamma,
        device=device,
    )


class _FilteringObjectiveBase(BaseMetric):
    def __init__(
        self,
        *,
        model: FilteringEmbedding,
        Fe_net: Callable,
        Fz_net: Callable,
        gamma: float,
        device: str,
    ) -> None:
        super().__init__(compute_type="sum", device=device)
        self.model = model
        self.Fe_net = Fe_net
        self.Fz_net = Fz_net
        self.gamma = float(gamma)

    def _prepare(self, rollout: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = rollout["model_state"].to(self.device).float()
        if z.ndim != 3:
            z = z.unsqueeze(0)
        batch, steps, latent_dim = z.shape
        e_bel = self.model.e
        z_bel = self.model.z
        e_m = e_bel["m"].to(self.device)
        if e_m.ndim == 1:
            e_m = e_m.unsqueeze(0)
        if e_m.shape[0] == 1 and batch > 1:
            e_rep = e_m.expand(batch, -1)
        else:
            e_rep = e_m
        e_rep_time = e_rep.unsqueeze(1).expand(batch, steps, -1)
        Fe = self.Fe_net(z, e_rep_time).detach()
        Fz = self.Fz_net(z, e_rep_time).detach()
        P_pred = z_bel["P"].to(self.device)
        if P_pred.ndim == 4:
            P_pred = P_pred.squeeze(1)
        elif P_pred.ndim == 2:
            P_pred = P_pred.unsqueeze(0)
        if P_pred.shape[0] == 1 and batch > 1:
            P_pred = P_pred.expand(batch, -1, -1)
        P_pred = symmetrize(P_pred)
        eye_latent = torch.eye(latent_dim, device=self.device).unsqueeze(0).expand(batch, -1, -1)
        dt = float(getattr(self.model, "dt", 1.0))
        q = softplus(self.model.dynamics.logvar).diag_embed().to(self.device) * dt
        if q.ndim == 2:
            q = q.unsqueeze(0)
        if q.shape[0] == 1 and batch > 1:
            q = q.expand(batch, -1, -1)
        q = symmetrize(q)
        return z, Fe, Fz, P_pred, eye_latent + 0.0 * q + 0.0 * z[:, :1, :1]


class StateInformationMetric(_FilteringObjectiveBase):
    def compute_stepwise(self, rollout: dict) -> torch.Tensor:
        z = rollout["model_state"].to(self.device).float()
        if z.ndim != 3:
            z = z.unsqueeze(0)
        batch, steps, latent_dim = z.shape
        z_bel = self.model.z
        p_pred = z_bel["P"].to(self.device)
        if p_pred.ndim == 4:
            p_pred = p_pred.squeeze(1)
        elif p_pred.ndim == 2:
            p_pred = p_pred.unsqueeze(0)
        if p_pred.shape[0] == 1 and batch > 1:
            p_pred = p_pred.expand(batch, -1, -1)
        p_pred = symmetrize(p_pred)
        dt = float(getattr(self.model, "dt", 1.0))
        q = softplus(self.model.dynamics.logvar).diag_embed().to(self.device) * dt
        if q.ndim == 2:
            q = q.unsqueeze(0)
        if q.shape[0] == 1 and batch > 1:
            q = q.expand(batch, -1, -1)
        q = symmetrize(q)
        eye = torch.eye(latent_dim, device=self.device).unsqueeze(0).expand(batch, -1, -1)
        Fz = self.Fz_net(z, self.model.e["m"].to(self.device).unsqueeze(1).expand(batch, steps, -1)).detach()
        current = torch.zeros(batch, device=self.device, dtype=z.dtype)
        for i in range(steps):
            z_i = z[:, i : i + 1]
            H_i = self.model.decoder.jacobian(z_i).to(self.device)
            if H_i.ndim == 4:
                H_i = H_i.squeeze(1)
            elif H_i.ndim == 2:
                H_i = H_i.unsqueeze(0)
            if isinstance(self.model.decoder.noise, torch.nn.Module) and hasattr(self.model.decoder.noise, "sigma"):
                pass
            if hasattr(self.model.decoder.noise, "__class__") and self.model.decoder.noise.__class__.__name__ == "PoissonNoise":
                r_diag = self.model.decoder(z_i).to(self.device)
            else:
                r_diag = self.model.decoder.var(z_i).to(self.device)
            if r_diag.ndim == 4:
                r = r_diag.squeeze(1)
            else:
                if r_diag.ndim == 2:
                    r_diag = r_diag.unsqueeze(0).unsqueeze(0)
                elif r_diag.ndim == 3 and r_diag.shape[1] != 1:
                    r_diag = r_diag.unsqueeze(1)
                if r_diag.shape[0] == 1 and batch > 1:
                    r_diag = r_diag.expand(batch, -1, -1)
                r = r_diag.diag_embed().squeeze(1)
            r = symmetrize(r)
            eye_obs = torch.eye(r.shape[-1], device=self.device).unsqueeze(0).expand(batch, -1, -1)
            chol_r = safe_cholesky(r + 1e-8 * eye_obs)
            invr_h = torch.cholesky_solve(H_i, chol_r)
            i_z = symmetrize(H_i.transpose(-1, -2) @ invr_h)
            chol_p = safe_cholesky(symmetrize(p_pred) + 1e-8 * eye)
            mat = symmetrize(eye + chol_p.transpose(-1, -2) @ i_z @ chol_p)
            chol = safe_cholesky(mat + 1e-8 * eye)
            current = current + (self.gamma**i) * torch.log(
                torch.diagonal(chol, dim1=-2, dim2=-1).clamp_min(eps)
            ).sum(dim=-1)
            dfdz = eye + Fz[:, i] * dt
            p_pred = symmetrize(dfdz @ p_pred @ dfdz.transpose(-1, -2) + q)
        self.current_cost = (-current).unsqueeze(-1)
        return self.current_cost


class DynamicsMetric(_FilteringObjectiveBase):
    """Covariance-weighted parameter-sensitivity objective.

    Scores the sensitivity Gramian ``G = S^T P S`` accumulated along the
    rollout, where ``S = d z / d theta`` is the propagated parameter
    sensitivity and ``P`` is the predicted state covariance. Two
    scalarizations:

    - ``"trace"`` (default): ``tr(G)`` -- total parameter-induced state
      variance. Rank-blind: rewards concentrating on the single most
      sensitive parameter direction.
    - ``"logdet"``: ``logdet(I + G)`` -- rank-aware (D-optimal on the
      Gramian). Still observation-free, so it shares the ``"trace"`` blind
      spot to the nuisance confound, but it rewards spreading sensitivity to
      full rank instead of piling onto one direction.
    """

    def __init__(self, *, scalarization: str = "trace", **kwargs) -> None:
        super().__init__(**kwargs)
        if scalarization not in {"trace", "logdet"}:
            raise ValueError(
                f"scalarization must be 'trace' or 'logdet', got {scalarization!r}"
            )
        self.scalarization = scalarization

    def compute_stepwise(self, rollout: dict) -> torch.Tensor:
        z = rollout["model_state"].to(self.device).float()
        if z.ndim != 3:
            z = z.unsqueeze(0)
        batch, steps, latent_dim = z.shape
        e_bel = self.model.e
        z_bel = self.model.z
        e_m = e_bel["m"].to(self.device)
        if e_m.ndim == 1:
            e_m = e_m.unsqueeze(0)
        if e_m.shape[0] == 1 and batch > 1:
            e_rep = e_m.expand(batch, -1)
        else:
            e_rep = e_m
        e_rep_time = e_rep.unsqueeze(1).expand(batch, steps, -1)
        Fe = self.Fe_net(z, e_rep_time).detach()
        Fz = self.Fz_net(z, e_rep_time).detach()
        P_pred = z_bel["P"].to(self.device)
        if P_pred.ndim == 4:
            P_pred = P_pred.squeeze(1)
        elif P_pred.ndim == 2:
            P_pred = P_pred.unsqueeze(0)
        if P_pred.shape[0] == 1 and batch > 1:
            P_pred = P_pred.expand(batch, -1, -1)
        P_pred = symmetrize(P_pred)
        dt = float(getattr(self.model, "dt", 1.0))
        q = softplus(self.model.dynamics.logvar).diag_embed().to(self.device) * dt
        if q.ndim == 2:
            q = q.unsqueeze(0)
        if q.shape[0] == 1 and batch > 1:
            q = q.expand(batch, -1, -1)
        q = symmetrize(q)
        eye = torch.eye(latent_dim, device=self.device).unsqueeze(0).expand(batch, -1, -1)
        embed_dim = e_bel["m"].shape[-1]
        s_sens = torch.zeros(batch, latent_dim, embed_dim, device=self.device, dtype=z.dtype)
        total = torch.zeros(batch, device=self.device, dtype=z.dtype)
        for i in range(steps):
            dfdz = eye + Fz[:, i] * dt
            dfde = Fe[:, i] * dt
            s_sens = dfdz @ s_sens + dfde
            if self.scalarization == "logdet":
                gram = torch.einsum("bde,bdk,bkf->bef", s_sens, P_pred, s_sens)
                eye_embed = torch.eye(
                    embed_dim, device=self.device, dtype=gram.dtype
                ).unsqueeze(0)
                score = torch.logdet(symmetrize(gram) + eye_embed)
            else:
                score = torch.einsum("bde,bdk,bke->b", s_sens, P_pred, s_sens)
            total = total + (self.gamma**i) * score
            P_pred = symmetrize(dfdz @ P_pred @ dfdz.transpose(-1, -2) + q)
        self.current_cost = (-total).unsqueeze(-1)
        return self.current_cost


class ObservationVarianceMetric(BaseMetric):
    def __init__(
        self,
        *,
        model: FilteringEmbedding,
        gamma: float,
        num_parameter_samples: int,
        sample_seed: int | None,
        device: str,
    ) -> None:
        super().__init__(compute_type="sum", device=device)
        self.model = model
        self.gamma = float(gamma)
        self.num_parameter_samples = max(1, int(num_parameter_samples))
        self._sample_seed = None if sample_seed is None else int(sample_seed)
        self._call_count = 0

    def _sample_theta_belief(self) -> torch.Tensor:
        mean = self.model.e["m"].to(self.device)
        if mean.ndim == 2:
            mean = mean[0]
        cov = self.model.e["P"].to(self.device)
        if cov.ndim == 3:
            cov = cov[0]
        cov = symmetrize(cov)
        eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=self.device)
        chol = safe_cholesky(cov + 1e-8 * eye)
        if self._sample_seed is None:
            noise = torch.randn(
                self.num_parameter_samples,
                cov.shape[-1],
                dtype=mean.dtype,
                device=self.device,
            )
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._sample_seed + self._call_count)
            noise = torch.randn(
                self.num_parameter_samples,
                cov.shape[-1],
                dtype=mean.dtype,
                generator=generator,
            ).to(self.device)
        self._call_count += 1
        return mean.unsqueeze(0) + noise @ chol.transpose(-1, -2)

    @staticmethod
    def _rollout_get(rollout, key: str):
        if isinstance(rollout, dict):
            return rollout.get(key)
        try:
            return rollout[key]
        except (KeyError, TypeError, IndexError):
            getter = getattr(rollout, "get", None)
            if getter is None:
                return None
            return getter(key, None)

    def _rollout_actions(self, rollout):
        action = self._rollout_get(rollout, "action")
        encoded_action = self._rollout_get(rollout, "encoded_action")
        if encoded_action is None:
            encoded_action = self._rollout_get(rollout, "env_action")
        if encoded_action is None:
            encoded_action = self._rollout_get(rollout, "model_action")
        return action, encoded_action

    def _encode_actions(self, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.model.action_encoder is None:
            return actions
        try:
            return self.model.action_encoder(actions)
        except TypeError:
            return self.model.action_encoder(actions, state)

    def _predict_lambda_samples(
        self,
        *,
        init_state: torch.Tensor,
        encoded_actions: torch.Tensor,
        theta_samples: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self.model.dynamics, "sample_forward"):
            return self._predict_lambda_samples_fallback(
                init_state=init_state,
                encoded_actions=encoded_actions,
                theta_samples=theta_samples,
            )
        batch, steps, _ = encoded_actions.shape
        num_samples = int(theta_samples.shape[0])
        state_batch = init_state.unsqueeze(0).expand(num_samples, -1, -1, -1).reshape(
            num_samples * batch, 1, -1
        )
        action_batch = encoded_actions.unsqueeze(0).expand(num_samples, -1, -1, -1).reshape(
            num_samples * batch, steps, -1
        )
        theta_batch = theta_samples.unsqueeze(1).expand(num_samples, batch, -1).reshape(
            num_samples * batch, -1
        )
        current_theta = self.model.e["m"].detach().clone()
        try:
            with torch.no_grad():
                self.model.dynamics.set_params(theta_batch)
                samples, _mus, _vars = self.model.dynamics.sample_forward(
                    init_z=state_batch,
                    action=action_batch,
                    k_step=steps,
                    return_traj=True,
                    add_noise=False,
                )
                traj = torch.cat(samples[1:], dim=-2)
                lam = self.model.decoder(traj).to(self.device)
        finally:
            self.model.dynamics.set_params(current_theta)
        return lam.reshape(num_samples, batch, steps, -1)

    def _predict_lambda_samples_fallback(
        self,
        *,
        init_state: torch.Tensor,
        encoded_actions: torch.Tensor,
        theta_samples: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self.model, "predict"):
            raise AttributeError("ObservationVarianceMetric requires model.predict for fallback mode")
        batch, _steps, _ = encoded_actions.shape
        original_state = None
        if hasattr(self.model, "_state"):
            original_state = self.model._state.detach().clone()
        current_theta = self.model.e["m"].detach().clone()
        lambda_samples = []
        try:
            for theta in theta_samples:
                theta_batch = theta.unsqueeze(0)
                if hasattr(self.model, "set_params"):
                    self.model.set_params(theta_batch)
                else:
                    self.model.dynamics.set_params(theta_batch)
                if original_state is not None:
                    state_seed = init_state if init_state.shape[0] == batch else init_state.expand(batch, -1, -1)
                    self.model._state = state_seed.detach().clone()
                traj = self.model.predict(encoded_actions)
                lam = self.model.decoder(traj).to(self.device)
                lambda_samples.append(lam)
        finally:
            if hasattr(self.model, "set_params"):
                self.model.set_params(current_theta)
            else:
                self.model.dynamics.set_params(current_theta)
            if original_state is not None:
                self.model._state = original_state
        return torch.stack(lambda_samples, dim=0)

    def compute_stepwise(self, rollout: dict) -> torch.Tensor:
        actions, encoded_action_value = self._rollout_actions(rollout)
        if actions is not None:
            actions = actions.to(self.device).float()
            if actions.ndim != 3:
                actions = actions.unsqueeze(0)
            batch, steps, _ = actions.shape
        else:
            if encoded_action_value is None:
                raise KeyError(
                    "ObservationVarianceMetric requires one of 'action', 'encoded_action', "
                    "'env_action', or 'model_action' in rollout"
                )
            encoded_actions = encoded_action_value.to(self.device).float()
            if encoded_actions.ndim != 3:
                encoded_actions = encoded_actions.unsqueeze(0)
            batch, steps, _ = encoded_actions.shape
        rollout_states_value = self._rollout_get(rollout, "model_state")
        if rollout_states_value is not None:
            rollout_states = rollout_states_value.to(self.device).float()
            if rollout_states.ndim != 3:
                rollout_states = rollout_states.unsqueeze(0)
            state0 = rollout_states[:, :1]
            if state0.shape[0] == 1 and batch > 1:
                state0 = state0.expand(batch, -1, -1).clone()
        else:
            if hasattr(self.model, "get_state"):
                state0 = self.model.get_state().to(self.device).float()
            else:
                state0 = self.model._state.to(self.device).float()
            if state0.ndim != 3:
                state0 = state0.unsqueeze(0)
            if state0.shape[0] == 1 and batch > 1:
                state0 = state0.expand(batch, -1, -1).clone()
        if encoded_action_value is not None:
            if "encoded_actions" not in locals():
                encoded_actions = encoded_action_value.to(self.device).float()
                if encoded_actions.ndim != 3:
                    encoded_actions = encoded_actions.unsqueeze(0)
        else:
            if actions is None:
                raise KeyError("ObservationVarianceMetric cannot encode actions when rollout['action'] is missing")
            encoded_actions = self._encode_actions(actions, state0)
        theta_samples = self._sample_theta_belief()
        lam_stack = self._predict_lambda_samples(
            init_state=state0,
            encoded_actions=encoded_actions,
            theta_samples=theta_samples,
        )
        var_diag = torch.var(
            lam_stack,
            dim=0,
            unbiased=self.num_parameter_samples > 1,
        )
        logdet_diag = torch.log1p(var_diag.clamp_min(0.0)).sum(dim=-1)
        gamma_scale = torch.pow(
            torch.full((steps,), self.gamma, dtype=lam_stack.dtype, device=self.device),
            torch.arange(steps, dtype=lam_stack.dtype, device=self.device),
        ).view(1, steps)
        total = torch.sum(gamma_scale * logdet_diag, dim=-1)
        self.current_cost = (-total).unsqueeze(-1)
        return self.current_cost


class CorrectedObservationVarianceMetric(ObservationVarianceMetric):
    """Observation-variance objective with Laplace-proposal importance correction.

    The current TBME filtering posterior is Gaussian. We treat that Gaussian as the
    Laplace proposal and reweight parameter samples toward a multivariate-Student-t
    corrected posterior with the same mode/covariance. This is a lightweight
    TBME-compatible analogue of the toy Laplace correction; it exposes ESS
    diagnostics and falls back to uniform weights when ESS is too low.
    """

    def __init__(
        self,
        *,
        correction_df: float = 3.0,
        ess_gate_fraction: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.correction_df = max(float(correction_df), 1.0 + 1e-6)
        self.ess_gate_fraction = float(max(0.0, min(1.0, ess_gate_fraction)))
        self.last_effective_sample_size: float | None = None
        self.last_effective_sample_fraction: float | None = None
        self.last_used_correction: bool | None = None

    def _sample_theta_belief_with_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        samples = self._sample_theta_belief()
        mean = self.model.e["m"].to(self.device)
        if mean.ndim == 2:
            mean = mean[0]
        cov = self.model.e["P"].to(self.device)
        if cov.ndim == 3:
            cov = cov[0]
        cov = symmetrize(cov)
        eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=self.device)
        chol = safe_cholesky(cov + 1e-8 * eye)
        diff = samples - mean.unsqueeze(0)
        solved = torch.cholesky_solve(diff.unsqueeze(-1), chol).squeeze(-1)
        mahal = torch.sum(diff * solved, dim=-1)
        dim = float(samples.shape[-1])
        df = float(self.correction_df)
        # Constants that do not depend on samples cancel in normalized weights.
        log_target = -0.5 * (df + dim) * torch.log1p(mahal / df)
        log_proposal = -0.5 * mahal
        logw = log_target - log_proposal
        logw = logw - torch.logsumexp(logw, dim=0)
        weights = torch.exp(logw)
        ess = torch.reciprocal(torch.sum(weights * weights).clamp_min(1e-12))
        self.last_effective_sample_size = float(ess.detach().cpu().item())
        self.last_effective_sample_fraction = self.last_effective_sample_size / float(self.num_parameter_samples)
        if self.last_effective_sample_fraction < self.ess_gate_fraction:
            weights = torch.full_like(weights, 1.0 / float(weights.numel()))
            self.last_used_correction = False
        else:
            self.last_used_correction = True
        return samples, weights

    @staticmethod
    def _weighted_variance(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        w = weights.to(values.device, dtype=values.dtype).view(-1, *([1] * (values.ndim - 1)))
        mean = torch.sum(w * values, dim=0, keepdim=True)
        var = torch.sum(w * (values - mean) ** 2, dim=0)
        return var

    def compute_stepwise(self, rollout: dict) -> torch.Tensor:
        actions, encoded_action_value = self._rollout_actions(rollout)
        if actions is not None:
            actions = actions.to(self.device).float()
            if actions.ndim != 3:
                actions = actions.unsqueeze(0)
            batch, steps, _ = actions.shape
        else:
            if encoded_action_value is None:
                raise KeyError(
                    "CorrectedObservationVarianceMetric requires one of 'action', 'encoded_action', "
                    "'env_action', or 'model_action' in rollout"
                )
            encoded_actions = encoded_action_value.to(self.device).float()
            if encoded_actions.ndim != 3:
                encoded_actions = encoded_actions.unsqueeze(0)
            batch, steps, _ = encoded_actions.shape
        rollout_states_value = self._rollout_get(rollout, "model_state")
        if rollout_states_value is not None:
            rollout_states = rollout_states_value.to(self.device).float()
            if rollout_states.ndim != 3:
                rollout_states = rollout_states.unsqueeze(0)
            state0 = rollout_states[:, :1]
            if state0.shape[0] == 1 and batch > 1:
                state0 = state0.expand(batch, -1, -1).clone()
        else:
            if hasattr(self.model, "get_state"):
                state0 = self.model.get_state().to(self.device).float()
            else:
                state0 = self.model._state.to(self.device).float()
            if state0.ndim != 3:
                state0 = state0.unsqueeze(0)
            if state0.shape[0] == 1 and batch > 1:
                state0 = state0.expand(batch, -1, -1).clone()
        if encoded_action_value is not None:
            if "encoded_actions" not in locals():
                encoded_actions = encoded_action_value.to(self.device).float()
                if encoded_actions.ndim != 3:
                    encoded_actions = encoded_actions.unsqueeze(0)
        else:
            if actions is None:
                raise KeyError("CorrectedObservationVarianceMetric cannot encode actions when rollout['action'] is missing")
            encoded_actions = self._encode_actions(actions, state0)
        theta_samples, weights = self._sample_theta_belief_with_weights()
        lam_stack = self._predict_lambda_samples(
            init_state=state0,
            encoded_actions=encoded_actions,
            theta_samples=theta_samples,
        )
        var_diag = self._weighted_variance(lam_stack, weights)
        logdet_diag = torch.log1p(var_diag.clamp_min(0.0)).sum(dim=-1)
        gamma_scale = torch.pow(
            torch.full((steps,), self.gamma, dtype=lam_stack.dtype, device=self.device),
            torch.arange(steps, dtype=lam_stack.dtype, device=self.device),
        ).view(1, steps)
        total = torch.sum(gamma_scale * logdet_diag, dim=-1)
        self.current_cost = (-total).unsqueeze(-1)
        return self.current_cost


def state_information(
    *,
    model: FilteringEmbedding,
    Fe_net: Callable,
    Fz_net: Callable,
    gamma: float,
    device: str,
) -> StateInformationMetric:
    return StateInformationMetric(
        model=model,
        Fe_net=Fe_net,
        Fz_net=Fz_net,
        gamma=gamma,
        device=device,
    )


def dynamics(
    *,
    model: FilteringEmbedding,
    Fe_net: Callable,
    Fz_net: Callable,
    gamma: float,
    device: str,
) -> DynamicsMetric:
    return DynamicsMetric(
        model=model,
        Fe_net=Fe_net,
        Fz_net=Fz_net,
        gamma=gamma,
        device=device,
        scalarization="trace",
    )


def dynamics_logdet(
    *,
    model: FilteringEmbedding,
    Fe_net: Callable,
    Fz_net: Callable,
    gamma: float,
    device: str,
) -> DynamicsMetric:
    """Rank-aware ``logdet(I + S^T P S)`` variant of the dynamics objective."""
    return DynamicsMetric(
        model=model,
        Fe_net=Fe_net,
        Fz_net=Fz_net,
        gamma=gamma,
        device=device,
        scalarization="logdet",
    )


def observation_variance(
    *,
    model: FilteringEmbedding,
    Fe_net: Callable,
    Fz_net: Callable,
    gamma: float,
    device: str,
    num_parameter_samples: int,
    sample_seed: int | None = None,
) -> ObservationVarianceMetric:
    del Fe_net, Fz_net
    return ObservationVarianceMetric(
        model=model,
        gamma=gamma,
        num_parameter_samples=num_parameter_samples,
        sample_seed=sample_seed,
        device=device,
    )


def corrected_observation_variance(
    *,
    model: FilteringEmbedding,
    Fe_net: Callable,
    Fz_net: Callable,
    gamma: float,
    device: str,
    num_parameter_samples: int,
    sample_seed: int | None = None,
    correction_df: float = 3.0,
    ess_gate_fraction: float = 0.05,
) -> CorrectedObservationVarianceMetric:
    del Fe_net, Fz_net
    return CorrectedObservationVarianceMetric(
        model=model,
        gamma=gamma,
        num_parameter_samples=num_parameter_samples,
        sample_seed=sample_seed,
        device=device,
        correction_df=correction_df,
        ess_gate_fraction=ess_gate_fraction,
    )


class StateVarianceMetric(ObservationVarianceMetric):
    """State-variance objective using theta samples from the current parameter posterior."""

    def __init__(self, *, aggregation: str = "sum", **kwargs) -> None:
        super().__init__(**kwargs)
        if aggregation not in {"sum", "terminal"}:
            raise ValueError(f"Unsupported aggregation={aggregation!r}")
        self.aggregation = aggregation

    def _predict_state_samples(
        self,
        *,
        init_state: torch.Tensor,
        encoded_actions: torch.Tensor,
        theta_samples: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self.model.dynamics, "sample_forward"):
            batch, steps, _ = encoded_actions.shape
            num_samples = int(theta_samples.shape[0])
            state_batch = init_state.unsqueeze(0).expand(num_samples, -1, -1, -1).reshape(
                num_samples * batch, 1, -1
            )
            action_batch = encoded_actions.unsqueeze(0).expand(num_samples, -1, -1, -1).reshape(
                num_samples * batch, steps, -1
            )
            theta_batch = theta_samples.unsqueeze(1).expand(num_samples, batch, -1).reshape(
                num_samples * batch, -1
            )
            current_theta = self.model.e["m"].detach().clone()
            try:
                with torch.no_grad():
                    self.model.dynamics.set_params(theta_batch)
                    samples, _mus, _vars = self.model.dynamics.sample_forward(
                        init_z=state_batch,
                        action=action_batch,
                        k_step=steps,
                        return_traj=True,
                        add_noise=False,
                    )
                    traj = torch.cat(samples[1:], dim=-2)
            finally:
                self.model.dynamics.set_params(current_theta)
            return traj.reshape(num_samples, batch, steps, -1)
        if not hasattr(self.model, "predict"):
            raise AttributeError("StateVarianceMetric requires dynamics.sample_forward or model.predict")
        batch, _steps, _ = encoded_actions.shape
        original_state = None
        if hasattr(self.model, "_state"):
            original_state = self.model._state.detach().clone()
        current_theta = self.model.e["m"].detach().clone()
        state_samples = []
        try:
            for theta in theta_samples:
                theta_batch = theta.unsqueeze(0)
                if hasattr(self.model, "set_params"):
                    self.model.set_params(theta_batch)
                else:
                    self.model.dynamics.set_params(theta_batch)
                if original_state is not None:
                    state_seed = init_state if init_state.shape[0] == batch else init_state.expand(batch, -1, -1)
                    self.model._state = state_seed.detach().clone()
                traj = self.model.predict(encoded_actions)
                state_samples.append(traj)
        finally:
            if hasattr(self.model, "set_params"):
                self.model.set_params(current_theta)
            else:
                self.model.dynamics.set_params(current_theta)
            if original_state is not None:
                self.model._state = original_state
        return torch.stack(state_samples, dim=0)

    def compute_stepwise(self, rollout: dict) -> torch.Tensor:
        actions, encoded_action_value = self._rollout_actions(rollout)
        if actions is not None:
            actions = actions.to(self.device).float()
            if actions.ndim != 3:
                actions = actions.unsqueeze(0)
            batch, steps, _ = actions.shape
        else:
            if encoded_action_value is None:
                raise KeyError(
                    "StateVarianceMetric requires one of 'action', 'encoded_action', "
                    "'env_action', or 'model_action' in rollout"
                )
            encoded_actions = encoded_action_value.to(self.device).float()
            if encoded_actions.ndim != 3:
                encoded_actions = encoded_actions.unsqueeze(0)
            batch, steps, _ = encoded_actions.shape
        rollout_states_value = self._rollout_get(rollout, "model_state")
        if rollout_states_value is not None:
            rollout_states = rollout_states_value.to(self.device).float()
            if rollout_states.ndim != 3:
                rollout_states = rollout_states.unsqueeze(0)
            state0 = rollout_states[:, :1]
            if state0.shape[0] == 1 and batch > 1:
                state0 = state0.expand(batch, -1, -1).clone()
        else:
            if hasattr(self.model, "get_state"):
                state0 = self.model.get_state().to(self.device).float()
            else:
                state0 = self.model._state.to(self.device).float()
            if state0.ndim != 3:
                state0 = state0.unsqueeze(0)
            if state0.shape[0] == 1 and batch > 1:
                state0 = state0.expand(batch, -1, -1).clone()
        if encoded_action_value is not None:
            if "encoded_actions" not in locals():
                encoded_actions = encoded_action_value.to(self.device).float()
                if encoded_actions.ndim != 3:
                    encoded_actions = encoded_actions.unsqueeze(0)
        else:
            if actions is None:
                raise KeyError("StateVarianceMetric cannot encode actions when rollout['action'] is missing")
            encoded_actions = self._encode_actions(actions, state0)
        theta_samples = self._sample_theta_belief()
        state_stack = self._predict_state_samples(
            init_state=state0,
            encoded_actions=encoded_actions,
            theta_samples=theta_samples,
        )
        var_diag = torch.var(
            state_stack,
            dim=0,
            unbiased=self.num_parameter_samples > 1,
        )
        trace = var_diag.sum(dim=-1)
        if self.aggregation == "terminal":
            total = trace[:, -1]
        else:
            gamma_scale = torch.pow(
                torch.full((steps,), self.gamma, dtype=state_stack.dtype, device=self.device),
                torch.arange(steps, dtype=state_stack.dtype, device=self.device),
            ).view(1, steps)
            total = torch.sum(gamma_scale * trace, dim=-1)
        self.current_cost = (-total).unsqueeze(-1)
        return self.current_cost


def state_variance(
    *,
    model: FilteringEmbedding,
    Fe_net: Callable,
    Fz_net: Callable,
    gamma: float,
    device: str,
    num_parameter_samples: int,
    sample_seed: int | None = None,
    aggregation: str = "sum",
) -> StateVarianceMetric:
    del Fe_net, Fz_net
    return StateVarianceMetric(
        model=model,
        gamma=gamma,
        num_parameter_samples=num_parameter_samples,
        sample_seed=sample_seed,
        device=device,
        aggregation=aggregation,
    )
