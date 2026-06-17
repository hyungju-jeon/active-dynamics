from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from einops import rearrange, repeat, einsum
from torch.nn.functional import softplus

from actdyn.environment.action import BaseAction
from actdyn.utils.torch_utils import safe_cholesky, symmetrize, eps, Belief
from actdyn.utils.rollout import RolloutBuffer

from .base import BaseDynamicsEnsemble, BaseModel
from .decoder import Decoder, diagonal_observation_information
from .dynamics import BaseDynamics, FunctionDynamics
from .encoder import BaseEncoder


class SeqVae(BaseModel):
    """Sequential Variational Autoencoder (SeqVAE) with dynamics."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.beta = 0.0

    @staticmethod
    def _kl_div_mc(mu_q, var_q, z_prior, mu_p, var_p):
        """Monte Carlo KL"""
        if z_prior.dim() == 3:
            z_prior = z_prior.unsqueeze(0)

        target_ndim = z_prior.dim()

        def _unsqueeze_to(tensor: torch.Tensor, ndim: int) -> torch.Tensor:
            while tensor.dim() < ndim:
                tensor = tensor.unsqueeze(0)
            return tensor

        mu_q = _unsqueeze_to(mu_q, target_ndim)
        var_q = _unsqueeze_to(var_q, target_ndim)
        mu_p = _unsqueeze_to(mu_p, target_ndim)
        var_p = _unsqueeze_to(var_p, target_ndim)

        var_q = var_q.clamp_min(eps)
        var_p = var_p.clamp_min(eps)

        log_q = -0.5 * (torch.log(2 * torch.pi * var_q) + (z_prior - mu_q) ** 2 / var_q).sum(
            dim=(-2, -1)
        )
        log_p = -0.5 * (torch.log(2 * torch.pi * var_p) + (z_prior - mu_p) ** 2 / var_p).sum(
            dim=(-2, -1)
        )

        return (log_q - log_p).mean(dim=0)

    def _compute_multistep_kl(
        self,
        mu_q,  # (B,T,D) posterior mean
        var_q,  # (B,T,D) posterior variance
        z_samples,  # (S,B,T,D) posterior samples
        u=None,  # (B,T,A) action sequence
        idx=None,
        t_mask=None,  # (T,1) temporal mask
        k_steps=1,
        decay_rate=0.8,
        detach_posterior=False,
        mc_estimate=False,
    ):
        """Compute multi-step KL terms KL[q(z_{t+k}) || p_k(z_{t+k}|z_t, u_{t+1:t+k})]"""
        dynamics = self._get_dynamics(idx)
        if z_samples.dim() == 3:  # (B,T,D) -> (1,B,T,D)
            z_samples = z_samples.unsqueeze(0)
        S, B, T, D = z_samples.shape

        # Prepare action tensor
        if u is None:
            u = torch.zeros(B, T, getattr(self.action_encoder, "action_dim", 0), device=self.device)

        # Shift actions for time alignment
        if u.ndim == 3:
            u_s = repeat(u, "b t a -> s b t a", s=S)
        elif u.ndim == 4:
            u_s = u

        z_init = z_samples  # (S,B,D)
        # if detach_posterior:
        #     z_init = z_samples.detach()

        # KL weights
        if decay_rate is None:
            decay_rate = 1.0

        samples_list, mus_list, vars_list = dynamics.sample_forward(
            init_z=z_init, action=u_s, k_step=k_steps, return_traj=True  # (S,B,T,D)
        )

        kl_terms = []
        for k in range(1, k_steps + 1):
            if T - k <= 0:
                kl_terms.append(torch.zeros(B, device=self.device))
                continue

            # Posterior slice
            mu_q_target = mu_q[:, k:, :]
            var_q_target = var_q[:, k:, :]
            if detach_posterior and k > 1:
                mu_q_target = mu_q_target.detach()
                var_q_target = var_q_target.detach()

            if mc_estimate:
                # TODO : FIX this part
                # Prior samples for MC KL
                z_prior = samples_list[k][:, :, :-k, :]  # shape (S,B,T,D)
                mu_p = mus_list[k - 1][:, :, :-k, :]
                var_p = vars_list[k - 1]
                kl_mc = self._kl_div_mc(mu_q_target, var_q_target, z_prior, mu_p, var_p)  # (B,)
                kl_terms.append(kl_mc)
            else:
                # Analytic KL
                mu_p = mus_list[k - 1][..., :-k, :]
                var_p = vars_list[k - 1]

                mu_q_target_s = repeat(mu_q_target, "b t d -> s b t d", s=S)
                var_q_target_s = repeat(var_q_target, "b t d -> s b t d", s=S)
                kl_k = self._kl_div(mu_q_target_s, var_q_target_s, mu_p, var_p)  # (S,B,T)
                if t_mask is not None:
                    kl_k = kl_k * t_mask[..., k:, :].T  # (S,B,T)

                kl_terms.append(
                    kl_k.mean(0).sum(-1)
                )  # average over particles S and sum over T -> (B,)

        # Stack KL per horizon
        kl_per_k = torch.stack(kl_terms, dim=-1)  # (B,K)

        # Weighted sum over horizons
        kl_weights = torch.tensor([decay_rate**k for k in range(k_steps)], device=self.device)
        kl_weights = kl_weights / kl_weights.sum()
        kl_weighted = (kl_per_k * kl_weights).sum(-1)  # (B,)

        return kl_per_k, kl_weighted

    def compute_elbo(self, y, u=None, n_samples=5, k_steps=5, beta=1.0, p_mask=0.0, idx=None):
        """Compute ELBO with multi-step KL"""

        # Sample mesaurement posterior
        z_me, mu_q_x, var_q_x = self.encoder(y=y, u=u, n_samples=n_samples)
        if z_me.dim() == 3:  # (B,T,D) -> (S,B,T,D)
            z_me = z_me.unsqueeze(0)
        S, B, T, D = z_me.shape
        # z_me = rearrange(z_me, "s b t d -> (s b) t d")  # (S*B,T,D)

        if self.action_encoder is not None and u is not None:
            u_encoded = self.action_encoder(u[..., 1:, :], z_me[..., :-1, :])
            # Align a_t with y_{t+1}
        else:
            u_encoded = u

        # Apply temporal masking
        t_mask = torch.bernoulli((1 - p_mask) * torch.ones((T, 1), device=mu_q_x.device))

        z_tr = self.dynamics.sample_forward(init_z=z_me[..., :-1, :], action=u_encoded, k_step=1)[1]
        z_tr = torch.cat([z_me[..., :1, :], z_tr], dim=-2)  # (S,B),T,D)

        z_samples = t_mask * z_me + (1 - t_mask) * z_tr  # (S,B),T,D)

        # Multi-step KL: (B,K), (B)
        _, kl_w = self._compute_multistep_kl(
            mu_q_x, var_q_x, z_samples, u=u_encoded, idx=idx, k_steps=k_steps, t_mask=t_mask
        )

        # Log-likelihood per sample (B,T,D)
        S, B, T, D = z_samples.shape
        z_flat = rearrange(z_samples, "s b t d -> (s b) t d")
        y_rep = repeat(y, "b t d -> (s b) t d", s=S)
        log_like_flat = self.decoder.compute_log_prob(z_flat, y_rep)  # (S*B)
        log_like_sb = rearrange(log_like_flat, "(s b) -> s b", s=S, b=B)

        # Monte Carlo expectation over samples
        log_like_b = log_like_sb.mean(dim=0)  # (B)
        elbo_b = log_like_b - beta * kl_w
        elbo = elbo_b.mean()

        if idx is None or idx == 0:
            return -elbo, log_like_b.mean(), kl_w.mean()
        else:
            return beta * kl_w, torch.zeros(1), kl_w.mean()

    def _train_single_model(
        self,
        dataloader,
        optimizer,
        param_list,
        lr,
        weight_decay,
        n_epochs,
        verbose,
        grad_clip_norm,
        n_samples,
        k_steps,
        beta,
        p_mask=0.0,
        warmup=1000,
        annealing_steps=1000,
        annealing_type="cyclic",  # "linear", "cyclic", "none"
        model_idx=None,
        model_name="Model",
    ):
        """
        Train a single model (or ensemble member) with the given parameters.
        """
        opt = self._get_optimizer(optimizer, param_list, lr, weight_decay)
        T = 0

        # Initialize epoch progress bar
        if verbose:
            from tqdm import tqdm

            epoch_pbar = tqdm(range(n_epochs), desc=f"{model_name}")
            epoch_iterator = epoch_pbar
        else:
            epoch_iterator = range(n_epochs)

        # Train for multiple epochs with DataLoader
        epoch_info = []
        for i in epoch_iterator:
            batch_info = []
            for batch in dataloader:
                obs = batch["next_obs"].to(self.device)
                action = batch["action"].to(self.device) if "action" in batch else None
                T = obs.shape[-2]

                # Ensure input shape (batch, time, obs_dim)
                while obs.dim() > 3 and obs.shape[0] == 1:
                    obs = obs.squeeze(0)  # Remove extra batch dimensions
                if action is not None:
                    while action is not None and action.dim() > 3 and action.shape[0] == 1:
                        action = action.squeeze(0)
                if obs.dim() != 3:
                    raise ValueError(
                        f"Expected 3D observation tensor (batch, time, obs_dim), got shape {obs.shape}"
                    )

                # Zero gradients, compute loss, backprop, and step optimizer
                opt.zero_grad()

                self.beta = beta
                self.beta_schedule(beta, warmup, annealing_steps, annealing_type)

                if self.step_count < warmup:
                    self.beta = 0.0
                    self.p_mask = 0

                loss, log_like, kl_d = self.compute_elbo(
                    obs,
                    u=action,
                    idx=model_idx,
                    n_samples=n_samples,
                    beta=self.beta,
                    k_steps=k_steps,
                    p_mask=p_mask,
                )
                loss.backward()

                # Apply gradient clipping over full parameter list once
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(param_list, grad_clip_norm)
                info = {
                    "ELBO": loss.detach(),
                    "log_L": log_like.detach().detach(),
                    "KL": kl_d.detach(),
                }
                # Update parameters
                opt.step()
                batch_info.append(info)

                # Explicit cleanup for gradient tensors
                del batch, obs, loss, log_like, kl_d

                if action is not None:
                    del action
            if model_idx is not None:
                self.step_count += 1 if model_idx == 0 else 0
            else:
                self.step_count += 1

            # Convert list of dict to dict of tensor
            batch_info = {
                key: torch.tensor([b[key] for b in batch_info]).mean(dim=0) for key in batch_info[0]
            }
            epoch_info.append(batch_info)

            # Convert list to tensor and average across batch

            # Update epoch progress bar with average ELBO
            if verbose and epoch_info and i % 10 == 0:
                current_info = epoch_info[-1]
                epoch_pbar.set_postfix({k: f"{v:.4f}" for k, v in current_info.items()})
                epoch_pbar.update(10)

        # Close progress bar
        if verbose:
            epoch_pbar.close()

        epoch_info = {key: torch.tensor([e[key] for e in epoch_info]) for key in epoch_info[0]}

        return epoch_info

    def update_posterior_embedding(self, y, u=None):
        """Update the posterior state given new observation and action."""
        with torch.no_grad():
            _, z_post, _ = self.encoder(y=y, u=u, n_samples=1)
        return z_post[:, -1, :].unsqueeze(1)


class SeqStateVae(BaseModel):
    """Sequential Variational Autoencoder (SeqVAE) with dynamics."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.beta = 0.0

    @staticmethod
    def _kl_div_mc(mu_q, var_q, z_prior, mu_p, var_p):
        """Monte Carlo KL"""
        if z_prior.dim() == 3:
            z_prior = z_prior.unsqueeze(0)

        target_ndim = z_prior.dim()

        def _unsqueeze_to(tensor: torch.Tensor, ndim: int) -> torch.Tensor:
            while tensor.dim() < ndim:
                tensor = tensor.unsqueeze(0)
            return tensor

        mu_q = _unsqueeze_to(mu_q, target_ndim)
        var_q = _unsqueeze_to(var_q, target_ndim)
        mu_p = _unsqueeze_to(mu_p, target_ndim)
        var_p = _unsqueeze_to(var_p, target_ndim)

        var_q = var_q.clamp_min(eps)
        var_p = var_p.clamp_min(eps)

        log_q = -0.5 * (torch.log(2 * torch.pi * var_q) + (z_prior - mu_q) ** 2 / var_q).sum(
            dim=(-2, -1)
        )
        log_p = -0.5 * (torch.log(2 * torch.pi * var_p) + (z_prior - mu_p) ** 2 / var_p).sum(
            dim=(-2, -1)
        )

        return (log_q - log_p).mean(dim=0)

    def _compute_multistep_kl(
        self,
        mu_q,  # (B,T,D) posterior mean
        var_q,  # (B,T,D) posterior variance
        z_samples,  # (S,B,T,D) posterior samples
        u=None,  # (B,T,A) action sequence
        idx=None,
        t_mask=None,  # (T,1) temporal mask
        k_steps=1,
        decay_rate=0.8,
        detach_posterior=False,
        mc_estimate=False,
    ):
        """Compute multi-step KL terms KL[q(z_{t+k}) || p_k(z_{t+k}|z_t, u_{t+1:t+k})]"""
        dynamics = self._get_dynamics(idx)
        if z_samples.dim() == 3:  # (B,T,D) -> (1,B,T,D)
            z_samples = z_samples.unsqueeze(0)
        S, B, T, D = z_samples.shape

        # Prepare action tensor
        if u is None:
            u = torch.zeros(B, T, getattr(self.action_encoder, "action_dim", 0), device=self.device)

        # Shift actions for time alignment
        if u.ndim == 3:
            u_s = repeat(u, "b t a -> s b t a", s=S)
        elif u.ndim == 4:
            u_s = u

        z_init = z_samples  # (S,B,D)
        # if detach_posterior:
        #     z_init = z_samples.detach()

        # KL weights
        if decay_rate is None:
            decay_rate = 1.0

        samples_list, mus_list, vars_list = dynamics.sample_forward(
            init_z=z_init, action=u_s, k_step=k_steps, return_traj=True  # (S,B,T,D)
        )

        kl_terms = []
        for k in range(1, k_steps + 1):
            if T - k <= 0:
                kl_terms.append(torch.zeros(B, device=self.device))
                continue

            # Posterior slice
            mu_q_target = mu_q[:, k:, :]
            var_q_target = var_q[:, k:, :]
            if detach_posterior and k > 1:
                mu_q_target = mu_q_target.detach()
                var_q_target = var_q_target.detach()

            if mc_estimate:
                # TODO : FIX this part
                # Prior samples for MC KL
                z_prior = samples_list[k][:, :, :-k, :]  # shape (S,B,T,D)
                mu_p = mus_list[k - 1][:, :, :-k, :]
                var_p = vars_list[k - 1]
                kl_mc = self._kl_div_mc(mu_q_target, var_q_target, z_prior, mu_p, var_p)  # (B,)
                kl_terms.append(kl_mc)
            else:
                # Analytic KL
                mu_p = mus_list[k - 1]
                var_p = vars_list[k - 1]

                mu_q_target_s = repeat(mu_q_target, "b t d -> s b t d", s=S)
                var_q_target_s = repeat(var_q_target, "b t d -> s b t d", s=S)
                kl_k = self._kl_div(mu_q_target_s, var_q_target_s, mu_p, var_p)  # (S,B,T)
                if t_mask is not None:
                    kl_k = kl_k * t_mask[..., k:, :].T  # (S,B,T)

                kl_terms.append(
                    kl_k.mean(0).sum(-1)
                )  # average over particles S and sum over T -> (B,)

        # Stack KL per horizon
        kl_per_k = torch.stack(kl_terms, dim=-1)  # (B,K)

        # Weighted sum over horizons
        kl_weights = torch.tensor([decay_rate**k for k in range(k_steps)], device=self.device)
        kl_weights = kl_weights / kl_weights.sum()
        kl_weighted = (kl_per_k * kl_weights).sum(-1)  # (B,)

        return kl_per_k, kl_weighted

    def compute_elbo(self, y, z, u=None, n_samples=5, k_steps=5, beta=1.0, p_mask=0.0, idx=None):
        """Compute ELBO with multi-step KL"""

        # Sample mesaurement posterior
        z_me, mu_q_x, var_q_x = self.encoder(y=y, u=u, n_samples=n_samples)
        if z_me.dim() == 3:  # (B,T,D) -> (S,B,T,D)
            z_me = z_me.unsqueeze(0)
        S, B, T, D = z_me.shape
        # z_me = rearrange(z_me, "s b t d -> (s b) t d")  # (S*B,T,D)

        if self.action_encoder is not None and u is not None:
            u_encoded = self.action_encoder(u[..., 1:, :], z_me[..., :-1, :])
            # Align a_t with y_{t+1}
        else:
            u_encoded = u

        # Apply temporal masking
        t_mask = torch.bernoulli((1 - p_mask) * torch.ones((T, 1), device=mu_q_x.device))

        z_tr = self.dynamics.sample_forward(init_z=z_me[..., :-1, :], action=u_encoded, k_step=1)[1]
        z_tr = torch.cat([z_me[..., :1, :], z_tr], dim=-2)  # (S,B),T,D)

        z_samples = t_mask * z_me + (1 - t_mask) * z_tr  # (S,B),T,D)

        # Multi-step KL: (B,K), (B)
        _, kl_w = self._compute_multistep_kl(
            mu_q_x, var_q_x, z_samples, u=u_encoded, idx=idx, k_steps=k_steps, t_mask=t_mask
        )

        # Log-likelihood per sample (B,T,D)
        S, B, T, D = z_samples.shape
        z_flat = rearrange(z_samples, "s b t d -> (s b) t d")
        y_rep = repeat(y, "b t d -> (s b) t d", s=S)
        log_like_flat = self.decoder.compute_log_prob(z_flat, y_rep)  # (S*B)
        log_like_sb = rearrange(log_like_flat, "(s b) -> s b", s=S, b=B)

        # Monte Carlo expectation over samples
        log_like_b = log_like_sb.mean(dim=0)  # (B)
        elbo_b = log_like_b - beta * kl_w
        elbo = elbo_b.mean()

        if idx is None or idx == 0:
            return -elbo, log_like_b.mean(), kl_w.mean()
        else:
            return beta * kl_w, torch.zeros(1), kl_w.mean()

    def _train_single_model(
        self,
        dataloader,
        optimizer,
        param_list,
        lr,
        weight_decay,
        n_epochs,
        verbose,
        grad_clip_norm,
        n_samples,
        k_steps,
        beta,
        p_mask=0.0,
        warmup=1000,
        annealing_steps=1000,
        annealing_type="cyclic",  # "linear", "cyclic", "none"
        model_idx=None,
        model_name="Model",
    ):
        """
        Train a single model (or ensemble member) with the given parameters.
        """
        opt = self._get_optimizer(optimizer, param_list, lr, weight_decay)
        T = 0

        # Initialize epoch progress bar
        if verbose:
            from tqdm import tqdm

            epoch_pbar = tqdm(range(n_epochs), desc=f"{model_name}")
            epoch_iterator = epoch_pbar
        else:
            epoch_iterator = range(n_epochs)

        # Train for multiple epochs with DataLoader
        epoch_info = []
        for i in epoch_iterator:
            batch_info = []
            for batch in dataloader:
                obs = batch["next_obs"].to(self.device)
                action = batch["action"].to(self.device) if "action" in batch else None
                T = obs.shape[-2]

                # Ensure input shape (batch, time, obs_dim)
                while obs.dim() > 3 and obs.shape[0] == 1:
                    obs = obs.squeeze(0)  # Remove extra batch dimensions
                if action is not None:
                    while action is not None and action.dim() > 3 and action.shape[0] == 1:
                        action = action.squeeze(0)
                if obs.dim() != 3:
                    raise ValueError(
                        f"Expected 3D observation tensor (batch, time, obs_dim), got shape {obs.shape}"
                    )

                # Zero gradients, compute loss, backprop, and step optimizer
                opt.zero_grad()

                self.beta = beta
                self.beta_schedule(beta, warmup, annealing_steps, annealing_type)

                if self.step_count < warmup:
                    self.beta = 0.0
                    self.p_mask = 0

                loss, log_like, kl_d = self.compute_elbo(
                    obs,
                    z=batch["next_env_state"].to(self.device).detach(),
                    u=action,
                    idx=model_idx,
                    n_samples=n_samples,
                    beta=self.beta,
                    k_steps=k_steps,
                    p_mask=self.p_mask,
                )
                loss.backward()

                # Apply gradient clipping over each parameter list once
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    for param in param_list:
                        torch.nn.utils.clip_grad_norm_(param, grad_clip_norm)

                info = {
                    "ELBO": loss.detach(),
                    "log_L": log_like.detach().detach(),
                    "KL": kl_d.detach(),
                }
                # Update parameters
                opt.step()
                batch_info.append(info)

                # Explicit cleanup for gradient tensors
                del batch, obs, loss, log_like, kl_d

                if action is not None:
                    del action
            if model_idx is not None:
                self.step_count += 1 if model_idx == 0 else 0
            else:
                self.step_count += 1

            # Convert list of dict to dict of tensor
            batch_info = {
                key: torch.tensor([b[key] for b in batch_info]).mean(dim=0) for key in batch_info[0]
            }
            epoch_info.append(batch_info)

            # Convert list to tensor and average across batch

            # Update epoch progress bar with average ELBO
            if verbose and epoch_info and i % 10 == 0:
                current_info = epoch_info[-1]
                epoch_pbar.set_postfix({k: f"{v:.4f}" for k, v in current_info.items()})
                epoch_pbar.update(10)

        # Close progress bar
        if verbose:
            epoch_pbar.close()

        epoch_info = {
            key: torch.tensor([e[key] for e in epoch_info]).mean(dim=0).item()
            for key in epoch_info[0]
        }

        return epoch_info

    def update_posterior_embedding(self, y, u=None, update_theta: bool = True, **kwargs):
        """Update the posterior state given new observation and action."""
        # with torch.no_grad():
        _, z_post, _ = self.encoder(y=y, u=u, n_samples=1)
        return z_post[:, -1, :].unsqueeze(1)


class DeepVariationalBayesFilter(SeqVae):
    """Deep Variational Bayes Filter (DVBF) model."""

    def __init__(
        self,
        dynamics: BaseDynamics,
        encoder: BaseEncoder,
        decoder: Decoder,
        action_encoder: Optional[BaseAction] = None,
        device: str = "cpu",
    ):
        super().__init__(
            dynamics=dynamics,
            encoder=encoder,
            decoder=decoder,
            action_encoder=action_encoder,
            device=device,
        )


class FilteringEmbedding(BaseModel):
    """Filtering embedding model."""

    def __init__(
        self,
        e: Belief,
        Fe: Callable = None,
        Fz: Callable = None,
        q_theta: float = 1e-4,
        k_theta: int = 10,
        e_clip: float = 5.0,
        state_init_uncertainty: float = 1.0,
        q_theta_meas_coeff: float = 0.0,
        q_theta_max_scale: float = 10.0,
        adaptive_update: bool = False,
        adaptive_update_min_interval: int = 1,
        adaptive_update_eig_threshold: float | None = None,
        shrinkage_map: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        shrinkage_min: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beta = 0.0
        self.e: Belief = e
        self.e_clip = max(float(e_clip), 1e-3)
        self._normalize_embedding_belief()
        self.state_init_uncertainty = max(float(state_init_uncertainty), 1e-9)
        initial_batch = self.e["m"].shape[0]
        self.z: Belief = {
            "m": torch.zeros(1, 1, self.latent_dim, device=self.device),
            "P": self._initial_state_covariance(batch_size=initial_batch),
        }
        self.Fe = Fe
        self.Fz = Fz
        self._state = torch.zeros(1, 1, self.latent_dim, device=self.device)
        self.q_theta = float(q_theta)
        self.q_theta_meas_coeff = max(float(q_theta_meas_coeff), 0.0)
        self.q_theta_max_scale = max(float(q_theta_max_scale), 1.0)
        self.k_theta = max(1, int(k_theta))
        self.adaptive_update = bool(adaptive_update)
        self.adaptive_update_min_interval = max(1, int(adaptive_update_min_interval))
        self.adaptive_update_eig_threshold = (
            None
            if adaptive_update_eig_threshold is None
            else float(adaptive_update_eig_threshold)
        )
        self.shrinkage_map = shrinkage_map
        self.shrinkage_min = float(shrinkage_min)
        self.gn_iter = 10
        self._last_innovation_statistic = None
        self._last_parameter_shrinkage = None
        self._last_parameter_update_reason = "none"
        self._reset_embedding_block_state(batch_size=self.e["m"].shape[0])
        self.last_information = {
            "I_z_t": 0.0,
            "I_theta_t": 0.0,
            "theta_block_eig": 0.0,
            "theta_block_steps": 0,
            "parameter_update_reason": "none",
            "Pz00": 0.0,
            "Pz01": 0.0,
            "Pz11": 0.0,
        }
        self.set_params(self.e["m"])

    def _normalize_embedding_belief(self) -> None:
        """Normalize embedding belief tensors to batched shapes on the model device."""
        e_m = self.e["m"].to(self.device)
        if e_m.dim() == 1:
            e_m = e_m.unsqueeze(0)
        e_m = torch.nan_to_num(e_m, nan=0.0, posinf=self.e_clip, neginf=-self.e_clip).clamp(
            -self.e_clip, self.e_clip
        )
        batch = e_m.shape[0]
        d_embed = e_m.shape[-1]
        eye = torch.eye(d_embed, device=self.device).unsqueeze(0)

        P = self.e.get("P", eye.clone())
        if P.dim() == 2:
            P = P.unsqueeze(0)
        P = P.to(self.device)
        if P.shape[0] == 1 and batch > 1:
            P = P.expand(batch, -1, -1).clone()
        if P.shape[0] != batch:
            raise ValueError(
                f"Embedding covariance batch mismatch: {P.shape} vs mean batch {batch}"
            )
        P = symmetrize(P)

        L = self.e.get("L")
        if L is None:
            chol_P = safe_cholesky(P)
            L = torch.cholesky_inverse(chol_P)
        if L.dim() == 2:
            L = L.unsqueeze(0)
        L = L.to(self.device)
        if L.shape[0] == 1 and batch > 1:
            L = L.expand(batch, -1, -1).clone()
        if L.shape[0] != batch:
            raise ValueError(f"Embedding precision batch mismatch: {L.shape} vs mean batch {batch}")
        L = symmetrize(L)

        self.e = {"m": e_m, "P": P, "L": L}

    def _ensure_state_belief_shapes(self, batch_size: int) -> None:
        """Normalize latent-state belief tensors to (B, 1, ...) shapes."""
        z_m = self.z["m"]
        if z_m.dim() == 2:
            z_m = z_m.unsqueeze(1)
        z_m = z_m.to(self.device)
        if z_m.shape[0] == 1 and batch_size > 1:
            z_m = z_m.expand(batch_size, -1, -1).clone()

        z_P = self.z["P"]
        if z_P.dim() == 2:
            z_P = z_P.unsqueeze(0).unsqueeze(0)
        elif z_P.dim() == 3:
            z_P = z_P.unsqueeze(1)
        z_P = z_P.to(self.device)
        if z_P.shape[0] == 1 and batch_size > 1:
            z_P = z_P.expand(batch_size, -1, -1, -1).clone()
        z_P = symmetrize(z_P)
        self.z = {"m": z_m, "P": z_P}

    def _reset_embedding_block_state(self, batch_size: int) -> None:
        d_embed = self.e["m"].shape[-1]
        self._theta_block_steps = 0
        self._theta_score_block = torch.zeros(batch_size, d_embed, device=self.device)
        self._theta_info_block = torch.zeros(batch_size, d_embed, d_embed, device=self.device)
        self._theta_sensitivity = torch.zeros(
            batch_size, self.latent_dim, d_embed, device=self.device
        )

    def _initial_state_covariance(self, batch_size: int) -> torch.Tensor:
        return (
            (
                self.state_init_uncertainty
                * torch.eye(self.latent_dim, device=self.device).unsqueeze(0).unsqueeze(0)
            )
            .expand(batch_size, -1, -1, -1)
            .clone()
        )

    def _compute_innovation_statistic(
        self,
        residual: torch.Tensor,
        chol_covariance: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized innovation energy used to moderate parameter updates."""
        residual_col = residual.unsqueeze(-1)
        inv_cov_residual = torch.cholesky_solve(residual_col, chol_covariance)
        innovation_quad = residual_col.transpose(-1, -2) @ inv_cov_residual
        innovation_quad = innovation_quad.squeeze(-1).squeeze(-1).squeeze(-1)
        obs_dim = residual.shape[-1]
        return innovation_quad / float(obs_dim)

    def _compute_parameter_shrinkage(self, innovation_statistic: torch.Tensor) -> torch.Tensor:
        """Map a mismatch statistic to a scalar shrinkage factor in [shrinkage_min, 1]."""
        if self.shrinkage_map is None:
            return torch.ones_like(innovation_statistic)

        tau = self.shrinkage_map(innovation_statistic)
        if not torch.is_tensor(tau):
            tau = torch.as_tensor(
                tau, device=innovation_statistic.device, dtype=innovation_statistic.dtype
            )
        tau = tau.to(device=innovation_statistic.device, dtype=innovation_statistic.dtype)
        if tau.ndim == 0:
            tau = tau.expand_as(innovation_statistic)
        elif tau.shape != innovation_statistic.shape:
            if tau.numel() == innovation_statistic.numel():
                tau = tau.reshape_as(innovation_statistic)
            else:
                tau = tau.expand_as(innovation_statistic)

        tau = torch.nan_to_num(tau, nan=1.0, posinf=1.0, neginf=self.shrinkage_min)
        return tau.clamp(min=self.shrinkage_min, max=1.0)

    def _apply_embedding_block_update(self) -> None:
        """Apply block-wise information-form update with drifting prior."""
        score = self._theta_score_block
        info = self._theta_info_block
        if score.shape[0] != self.e["m"].shape[0]:
            # Shared parameter belief across batch: aggregate per-step statistics.
            score = score.mean(dim=0, keepdim=True)
            info = info.mean(dim=0, keepdim=True)
        score = torch.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
        info = torch.nan_to_num(info, nan=0.0, posinf=1e6, neginf=-1e6)
        self._last_theta_score_block_applied = score.detach().clone()
        self._last_theta_info_block_applied = info.detach().clone()
        self._last_theta_block_steps_applied = int(self._theta_block_steps)

        d_embed = self.e["m"].shape[-1]
        eye = torch.eye(d_embed, device=self.device).unsqueeze(0)

        # No measurement-error-dependent scaling: process drift uses fixed q_theta.
        q_theta_eff = torch.full((self.e["m"].shape[0],), float(self.q_theta), device=self.device)

        P_prior = self._project_spd(self.e["P"] + q_theta_eff.view(-1, 1, 1) * eye)
        try:
            chol_P_prior = safe_cholesky(P_prior)
        except Exception:
            P_prior = self._project_spd(P_prior + 1e-6 * eye, min_eig=1e-6)
            chol_P_prior = safe_cholesky(P_prior)
        L_prior = torch.cholesky_inverse(chol_P_prior)

        L_new = self._project_spd(L_prior + info)
        try:
            chol_L_new = safe_cholesky(L_new)
        except Exception:
            L_new = self._project_spd(L_new + 1e-4 * eye, min_eig=1e-4)
            chol_L_new = safe_cholesky(L_new)
        P_new = torch.cholesky_inverse(chol_L_new)
        m_new = self.e["m"] + (P_new @ score.unsqueeze(-1)).squeeze(-1)
        m_new = torch.nan_to_num(m_new, nan=0.0, posinf=self.e_clip, neginf=-self.e_clip).clamp(
            -self.e_clip, self.e_clip
        )

        self.e = {"m": m_new.detach(), "P": P_new.detach(), "L": L_new.detach()}
        self.set_params(self.e["m"].detach())
        self._reset_embedding_block_state(batch_size=self.e["m"].shape[0])

    def _theta_block_eig(self) -> torch.Tensor:
        """Return posterior-scaled EIG for the currently accumulated parameter block."""
        d_embed = self.e["m"].shape[-1]
        if d_embed <= 0 or self._theta_block_steps <= 0:
            return torch.tensor(0.0, device=self.device)
        info = torch.nan_to_num(
            self._theta_info_block, nan=0.0, posinf=1e6, neginf=-1e6
        )
        if info.shape[0] != self.e["P"].shape[0]:
            info = info.mean(dim=0, keepdim=True)
        P = self._project_spd(self.e["P"])
        chol_P = safe_cholesky(P)
        eye = torch.eye(d_embed, device=self.device).unsqueeze(0).expand(P.shape[0], -1, -1)
        scaled_info = self._project_spd(
            eye + chol_P.transpose(-1, -2) @ info @ chol_P,
            min_eig=1e-9,
        )
        chol_scaled = safe_cholesky(scaled_info)
        logdet = 2.0 * torch.log(
            torch.diagonal(chol_scaled, dim1=-2, dim2=-1).clamp_min(eps)
        ).sum(dim=-1)
        eig = 0.5 * logdet / float(d_embed)
        return torch.nan_to_num(eig.mean(), nan=0.0, posinf=1e6, neginf=0.0)

    def _embedding_block_update_reason(self) -> str | None:
        """Return why the current parameter block should be applied, if at all."""
        steps = int(self._theta_block_steps)
        if steps >= self.k_theta:
            return "max_interval"
        if not self.adaptive_update:
            return None
        if steps < min(self.adaptive_update_min_interval, self.k_theta):
            return None
        if self.adaptive_update_eig_threshold is None:
            return None
        if float(self._theta_block_eig().item()) >= float(self.adaptive_update_eig_threshold):
            return "block_eig"
        return None

    @staticmethod
    def _project_spd(M: torch.Tensor, min_eig: float = 1e-6) -> torch.Tensor:
        M = symmetrize(torch.nan_to_num(M.float(), nan=0.0, posinf=1e6, neginf=-1e6))
        floor = max(float(min_eig), 1e-8)
        eye = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype).expand_as(M)
        work = M
        for _ in range(4):
            try:
                eigvals, eigvecs = torch.linalg.eigh((work + floor * eye).double())
                eigvals = eigvals.clamp_min(floor)
                proj = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
                proj = torch.nan_to_num(proj, nan=0.0, posinf=1e6, neginf=-1e6).to(M.dtype)
                return symmetrize(proj)
            except RuntimeError:
                work = symmetrize(work + floor * eye)
                floor *= 10.0

        diag = torch.diagonal(M, dim1=-2, dim2=-1)
        diag = torch.nan_to_num(diag, nan=floor, posinf=1e6, neginf=floor).clamp_min(floor)
        return torch.diag_embed(diag)

    def set_params(self, e: torch.Tensor):
        self.e["m"] = torch.nan_to_num(
            e.to(self.device), nan=0.0, posinf=self.e_clip, neginf=-self.e_clip
        ).clamp(-self.e_clip, self.e_clip)
        self.dynamics.set_params(self.e["m"])

    def reset(self, observation: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment to initial state."""
        observation, info = super().reset(observation)
        d_embed = self.e["m"].shape[-1]
        batch = self.e["m"].shape[0]
        eye_embed = (
            torch.eye(d_embed, device=self.device).unsqueeze(0).expand(batch, -1, -1).clone()
        )
        self.e.update(
            {
                "P": eye_embed,
                "L": eye_embed.clone(),
            }
        )
        self._normalize_embedding_belief()
        self.z = {
            "m": self._state,
            "P": self._initial_state_covariance(batch_size=batch),
        }
        self._ensure_state_belief_shapes(batch_size=batch)
        self.set_params(self.e["m"])
        self._reset_embedding_block_state(batch_size=self.e["m"].shape[0])
        self._last_theta_score_block_applied = torch.zeros_like(self._theta_score_block)
        self._last_theta_info_block_applied = torch.zeros_like(self._theta_info_block)
        self._last_theta_block_steps_applied = 0
        self._last_parameter_update_reason = "none"
        self.last_information = {
            "I_z_t": 0.0,
            "I_theta_t": 0.0,
            "theta_block_eig": 0.0,
            "theta_block_steps": 0,
            "parameter_update_reason": "none",
            "Pz00": 0.0,
            "Pz01": 0.0,
            "Pz11": 0.0,
        }

        return observation, info

    def set_state(self, state: torch.Tensor):
        self.z["m"] = state
        super().set_state(state)

    @property
    def embedding(self):
        return self.e["m"]

    @torch.no_grad()
    def predict_state(self, u=None):
        Q = softplus(self.dynamics.logvar).diag_embed().unsqueeze(0) * self.dt
        I = torch.eye(self.latent_dim, device=self.device).unsqueeze(0).unsqueeze(0)

        # Transition linearization at current posterior mean
        Fz = self.Fz(self.z["m"], self.e["m"])
        dfdz = Fz * self.dt + I

        if u is not None and self.action_encoder is not None:
            u_enc = self.action_encoder(u, self.z["m"])
        else:
            u_enc = u

        # Predict
        z_pred = {
            "m": self.predict(action=u_enc),
            "P": dfdz @ self.z["P"] @ dfdz.transpose(-1, -2) + Q,
        }

        model_info = {"env_action": u_enc, "latent_state": z_pred["m"]}
        # self.z = {"m": z_pred["m"].detach(), "P": z_pred["P"].detach()}
        # self._state = z_pred["m"].detach()

        return model_info

    @torch.no_grad()
    def update_information(self, replay: RolloutBuffer):
        de = self.embedding.shape[-1]
        B, T, dz = replay["model_state"].shape

        Q = softplus(self.dynamics.logvar).diag_embed().unsqueeze(0) * self.dt
        I = torch.eye(self.latent_dim, device=self.device).unsqueeze(0).unsqueeze(0)

        # We'll accumulate curvature G (De,De) and gradient g (De,1)
        curv_total = torch.zeros(B, de, de, device=self.device)
        grad_total = torch.zeros(B, de, 1, device=self.device)
        dzde = torch.zeros(B, 1, dz, de, device=self.device)
        P = self.z["P"]

        for t in range(T):
            pred_m = torch.cat(
                self.dynamics.sample_forward(
                    init_z=replay["model_state"][:, t : t + 1],
                    action=replay["model_action"][:, t : t + 1],
                    return_traj=True,
                    add_noise=False,
                )[1]
            )
            # 1. Propagate dzde forward using dynamics sensitivity
            Fz = self.Fz(replay["model_state"][:, t : t + 1], self.e["m"])
            dfdz = Fz * self.dt + I
            Fe = self.Fe(replay["model_state"][:, t : t + 1], self.e["m"])
            dfde = Fe * self.dt
            P = dfdz @ P @ dfdz.transpose(-1, -2) + Q  # (B, 1, Dz, Dz)

            dzde = einsum(dfdz, dzde, "b t z z, b t z e -> b t z e") + dfde  # (B, Dz, De)

            # 2. Decoder linearization at that latent
            H = self.decoder.jacobian(pred_m)  # (B, 1, Dy, Dz)
            R = self.decoder.var(pred_m).diag_embed()  # (B, 1, Dy, Dy)
            R = torch.exp(self.decoder.logvar).diag_embed().unsqueeze(0)  # (B, 1, Dy, Dy)
            y_pred = self.decoder(pred_m)  # (B, 1, Dy)

            # 3. Innovation covariance in observation space
            S = H @ P @ H.transpose(-1, -2) + R + torch.ones_like(R) * 1e-3  # (B, Dy, Dy)
            chol_S = safe_cholesky(symmetrize(S))  # (B, Dy, Dy)
            # 4. Residual in obs space
            y = replay["next_obs"][:, t : t + 1]  # (B, 1, Dy)
            r = (y - y_pred).unsqueeze(-1)  # (B, 1, Dy, 1)
            invS_r = torch.cholesky_solve(r, chol_S)  # (B, 1, Dy, 1)
            # 5. Map embedding -> observation via latent:
            J = H @ dzde  # (B, 1, Dy, De)

            # 6. Accumulate curvature and gradient
            X = torch.cholesky_solve(J, chol_S)

            grad_total += einsum(J, invS_r, "b t y d, b t y k->b t d k").sum(dim=0)
            curv_total += einsum(J, X, "b t y d, b t y e->b t d e").sum(dim=0)

        # ---- 7. Damped Gauss–Newton solve for Δe
        # Add damping λ I
        # lamI = damping * torch.eye(De, device=device).unsqueeze(0)  # (1, De, De)
        # G_damped = curv_total + lamI  # (B, 1, De, De)
        # curv_total = torch.sum(curv_total, dim=0, keepdim=True)  # (1, De, De)
        # grad_total = torch.sum(grad_total, dim=0, keepdim=True)  # (1, De, 1)

        curv_norm = curv_total / torch.norm(curv_total, dim=(1, 2), keepdim=True)
        L_new = self.e["L"] + curv_norm
        step_dir = torch.linalg.solve(L_new, grad_total)

        chol_L_new = safe_cholesky(L_new)
        Sigma_e = torch.cholesky_inverse(chol_L_new)  # (1, De, De)

        eta = (self.e["L"] @ self.e["m"].unsqueeze(-1)).squeeze(-1)  # [1, De]
        eta_new = eta + grad_total.squeeze(-1)  # (1, De)
        mu_e = (Sigma_e @ eta_new.unsqueeze(-1)).squeeze(-1)
        step_norm = step_dir.norm()

        if step_norm > 1e-1:
            step_dir = step_dir * (1e-1 / (step_norm + 1e-12))

        mu_e = self.e["m"] + step_dir.squeeze(-1)

        self.e = {"m": mu_e.detach(), "P": Sigma_e.detach(), "L": L_new.detach()}

    @torch.no_grad()
    def update_prediction(self, y, u=None):
        """Update the posterior state given new observation and action."""

        y = y[:, -1:, :]
        u = u[:, -1:, :] if u is not None else None
        Q = softplus(self.dynamics.logvar).diag_embed().unsqueeze(0) * self.dt
        I = torch.eye(self.latent_dim, device=self.device).unsqueeze(0).unsqueeze(0)

        if u is not None and self.action_encoder is not None:
            u_enc = self.action_encoder(u, self.z["m"])
        else:
            u_enc = u

        # Final EKF update for latent state
        # Re-propagate dynamics with updated e
        Fz = self.Fz(self.z["m"], self.e["m"])
        dfdz = Fz * self.dt + I

        z_pred = {
            "m": self.predict(action=u_enc),
            "P": dfdz @ self.z["P"] @ dfdz.transpose(-1, -2) + Q,
        }

        # Re-linearize observation and variance at new z_pred
        dhdz = self.decoder.jacobian(z_pred["m"])
        R = self.decoder.var(z_pred["m"]).diag_embed()

        # Predict observation and compute residual
        y_pred = self.decoder(z_pred["m"])
        r = y - y_pred

        S = symmetrize(dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R)
        chol_S = safe_cholesky(S)

        # Gt = self.Fe(self.z["m"], self.e["m"]) * self.dt
        # HzGt = dhdz @ Gt
        # # GN curvature
        # X = torch.cholesky_solve(HzGt, chol_S)
        # curv_ll = einsum(HzGt, X, "b t y d, b t y e->b t d e")
        # curv_ll = symmetrize(curv_ll)  # ensure symmetry)
        # self.update_embedding(r, chol_S, HzGt, curv_ll)

        # Fz = self.Fz(self.z["m"], self.e["m"])
        # dfdz = Fz * self.dt + I

        z_pred = {
            "m": self.predict(action=u_enc),
            "P": dfdz @ self.z["P"] @ dfdz.transpose(-1, -2) + Q,
        }

        # Re-linearize observation and variance at new z_pred
        dhdz = self.decoder.jacobian(z_pred["m"])
        R = self.decoder.var(z_pred["m"]).diag_embed()
        S = symmetrize(dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2) + R)
        chol_S = safe_cholesky(S)

        # Compute Kalman Gain and update posterior with observation y_t
        HPt = dhdz @ z_pred["P"]
        K = torch.cholesky_solve(HPt, chol_S).transpose(-1, -2)
        KH = K @ dhdz
        P_upd = (I - KH) @ z_pred["P"] @ (I - KH).transpose(-1, -2) + K @ R @ K.transpose(-1, -2)

        # innovation uses current y_pred; recompute for consistency
        y_pred = self.decoder(z_pred["m"])
        r = y - y_pred

        z_post = {
            "m": z_pred["m"] + (K @ r.unsqueeze(-1)).squeeze(-1),
            "P": symmetrize(P_upd),
        }

        self.z = {"m": z_post["m"].detach(), "P": z_post["P"].detach()}
        self._state = z_post["m"].detach()

        return self._state

    @torch.no_grad()
    def update_posterior_embedding(self, y, u=None, update_theta: bool = True, **kwargs):
        """Update the posterior state given new observation and action."""

        self._last_parameter_update_reason = "none"
        self._normalize_embedding_belief()
        self._ensure_state_belief_shapes(batch_size=self.e["m"].shape[0])
        y = y[:, -1:, :]
        u = u[:, -1:, :] if u is not None else None
        Q = softplus(self.dynamics.logvar).diag_embed().unsqueeze(0) * self.dt
        I = torch.eye(self.latent_dim, device=self.device).unsqueeze(0).unsqueeze(0)

        if self.Fe is None or self.Fz is None:
            raise ValueError(
                "FilteringEmbedding requires both Fe and Fz callables for parameter updates."
            )

        batch_size = y.shape[0]
        if self._theta_score_block.shape[0] != batch_size:
            self._reset_embedding_block_state(batch_size=batch_size)

        z_prev = torch.nan_to_num(self.z["m"], nan=0.0, posinf=1e6, neginf=-1e6)
        e_eval = self.e["m"]
        if e_eval.shape[0] == 1 and batch_size > 1:
            e_eval = e_eval.expand(batch_size, -1)

        # Transition linearization at current posterior mean
        Fz = self.Fz(z_prev, e_eval)
        dfdz = Fz * self.dt + I
        F_theta = self.Fe(z_prev, e_eval) * self.dt

        if u is not None and self.action_encoder is not None:
            u_enc = self.action_encoder(u, self.z["m"])
        else:
            u_enc = u

        # Predict

        pred_m = torch.nan_to_num(self.predict(action=u_enc), nan=0.0, posinf=1e6, neginf=-1e6)
        pred_cov = dfdz @ self.z["P"] @ dfdz.transpose(-1, -2) + Q + 1e-6 * I
        pred_cov = torch.nan_to_num(pred_cov, nan=0.0, posinf=1e6, neginf=-1e6)
        z_pred = {
            "m": pred_m,
            "P": self._project_spd(pred_cov),
        }

        # Re-linearize observation at new z_pred. Observation noise is diagonal
        # for the Gaussian and Poisson decoders used here, so the EKF update can
        # use the equivalent information form without materializing dense R/S.
        z_obs = torch.nan_to_num(z_pred["m"], nan=0.0, posinf=1e6, neginf=-1e6)
        y_pred, I_z, obs_score, r_Rinv_r = diagonal_observation_information(
            self.decoder, z_obs, observation=y
        )
        assert obs_score is not None and r_Rinv_r is not None

        chol_P_pred = safe_cholesky(z_pred["P"])
        P_pred_inv = torch.cholesky_inverse(chol_P_pred)
        L_post = symmetrize(P_pred_inv + I_z)
        try:
            chol_L_post = safe_cholesky(L_post)
        except Exception:
            L_post = self._project_spd(L_post + 1e-4 * I, min_eig=1e-4)
            chol_L_post = safe_cholesky(L_post)
        P_upd = torch.cholesky_inverse(chol_L_post)
        z_post_m = z_pred["m"] + (P_upd @ obs_score.unsqueeze(-1)).squeeze(-1)
        z_post_m = torch.nan_to_num(z_post_m, nan=0.0, posinf=1e6, neginf=-1e6)

        correction = (obs_score.unsqueeze(-2) @ P_upd @ obs_score.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        delta_t = torch.nan_to_num(
            (r_Rinv_r - correction).clamp_min(0.0) / float(y.shape[-1]),
            nan=0.0,
            posinf=1e6,
            neginf=0.0,
        ).squeeze(-1)
        tau_t = self._compute_parameter_shrinkage(delta_t)
        self._last_innovation_statistic = delta_t.detach()
        self._last_parameter_shrinkage = tau_t.detach()

        z_post = {
            "m": z_post_m,
            "P": symmetrize(
                torch.nan_to_num(P_upd, nan=0.0, posinf=1e6, neginf=-1e6) + 1e-8 * I
            ),
        }

        self.z = {"m": z_post["m"].detach(), "P": z_post["P"].detach()}
        self._state = z_post["m"].detach()

        # Information diagnostics are still useful in state-only update mode, but
        # parameter score / posterior updates are disabled when update_theta=False.
        info_t = None
        if update_theta:
            # Parameter sensitivity recursion S_t = F_theta,t + F_z,t S_{t-1}.
            S_prev = self._theta_sensitivity
            S_t = F_theta.squeeze(1) + dfdz.squeeze(1) @ S_prev  # (B, Dz, De)

            # Score: s_t = S_t^T (Pz_pred)^-1 (z_post - z_pred).
            delta_z = z_post["m"] - z_pred["m"]  # (B, 1, Dz)
            invP_delta = (
                torch.cholesky_solve(delta_z.unsqueeze(-1), chol_P_pred).squeeze(-1).squeeze(1)
            )
            score_t = torch.einsum("bze,bz->be", S_t, invP_delta)
            score_t = tau_t.unsqueeze(-1) * score_t

            # Information: I_t = S_t^T (I + Pz_pred I_z)^-1 I_z S_t.
            atten_mat = self._project_spd(I + z_pred["P"] @ I_z)
            chol_atten = safe_cholesky(atten_mat)
            atten_Iz = torch.cholesky_solve(I_z, chol_atten).squeeze(1)
            info_t = symmetrize(S_t.transpose(-1, -2) @ atten_Iz @ S_t)

        # Diagnostic traces for visualization: average matrix trace over batch.
        I_z_scalar = torch.diagonal(I_z.squeeze(1), dim1=-2, dim2=-1).sum(dim=-1).mean()
        if info_t is not None:
            I_theta_scalar = torch.diagonal(info_t, dim1=-2, dim2=-1).sum(dim=-1).mean()
        else:
            I_theta_scalar = torch.tensor(0.0, device=self.device)
        Pz_eval = z_pred["P"].squeeze(1)
        Pz00 = Pz_eval[:, 0, 0].mean()
        Pz01 = Pz_eval[:, 0, 1].mean()
        Pz11 = Pz_eval[:, 1, 1].mean()
        theta_block_eig = self._theta_block_eig() if update_theta else torch.tensor(0.0, device=self.device)
        self.last_information = {
            "I_z_t": float(torch.nan_to_num(I_z_scalar, nan=0.0, posinf=1e6, neginf=0.0).item()),
            "I_theta_t": float(
                torch.nan_to_num(I_theta_scalar, nan=0.0, posinf=1e6, neginf=0.0).item()
            ),
            "theta_block_eig": float(
                torch.nan_to_num(theta_block_eig, nan=0.0, posinf=1e6, neginf=0.0).item()
            ),
            "theta_block_steps": int(self._theta_block_steps),
            "parameter_update_reason": "none",
            "Pz00": float(torch.nan_to_num(Pz00, nan=0.0, posinf=1e6, neginf=0.0).item()),
            "Pz01": float(torch.nan_to_num(Pz01, nan=0.0, posinf=1e6, neginf=0.0).item()),
            "Pz11": float(torch.nan_to_num(Pz11, nan=0.0, posinf=1e6, neginf=0.0).item()),
        }
        if update_theta and info_t is not None:
            info_t = tau_t.view(-1, 1, 1) * info_t

            self._theta_score_block += score_t
            self._theta_info_block += info_t
            self._theta_sensitivity = S_t.detach()
            self._theta_block_steps += 1
            reason = self._embedding_block_update_reason()
            self.last_information["theta_block_eig"] = float(
                torch.nan_to_num(
                    self._theta_block_eig(), nan=0.0, posinf=1e6, neginf=0.0
                ).item()
            )
            self.last_information["theta_block_steps"] = int(self._theta_block_steps)
            self.last_information["parameter_update_reason"] = "none" if reason is None else reason
            if reason is not None:
                self._last_parameter_update_reason = reason
                self._apply_embedding_block_update()

        return self._state

    def update_embedding(self, r, chol_S, HzGt, curv_ll):
        # predictive covariance and Cholesky solve (as fixed earlier)

        L = self.e["L"]
        eta = L @ self.e["m"].unsqueeze(-1)
        beta = 1
        for _ in range(self.gn_iter):
            invS_r = torch.cholesky_solve(r.mT, chol_S)
            grad_ll = einsum(HzGt, invS_r, "b t y d, b t y k->b t d")  # (1, De)
            L_new = L + beta * curv_ll
            eta_new = eta + beta * grad_ll.unsqueeze(-1)

            chol_L_new = safe_cholesky(L_new)
            Sigma_e = torch.cholesky_inverse(chol_L_new)  # (1, De, De)
            mu_e = (Sigma_e @ eta_new).squeeze(-1)  # (1, De)

            L, eta = L_new, eta_new

        # Detach after all refinements
        self.e = {"m": mu_e.detach(), "P": Sigma_e.detach(), "L": L_new.detach()}
        self.set_params(self.e["m"].detach())

    def _train_single_model(
        self,
        dataloader,
        optimizer,
        param_list,
        lr,
        weight_decay,
        n_epochs,
        verbose,
        grad_clip_norm,
        n_samples,
        k_steps,
        beta,
        p_mask=0.0,
        warmup=1000,
        annealing_steps=1000,
        annealing_type="cyclic",  # "linear", "cyclic", "none"
        model_idx=None,
        model_name="Model",
    ):
        """
        Train a single model (or ensemble member) with the given parameters.
        """
        opt = self._get_optimizer(optimizer, param_list, lr, weight_decay)
        T = 0

        # Initialize epoch progress bar
        if verbose:
            from tqdm import tqdm

            epoch_pbar = tqdm(range(n_epochs), desc=f"{model_name}")
            epoch_iterator = epoch_pbar
        else:
            epoch_iterator = range(n_epochs)

        # Train for multiple epochs with DataLoader
        epoch_info = []
        for i in epoch_iterator:
            batch_info = []
            for batch in dataloader:
                # Zero gradients, compute loss, backprop, and step optimizer
                opt.zero_grad()

                loss, info = self(batch)
                loss.backward()
                # Apply gradient clipping over full parameter list once
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(param_list, grad_clip_norm)

                # Update parameters
                opt.step()

                # Store normalized losses
                batch_info.append(info)  # Assuming info contains the relevant metrics

                # Explicit cleanup for gradient tensors
                del batch, loss

            if model_idx is not None:
                self.step_count += 1 if model_idx == 0 else 0
            else:
                self.step_count += 1

            # Convert list of dict to dict of tensor
            batch_info = {
                key: torch.tensor([b[key] for b in batch_info]).mean(dim=0) for key in batch_info[0]
            }
            epoch_info.append(batch_info)

            # Update epoch progress bar with average ELBO
            if verbose and epoch_info and i % 10 == 0:
                current_info = epoch_info[-1]
                epoch_pbar.set_postfix({k: f"{v:.4f}" for k, v in current_info.items()})
                epoch_pbar.update(10)

        # Close progress bar
        if verbose:
            epoch_pbar.close()

        epoch_info = {
            key: torch.tensor([e[key] for e in epoch_info]).mean(dim=0).item()
            for key in epoch_info[0]
        }
        return epoch_info

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        z = batch["next_model_state"].to(self.device)
        y = batch["next_obs"].to(self.device)

        ll = self.decoder.compute_log_prob(z, y)
        loss = -ll.mean()
        info = {"log_L": ll.mean().detach()}
        return loss, info
