from calendar import c
import re
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from einops import rearrange, repeat, einsum
from torch.nn.functional import softplus

from actdyn.environment.action import BaseAction
from actdyn.utils.helper import safe_cholesky, symmetrize, eps
from actdyn.utils.rollout import RolloutBuffer

from .base import BaseDynamicsEnsemble, BaseModel
from .decoder import Decoder
from .dynamics import BaseDynamics, FunctionDynamics
from .encoder import BaseEncoder

Belief = Dict[str, torch.Tensor]


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

        z_tr = self.dynamics.sample_forward(init_z=z_me, action=u_encoded, k_step=1)[1]
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
                batch_info.append(info)
                # Update parameters
                opt.step()
                loss, log_like, kl_d = (
                    loss.detach().item(),
                    log_like.detach().item(),
                    kl_d.detach().item(),
                )
                # Store normalized losses
                batch_info.append(info)  # Assuming info contains the relevant metrics

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

            # End of epoch cleanup
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

        # Close progress bar
        if verbose:
            epoch_pbar.close()

        epoch_info = {
            key: torch.tensor([e[key] for e in epoch_info]).mean(dim=0).item()
            for key in epoch_info[0]
        }

        return epoch_info

    # DESIGN NOTE: Missing save/load methods
    # =======================================
    # Issue: experiment.py expects model.save_model(path) method
    # Impact: Experiment tests fail - cannot save checkpoints
    #
    # Recommended implementation:
    #
    # def save_model(self, filepath):
    #     """Save model state dict to file."""
    #     torch.save({
    #         'encoder': self.encoder.state_dict(),
    #         'decoder': self.decoder.state_dict(),
    #         'dynamics': self.dynamics.state_dict(),
    #         'action_encoder': self.action_encoder.state_dict() if self.action_encoder else None,
    #         'step_count': self.step_count,
    #         'beta': self.beta,
    #     }, filepath)
    #
    # def load_model(self, filepath):
    #     """Load model state dict from file."""
    #     checkpoint = torch.load(filepath, map_location=self.device)
    #     self.encoder.load_state_dict(checkpoint['encoder'])
    #     self.decoder.load_state_dict(checkpoint['decoder'])
    #     self.dynamics.load_state_dict(checkpoint['dynamics'])
    #     if self.action_encoder and checkpoint['action_encoder']:
    #         self.action_encoder.load_state_dict(checkpoint['action_encoder'])
    #     self.step_count = checkpoint.get('step_count', 0)
    #     self.beta = checkpoint.get('beta', 0.0)
    #
    # Usage: experiments/run_experiment.py saves checkpoints during training

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

        z_tr = self.dynamics.sample_forward(init_z=z_me, action=u_encoded, k_step=1)[1]
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
                batch_info.append(info)
                # Update parameters
                opt.step()
                loss, log_like, kl_d = (
                    loss.detach().item(),
                    log_like.detach().item(),
                    kl_d.detach().item(),
                )
                # Store normalized losses
                batch_info.append(info)  # Assuming info contains the relevant metrics

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

            # End of epoch cleanup
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

        # Close progress bar
        if verbose:
            epoch_pbar.close()

        epoch_info = {
            key: torch.tensor([e[key] for e in epoch_info]).mean(dim=0).item()
            for key in epoch_info[0]
        }

        return epoch_info

    def update_posterior_embedding(self, y, z, u=None):
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beta = 0.0
        self.e: Belief = e
        self.z: Belief = {
            "m": torch.zeros(1, 1, self.latent_dim, device=self.device),
            "P": torch.eye(self.latent_dim, device=self.device).unsqueeze(0).unsqueeze(0),
        }
        self.Fe = Fe
        self.Fz = Fz
        self._state = torch.zeros(1, 1, self.latent_dim, device=self.device)
        self.gn_iter = 10
        self.set_params(e["m"])

    def set_params(self, e: torch.Tensor):
        self.e["m"] = e.to(self.device)
        self.dynamics.set_params(e)

    def reset(self, observation: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment to initial state."""
        observation, info = super().reset(observation)
        self.e.update(
            {
                "P": torch.eye(self.e["m"].shape[-1], device=self.device),
                "L": torch.eye(self.e["m"].shape[-1], device=self.device),
            }
        )
        self.z = {
            "m": self._state,
            "P": torch.eye(self.latent_dim, device=self.device),
        }
        self.set_params(self.e["m"])

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
    def update_posterior(self, y, u=None):
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
    def update_posterior_embedding(self, y, u=None, **kwargs):
        """Update the posterior state given new observation and action."""

        y = y[:, -1:, :]
        u = u[:, -1:, :] if u is not None else None
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

        # Re-linearize observation and variance at new z_pred
        dhdz = self.decoder.jacobian(z_pred["m"])
        R = self.decoder.var(z_pred["m"]).diag_embed()
        S = symmetrize(dhdz @ z_pred["P"] @ dhdz.transpose(-1, -2)) + R
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
        # Sensitivity wrt embedding via dynamics (ignore Fz dependence)
        Gt = self.Fe(self.z["m"], self.e["m"]) * self.dt
        HzGt = dhdz @ Gt

        # GN curvature
        X = torch.cholesky_solve(HzGt, chol_S)
        curv_ll = einsum(HzGt, X, "b t y d, b t y e->b t d e")
        curv_ll = symmetrize(curv_ll)  # ensure symmetry
        self.update_embedding(r, chol_S, HzGt, curv_ll)

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
        alpha = 0.5
        mu_e = (1 - alpha) * mu_e + alpha * self.e["m"].detach()
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

            # End of epoch cleanup
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

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
