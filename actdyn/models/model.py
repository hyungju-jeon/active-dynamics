from typing import Optional
import torch
from einops import rearrange, repeat

from actdyn.environment.action import BaseAction
from .base import BaseDynamicsEnsemble, BaseModel
from .decoder import Decoder
from .dynamics import BaseDynamics
from .encoder import BaseEncoder


class SeqVae(BaseModel):
    """Sequential Variational Autoencoder (SeqVAE) with dynamics."""

    def __init__(
        self,
        dynamics: BaseDynamics | BaseDynamicsEnsemble,
        encoder: BaseEncoder,
        decoder: Decoder,
        action_encoder: Optional[BaseAction] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.dynamics = dynamics
        self.encoder = encoder
        self.decoder = decoder
        self.action_encoder = action_encoder
        self.input_dim = encoder.obs_dim
        self.latent_dim = getattr(encoder, "latent_dim", None)
        self.action_dim = getattr(action_encoder, "action_dim", 0)
        self.is_ensemble = hasattr(dynamics, "ensemble") and hasattr(dynamics, "n_models")
        self.step_count = 0
        self.beta = 0.0

    def _get_dynamics(self, idx=None):
        if self.is_ensemble:  # --- IGNORE ---
            if idx is None:
                raise ValueError("Ensemble mode requires an idx argument.")
            return self.dynamics.ensemble[idx]
        else:
            return self.dynamics

    def _get_optimizer(self, optimizer, param_list, lr, weight_decay):
        if optimizer == "SGD":
            return torch.optim.SGD(params=param_list, lr=lr)
        elif optimizer == "Adam":
            return torch.optim.Adam(params=param_list, lr=lr, weight_decay=weight_decay)
        elif optimizer == "AdamW":
            return torch.optim.AdamW(params=param_list, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    @staticmethod
    def _kl_div(mu_q, var_q, mu_p, var_p):
        """Analytic KL divergence between two multivariate normal distributions."""
        kl_d = 0.5 * (torch.log(var_p / var_q) + ((mu_q - mu_p) ** 2) / var_p + (var_q / var_p) - 1)
        return torch.sum(kl_d, (-1))

    @staticmethod
    def _kl_div_mc(mu_q, var_q, z_prior, mu_p, var_p):
        """Monte Carlo KL"""
        # TODO : FIX this part
        S, B, T, D = z_prior.shape

        # Expand posterior to match particles
        mu_q_exp = repeat(mu_q, "b t d -> s b t d", s=S)
        var_q_exp = repeat(var_q, "b t d -> s b t d", s=S)

        # Posterior log-density at prior samples
        log_q = -0.5 * (
            (torch.log(2 * torch.pi * var_q_exp) + (z_prior - mu_q_exp) ** 2 / var_q_exp).sum(
                dim=(-2, -1)
            )
        )

        # Prior log-density using dynamics mean and variance
        log_p = -0.5 * (
            (torch.log(2 * torch.pi * var_p) + (z_prior - mu_p) ** 2 / var_p).sum(dim=(-2, -1))
        )

        # Monte Carlo KL estimate: mean over particles
        kl_est = (log_q - log_p).mean(dim=0)  # (B,)

        return kl_est

    def _compute_multistep_kl(
        self,
        mu_q,  # (B,T,D) posterior mean
        var_q,  # (B,T,D) posterior variance
        z_samples,  # (S,B,T,D) posterior samples
        u=None,  # (B,T,A) action sequence
        idx=None,
        t_mask=None,  # (T,1) temporal mask
        k_steps=5,
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

        if self.action_encoder is not None and u is not None:
            u_encoded = self.action_encoder(u[..., 1:, :], z_me[..., :-1, :])
            # Align a_t with y_{t+1}
        else:
            u_encoded = u

        # Apply temporal masking
        t_mask = torch.bernoulli((1 - p_mask) * torch.ones((T, 1), device=mu_q_x.device))

        z_tr = self.dynamics.sample_forward(init_z=z_me, action=u_encoded, k_step=1)[1]
        z_tr = torch.cat([z_me[..., :1, :], z_tr], dim=-2)  # (S,B,T,D)

        z_samples = t_mask * z_me + (1 - t_mask) * z_tr  # (S,B,T,D)

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

    def _add_param_perturbation(self, model, perturbation):
        if perturbation > 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.data += perturbation * torch.randn_like(param.data)

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
        epoch_losses = []
        for i in epoch_iterator:
            batch_losses = []
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

                # Update parameters
                opt.step()
                loss, log_like, kl_d = (
                    loss.detach().item(),
                    log_like.detach().item(),
                    kl_d.detach().item(),
                )
                # Store normalized losses
                batch_losses.append(
                    [loss / T, log_like / T / self.input_dim, kl_d / T / self.latent_dim]
                )

                # Explicit cleanup for gradient tensors
                del batch, obs, loss, log_like, kl_d

                if action is not None:
                    del action
            if model_idx is not None:
                self.step_count += 1 if model_idx == 0 else 0
            else:
                self.step_count += 1

            # Convert list to tensor and average across batch
            epoch_losses.append(torch.tensor(batch_losses).mean(dim=0))

            # Update epoch progress bar with average ELBO
            if verbose and epoch_losses and i % 10 == 0:
                current_elbo = -epoch_losses[-1][0]
                epoch_pbar.set_postfix(ELBO=f"{current_elbo:.6f}, beta={self.beta:.4f}")
                epoch_pbar.update(10)

            # End of epoch cleanup
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

        # Close progress bar
        if verbose:
            epoch_pbar.close()

        return epoch_losses

    def train_model(
        self,
        dataloader=None,
        lr=1e-4,
        weight_decay=1e-3,
        n_epochs=1,
        optimizer="SGD",
        verbose=True,
        perturbation=0.0,
        grad_clip_norm=1.0,
        n_samples=5,
        k_steps=1,
        **kwargs,
    ):
        """
        Train model using PyTorch DataLoader.
        """
        all_losses = []
        train_args = {
            "dataloader": dataloader,
            "optimizer": optimizer,
            "lr": lr,
            "weight_decay": weight_decay,
            "n_epochs": n_epochs,
            "verbose": verbose,
            "grad_clip_norm": grad_clip_norm,
            "n_samples": n_samples,
            "k_steps": k_steps,
            "p_mask": kwargs.get("p_mask", 0.0),
            "beta": kwargs.get("beta", 1.0),
            "warmup": kwargs.get("warmup", 1000),
            "annealing_steps": kwargs.get("annealing_steps", 1000),
            "annealing_type": kwargs.get("annealing_type", "cyclic"),
            "param_list": kwargs.get("param_list", "all"),  # To be set per model
        }

        param_lists = self.get_param_list(train_args["param_list"])
        # print(f"Training with {param_lists} parameter groups.")

        if self.is_ensemble:
            # Train each ensemble member
            for i in range(self.dynamics.n_models):
                # First model trains encoder/decoder, others only train dynamics
                param_list = param_lists[i]

                model_name = f"Ensemble model {i+1}/{self.dynamics.n_models}"
                train_args["model_idx"] = i
                train_args["model_name"] = model_name
                train_args["param_list"] = param_list

                # Train this ensemble member
                epoch_losses = self._train_single_model(**train_args)
                all_losses.extend(epoch_losses)

                # Apply parameter perturbation for ensemble diversity
                self._add_param_perturbation(self.dynamics.ensemble[i], perturbation)
        else:
            # Train single model
            param_list = param_lists[0]
            train_args["param_list"] = param_list
            train_args["model_idx"] = None
            train_args["model_name"] = "Model"

            epoch_losses = self._train_single_model(**train_args)
            all_losses.extend(epoch_losses)

            # Apply parameter perturbation
            self._add_param_perturbation(self.dynamics, perturbation)

        # Final cleanup
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()

        return all_losses

    def get_param_list(self, param_list_type):
        if param_list_type == "all":
            return [list(self.parameters())]
        else:
            params = []
            if "dynamics" in param_list_type:
                if self.is_ensemble:
                    for i in range(self.dynamics.n_models):
                        params.append(list(self.dynamics.ensemble[i].parameters()))
                else:
                    params.append(list(self.dynamics.parameters()))
            if "encoder" in param_list_type:
                params.append(list(self.encoder.parameters()))
            if "decoder" in param_list_type:
                params.append(list(self.decoder.parameters()))
            if "action" in param_list_type and self.action_encoder is not None:
                params.append(list(self.action_encoder.parameters()))

        return params

    def beta_schedule(self, beta, warmup=1000, annealing_steps=1000, annealing_type="cyclic"):
        if annealing_type == "linear":
            self.beta = min(beta, beta * (self.step_count - warmup) / annealing_steps)
        elif annealing_type == "cyclic":
            cycle_length = 2 * annealing_steps
            cycle_pos = (self.step_count - warmup) % cycle_length
            if cycle_pos < annealing_steps:
                self.beta = min(beta, beta * cycle_pos / annealing_steps)
            else:
                self.beta = beta
        elif annealing_type == "none":
            self.beta = beta


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
