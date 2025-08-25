from typing import Optional
import torch
from einops import rearrange, repeat

from actdyn.environment.action import BaseAction
from .base import BaseModel
from .decoder import Decoder
from .dynamics import BaseDynamics
from .encoder import BaseEncoder


class SeqVae(BaseModel):
    """Sequential Variational Autoencoder (SeqVAE) with dynamics."""

    def __init__(
        self,
        dynamics: BaseDynamics,
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
        self.is_ensemble = hasattr(dynamics, "models") and hasattr(dynamics, "n_models")

    def _get_dynamics(self, idx=None):
        if self.is_ensemble:
            if idx is None:
                raise ValueError("Ensemble mode requires an idx argument.")
            return self.dynamics.models[idx]
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
        """Computes KL divergence between two multivariate normal distributions."""
        kl_d = 0.5 * (torch.log(var_p / var_q) + ((mu_q - mu_p) ** 2) / var_p + (var_q / var_p) - 1)
        return torch.sum(kl_d, (-1, -2))

    def _compute_multistep_kl(self, mu_q, var_q, x_samples, u=None, idx=None, k_steps=5):
        """Compute multi-step KL terms KL[q(x_{t+k}) || p_k(x_{t+k}|x_t, u_{t+1:t+k})] for k_steps."""
        dynamics = self._get_dynamics(idx)
        if x_samples.dim() == 3:  # (B,T,D) -> (1,B,T,D)
            x_samples = x_samples.unsqueeze(0)
        S, B, T, D = x_samples.shape

        # Prepare action tensor
        if u is None:
            u = torch.zeros(B, T, getattr(self.action_encoder, "action_dim", 0), device=self.device)
        if u.dim() != 3:
            raise ValueError("Action tensor must have shape (B,T,A)")
        # Broadcast actions across sample dimension
        u_exp = repeat(u, "b t a -> s b t a", s=S)

        kl_list = []
        for k in range(1, k_steps + 1):
            if T - k <= 0:
                # Not enough horizon left; append zeros
                kl_list.append(torch.zeros(S, B, device=self.device))
                continue

            # Starting states x_t for all valid t
            start_states = x_samples[:, :, : T - k, :]  # (S,B,T-k,D)

            # Roll forward k steps vectorized over (S,B,T-k)
            pred = start_states
            var_accum = None
            for step in range(k):
                mu_step, var_step = dynamics.compute_param(pred)
                pred = pred + mu_step * getattr(dynamics, "dt", 1.0)
                if u is not None and u_exp.shape[2] >= (step + 1) + (T - k):
                    # Actions indices aligned: want a_{t+step+1}
                    action_slice = u_exp[:, :, step + 1 : step + 1 + (T - k), :]
                    if action_slice.shape[2] == pred.shape[2]:  # time dims align
                        pred = pred + action_slice * getattr(dynamics, "dt", 1.0)
                # Accumulate variance assuming independence each step
                var_accum = var_step.clone() if var_accum is None else (var_accum + var_step)

            mu_p = pred  # (S,B,T-k,D)

            # Broadcast posterior slices to sample dimension
            mu_q_target = repeat(mu_q[:, k:, :], "b t d -> s b t d", s=S)
            var_q_target = repeat(var_q[:, k:, :], "b t d -> s b t d", s=S)

            # Prior variance broadcast & match shape (var_accum: (1,D) broadcast)
            if var_accum is None:
                var_accum = torch.full_like(mu_p, 1e-3)
            if var_accum.dim() == 2:
                var_p = var_accum.view(1, 1, 1, D).expand(S, B, T - k, D)
            else:
                var_p = var_accum
            kl_k = self._kl_div(mu_q_target, var_q_target, mu_p, var_p)  # (S,B)
            kl_list.append(kl_k)

        kl_per_sample_per_k = torch.stack(kl_list, dim=-1)  # (S,B,K)
        return kl_per_sample_per_k

    def compute_elbo(self, y, u=None, n_samples=5, k_steps=5, beta=1.0, idx=None):
        """Compute ELBO with proper multi-step KL and Monte Carlo averaging.

        Returns:
            loss (scalar), recon_loss (scalar), kl_loss (scalar)
        """
        x_samples, mu_q_x, var_q_x = self.encoder(y=y, u=u, n_samples=n_samples)
        if x_samples.dim() == 3:  # (B,T,D) -> add samples dim
            x_samples = x_samples.unsqueeze(0)
        if self.action_encoder is not None and u is not None:
            u_encoded = self.action_encoder(u)
        else:
            u_encoded = u

        # Multi-step KL: (S,B,K)
        kl_sbk = self._compute_multistep_kl(
            mu_q_x, var_q_x, x_samples, u=u_encoded, idx=idx, k_steps=k_steps
        )
        # Average over k horizons equally
        kl_sb = kl_sbk.mean(dim=-1)  # (S,B)

        # Log-likelihood per sample: decoder expects (B,T,D)
        S, B, T, D = x_samples.shape
        z_flat = rearrange(x_samples, "s b t d -> (s b) t d")
        y_rep = repeat(y, "b t d -> (s b) t d", s=S)
        log_like_flat = self.decoder.compute_log_prob(z_flat, y_rep)  # (S*B)
        log_like_sb = rearrange(log_like_flat, "(s b) -> s b", s=S, b=B)

        # Monte Carlo expectation over samples
        log_like_b = log_like_sb.mean(dim=0)  # (B)
        kl_b = kl_sb.mean(dim=0)  # (B)

        elbo_b = log_like_b - beta * kl_b
        elbo = elbo_b.mean()

        # Return negatives for losses (training minimizes)
        return -elbo, -log_like_b.mean(), kl_b.mean()

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
        for _ in epoch_iterator:
            batch_losses = []
            for batch in dataloader:
                # Move batch to device
                obs = batch["next_obs"].to(self.device)
                action = batch["action"].to(self.device) if "action" in batch else None
                T = obs.shape[-2]  # Time dimension

                # Handle sequence data: if we have extra dimensions, squeeze them
                # This happens when DataLoader adds batch dimension to already-batched sequence data
                while obs.dim() > 3 and obs.shape[0] == 1:
                    obs = obs.squeeze(0)  # Remove extra batch dimensions
                if action is not None:
                    while action is not None and action.dim() > 3 and action.shape[0] == 1:
                        action = action.squeeze(0)

                # Ensure we have the right shape: (batch, time, obs_dim)
                if obs.dim() != 3:
                    raise ValueError(
                        f"Expected 3D observation tensor (batch, time, obs_dim), got shape {obs.shape}"
                    )

                opt.zero_grad()
                loss, log_like, kl_d = self.compute_elbo(
                    obs, u=action, idx=model_idx, n_samples=n_samples, k_steps=k_steps
                )
                loss.backward()

                # Apply gradient clipping over full parameter list once
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(param_list, grad_clip_norm)

                opt.step()
                loss, log_like, kl_d = loss.item(), log_like.detach().item(), kl_d.detach().item()
                # Store normalized losses
                batch_losses.append([loss / T, log_like / T, kl_d / T])

                # Explicit cleanup for gradient tensors
                del batch, obs, loss, log_like, kl_d
                if action is not None:
                    del action

            # Convert list to tensor and average across batch
            epoch_losses.append(torch.tensor(batch_losses).mean(dim=0))

            # Update epoch progress bar with average ELBO
            if verbose and epoch_losses:
                current_loss = epoch_losses[-1][0]
                epoch_pbar.set_postfix(ELBO=f"{current_loss:.6f}")

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
        k_steps=5,
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
        }

        if self.is_ensemble:
            # Train each ensemble member
            for i in range(self.dynamics.n_models):
                # First model trains encoder/decoder, others only train dynamics
                if i == 0:
                    param_list = (
                        list(self.dynamics.models[i].parameters())
                        + list(self.encoder.parameters())
                        + list(self.decoder.parameters())
                    )
                else:
                    param_list = list(self.dynamics.models[i].parameters())

                model_name = f"Ensemble model {i+1}/{self.dynamics.n_models}"
                train_args["model_idx"] = i
                train_args["model_name"] = model_name
                train_args["param_list"] = param_list

                # Train this ensemble member
                epoch_losses = self._train_single_model(**train_args)
                all_losses.extend(epoch_losses)

                # Apply parameter perturbation for ensemble diversity
                self._add_param_perturbation(self.dynamics.models[i], perturbation)
        else:
            # Train single model
            param_list = list(self.parameters())
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
