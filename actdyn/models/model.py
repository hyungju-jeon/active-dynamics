import torch
from tqdm import tqdm
from typing import Optional

from .base import BaseModel
from .dynamics import BaseDynamics
from .encoder import BaseEncoder
from .decoder import Decoder
from actdyn.environment.action import BaseAction
from actdyn.utils.rollout import Rollout
from torch.utils.data import DataLoader


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
        self.input_dim = encoder.input_dim
        self.latent_dim = getattr(dynamics, "latent_dim", None)
        self.action_dim = getattr(action_encoder, "input_dim", 0)
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
            return torch.optim.AdamW(
                params=param_list, lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    @staticmethod
    def _kl_div(mu_q, var_q, mu_p, var_p):
        """Computes KL divergence between two multivariate normal distributions."""
        kl_d = 0.5 * (
            torch.log(var_p / var_q)
            + ((mu_q - mu_p) ** 2) / var_p
            + (var_q / var_p)
            - 1
        )
        return torch.sum(kl_d, (-1, -2))

    def _compute_kld_x(self, mu_q, var_q, x_samples, u=None, idx=None):
        dynamics = self._get_dynamics(idx)
        if u is None:
            u = torch.zeros_like(x_samples, device=self.device)
        mu_p_x, var_p_x = dynamics.sample_forward(
            x_samples[..., :-1, :], u[..., :-1, :]
        )
        kl_d = self._kl_div(mu_q[..., 1:, :], var_q[..., 1:, :], mu_p_x, var_p_x)
        return kl_d

    def compute_elbo(self, y, u=None, n_samples=1, beta=10.0, idx=None):
        x_samples, mu_q_x, var_q_x, log_q = self.encoder(y, n_samples=n_samples)
        if self.action_encoder is not None and u is not None:
            u_encoded = self.action_encoder(u)
        else:
            u_encoded = u
        kl_d_x = self._compute_kld_x(mu_q_x, var_q_x, x_samples, u=u_encoded, idx=idx)
        log_like = self.decoder.compute_log_prob(x_samples, y)
        elbo = torch.mean(log_like - beta * kl_d_x)
        return -elbo

    def _add_param_perturbation(self, model, perturbation):
        if perturbation > 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.data += perturbation * torch.randn_like(param.data)

    def _train_loop(self, data, opt, n_epochs, verbose=False, idx=None):
        training_losses = []
        if isinstance(data, Rollout) or isinstance(data, dict):
            batch_iter = [data]
        else:
            batch_iter = data
        if verbose:
            pbar = tqdm(range(n_epochs))
        else:
            pbar = range(n_epochs)
        for epoch in pbar:
            last_loss = None
            for batch in batch_iter:
                if not isinstance(batch, dict):
                    continue  # skip invalid batches
                obs = batch["obs"].to(self.device)
                action = batch.get("action", None)
                if action is not None:
                    action = action.to(self.device)
                opt.zero_grad()
                loss = self.compute_elbo(obs, u=action, idx=idx)
                loss.backward()
                opt.step()
                training_losses.append(loss.item())
                last_loss = loss.item()
            if verbose:
                pbar.set_postfix({"loss": last_loss})  # type: ignore
        return training_losses

    def train(
        self,
        data,
        lr=1e-4,
        weight_decay=1e-4,
        n_epochs=100,
        optimizer="SGD",
        verbose=True,
        perturbation=0.0,
    ):
        if self.is_ensemble:
            for i in range(self.dynamics.n_models):
                param_list = (
                    list(self.dynamics.models[i].parameters())
                    + list(self.encoder.parameters())
                    + list(self.decoder.parameters())
                )
                opt = self._get_optimizer(optimizer, param_list, lr, weight_decay)
                self._train_loop(data, opt, n_epochs, verbose, idx=i)
                self._add_param_perturbation(self.dynamics.models[i], perturbation)
        else:
            param_list = list(self.parameters())
            opt = self._get_optimizer(optimizer, param_list, lr, weight_decay)
            self._train_loop(data, opt, n_epochs, verbose)
            self._add_param_perturbation(self.dynamics, perturbation)
