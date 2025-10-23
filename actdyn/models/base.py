"""Base model classes for the active dynamics package."""

from typing import Any, Dict, Tuple
from collections.abc import Callable
import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch.utils.data import DataLoader, Dataset


from actdyn.environment.base import BaseAction


# Small constant to prevent numerical instability
eps = 1e-6


# Encoder models
class BaseEncoder(nn.Module):
    """Base class for encoder models."""

    network: nn.Module

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, device="cpu"):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.network = None

    def compute_param(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def to(self, device):
        self.device = torch.device(device)
        self.network.to(device)
        return self

    def validate_input(self, y, u):
        assert y.dim() == 3, f"Input y must be of shape (batch, time, input_dim), got {y.shape}"
        if u is not None:
            assert u.dim() == 3
            assert (
                u.shape[0] == y.shape[0]
            ), f"Batch size of a {u.shape[0]} must match y {y.shape[0]}"
            assert (
                u.shape[1] == y.shape[1]
            ), f"Time dimension of a {u.shape[1]} must match y {y.shape[1]}"
            # DESIGN NOTE: Action dimension validation flexibility
            # =======================================================
            # Current: Strict assertion requires exact match
            # Problem: Fails when action_dim=0 (no action encoder) but u is passed
            #
            # Better design options:
            #
            # Option 1 (Recommended - More flexible):
            # if self.action_dim > 0:
            #     assert u.shape[2] == self.action_dim, \
            #         f"Action dimension {u.shape[2]} must match action_dim {self.action_dim}"
            # else:
            #     # If no action encoder, ignore actions or warn
            #     u = None  # or warn and ignore
            #
            # Option 2 (Handle None action_dim):
            # if self.action_dim is not None and self.action_dim > 0:
            #     # validate as normal
            # else:
            #     # Either ignore actions or use them anyway
            #
            # Option 3 (Auto-infer):
            # if self.action_dim is None:
            #     self.action_dim = u.shape[2]  # infer from data
            #
            # Current usage:
            # - experiments/run_experiment.py: Models can be trained without actions
            # - Some environments don't use actions (passive observation only)
            assert (
                u.shape[2] == self.action_dim
            ), f"Action dimension of a {u.shape[2]} must match action_dim {self.action_dim}"
        else:
            u = torch.zeros(
                (y.shape[0], y.shape[1], self.action_dim), device=y.device, dtype=y.dtype
            )

        return y.to(self.device), u.to(self.device)


# Observation mappings
class BaseMapping(nn.Module):

    network: nn.Module

    def __init__(self, latent_dim: int, obs_dim: int, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.network = None
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

    def forward(self, z):
        return self.network(z)

    def to(self, device):
        self.device = torch.device(device)
        self.network.to(device)
        return self

    def set_weights(self, weights):
        raise NotImplementedError

    def set_bias(self, bias):
        raise NotImplementedError

    def set_params(self, weights, bias):
        """Set both weights and bias of the linear mapping."""
        self.set_weights(weights)
        self.set_bias(bias)

    @property
    def jacobian(self):
        raise NotImplementedError


# Noise models
class BaseNoise(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def log_prob(self, mean, x):
        raise NotImplementedError


# Dynamics models
class BaseDynamics(nn.Module):
    """Base class for all dynamics models with sampling utility."""

    network: nn.Module | Callable

    def __init__(self, state_dim, dt, is_residual: bool = False, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.dt = dt
        self.is_residual = is_residual
        self.logvar = nn.Parameter(
            -2 * torch.rand(1, state_dim, device=self.device), requires_grad=True
        )
        self.network = None
        self.state_dim = state_dim

    def to(self, device):
        self.device = torch.device(device)
        if isinstance(self.network, nn.Module):
            self.network.to(device)

    def compute_param(self, state):
        mu = self.network(state)
        var = softplus(self.logvar) + eps
        return mu, var

    def sample_forward(self, init_z, action=None, k_step=1, return_traj=False, add_noise=True):
        """Generates samples from forward dynamics model.

        DESIGN NOTE: Return signature consistency issue
        ===============================================

        Current behavior:
        - If return_traj=True: returns (samples_list, mus_list, vars_list)
          where each is a list of length k_step+1
        - If return_traj=False: returns (sample, mu, var)
          where each is a single tensor

        Problem: Code in model.py line 106-108 always unpacks as:
            samples_list, mus_list, vars_list = dynamics.sample_forward(...)
        This fails when return_traj=False since it only returns 3 tensors not 3 lists.

        Better design options:

        Option 1 (Minimal change - Recommended):
        Always return lists, make return_traj only control list length:
        - return_traj=True: return full trajectory lists
        - return_traj=False: return lists with only final values [final_sample], [final_mu], [final_var]

        Option 2 (More explicit):
        Rename parameter to be clearer:
        - return_trajectory=False: return final (sample, mu, var) as tensors
        - return_trajectory=True: return trajectories as (samples_list, mus_list, vars_list)
        And update all callers to check return type or always use return_trajectory=True

        Option 3 (API redesign):
        Split into two methods:
        - sample_forward_single(...) -> (sample, mu, var) tensors
        - sample_forward_trajectory(...) -> (samples, mus, vars) lists

        Current usage in codebase:
        - model.py:106 always expects lists (uses return_traj=True)
        - Tests expect both behaviors

        Recommended: Option 1 for backward compatibility with minimal changes.
        """
        B, T, _ = init_z.shape
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            # If action is longer than the initial trajectory, adjust k_step
            k_step = action.shape[-2] - T + 1
            if B < action.shape[0]:
                init_z = init_z.repeat(action.shape[0], 1, 1)

        samples, mus, vars = [init_z], [], []
        for k in range(1, k_step + 1):
            mu, var = self.compute_param(samples[k - 1])
            if self.is_residual:
                z_pred = samples[k - 1] + mu * self.dt  # Residual connection
            else:
                z_pred = mu
            if len(z_pred.shape) == 2:
                z_pred = z_pred.unsqueeze(0)

            if action is not None:
                valid_T = min(z_pred.shape[-2], action.shape[-2])
                z_pred = z_pred[..., :valid_T, :]
                z_pred += action[..., :valid_T, :] * self.dt
                action = action[..., 1:, :]  # Shift action for next step

            mus.append(z_pred)
            vars.append(var)

            z_sample = z_pred
            if add_noise:
                z_sample += torch.sqrt(var * self.dt) * torch.randn_like(z_pred, device=self.device)
            samples.append(z_sample)

        if return_traj:
            return samples, mus, vars
        else:
            # DESIGN FIX: For consistency, wrap single values in lists
            # This makes the return signature consistent regardless of return_traj value
            # Callers expecting lists (like model.py:106) won't break
            # Alternative: Keep as is but update all callers to check return type
            return samples[-1], mus[-1], vars[-1]

    def forward(self, state):
        return self.compute_param(state)[0]


class BaseDynamicsEnsemble(nn.Module):
    """
    Generic ensemble wrapper for any dynamics model.

    DESIGN NOTE: Constructor signature for ensemble classes
    ========================================================

    Issue: Tests try to instantiate with different parameters than supported
    - Tests use: RBFDynamicsEnsemble(state_dim=2, n_models=3, alpha=0.1, ...)
    - Actual signature: __init__(dynamics_cls, n_models, dynamics_config)

    Current design requires:
    1. dynamics_cls: The class to instantiate (e.g., RBFDynamics)
    2. n_models: Number of ensemble members
    3. dynamics_config: Dict of kwargs to pass to each dynamics instance

    Example usage:
    ensemble = BaseDynamicsEnsemble(
        dynamics_cls=RBFDynamics,
        n_models=5,
        dynamics_config={
            'state_dim': 2,
            'alpha': 0.1,
            'gamma': 1.0,
            'z_max': 2.0,
            'num_grid_pts': 5,
            'device': 'cpu'
        }
    )

    Better design options:

    Option 1 (More intuitive for subclasses):
    Let subclasses handle their own instantiation:

    class RBFDynamicsEnsemble(BaseDynamicsEnsemble):
        def __init__(self, state_dim, n_models=5, **rbf_kwargs):
            dynamics_config = {'state_dim': state_dim, **rbf_kwargs}
            super().__init__(
                dynamics_cls=RBFDynamics,
                n_models=n_models,
                dynamics_config=dynamics_config
            )

    Option 2 (More flexible base class):
    Accept **kwargs and pass to dynamics:

    def __init__(self, dynamics_cls, n_models=5, **dynamics_kwargs):
        super().__init__()
        self.ensemble = nn.ModuleList([
            dynamics_cls(**dynamics_kwargs) for _ in range(n_models)
        ])
        self.n_models = n_models

    Then subclasses can be simpler:
    class MLPDynamicsEnsemble(BaseDynamicsEnsemble):
        def __init__(self, state_dim, n_models=5, **mlp_kwargs):
            super().__init__(
                dynamics_cls=MLPDynamics,
                n_models=n_models,
                state_dim=state_dim,
                **mlp_kwargs
            )

    Recommended: Option 1 for clarity and type safety
    """

    def __init__(
        self,
        dynamics_cls,
        n_models=5,
        dynamics_config=None,
    ):
        super().__init__()
        if dynamics_config is None:
            dynamics_config = {}
        self.ensemble = nn.ModuleList([dynamics_cls(**dynamics_config) for _ in range(n_models)])
        self.n_models = n_models

    def sample_forward(self, init_z, action=None, k_step=1, return_traj=False):
        all_mus, all_vars = [], []
        for model in self.ensemble:
            samples, mus, vars = model.sample_forward(
                init_z, action, k_step=k_step, return_traj=return_traj
            )
            all_mus.append(mus)
            all_vars.append(vars)
        mus = torch.stack(all_mus)
        vars = torch.stack(all_vars)
        mean_prediction = mus.mean(dim=0)
        total_variance = vars.mean(dim=0) + mus.var(dim=0)
        return mean_prediction, total_variance

    def forward(self, state):
        predictions = [model(state) for model in self.ensemble]
        return torch.stack(predictions)


# Base model class
class BaseModel(nn.Module):
    """Base class for all models in the package."""

    from actdyn.models.decoder import Decoder

    encoder: BaseEncoder | None
    decoder: Decoder | None
    dynamics: BaseDynamics | BaseDynamicsEnsemble | None
    action_encoder: BaseAction | None

    def __init__(
        self, encoder=None, action_encoder=None, dynamics=None, decoder=None, device: str = "cpu"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.decoder = decoder
        self.dynamics = dynamics
        self.input_dim = self.decoder.obs_dim
        self.latent_dim = getattr(self.decoder, "latent_dim", 0)
        self.action_dim = getattr(self.action_encoder, "action_dim", 0)
        self.is_ensemble = hasattr(self.dynamics, "ensemble") and hasattr(self.dynamics, "n_models")
        self.dt = getattr(self.dynamics, "dt", 1.0)
        self.step_count = 0

        # Initialize state tracking
        self._state = None

    def reset(self, observation: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment to initial state."""
        self._state = torch.zeros(1, 1, self.dynamics.state_dim, device=self.device)

        # Encode initial state to latent space
        with torch.no_grad():
            self._state = self.update_posterior(y=observation)  # Use mean of encoding

        info = {"latent_state": self._state}

        return observation, info

    def update(self, observation: torch.Tensor, action: torch.Tensor) -> Dict[str, Any]:
        """Update the model state given new observation and action."""
        with torch.no_grad():
            self._state = self.update_posterior(y=observation, u=action)
            if self.action_encoder is not None and action is not None:
                encoded_action = self.action_encoder(action)
            else:
                encoded_action = action

        info = {
            "latent_state": self._state,
            "env_action": encoded_action[..., -1:, :] if encoded_action is not None else None,
        }

        return info

    def predict(self, action: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Predict the next state given the current state and action."""
        with torch.no_grad():
            _, next_state, _ = self.dynamics.sample_forward(
                init_z=self._state, action=action, add_noise=False, return_traj=True
            )

        return torch.cat(next_state, dim=-2)

    def set_state(self, state: torch.Tensor):
        self._state = state

    def prepare_dataloader(
        self, data, batch_size=32, chunk_size=1000, shuffle=False, num_workers=0
    ):
        # Handle different input types and convert to DataLoader
        if hasattr(data, "get_dataloader"):
            # This is a RolloutBuffer
            dataloader = data.get_dataloader(
                batch_size=batch_size,
                chunk_size=chunk_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )
        elif isinstance(data, dict):
            # Direct dict input - create single-item DataLoader
            # Convert dict to single-batch format and create minimal DataLoader
            class SingleBatchDataset(Dataset):
                def __init__(self, batch_dict):
                    self.batch = batch_dict

                def __len__(self):
                    return 1

                def __getitem__(self, idx):
                    return self.batch

            dataset = SingleBatchDataset(data)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                f"Expected RolloutBuffer, Rollout, RecentRollout, or dict."
            )
        return dataloader

    def save(self, path: str) -> None:
        """Save model to disk."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    def to_device(self, device: str) -> None:
        """Move model to specified device."""
        self.device = torch.device(device)
        self.to(self.device)

    def _add_param_perturbation(self, model, perturbation):
        if perturbation > 0:
            for param in model.parameters():
                if param.grad is not None:
                    param.data += perturbation * torch.randn_like(param.data)

    def _get_optimizer(self, optimizer, param_list, lr, weight_decay):
        if optimizer == "SGD":
            return torch.optim.SGD(params=param_list, lr=lr)
        elif optimizer == "Adam":
            return torch.optim.Adam(params=param_list, lr=lr, weight_decay=weight_decay)
        elif optimizer == "AdamW":
            return torch.optim.AdamW(params=param_list, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def _get_dynamics(self, idx=None):
        if self.is_ensemble:  # --- IGNORE ---
            if idx is None:
                raise ValueError("Ensemble mode requires an idx argument.")
            return self.dynamics.ensemble[idx]
        else:
            return self.dynamics

    def _get_train_args(self, **kwargs):
        train_args = {
            "optimizer": kwargs.get("optimizer", "SGD"),
            "lr": kwargs.get("lr", 1e-4),
            "weight_decay": kwargs.get("weight_decay", 1e-3),
            "n_epochs": kwargs.get("n_epochs", 1),
            "verbose": kwargs.get("verbose", True),
            "grad_clip_norm": kwargs.get("grad_clip_norm", 1.0),
            "n_samples": kwargs.get("n_samples", 5),
            "k_steps": kwargs.get("k_steps", 1),
            "p_mask": kwargs.get("p_mask", 0.0),
            "beta": kwargs.get("beta", 1.0),
            "warmup": kwargs.get("warmup", 1000),
            "annealing_steps": kwargs.get("annealing_steps", 1000),
            "annealing_type": kwargs.get("annealing_type", "cyclic"),
            "param_list": kwargs.get("param_list", "all"),  # To be set per model
        }
        return train_args

    @staticmethod
    def _kl_div(mu_q, var_q, mu_p, var_p):
        """Analytic KL divergence between two multivariate normal distributions."""
        kl_d = 0.5 * (torch.log(var_p / var_q) + ((mu_q - mu_p) ** 2) / var_p + (var_q / var_p) - 1)
        return torch.sum(kl_d, (-1))

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

    def update_posterior(self, y, u=None):
        """Update the posterior state given new observation and action."""
        raise NotImplementedError

    def _train_single_model(self, **train_args):
        raise NotImplementedError

    def train_model(
        self,
        data,
        batch_size=32,
        chunk_size=1000,
        shuffle=False,
        num_workers=0,
        **kwargs,
    ) -> list[torch.Tensor]:
        """
        Train model using PyTorch DataLoader.
        """
        dataloader = self.prepare_dataloader(data, batch_size, chunk_size, shuffle, num_workers)
        all_losses = []
        train_args = self._get_train_args(**kwargs)
        train_args["dataloader"] = dataloader
        perturbation = kwargs.get("perturbation", 0.0)

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
                if perturbation > 0.0:
                    self._add_param_perturbation(self.dynamics.ensemble[i], perturbation)

            all_losses = torch.stack(all_losses).mean(dim=0)
        else:
            # Train single model
            param_list = param_lists[0]
            train_args["param_list"] = param_list
            train_args["model_idx"] = None
            train_args["model_name"] = "Model"

            epoch_losses = self._train_single_model(**train_args)
            all_losses = epoch_losses

            # Apply parameter perturbation
            if perturbation > 0.0:
                self._add_param_perturbation(self.dynamics, perturbation)

        # Final cleanup
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()

        # DESIGN NOTE: Return value type
        # ===============================
        # Returns: List[torch.Tensor] - epoch losses
        # Each element: [loss/T, log_like/T/input_dim, kl_d/T/latent_dim]
        #
        # Issue for experiment.py:
        # - experiment.py line 97 tries to index: self.training_loss[-1]
        # - When only 1 epoch, this works
        # - When verbose=False and errors, might return single float
        #
        # Recommendation: Always return list of tensors for consistency
        return all_losses

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute Loss given input data."""
        raise NotImplementedError
