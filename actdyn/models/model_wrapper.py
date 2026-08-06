import os
from typing import Tuple, Dict, Any
import torch
import gymnasium as gym

from actdyn.models.base import BaseDynamicsEnsemble
from actdyn.utils.plotting import plot_vector_field
from .base import BaseModel


class ModelWrapper(gym.Env):
    """A wrapper class that converts a VAE model into a gym-like environment.

    This wrapper allows the VAE model to be used as a simulated environment for model-based RL.
    It handles state encoding/decoding and dynamics prediction in a gym-like interface.

    Args:
        model (Union[VAE, EnsembleVAE]): The VAE model to wrap
        observation_space (gym.Space): The observation space of the environment
        action_space (gym.Space): The action space of the environment
        device (str, optional): Device to run the model on. Defaults to "cpu".
    """

    def __init__(
        self,
        model: BaseModel,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
    ):
        super().__init__()
        self.model = model
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)

        # Initialize state tracking
        self._state = None

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the wrapped model."""
        if name != "model" and hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def reset(self, observation: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment to initial state."""

        # Encode initial state to latent space
        with torch.no_grad():
            _samples, mu, _var = self.model.encoder(y=observation, n_samples=1)
            self._state = mu[:, -1:, :]
            self.model.set_state(self._state)

        info = {"latent_state": self._state}

        return observation, info

    def set_state(self, state: torch.Tensor):
        self._state = state
        self.model.set_state(state)

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the environment forward one timestep."""

        # Predict next latent state
        env_action = (
            self.model.action_encoder(action, self._state)
            if self.model.action_encoder is not None
            else action
        )

        with torch.no_grad():
            next_state = self.model.dynamics.sample_forward(self._state, env_action)[0]
            # Decode next state
            next_observation = self.model.decoder(next_state)
        # Update states
        self._state = next_state
        self.model.set_state(next_state)

        # For now, return zero reward and not done
        # These can be modified based on your specific needs
        reward = torch.tensor(0.0, device=self.device)
        terminated = torch.tensor(False, device=self.device)
        truncated = torch.tensor(False, device=self.device)
        info = {
            "latent_state": next_state,
            "env_action": env_action,
        }

        return next_observation, reward, terminated, truncated, info

    def render(self, ax=None):
        if isinstance(self.model.dynamics, BaseDynamicsEnsemble):
            plot_vector_field(self.model.dynamics.ensemble[0], ax=ax, x_range=1, device=self.device)
        else:
            plot_vector_field(self.model.dynamics, ax=ax, x_range=1, device=self.device)

    def close(self):
        """Clean up resources."""
        return None

    def train_model(
        self, data, batch_size=32, chunk_size=1000, shuffle=False, num_workers=0, **kwargs
    ):
        return self.model.train_model(
            data,
            batch_size=batch_size,
            chunk_size=chunk_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path: str):
        self.model.load(path)

    def save_model(self, path: str):
        self.save(path)
