from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader, Dataset
import gymnasium as gym

from actdyn.models.base import BaseDynamicsEnsemble
from actdyn.utils.visualize import plot_vector_field
from actdyn.utils.rollout import RecentRollout, RolloutBuffer, Rollout
from .model import SeqVae


class VAEWrapper(gym.Env):
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
        model: SeqVae,
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

    def reset(self, observation: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment to initial state."""

        # Encode initial state to latent space
        with torch.no_grad():
            self._state = self.model.encoder(y=observation)[0]  # Use mean of encoding

        info = {"latent_state": self._state}

        return observation, info

    def set_state(self, state: torch.Tensor):
        self._state = state

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the environment forward one timestep."""

        # Predict next latent state
        with torch.no_grad():
            next_state = self.model.dynamics.sample_forward(self._state, action)[0]
            # Decode next state
            next_observation = self.model.decoder(next_state)
        # Update states
        self._state = next_state

        # For now, return zero reward and not done
        # These can be modified based on your specific needs
        reward = torch.tensor(0.0, device=self.device)
        terminated = torch.tensor(False, device=self.device)
        truncated = torch.tensor(False, device=self.device)
        env_action = (
            self.model.action_encoder(action) if self.model.action_encoder is not None else action
        )
        info = {
            "latent_state": next_state,
            "env_action": env_action,
        }

        return next_observation, reward, terminated, truncated, info

    def render(self, ax=None):
        if isinstance(self.model.dynamics, BaseDynamicsEnsemble):
            plot_vector_field(self.model.dynamics.models[0], ax=ax, x_range=1, device=self.device)
        else:
            plot_vector_field(self.model.dynamics, ax=ax, x_range=1, device=self.device)

    def close(self):
        """Clean up resources."""
        return None

    def train_model(
        self, data, batch_size=32, chunk_size=1000, shuffle=False, num_workers=0, **kwargs
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

        return self.model.train_model(dataloader=dataloader, **kwargs)
