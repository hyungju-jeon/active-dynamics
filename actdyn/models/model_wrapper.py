import torch
import gymnasium as gym
from typing import Tuple, Dict, Any

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
        """Reset the environment to initial state.

        Args:
            seed (int, optional): Random seed for reproducibility
            options (Dict[str, Any], optional): Additional options for reset

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]:
                - Initial observation
                - Additional information
        """

        # Encode initial state to latent space
        with torch.no_grad():
            self._state = self.model.encoder(observation)[0]  # Use mean of encoding

        info = {
            "latent_state": self._state,
        }

        return observation, info

    def set_state(self, state: torch.Tensor):
        self._state = state

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the environment forward one timestep.

        Args:
            action (torch.Tensor): Action to take

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
                - observation: Next observation
                - reward: Reward (always 0 for now)
                - terminated: Whether episode is terminated
                - truncated: Whether episode is truncated
                - info: Additional information
        """

        # Predict next latent state
        with torch.no_grad():
            next_state = self.model.dynamics(self._state, action)
            # Decode next state
            next_observation = self.model.decoder(next_state)

        # Update states
        self._state = next_state

        # For now, return zero reward and not done
        # These can be modified based on your specific needs
        reward = torch.tensor(0.0, device=self.device)
        terminated = torch.tensor(False, device=self.device)
        truncated = torch.tensor(False, device=self.device)
        env_action = self.model.action_encoder(action)
        info = {
            "latent_state": next_state,
            "env_action": env_action,
        }

        return next_observation, reward, terminated, truncated, info

    def render(self):
        """Render the current state (not implemented)."""
        raise NotImplementedError("Rendering not implemented for VAE environment")

    def close(self):
        """Clean up resources."""
        pass

    def train(self, data, **kwargs):
        return self.model.train(data, **kwargs)
