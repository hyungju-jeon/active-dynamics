import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
import numpy as np

from .observation import BaseObservation
from .action import BaseAction
from .vectorfield import VectorFieldEnv


class GymObservationWrapper(gym.Wrapper):
    """A wrapper that adds observation models to any gym environment.

    This wrapper allows you to use any observation model with any gym environment,
    while maintaining the original environment's interface and adding observation
    information to the info dictionary.

    Args:
        env: The gym environment to wrap
        obs_model: The observation model to use
        device: The device to use for tensors (default: "cpu")
    """

    def __init__(
        self,
        env: gym.Env,
        obs_model: BaseObservation,
        action_model: BaseAction,
        device: str = "cpu",
    ):
        super().__init__(env)
        self.obs_model = obs_model
        self.action_model = action_model
        self.device = device

        # Detect if the environment is torch-native (returns torch.Tensor from reset/step)
        self._torch_native = self._is_torch_native_env()

        # Update observation space if needed
        if hasattr(obs_model, "dy"):
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_model.dy,), dtype=np.float32
            )

    def _is_torch_native_env(self) -> bool:
        # Try a dummy reset to check if the env returns torch.Tensor
        try:
            obs, _ = self.env.reset()
            return isinstance(obs, torch.Tensor)
        except Exception:
            return False

    def _to_tensor(self, x: Any) -> torch.Tensor:
        """Convert input to torch tensor."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(x, device=self.device, dtype=torch.float32)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment and return initial observation."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Convert to tensor and apply observation model
        latent_state = self._to_tensor(obs)
        observed = self.obs_model.observe(latent_state)

        # Add observation info (all torch tensors)
        info.update(
            {
                "latent_state": latent_state,
            }
        )

        return observed, info

    def step(
        self, action: Any
    ) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """Step the environment and return observation."""
        # Pass action through action_model (should output torch.Tensor)
        env_action = self.action_model(action)
        # Only convert to numpy if env is not torch-native
        if not self._torch_native and isinstance(env_action, torch.Tensor):
            env_action = env_action.cpu().numpy()

        obs, reward, terminated, truncated, info = self.env.step(env_action)

        # Convert to tensor and apply observation model
        latent_state = self._to_tensor(obs)
        observed = self.obs_model.observe(latent_state)

        # Add observation info (all torch tensors)
        info.update(
            {
                "latent_state": latent_state,
            }
        )

        return observed, reward, terminated, truncated, info
