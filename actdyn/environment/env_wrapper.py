import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, SupportsFloat
import numpy as np

from .observation import BaseObservation
from .action import BaseAction


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
        dt: float,
        device: str = "cpu",
    ):
        super().__init__(env)
        self.obs_model = obs_model
        self.action_model = action_model
        self.dt = dt

        # Auto-detect device from observation model if available
        if hasattr(obs_model, "network") and obs_model.network is not None:
            try:
                detected_device = next(obs_model.network.parameters()).device
                self.device = detected_device
            except StopIteration:
                # Network has no parameters, use provided device
                self.device = torch.device(device)
        else:
            self.device = torch.device(device)

        # Ensure action model is on the same device
        if hasattr(action_model, "to"):
            action_model.to(self.device)

        # Detect if the environment is torch-native (returns torch.Tensor from reset/step)
        self._torch_native = self._is_torch_native_env()

        # Update observation space if needed
        if hasattr(obs_model, "obs_dim"):
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_model.obs_dim,), dtype=np.float32
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
        # Get the device from the observation model's network if available
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        else:
            x = torch.tensor(x, device=self.device, dtype=torch.float32)

        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch dimension if needed
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # Add time dimension if needed

        return x

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reset the environment and return initial observation."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Convert to tensor and apply observation model
        if "latent_state" in info:
            latent_state = self._to_tensor(info["latent_state"])
        else:
            latent_state = self._to_tensor(obs)
        observed = self.obs_model.observe(self._to_tensor(obs))

        # Add observation info (all torch tensors)
        info.update(
            {
                "latent_state": latent_state,
            }
        )

        return observed, info

    def step(self, action: Any) -> Tuple[torch.Tensor, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Step the environment and return observation."""
        # Convert action to tensor on correct device before passing to action_model
        action_tensor = self._to_tensor(action)
        # Pass action through action_model (should output torch.Tensor)
        env_action = self.action_model(action_tensor)

        # Only convert to numpy if env is not torch-native
        if not self._torch_native and isinstance(env_action, torch.Tensor):
            env_action = env_action.cpu().numpy()[0, 0]

        obs, reward, terminated, truncated, info = self.env.step(env_action)

        # Convert to tensor and apply observation model
        # if info has "latent_state", use it directly
        if "latent_state" in info:
            latent_state = self._to_tensor(info["latent_state"])
        else:
            latent_state = self._to_tensor(obs)
        observed = self.obs_model.observe(self._to_tensor(obs))

        # Add observation info (all torch tensors)
        info.update(
            {
                "latent_state": latent_state,
                "env_action": (
                    self._to_tensor(env_action)
                    if isinstance(env_action, (np.ndarray, list))
                    else env_action
                ),
            }
        )

        return observed, reward, terminated, truncated, info
