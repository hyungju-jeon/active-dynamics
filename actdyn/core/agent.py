from actdyn.utils.rollout import RecentRollout
import torch
from actdyn.environment.env_wrapper import GymObservationWrapper
from actdyn.models.model_wrapper import VAEWrapper
from actdyn.policy.base import BasePolicy


class Agent:
    """Agent class for active learning in dynamical systems.

    Args:
        env: The environment to interact with
        model_env: The internal model(VAE) for state estimation
        policy: The policy for action selection
        action_encoder: The learnable transfer function g(.)
        device (str, optional): Device to run on. Defaults to "cuda".
    """

    def __init__(
        self,
        env: GymObservationWrapper,
        model_env: VAEWrapper,
        policy: BasePolicy,
        device="cuda",
    ):
        self.env = env
        self.model_env = model_env
        self.policy = policy

        self.device = torch.device(device)

        # Buffers on GPU for training
        self.recent = RecentRollout(max_len=20, device=device)

        # State tracking
        self._observation = None
        self._model_state = None  # Agent's internal state estimate
        self._env_state = None  # Current environment state

    def reset(self):
        """Reset the agent and environment.

        Returns:
            torch.Tensor: Initial observation
        """
        # Get initial observation from environment
        with torch.no_grad():
            obs, info = self.env.reset()
            self._observation = obs.unsqueeze(0)
            _, model_info = self.model_env.reset(self._observation)
        self._env_state = info["latent_state"]
        self._model_state = model_info["latent_state"]

        self.recent = RecentRollout(max_len=20, device=self.device)

        return self._observation

    def step(self, action):
        """Take a step in the environment.

        Args:
            action: Action to take (raw action, will be encoded by action_encoder)

        Returns:
            Tuple containing:
                - obs: Next observation
                - reward: Reward received
                - done: Whether episode is done
                - info: Additional information
        """
        # Step both environments with the encoded action
        obs, reward, terminated, truncated, env_info = self.env.step(action)
        _, reward, _, _, model_info = self.model_env.step(action)
        done = terminated or truncated

        # Store transition for training
        transition = {
            "obs": self._observation,
            "next_obs": obs,
            "action": action,  # Store original action
            "env_action": env_info["env_action"],  # Store encoded action
            "model_action": model_info["env_action"],  # Store encoded action
            "reward": reward,
            "env_state": self._env_state,  # Current environment state
            "next_env_state": env_info["latent_state"],  # Next environment state
            "model_state": self._model_state,  # Current belief state
            "next_model_state": model_info["latent_state"],  # Next belief state
        }

        self.recent.add(**transition)

        # Update observation and environment/model state
        _, model_info = self.model_env.reset(obs)

        self._observation = obs
        self._model_state = model_info["latent_state"]
        self._env_state = env_info["latent_state"]

        return obs, reward, done, env_info, model_info

    def plan(self):
        """Plan next action using the policy.

        Returns:
            torch.Tensor: Selected action
        """
        # Use policy to plan and get action
        action = self.policy(self._model_state)
        return action

    def train_model(self, **kwargs):
        """Train the model using recent transitions."""
        # Sample from GPU-stored recent rollout
        batch = self.recent.as_batch()
        self.model_env.train(batch, **kwargs)
