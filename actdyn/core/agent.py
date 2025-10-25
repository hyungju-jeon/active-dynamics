from typing import Dict
import torch
from actdyn.utils.rollout import RecentRollout
from actdyn.environment.env_wrapper import GymObservationWrapper
from actdyn.models.base import BaseModel
from actdyn.policy.base import BasePolicy
from actdyn.policy.mpc import BaseMPC
from actdyn.metrics.base import CompositeMetric
from actdyn.metrics.information import FisherInformationMetric
from einops import rearrange

Belief = Dict[str, torch.Tensor]


class Agent:
    """Agent class for active learning in dynamical systems.

    Args:
        env: The environment to interact with
        model_env: The internal model(VAE) for state estimation
        policy: The policy for action selection
        action_encoder: The learnable action function g(.)
        device (str, optional): Device to run on. Defaults to "cuda".
    """

    def __init__(
        self,
        env: GymObservationWrapper,
        model: BaseModel,
        policy: BasePolicy | BaseMPC,
        buffer_length: int = 20,
        device="cuda",
    ):
        self.env = env
        self.model = model
        self.policy = policy

        self.device = torch.device(device)

        # Buffers on GPU for training
        self.buffer_length = buffer_length
        self.recent = RecentRollout(max_len=self.buffer_length, device=str(self.device))

        # State tracking
        self._observation = None  # Current observation from the environment
        self._env_state = None  # Current environment state
        self._model_state = None  # Agent's internal state estimate

    def reset(self, seed: int | None = None):
        """Reset the agent and environment.

        Returns:
            torch.Tensor: Initial observation
        """
        # Get initial observation from environment
        with torch.no_grad():
            obs, info = self.env.reset(seed=seed)
            # Seed action space as well for deterministic sampling
            try:
                if seed is not None and hasattr(self.env, "action_space"):
                    self.env.action_space.seed(int(seed))
            except Exception:
                pass
            self._observation = obs
            _, model_info = self.model.reset(self._observation)

        # Initialize internal states
        self._env_state = info["latent_state"]
        self._model_state = model_info["latent_state"]

        self.recent = RecentRollout(max_len=self.buffer_length, device=str(self.device))

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
        # Update the policy
        # self.update_policy(self.recent)

        # Step environment with the encoded action
        obs, reward, terminated, truncated, env_info = self.env.step(action)
        done = terminated or truncated

        # Store transition for training
        env_transition = {
            "obs": self._observation,  # Observation  y_t
            "next_obs": obs,  # New Observation y_{t+1}
            "action": action,  # Action a_t
            "env_action": env_info["env_action"],  # Env encoded action g(a_t)
            "reward": reward,  # Env Reward
            "env_state": self._env_state,  # Environment state z_t
            "next_env_state": env_info["latent_state"],  # Next environment state z_{t+1}
        }
        self.recent.add(**env_transition)

        # Update model with the action taken
        model_info = self.model.update(self.recent["next_obs"], self.recent["action"])
        model_transition = {
            "model_action": model_info["env_action"],  # Model encoded action g'(a_t)
            "model_state": self._model_state,  # Current belief state z'_t
            "next_model_state": model_info["latent_state"],  # Next belief state z'_{t+1}
        }
        self.recent.add(**model_transition)

        transition = {**env_transition, **model_transition}

        # Update observation and environment/model state
        self._observation = obs
        self._env_state = env_info["latent_state"]
        self._model_state = self.model._state

        return transition, done

    def plan(self):
        """Plan next action using the policy.

        Returns:
            torch.Tensor: Selected action
        """
        # Use policy to plan and get action
        action = self.policy(self._model_state)
        return action

    def update_policy(self, transition):
        """Update the policy based on the latest transition."""
        # Update policy if necessary
        if isinstance(self.policy, BaseMPC):
            if isinstance(self.policy.metric, CompositeMetric):
                for metric in self.policy.metric.metrics:
                    if isinstance(metric, FisherInformationMetric):
                        metric.update_fim(transition)
            if isinstance(self.policy.metric, FisherInformationMetric):
                self.policy.metric.update_fim(transition)

    def train_model(self, sampling_ratio=1, **kwargs) -> Dict[str, float]:
        """Train the model using recent transitions."""
        data = self.recent.copy()
        data.downsample(n=int(sampling_ratio))

        train_info = self.model.train_model(data, batch_size=len(data), **kwargs)
        return train_info
