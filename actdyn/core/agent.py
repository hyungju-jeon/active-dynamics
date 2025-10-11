from typing import Optional, Dict, Any, Tuple, Union
import torch
from actdyn.utils.rollout import RecentRollout
from actdyn.environment.env_wrapper import GymObservationWrapper
from actdyn.models.model_wrapper import VAEWrapper
from actdyn.policy.base import BasePolicy
from actdyn.policy.mpc import BaseMPC
from actdyn.metrics.base import CompositeMetric
from actdyn.metrics.information import FisherInformationMetric


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
        model_env: VAEWrapper,
        policy: Union[BasePolicy, BaseMPC],
        buffer_length: int = 20,
        device: str = "cuda",
    ):
        self.env = env
        self.model_env = model_env
        self.policy = policy

        self.device = torch.device(device)

        # Buffers on GPU for training
        self.buffer_length = buffer_length
        self.recent = RecentRollout(max_len=self.buffer_length, device=str(self.device))

        # State tracking
        self._observation: Optional[torch.Tensor] = None  # Current observation from the environment
        self._env_state: Optional[torch.Tensor] = None  # Current environment state
        self._model_state: Optional[torch.Tensor] = None  # Agent's internal state estimate

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """Reset the agent and environment.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Initial observation tensor
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
            _, model_info = self.model_env.reset(self._observation)

        # Initialize internal states
        self._env_state = info["latent_state"]
        self._model_state = model_info["latent_state"]

        self.recent = RecentRollout(max_len=self.buffer_length, device=str(self.device))

        return self._observation

    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], bool]:
        """Take a step in the environment.

        Args:
            action: Action to take (raw action, will be encoded by action_encoder)

        Returns:
            Tuple containing:
                - transition: Dictionary with transition data
                - done: Whether episode is done
        """
        # Step both environments with the encoded action
        obs, reward, terminated, truncated, env_info = self.env.step(action)
        _, _, _, _, model_info = self.model_env.step(action)
        done = terminated or truncated

        # Store transition with optimized tensor handling
        transition = {
            "obs": self._observation,  # Observation  y_t
            "next_obs": obs,  # New Observation y_{t+1}
            "action": action,  # Action a_t
            "env_action": env_info["env_action"],  # Env encoded action g(a_t)
            "model_action": model_info["env_action"],  # Model encoded action g'(a_t)
            "reward": reward,  # Env Reward
            "env_state": self._env_state,  # Environment state z_t
            "next_env_state": env_info["latent_state"],  # Next environment state z_{t+1}
            "model_state": self._model_state,  # Current belief state z'_t
            "next_model_state": model_info["latent_state"],  # Next belief state z'_{t+1}
        }

        self.recent.add(**transition)

        # Optimize model state update - avoid redundant encoder calls
        if self._model_state is not None:
            # More efficient: update state incrementally rather than full encoder pass
            self._model_state = model_info["latent_state"]
        else:
            # Fallback to encoder if state is None
            obs_data = self.recent.get("obs")
            if obs_data is not None:
                current_obs = obs_data[-1:, :]  # Get last observation
                with torch.no_grad():  # No gradients needed for state update
                    self._model_state = self.model_env.model.encoder(y=current_obs)[1][:, -1:, :]

        # Update states efficiently
        self._observation = obs
        self._env_state = env_info["latent_state"]

        return transition, done

    def plan(self) -> torch.Tensor:
        """Plan next action using the policy.

        Returns:
            Selected action tensor
        """
        # Use policy to plan and get action
        action = self.policy(self._model_state)
        return action

    def update_policy(self, transition: Dict[str, torch.Tensor]) -> None:
        """Update the policy based on the latest transition.
        
        Args:
            transition: Dictionary containing transition data
        """
        # Update policy if necessary
        if isinstance(self.policy, BaseMPC):
            if isinstance(self.policy.metric, CompositeMetric):
                for metric in self.policy.metric.metrics:
                    if isinstance(metric, FisherInformationMetric):
                        metric.update_fim(transition)
            if isinstance(self.policy.metric, FisherInformationMetric):
                self.policy.metric.update_fim(transition)

    def train_model(self, sampling_ratio: float = 1, **kwargs) -> list:
        """Train the model using recent transitions.
        
        Args:
            sampling_ratio: Ratio for downsampling the data
            **kwargs: Additional training parameters
            
        Returns:
            List of training losses [elbo, log_likelihood, kl_divergence]
        """
        data = self.recent.copy()
        data.downsample(n=int(sampling_ratio))

        elbo = self.model_env.train_model(data, batch_size=len(data), **kwargs)
        return elbo
