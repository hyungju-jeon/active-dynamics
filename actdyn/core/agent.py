from typing import Dict, Tuple
import torch
from actdyn.environment.vectorfield import VectorFieldEnv
from actdyn.metrics.uncertainty import RandomNetworkDistillation
from actdyn.models.decoder import GaussianNoise, PoissonNoise
from actdyn.utils.rollout import RecentRollout, RolloutBuffer
from actdyn.environment.env_wrapper import GymObservationWrapper
from actdyn.models.base import BaseModel
from actdyn.policy.base import BasePolicy
from actdyn.policy.mpc import BaseMPC
from actdyn.metrics.base import CompositeMetric
from actdyn.metrics.information import FisherInformationMetric
from einops import rearrange
from actdyn.utils import Transition, Belief


class Agent:
    """Agent class for active learning in dynamical systems."""

    def __init__(
        self,
        env: GymObservationWrapper,
        model: BaseModel,
        policy: BasePolicy,
        buffer_length: int = 20,
        device="cuda",
    ):
        self.env = env
        self.model = model
        self.policy = policy

        self.device = torch.device(device)

        # Buffers on GPU for training
        self.buffer_length = buffer_length
        self.recent = RecentRollout(max_len=self.buffer_length, device=device)

        # State tracking
        self._observation = None  # Current observation from the environment
        self._env_state = None  # Current environment state
        self._model_state = None  # Agent's internal state estimate

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Reset the agent and environment."""
        # Reset environment and get initial observation
        with torch.no_grad():
            obs, info = self.env.reset(seed=seed)
            # Seed action space as well for deterministic sampling
            try:
                if seed is not None and hasattr(self.env, "action_space"):
                    self.env.action_space.seed(int(seed))
            except Exception:
                pass
            self._observation = obs
            # Reset model
            _, model_info = self.model.reset(self._observation)

        # Initialize internal states
        self._env_state = info["latent_state"]
        self._model_state = model_info["latent_state"]

        # Reset recent buffer
        self.recent = RecentRollout(max_len=self.buffer_length, device=str(self.device))

        return self._observation

    def step(self, action: torch.Tensor | None = None) -> Tuple[Transition, bool]:
        """Take a step in the environment."""

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
            "model_state": self._model_state,
        }
        self.recent.add(**env_transition)

        # Update and take a step in the model environment with the action taken
        model_info = self.model.update(self.recent)
        model_transition = {
            "model_action": model_info["env_action"],  # Model encoded action g'(a_t)
            "next_model_state": model_info["latent_state"],  # Next belief state z'_{t+1}
        }
        self.recent.add(**model_transition)
        transition = {**env_transition, **model_transition}

        # Update the policy state based on the recent transition
        self.update_policy(self.recent)

        # Update observation and environment/model state
        self._observation = obs
        self._env_state = env_info["latent_state"]
        self._model_state = self.model._state

        return transition, done

    def plan(self) -> torch.Tensor:
        """Plan next action using the policy."""
        # Use policy to plan and get action
        action = self.policy(self._model_state)
        return action

    def update_policy(self, transition: Transition) -> None:
        """Update the policy based on the latest transition."""
        # Update policy if necessary (only on MPC for now)
        if isinstance(self.policy, BaseMPC):
            for metric in self.policy.metric.metric_list:
                metric.update(transition)

    def train_model(self, sampling_ratio: int = 1, **kwargs) -> Dict[str, float | torch.Tensor]:
        """Train the model using recent transitions."""
        data = self.recent.copy()
        data.downsample(n=int(sampling_ratio))

        train_info = self.model.train_model(data, batch_size=len(data), **kwargs)
        return train_info


class AsyncAgent(Agent):
    """Agent with asynchronous filtering and update frequency for Poisson observation"""

    def __init__(
        self,
        env: GymObservationWrapper,
        model: BaseModel,
        policy: BasePolicy,
        buffer_length: int = 20,
        device="cuda",
    ):
        super().__init__(env, model, policy, buffer_length, device)
        if not isinstance(self.model.decoder.noise, PoissonNoise):
            print("Warning: AsyncAgent is designed for models with Poisson observation noise.")

    def step(self, action: torch.Tensor | None = None) -> Tuple[Transition, bool]:
        """Take a step in the environment."""

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

        # Prediction update in the model with the new observation and action
        model_info = self.model.predict_state(action)
        model_info["latent_state"] = self.model.update_prediction(self.recent)

        model_transition = {
            "model_action": model_info["env_action"],  # Model encoded action g'(a_t)
            "model_state": self._model_state,  # Current belief state z'_t
            "next_model_state": model_info["latent_state"],  # Next belief state z'_{t+1}
        }
        self.recent.add(**model_transition)

        # Measurement update if enough data is collected
        if len(self.recent) >= self.buffer_length:
            self.model.update_posterior(self.recent)

        transition = {**env_transition, **model_transition}

        # Update the policy state based on the recent transition
        self.update_policy(self.recent)

        # Update observation and environment/model state
        self._observation = obs
        self._env_state = env_info["latent_state"]
        self._model_state = self.model._state

        return transition, done
