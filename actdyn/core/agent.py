from __future__ import annotations

"""actdyn.core.agent.py: Agent class for active learning in dynamical systems."""

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from actdyn.environment.env_wrapper import EnvWrapper
from actdyn.models.base import BaseModel
from actdyn.models.decoder import PoissonNoise
from actdyn.policy.base import BasePolicy
from actdyn.policy.mpc import BaseMPC
from actdyn.utils import Transition
from actdyn.utils.rollout import RecentRollout


def _clone_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clone_value(v) for k, v in value.items()}
    if hasattr(value, 'detach'):
        return value.detach().clone()
    return value


def _clone_filter_belief_state(model: Any) -> dict[str, Any]:
    return {
        'e': _clone_value(getattr(model, 'e', None)),
        'z': _clone_value(getattr(model, 'z', None)),
        '_state': _clone_value(getattr(model, '_state', None)),
        'last_information': dict(getattr(model, 'last_information', {}) or {}),
        '_theta_score_block': _clone_value(getattr(model, '_theta_score_block', None)),
        '_theta_info_block': _clone_value(getattr(model, '_theta_info_block', None)),
        '_theta_info_diag_block': _clone_value(getattr(model, '_theta_info_diag_block', None)),
        '_theta_active_mask_block': _clone_value(
            getattr(model, '_theta_active_mask_block', None)
        ),
        '_theta_sensitivity': _clone_value(getattr(model, '_theta_sensitivity', None)),
        '_theta_block_steps': int(getattr(model, '_theta_block_steps', 0)),
    }


def _restore_filter_belief_state(model: Any, snapshot: dict[str, Any]) -> None:
    if snapshot.get('e') is not None:
        model.e = snapshot['e']
    if snapshot.get('z') is not None:
        model.z = snapshot['z']
    if snapshot.get('_state') is not None:
        model._state = snapshot['_state']
    model.last_information = dict(snapshot.get('last_information', {}) or {})
    if snapshot.get('_theta_score_block') is not None:
        model._theta_score_block = snapshot['_theta_score_block']
    if snapshot.get('_theta_info_block') is not None:
        model._theta_info_block = snapshot['_theta_info_block']
    if snapshot.get('_theta_info_diag_block') is not None:
        model._theta_info_diag_block = snapshot['_theta_info_diag_block']
    if snapshot.get('_theta_active_mask_block') is not None:
        model._theta_active_mask_block = snapshot['_theta_active_mask_block']
    if snapshot.get('_theta_sensitivity') is not None:
        model._theta_sensitivity = snapshot['_theta_sensitivity']
    model._theta_block_steps = int(snapshot.get('_theta_block_steps', 0))
    if hasattr(model, 'set_params') and getattr(model, 'e', None) is not None:
        model.set_params(model.e['m'])


def _ensure_batch_time_tensor(value: Any, *, device: Any):
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, 1, -1)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(1) if tensor.shape[0] == 1 else tensor.unsqueeze(0)
    return tensor


def _predictive_only_embedding_step(model: Any, action: Any) -> dict[str, Any]:
    required = [
        '_normalize_embedding_belief',
        '_ensure_state_belief_shapes',
        'dynamics',
        'Fz',
        'predict',
        '_project_spd',
        'z',
        'e',
    ]
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(
            'predictive_only_window requires a filtering-style model; '
            f'missing attributes: {missing}'
        )
    model._normalize_embedding_belief()
    model._ensure_state_belief_shapes(batch_size=model.e['m'].shape[0])
    q = F.softplus(model.dynamics.logvar).diag_embed().unsqueeze(0) * model.dt
    eye = torch.eye(model.latent_dim, device=model.device).unsqueeze(0).unsqueeze(0)
    action_bt = _ensure_batch_time_tensor(action, device=model.device)
    if action_bt is not None and model.action_encoder is not None:
        u_enc = model.action_encoder(action_bt, model.z['m'])
    else:
        u_enc = action_bt
    fz = model.Fz(model.z['m'], model.e['m'])
    dfdz = fz * model.dt + eye
    pred_m = model.predict(action=u_enc)
    pred_cov = model._project_spd(dfdz @ model.z['P'] @ dfdz.transpose(-1, -2) + q + 1e-6 * eye)
    model.z = {'m': pred_m.detach(), 'P': pred_cov.detach()}
    model._state = pred_m.detach()
    model.last_information = {
        'I_z_t': 0.0,
        'I_theta_t': 0.0,
        'Pz00': float(pred_cov[..., 0, 0].mean().item()),
        'Pz01': float(pred_cov[..., 0, 1].mean().item()) if pred_cov.shape[-1] > 1 else 0.0,
        'Pz11': float(pred_cov[..., 1, 1].mean().item()) if pred_cov.shape[-1] > 1 else 0.0,
    }
    return {
        'env_action': u_enc[..., -1:, :] if u_enc is not None else None,
        'latent_state': model._state,
    }


def _sync_policy_parameter_state(*, model: Any, policy: Any) -> None:
    if not bool(getattr(policy, 'owns_parameter_estimate', False)):
        return
    if not hasattr(policy, 'get_parameter_mean') or not hasattr(model, 'set_params'):
        return
    mean = policy.get_parameter_mean().detach()
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)
    if getattr(model, 'e', None) is not None and 'm' in model.e:
        model.e['m'] = mean.to(model.device).clone()
    model.set_params(mean.to(model.device))


class Agent:
    """Agent class for active learning in dynamical systems."""

    def __init__(
        self,
        env: EnvWrapper,
        model: BaseModel,
        policy: BasePolicy,
        buffer_length: int = 20,
        device='cuda',
        *,
        state_update_interval: int = 1,
        predictive_only_window: bool = False,
    ):
        self.env = env
        self.model = model
        self.policy = policy

        self.device = torch.device(device)

        # Buffers on GPU for training
        self.buffer_length = buffer_length
        self.recent = RecentRollout(max_len=self.buffer_length, device=device)

        # State tracking
        self._observation = None
        self._env_state = None
        self._model_state = None

        # Optional cadenced-state-update behavior
        self.state_update_interval = max(1, int(state_update_interval))
        self.predictive_only_window = bool(predictive_only_window)
        self._window_buffer: list[dict[str, Any]] = []
        self._window_start_snapshot: dict[str, Any] | None = None

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Reset the agent and environment."""
        with torch.no_grad():
            obs, info = self.env.reset(seed=seed)
            try:
                if seed is not None and hasattr(self.env, 'action_space'):
                    self.env.action_space.seed(int(seed))
            except (AttributeError, TypeError):
                pass
            self._observation = obs
            _, model_info = self.model.reset(self._observation)

        reset_policy_state = getattr(self.policy, 'reset_policy_state', None)
        if callable(reset_policy_state):
            reset_policy_state(seed=seed)

        self._env_state = info['latent_state']
        self._model_state = model_info['latent_state']

        self.recent = RecentRollout(max_len=self.buffer_length, device=str(self.device))
        self._window_buffer = []
        self._window_start_snapshot = (
            _clone_filter_belief_state(self.model) if self.predictive_only_window else None
        )

        beginning_of_rollout = getattr(self.policy, 'beginning_of_rollout', None)
        if callable(beginning_of_rollout):
            beginning_of_rollout(self._env_state)

        return self._observation

    def step(self, action: torch.Tensor | None = None) -> Tuple[Transition, bool]:
        """Take a step in the environment."""
        obs, reward, terminated, truncated, env_info = self.env.step(action)
        done = terminated or truncated

        policy_owns_theta = bool(getattr(self.policy, 'owns_parameter_estimate', False))
        env_transition = {
            'obs': self._observation,
            'next_obs': obs,
            'action': action,
            'env_action': env_info['env_action'],
            'reward': reward,
            'env_state': self._env_state,
            'next_env_state': env_info['latent_state'],
            'model_state': self._model_state,
        }
        self.recent.add(**env_transition)

        state_posterior_updated = False
        parameter_posterior_updated = False
        if policy_owns_theta:
            _sync_policy_parameter_state(model=self.model, policy=self.policy)

        if self.predictive_only_window and self.state_update_interval > 1:
            model_info = _predictive_only_embedding_step(self.model, action)
            self._window_buffer.append(
                {
                    'next_obs': _ensure_batch_time_tensor(obs, device=self.device),
                    'action': _ensure_batch_time_tensor(action, device=self.device),
                }
            )
            if len(self._window_buffer) >= self.state_update_interval:
                if self._window_start_snapshot is None:
                    self._window_start_snapshot = _clone_filter_belief_state(self.model)
                _restore_filter_belief_state(self.model, self._window_start_snapshot)
                if policy_owns_theta:
                    _sync_policy_parameter_state(model=self.model, policy=self.policy)
                for buffered in self._window_buffer:
                    prev_block_steps = int(getattr(self.model, '_theta_block_steps', 0))
                    self.model.update_posterior_embedding(
                        y=buffered['next_obs'],
                        u=buffered['action'],
                        update_theta=not policy_owns_theta,
                    )
                    if not policy_owns_theta:
                        update_reason = str(
                            getattr(self.model, '_last_parameter_update_reason', 'none')
                        )
                        parameter_posterior_updated = parameter_posterior_updated or (
                            update_reason != 'none'
                            or (
                                prev_block_steps
                                + 1
                                >= max(1, int(getattr(self.model, 'k_theta', 1)))
                                and int(getattr(self.model, '_theta_block_steps', 0)) == 0
                            )
                        )
                model_info['latent_state'] = self.model.get_state()
                state_posterior_updated = True
                self._window_buffer = []
                self._window_start_snapshot = _clone_filter_belief_state(self.model)
        else:
            prev_block_steps = int(getattr(self.model, '_theta_block_steps', 0))
            model_info = self.model.update(self.recent, update_theta=not policy_owns_theta)
            state_posterior_updated = True
            if not policy_owns_theta:
                update_reason = str(getattr(self.model, '_last_parameter_update_reason', 'none'))
                parameter_posterior_updated = (
                    update_reason != 'none'
                    or (
                        prev_block_steps + 1 >= max(1, int(getattr(self.model, 'k_theta', 1)))
                        and int(getattr(self.model, '_theta_block_steps', 0)) == 0
                    )
                )

        model_transition = {
            'model_action': model_info['env_action'],
            'next_model_state': model_info['latent_state'],
        }
        self.recent.add(**model_transition)
        self.update_policy(self.recent)
        parameter_update_forces_replan = bool(
            getattr(self.policy, 'force_replan_on_parameter_update', False)
        )
        if parameter_posterior_updated and parameter_update_forces_replan:
            request_replan = getattr(self.policy, 'request_replan', None)
            if callable(request_replan):
                request_replan('parameter_update')
        if policy_owns_theta:
            policy_update_info = getattr(self.policy, 'last_update_info', {}) or {}
            parameter_posterior_updated = bool(
                policy_update_info.get('parameter_posterior_updated', False)
            )
        policy_update_info = getattr(self.policy, 'last_update_info', {}) or {}
        if (
            parameter_posterior_updated
            and parameter_update_forces_replan
            and hasattr(self.policy, 'request_replan')
        ):
            policy_update_info = {
                **policy_update_info,
                'adaptive_replan_triggered': True,
                'adaptive_replan_reason': 'parameter_update',
            }

        transition = {
            **env_transition,
            **model_transition,
            'policy_action': action,
            'state_posterior_updated': state_posterior_updated,
            'parameter_posterior_updated': parameter_posterior_updated,
            'window_buffer_length': len(self._window_buffer),
            'state_update_interval': self.state_update_interval,
            'adaptive_replan_triggered': bool(
                policy_update_info.get('adaptive_replan_triggered', False)
            ),
            'adaptive_replan_reason': str(
                policy_update_info.get('adaptive_replan_reason', 'none')
            ),
            'adaptive_replan_interval': int(
                policy_update_info.get('adaptive_replan_interval', getattr(self.policy, 'chunk', 1))
            ),
            'adaptive_state_tracking_error': policy_update_info.get(
                'adaptive_state_tracking_error'
            ),
        }

        self._observation = obs
        self._env_state = env_info['latent_state']
        self._model_state = self.model.get_state()

        return transition, done

    def plan(self) -> torch.Tensor:
        """Plan next action using the policy."""
        action = self.policy(self._model_state, observed_state=self._env_state)
        return action

    def update_policy(self, rollout: RecentRollout) -> None:
        """Update the policy based on the latest transition."""
        if isinstance(self.policy, BaseMPC):
            for metric in self.policy.metric.metric_list:
                metric.update(rollout)
        update_info = None
        policy_update = getattr(type(self.policy), 'update', BasePolicy.update)
        if policy_update is not BasePolicy.update:
            update_info = self.policy.update(rollout)
        if isinstance(update_info, dict):
            setattr(self.policy, 'last_update_info', update_info)

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
        env: EnvWrapper,
        model: BaseModel,
        policy: BasePolicy,
        buffer_length: int = 20,
        device='cuda',
        *,
        state_update_interval: int = 1,
        predictive_only_window: bool = False,
    ):
        super().__init__(
            env,
            model,
            policy,
            buffer_length,
            device,
            state_update_interval=state_update_interval,
            predictive_only_window=predictive_only_window,
        )
        if not isinstance(self.model.decoder.noise, PoissonNoise):
            print('Warning: AsyncAgent is designed for models with Poisson observation noise.')

    def step(self, action: torch.Tensor | None = None) -> Tuple[Transition, bool]:
        """Take a step in the environment."""

        obs, reward, terminated, truncated, env_info = self.env.step(action)
        done = terminated or truncated

        env_transition = {
            'obs': self._observation,
            'next_obs': obs,
            'action': action,
            'env_action': env_info['env_action'],
            'reward': reward,
            'env_state': self._env_state,
            'next_env_state': env_info['latent_state'],
        }
        self.recent.add(**env_transition)

        model_info = self.model.predict_state(action)
        model_info['latent_state'] = self.model.update_prediction(self.recent)

        model_transition = {
            'model_action': model_info['env_action'],
            'model_state': self._model_state,
            'next_model_state': model_info['latent_state'],
        }
        self.recent.add(**model_transition)

        if len(self.recent) >= self.buffer_length:
            self.model.update_posterior(self.recent)

        transition = {**env_transition, **model_transition}
        self.update_policy(self.recent)
        self._observation = obs
        self._env_state = env_info['latent_state']
        self._model_state = self.model.get_state()

        return transition, done
