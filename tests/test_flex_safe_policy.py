from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from actdyn.policy import baseline_flex
from actdyn.policy.baseline_flex import FLEXPolicy, FLEXUpstreamPolicy


class _JumpingFlexAgent:
    def __init__(
        self,
        model: torch.nn.Module,
        d: int,
        m: int,
        gamma: float,
        *,
        dt: float,
        regularization: float,
    ) -> None:
        self.model = model
        self.m = int(m)
        n_params = sum(int(param.numel()) for param in model.parameters())
        self.M = np.eye(n_params, dtype=np.float64) * float(regularization)
        self.M_inv = np.eye(n_params, dtype=np.float64) / float(regularization)

    def policy(self, x: np.ndarray, t: int) -> np.ndarray:
        return np.zeros(self.m, dtype=np.float32)

    def learning_step(self, x: np.ndarray, u: np.ndarray, dx_dt: np.ndarray) -> None:
        self.M = self.M + np.eye(self.M.shape[0], dtype=np.float64)
        self.M_inv = np.linalg.pinv(self.M)
        for param in self.model.parameters():
            param.data.add_(10.0)


def _env_preset() -> SimpleNamespace:
    return SimpleNamespace(
        dt=0.01,
        dynamics_alpha=1.0,
        latent_dim=2,
        action_dim=2,
        action_max=1.0,
        embedding_dim=4,
        resolved_true_params=lambda estimator=False: (-1.2, -0.8, 0.5, 1.1),
        resolved_dynamics_type=lambda estimator=False: "gated_duffing",
        true_embedding_vector=lambda embedding_dim=None, estimator=False: np.array(
            [-1.2, -0.8, 0.5, 1.1], dtype=np.float32
        )[:embedding_dim],
    )


def test_flex_safe_rolls_back_unstable_update(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(baseline_flex, "_FLEX_POLICY_CLASS", _JumpingFlexAgent)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    policy = FLEXPolicy(
        action_space=action_space,
        model=SimpleNamespace(e={"m": torch.zeros(1, 4)}),
        env_preset=_env_preset(),
        rollback_unstable_update=True,
        regularization=0.1,
        parameter_step_clip=0.25,
    )
    agent = policy._flex_agent
    assert isinstance(agent, _JumpingFlexAgent)
    previous_mean = policy.get_parameter_mean().detach().clone()
    previous_gram = np.asarray(agent.M).copy()

    info = policy.update(
        {
            "model_state": torch.zeros(1, 1, 2),
            "next_model_state": torch.ones(1, 1, 2),
            "env_action": torch.zeros(1, 1, 2),
        }
    )

    np.testing.assert_allclose(policy.get_parameter_mean().numpy(), previous_mean.numpy())
    np.testing.assert_allclose(agent.M, previous_gram)
    assert info["parameter_posterior_updated"] is False
    assert info["flex_update_rejected"] is True


def test_existing_flex_clips_parameters_but_keeps_gram_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baseline_flex, "_FLEX_POLICY_CLASS", _JumpingFlexAgent)
    policy = FLEXPolicy(
        action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        model=SimpleNamespace(e={"m": torch.zeros(1, 4)}),
        env_preset=_env_preset(),
        regularization=0.01,
        parameter_step_clip=0.25,
    )
    agent = policy._flex_agent
    previous_mean = policy.get_parameter_mean().detach().clone()
    previous_gram = np.asarray(agent.M).copy()

    info = policy.update(
        {
            "model_state": torch.zeros(1, 1, 2),
            "next_model_state": torch.ones(1, 1, 2),
            "env_action": torch.zeros(1, 1, 2),
        }
    )

    update = policy.get_parameter_mean() - previous_mean
    assert torch.linalg.norm(update).item() == pytest.approx(0.25)
    np.testing.assert_allclose(agent.M, previous_gram + np.eye(previous_gram.shape[0]))
    assert info["parameter_posterior_updated"] is True
    assert info["flex_update_rejected"] is False


def test_upstream_flex_accepts_raw_parameter_and_gram_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baseline_flex, "_FLEX_POLICY_CLASS", _JumpingFlexAgent)
    policy = FLEXUpstreamPolicy(
        action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        model=SimpleNamespace(e={"m": torch.zeros(1, 4)}),
        env_preset=_env_preset(),
        regularization=0.01,
        parameter_step_clip=None,
        parameter_min=None,
        parameter_max=None,
    )
    agent = policy._flex_agent
    previous_mean = policy.get_parameter_mean().detach().clone()
    previous_gram = np.asarray(agent.M).copy()

    info = policy.update(
        {
            "model_state": torch.zeros(1, 1, 2),
            "next_model_state": torch.ones(1, 1, 2),
            "env_action": torch.zeros(1, 1, 2),
        }
    )

    np.testing.assert_allclose(
        policy.get_parameter_mean().numpy(),
        previous_mean.numpy() + 10.0,
    )
    np.testing.assert_allclose(agent.M, previous_gram + np.eye(previous_gram.shape[0]))
    assert info["flex_update_norm"] == pytest.approx(20.0)
    assert info["parameter_posterior_updated"] is True
    assert info["flex_update_rejected"] is False
