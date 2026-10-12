"""Ball constraint ||u|| <= r for the baselines (PRBS, Random, RHC-US) and the runner's radius rule."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from actdyn.policy.baseline_prbs import BaselinePRBSPolicy
from actdyn.policy.baseline_random import BaselineRandomPolicy
from actdyn.policy.policy import RandomPolicy


def _box(bound=1.25, dim=2):
    return gym.spaces.Box(low=-bound, high=bound, shape=(dim,), dtype=np.float32)


def _actions(policy, n):
    return np.stack([policy.get_action(torch.zeros(1, 2))[0].reshape(-1).numpy() for _ in range(n)])


def test_prbs_ball_levels_lie_on_the_ball_and_box_levels_on_the_corners():
    ball = _actions(BaselinePRBSPolicy(_box(), hold_steps=1, seed=0, action_radius=1.25), 200)
    np.testing.assert_allclose(np.linalg.norm(ball, axis=1), 1.25, rtol=1e-6)
    np.testing.assert_allclose(np.abs(ball), 1.25 / np.sqrt(2), rtol=1e-6)
    box = _actions(BaselinePRBSPolicy(_box(), hold_steps=1, seed=0), 200)
    np.testing.assert_allclose(np.abs(box), 1.25, rtol=1e-6)


@pytest.mark.parametrize("policy_class", [RandomPolicy, BaselineRandomPolicy])
def test_random_ball_is_uniform_in_the_disk(policy_class):
    space = _box()
    space.seed(0)
    policy = policy_class(space, action_radius=1.25)
    norms = np.linalg.norm(_actions(policy, 4000), axis=1)
    assert norms.max() <= 1.25 + 1e-6
    # Uniform in a disk of radius r: E||u|| = 2r/3, and a quarter of the mass lies within r/2.
    assert norms.mean() == pytest.approx(2 * 1.25 / 3, abs=0.02)
    assert np.mean(norms < 0.625) == pytest.approx(0.25, abs=0.03)
    box_space = _box()
    box_space.seed(0)
    box = _actions(policy_class(box_space), 4000)
    assert np.abs(box).max() <= 1.25 + 1e-6 and np.linalg.norm(box, axis=1).max() > 1.3


def test_rhc_ball_constraint_bounds_every_planned_input():
    pytest.importorskip("casadi")
    from actdyn.policy.baseline_rhc import RecedingHorizonCuriosityPolicy

    kwargs = dict(action_space=_box(1.0), horizon=4, num_features=8, planner_maxiter=50,
                  optimize_hyperparams=False, device="cpu", seed=0)
    ball, _ = RecedingHorizonCuriosityPolicy(action_radius=1.0, **kwargs).get_action(
        torch.zeros(1, 2), observed_state=torch.zeros(1, 2))
    assert np.linalg.norm(ball[0].numpy(), axis=1).max() <= 1.0 + 1e-6
    box, _ = RecedingHorizonCuriosityPolicy(**kwargs).get_action(torch.zeros(1, 2), observed_state=torch.zeros(1, 2))
    assert np.abs(box[0].numpy()).max() <= 1.0 + 1e-6
    assert np.linalg.norm(box[0].numpy(), axis=1).max() > 1.0 + 1e-3  # the box plan leaves the ball


def test_runner_ball_radius_follows_the_policy_constraint():
    from experiments.run import _action_ball_radius

    preset = SimpleNamespace(action_max=1.25)
    assert _action_ball_radius(SimpleNamespace(action_constraint="radius", action_radius=None), preset) == 1.25
    assert _action_ball_radius(SimpleNamespace(action_constraint="radius", action_radius=0.5), preset) == 0.5
    assert _action_ball_radius(SimpleNamespace(action_constraint=None, action_radius=None), preset) is None
    assert _action_ball_radius(SimpleNamespace(action_constraint="box", action_radius=None), preset) is None
