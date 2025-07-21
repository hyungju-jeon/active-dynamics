import pytest
import torch
from actdyn.policy.mpc import MpcICem
from actdyn.utils.rollout import RolloutBuffer
from gym.spaces import Box
import numpy as np
from actdyn.config import PolicyConfig


class MockModel:
    def __init__(self, state_dim=2, action_dim=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

    def dynamics(self, state):
        # Simple linear dynamics for testing
        return state + 0.1

    def action_encoder(self, action):
        # Simple action encoding for testing
        return action * 0.1


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def icem_config():
    config = PolicyConfig()
    config.horizon = 10
    config.num_samples = 100
    config.num_elites = 10
    config.opt_iterations = 5
    config.init_std = 0.5
    config.alpha = 0.1
    config.noise_beta = 0.0
    config.frac_prev_elites = 0.2
    config.factor_decrease_num = 1.25
    config.frac_elites_reused = 0.3
    config.use_mean_actions = True
    config.shift_elites = True
    config.keep_elites = True
    return config


def simple_cost_fn(rollout):
    # Access action data from _data dict in each Rollout
    actions = rollout.as_array("action")
    return torch.sum(actions**2, dim=-1)


@pytest.fixture
def icem_policy(mock_model, icem_config):
    mpc_params = {
        "horizon": icem_config.horizon,
        "num_samples": icem_config.num_samples,
    }
    return MpcICem(
        model=mock_model,
        icem_params=icem_config,
        mpc_params=mpc_params,
        metric=simple_cost_fn,
        device="cpu",
    )


def test_initialization(icem_policy, icem_config):
    """Test if MpcICem is initialized correctly"""
    assert icem_policy.horizon == icem_config.horizon
    assert icem_policy.num_samples == icem_config.num_samples
    assert icem_policy.num_elites == icem_config.num_elites
    assert icem_policy.opt_iter == icem_config.opt_iterations
    assert icem_policy.init_std == icem_config.init_std
    assert icem_policy.alpha == icem_config.alpha
    assert icem_policy.noise_beta == icem_config.noise_beta


def test_beginning_of_rollout(icem_policy):
    """Test beginning_of_rollout method"""
    state = torch.tensor([1.0, 2.0])
    icem_policy.beginning_of_rollout(state)

    assert icem_policy.was_reset
    assert isinstance(icem_policy.mean, torch.Tensor)
    assert isinstance(icem_policy.std, torch.Tensor)
    assert isinstance(icem_policy.elite_samples, RolloutBuffer)
    assert icem_policy.mean.shape == (icem_policy.horizon, icem_policy.action_dim)
    assert icem_policy.std.shape == (icem_policy.horizon, icem_policy.action_dim)


def test_get_init_mean(icem_policy):
    """Test get_init_mean method"""
    mean = icem_policy.get_init_mean()
    assert isinstance(mean, torch.Tensor)
    assert mean.shape == (icem_policy.horizon, icem_policy.action_dim)
    assert torch.all(mean == 0)  # Default initialization should be zeros


def test_get_init_std(icem_policy):
    """Test get_init_std method"""
    std = icem_policy.get_init_std()
    assert isinstance(std, torch.Tensor)
    assert std.shape == (icem_policy.horizon, icem_policy.action_dim)
    assert torch.all(std == icem_policy.init_std)  # Should match init_std


def test_sample_action_sequences(icem_policy):
    """Test sample_action_sequences method"""
    state = torch.tensor([1.0, 2.0])
    icem_policy.beginning_of_rollout(state)
    num_samples = 50
    actions = icem_policy.sample_action_sequences(num_samples)

    assert isinstance(actions, torch.Tensor)
    assert actions.shape == (num_samples, icem_policy.horizon, icem_policy.action_dim)


def test_simulate(icem_policy):
    """Test simulate method"""
    initial_state = torch.tensor([1.0, 2.0])
    actions = torch.randn(10, icem_policy.horizon, icem_policy.action_dim)

    rollout = icem_policy.simulate(initial_state, actions)

    assert isinstance(rollout, RolloutBuffer)
    # Check all rollouts in the buffer
    for r in rollout.buffer:
        assert "action" in r._data
        assert "model_state" in r._data
        assert "next_model_state" in r._data
        assert r._data["action"].shape == actions.shape[1:]
        assert r._data["model_state"].shape == (
            icem_policy.horizon,
            initial_state.shape[-1],
        )
        assert r._data["next_model_state"].shape == (
            icem_policy.horizon,
            initial_state.shape[-1],
        )


def test_get_action(icem_policy):
    """Test get_action method"""
    state = torch.tensor([1.0, 2.0])
    icem_policy.beginning_of_rollout(state)

    action = icem_policy.get_action(state)

    assert isinstance(action, torch.Tensor)
    assert action.shape == (icem_policy.action_dim,)


def test_action_bounds(icem_policy):
    """Test if action bounds are respected"""
    # Set different bounds for each dimension
    icem_policy.action_bounds = [
        (-1.0, 1.0),
        (-0.5, 0.5),
    ]  # Different bounds for each dimension
    state = torch.tensor([1.0, 2.0])
    icem_policy.beginning_of_rollout(state)

    action = icem_policy.get_action(state)
    # Check bounds for each dimension
    assert torch.all(action[0] >= -1.0) and torch.all(
        action[0] <= 1.0
    )  # First dimension
    assert torch.all(action[1] >= -0.5) and torch.all(
        action[1] <= 0.5
    )  # Second dimension


def test_elite_reuse(icem_policy):
    """Test if elite samples are properly reused"""
    state = torch.tensor([1.0, 2.0])
    icem_policy.beginning_of_rollout(state)

    # First action
    action1 = icem_policy.get_action(state)

    # Second action should reuse some elites
    action2 = icem_policy.get_action(state)

    assert isinstance(action1, torch.Tensor)
    assert isinstance(action2, torch.Tensor)
    assert action1.shape == action2.shape


@pytest.fixture
def icem_params():
    config = PolicyConfig()
    config.num_elites = 5
    config.num_samples = 20
    config.horizon = 10
    config.alpha = 0.1
    config.beta = 1.0
    config.device = "cpu"
    config.noise_beta = 1.0  # Use pink noise (1/f noise)
    config.init_std = 0.5  # Initial standard deviation for noise
    return config


@pytest.fixture
def mpc_params():
    config = PolicyConfig()
    config.horizon = 10
    config.num_samples = 20
    config.num_elites = 5
    config.alpha = 0.1
    config.device = "cpu"
    return config


@pytest.fixture
def icem(icem_params, mpc_params, mock_model):
    return MpcICem(
        model=mock_model,
        icem_params=icem_params,
        mpc_params=mpc_params,
        metric=simple_cost_fn,
        device="cpu",
    )


def test_icem_optimize(icem):
    """Test if get_action optimizes actions correctly"""
    state = torch.tensor([1.0, 2.0])
    icem.beginning_of_rollout(state)

    # Get optimized action
    action = icem.get_action(state)

    # Check that action has the correct shape
    assert action.shape == (icem.action_dim,)

    # Check that action is within bounds
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_icem_optimize_with_constraints(icem):
    """Test if get_action respects action bounds"""
    state = torch.tensor([1.0, 2.0])
    icem.beginning_of_rollout(state)

    # Set action bounds
    icem.action_bounds = [(-1.0, 1.0), (-1.0, 1.0)]

    # Get optimized action
    action = icem.get_action(state)

    # Check that action has the correct shape
    assert action.shape == (icem.action_dim,)

    # Check that action is within bounds
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__])
