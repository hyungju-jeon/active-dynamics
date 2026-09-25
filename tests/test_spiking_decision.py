"""Contracts for the Wang (2002) spiking decision environment."""

from __future__ import annotations

import numpy as np
import pytest
import torch

brian2 = pytest.importorskip("brian2")

from actdyn.environment.spiking_decision import (  # noqa: E402
    SpikeCountObservation,
    SpikingDecisionEnv,
    WangDecisionNetwork,
    calibration_input_program,
    fit_loglinear_readout,
)


@pytest.fixture(scope="module")
def small_network():
    # 200 excitatory neurons keep the Cython build and stepping fast; pools of 30.
    net = WangDecisionNetwork(
        n_e=200, n_i=50, observed_per_pool=10, control_bin_ms=5.0, sim_dt_ms=0.1,
        stim_gain_pa=40.0, seed=0,
    )
    yield net
    net.close()


def test_network_steps_report_counts_and_latent_proxy(small_network):
    z0 = small_network.reset(0)
    assert z0.shape == (2,)
    np.testing.assert_allclose(z0, -2.0, atol=1e-6)  # gating starts at zero
    counts, z = small_network.step(np.array([1.0, 0.0]))
    assert counts.shape == (20,) and counts.dtype == np.float32
    assert np.all(counts >= 0) and np.all(counts == np.round(counts))
    assert z.shape == (2,) and np.all(np.abs(z) <= 2.0 + 1e-6)
    assert small_network.dt_latent == pytest.approx(0.05)


def test_driving_a_pool_raises_its_gating(small_network):
    small_network.reset(1)
    recent = []
    for t in range(100):  # 0.5 s of drive to pool 1
        counts, z = small_network.step(np.array([1.0, 0.0]))
        if t >= 80:
            recent.append(counts)
    assert z[0] > z[1] + 0.5
    recent = np.sum(recent, axis=0)  # counts over the last 100 ms
    assert recent[:10].sum() > recent[10:].sum()


def test_step_rejects_wrong_action_shape(small_network):
    with pytest.raises(ValueError, match="2 entries"):
        small_network.step(np.zeros(3))


def test_env_and_observation_model_pass_counts_through(small_network):
    env = SpikingDecisionEnv(small_network, action_max=1.0, reference_params=np.array([1.2, 0.8]))
    z, info = env.reset(seed=2)
    assert info["latent_state"].shape == (2,)
    weight = torch.zeros((small_network.n_observed, 2))
    bias = torch.zeros(small_network.n_observed)
    obs = SpikeCountObservation(env, weight=weight, bias=bias, dt=env.dt)
    z, _reward, _term, _trunc, info = env.step(torch.tensor([0.5, -0.5]))
    observed = obs.observe(torch.zeros(1, 1, 2))
    assert observed.shape == (1, 1, small_network.n_observed)
    np.testing.assert_array_equal(observed.reshape(-1).numpy(), info["spike_counts"])
    assert torch.equal(env.get_params(), torch.tensor([1.2, 0.8]))
    # The calibrated readout is what the estimator's decoder copies.
    assert obs.network[0].weight.shape == (small_network.n_observed, 2)


def test_calibration_program_is_seeded_and_piecewise_constant():
    a = calibration_input_program(120, hold_steps=50, seed=3)
    b = calibration_input_program(120, hold_steps=50, seed=3)
    assert a.shape == (120, 2)
    np.testing.assert_array_equal(a, b)
    assert np.all(a[:50] == a[0]) and np.all(a[50:100] == a[50])
    assert set(np.unique(a).tolist()) <= {-1.0, 0.0, 1.0}


def test_fit_loglinear_readout_recovers_planted_parameters():
    rng = np.random.default_rng(0)
    c_true = np.array([[0.8, -0.3], [-0.5, 0.6], [0.2, 0.9]])
    b_true = np.array([2.0, 2.5, 1.5])
    dt = 0.05
    z = rng.uniform(-2, 2, size=(4000, 2))
    lam = dt * np.exp(z @ c_true.T + b_true)
    counts = rng.poisson(lam)
    c, b = fit_loglinear_readout(z, counts, dt=dt, iterations=800, learning_rate=0.05, ridge=0.0)
    np.testing.assert_allclose(c.numpy(), c_true, atol=0.08)
    np.testing.assert_allclose(b.numpy(), b_true, atol=0.08)


def test_reset_with_same_seed_repeats_the_trial_exactly(small_network):
    def trial(seed, action, n):
        small_network.reset(seed)
        return np.array([small_network.step(np.array(action))[0] for _ in range(n)])

    first = trial(11, (1.0, 0.0), 40)
    trial(12, (-0.5, 0.8), 17)  # different history, ending mid-trial
    second = trial(11, (1.0, 0.0), 40)
    np.testing.assert_array_equal(first, second)
