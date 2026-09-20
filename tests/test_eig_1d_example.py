from __future__ import annotations

import numpy as np
import pytest

from experiments.eig_1d_example import compute_eig_curve


def test_final_scalar_setting_switches_at_horizon_three() -> None:
    candidates = np.array([0.2, 1.5, 3., 4.5])
    curve = compute_eig_curve(candidates, theta_mean=1., theta_var=2., c=-1.6,
                              b=0., state_var=.02, state_noise=.05, dt=1., horizon=5)
    scores = .5 * np.log1p(2. * np.cumsum(curve["theta_information_steps"], axis=0))
    np.testing.assert_array_equal(candidates[scores.argmax(axis=1)], [3., 3., .2, .2, .2])


@pytest.mark.parametrize("mode", ["prediction_only", "measurement_conditioned"])
def test_sensitivity_matches_fixed_observation_finite_difference(mode) -> None:
    initial = np.array([.2, .75, 1.5, 3., 4.5], dtype=np.float64)
    cfg = dict(theta_mean=1., theta_var=2., c=-1.6, b=0., state_var=.02,
               state_noise=.05, dt=1., horizon=5, planning_rollout=mode)
    curve = compute_eig_curve(initial, **cfg)
    fixed_observations = np.exp(cfg["c"] * curve["z_path"][1:] + cfg["b"]) * cfg["dt"]

    def replay(theta):
        z = initial.copy()
        p = np.full_like(z, cfg["state_var"])
        predicted, carried = [], []
        for observation in fixed_observations:
            a = 1 + cfg["dt"] * theta * np.cos(theta*z)
            p_minus = a*a*p + cfg["state_noise"]*cfg["dt"]
            z_minus = z + cfg["dt"] * np.sin(theta*z)
            predicted.append(z_minus.copy())
            if mode == "measurement_conditioned":
                mean = np.exp(cfg["c"]*z_minus + cfg["b"]) * cfg["dt"]
                p = p_minus / (1 + p_minus * cfg["c"]**2 * mean)
                z = z_minus + p * cfg["c"] * (observation-mean)
            else:
                p, z = p_minus, z_minus
            carried.append(z.copy())
        return np.stack(predicted), np.stack(carried)

    epsilon = 1e-6
    plus, minus = replay(1.+epsilon), replay(1.-epsilon)
    for index, field in enumerate(["sensitivity_path", "carried_sensitivity_path"]):
        fd = (plus[index]-minus[index]) / (2*epsilon)
        np.testing.assert_allclose(curve[field][1:], fd, rtol=2e-6, atol=2e-7)
        assert curve[field].shape == (6, len(initial))
        assert curve[field].dtype == np.float64


def test_prediction_only_covariance_carries_prior_forward() -> None:
    initial = np.array([.2, 1.5, 3., 4.5])
    curve = compute_eig_curve(initial, theta_mean=1., theta_var=2., c=-1.6, b=0.,
                              state_var=.02, state_noise=.05, dt=1., horizon=5,
                              planning_rollout="prediction_only")
    expected = np.full_like(initial, .02)
    for k in range(5):
        expected = (1+np.cos(curve["z_path"][k]))**2 * expected + .05
        np.testing.assert_allclose(curve["state_variance_path"][k+1], expected)
    np.testing.assert_allclose(curve["carried_state_variance_path"], curve["state_variance_path"])


def test_eig_curve_matches_closed_form_without_state_uncertainty() -> None:
    z = np.array([0.0, 0.7, 1.4])
    theta_mean = 0.9
    theta_var = 0.4
    c = 1.2
    b = -0.1

    dt = 0.1
    curve = compute_eig_curve(
        z,
        theta_mean=theta_mean,
        theta_var=theta_var,
        c=c,
        b=b,
        state_var=0.0,
        state_noise=0.0,
        dt=dt,
    )

    z_next = z + dt * np.sin(z * theta_mean)
    sensitivity = dt * z * np.cos(z * theta_mean)
    expected_state_info = c**2 * np.exp(c * z_next + b) * dt
    expected_fisher = sensitivity**2 * expected_state_info
    expected_eig = 0.5 * np.log1p(theta_var * expected_fisher)
    assert np.allclose(curve["state_information_steps"][0], expected_state_info)
    assert np.allclose(curve["theta_fisher"], expected_fisher)
    assert np.allclose(curve["eig"], expected_eig)
    assert curve["eig"][0] == 0.0


def test_two_step_eig_matches_sensitivity_recursion() -> None:
    z0 = np.array([-0.4, 0.8])
    theta_mean = 1.1
    theta_var = 0.7
    c = 1.3
    b = -0.2

    dt = 0.1
    curve = compute_eig_curve(
        z0,
        theta_mean=theta_mean,
        theta_var=theta_var,
        c=c,
        b=b,
        state_var=0.0,
        horizon=2,
        state_noise=0.0,
        dt=dt,
    )

    z1 = z0 + dt * np.sin(z0 * theta_mean)
    s1 = dt * z0 * np.cos(z0 * theta_mean)
    fisher1 = s1**2 * c**2 * np.exp(c * z1 + b) * dt

    transition_z2 = 1.0 + dt * theta_mean * np.cos(z1 * theta_mean)
    residual_theta2 = dt * z1 * np.cos(z1 * theta_mean)
    z2 = z1 + dt * np.sin(z1 * theta_mean)
    s2 = transition_z2 * s1 + residual_theta2
    fisher2 = s2**2 * c**2 * np.exp(c * z2 + b) * dt

    expected_fisher = fisher1 + fisher2
    expected_eig = 0.5 * np.log1p(theta_var * expected_fisher)
    assert np.allclose(curve["sensitivity_path"][0], 0.0)
    assert np.allclose(curve["sensitivity_path"][1], s1)
    assert np.allclose(curve["sensitivity_path"][2], s2)
    assert np.allclose(
        curve["theta_information_steps"].sum(axis=0),
        curve["theta_fisher"],
    )
    assert np.allclose(curve["theta_fisher"], expected_fisher)
    assert np.allclose(curve["eig"], expected_eig)


def test_state_covariance_path_uses_prior_then_posterior_update() -> None:
    z0 = np.array([0.2, 0.6])
    theta_mean = 1.3
    c = 1.3
    b = -0.2
    state_var = 0.4
    state_noise = 0.05

    dt = 0.1
    curve = compute_eig_curve(
        z0,
        theta_mean=theta_mean,
        theta_var=0.7,
        c=c,
        b=b,
        state_var=state_var,
        horizon=2,
        state_noise=state_noise,
        dt=dt,
    )

    transition_z1 = 1.0 + dt * theta_mean * np.cos(z0 * theta_mean)
    p1_prior = transition_z1**2 * state_var + state_noise * dt
    z1 = z0 + dt * np.sin(z0 * theta_mean)
    state_info1 = c**2 * np.exp(c * z1 + b) * dt
    p1_posterior = p1_prior / (1.0 + p1_prior * state_info1)
    transition_z2 = 1.0 + dt * theta_mean * np.cos(z1 * theta_mean)
    p2_prior = transition_z2**2 * p1_posterior + state_noise * dt

    assert np.allclose(curve["state_variance_path"][0], state_var)
    assert np.allclose(curve["state_variance_path"][1], p1_prior)
    assert np.allclose(curve["state_variance_path"][2], p2_prior)


def test_state_noise_attenuates_parameter_information() -> None:
    z = np.linspace(-2.0, 2.0, 101)
    full = compute_eig_curve(
        z,
        theta_mean=1.1,
        theta_var=0.5,
        c=1.4,
        b=-0.3,
        state_var=0.0,
        horizon=3,
        state_noise=0.0,
    )
    noisy = compute_eig_curve(
        z,
        theta_mean=1.1,
        theta_var=0.5,
        c=1.4,
        b=-0.3,
        state_var=0.0,
        horizon=3,
        state_noise=0.2,
    )

    assert np.all(noisy["theta_fisher"] <= full["theta_fisher"] + 1e-12)
    assert np.max(noisy["eig"]) > 0.0


def test_conditioned_information_matches_joint_gaussian_likelihood() -> None:
    """The innovation sum must match a joint likelihood with correlated states."""
    initial = np.array([.2, 1.5, 3., 4.5])
    theta, dt, p0, q, c = .9, .7, .12, .05, -1.6
    curve = compute_eig_curve(initial, theta_mean=theta, theta_var=2., c=c,
                              b=0., state_var=p0, state_noise=q, dt=dt,
                              horizon=2, planning_rollout="measurement_conditioned")
    for i, z0 in enumerate(initial):
        z1 = z0 + dt*np.sin(theta*z0)
        z2 = z1 + dt*np.sin(theta*z1)
        a1, a2 = 1+dt*theta*np.cos(theta*np.array([z0, z1]))
        b1, b2 = dt*np.array([z0, z1])*np.cos(theta*np.array([z0, z1]))
        # Independent initial/process noise induces off-diagonal covariance
        # between the two observations in the frozen local Gaussian model.
        noise_map = np.array([[a1, 1., 0.], [a2*a1, a2, 1.]])
        state_cov = (noise_map*np.array([p0, q*dt, q*dt])) @ noise_map.T
        observation_variance = 1/(c*c*dt*np.exp(c*np.array([z1, z2])))
        joint_cov = state_cov + np.diag(observation_variance)
        joint_mean_derivative = np.array([b1, a2*b1+b2])
        joint_information = joint_mean_derivative @ np.linalg.solve(
            joint_cov, joint_mean_derivative)
        np.testing.assert_allclose(curve["theta_fisher"][i], joint_information,
                                   rtol=1e-12, atol=1e-12)
