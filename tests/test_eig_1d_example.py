from __future__ import annotations

import numpy as np

from experiments.eig_1d_example import compute_eig_curve


def test_eig_curve_matches_closed_form_without_state_uncertainty() -> None:
    z = np.array([0.0, 0.7, 1.4])
    theta_mean = 0.9
    theta_var = 0.4
    c = 1.2
    b = -0.1

    curve = compute_eig_curve(
        z,
        theta_mean=theta_mean,
        theta_var=theta_var,
        c=c,
        b=b,
        state_var=0.0,
        state_noise=0.0,
    )

    z_next = z + np.sin(z * theta_mean)
    sensitivity = z * np.cos(z * theta_mean)
    expected_state_info = c**2 * np.exp(c * z_next + b)
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

    curve = compute_eig_curve(
        z0,
        theta_mean=theta_mean,
        theta_var=theta_var,
        c=c,
        b=b,
        state_var=0.0,
        horizon=2,
        state_noise=0.0,
    )

    z1 = z0 + np.sin(z0 * theta_mean)
    s1 = z0 * np.cos(z0 * theta_mean)
    fisher1 = s1**2 * c**2 * np.exp(c * z1 + b)

    transition_z2 = 1.0 + theta_mean * np.cos(z1 * theta_mean)
    residual_theta2 = z1 * np.cos(z1 * theta_mean)
    z2 = z1 + np.sin(z1 * theta_mean)
    s2 = transition_z2 * s1 + residual_theta2
    fisher2 = s2**2 * c**2 * np.exp(c * z2 + b)

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

    curve = compute_eig_curve(
        z0,
        theta_mean=theta_mean,
        theta_var=0.7,
        c=c,
        b=b,
        state_var=state_var,
        horizon=2,
        state_noise=state_noise,
    )

    transition_z1 = 1.0 + theta_mean * np.cos(z0 * theta_mean)
    p1_prior = transition_z1**2 * state_var + state_noise
    z1 = z0 + np.sin(z0 * theta_mean)
    state_info1 = c**2 * np.exp(c * z1 + b)
    p1_posterior = p1_prior / (1.0 + p1_prior * state_info1)
    transition_z2 = 1.0 + theta_mean * np.cos(z1 * theta_mean)
    p2_prior = transition_z2**2 * p1_posterior + state_noise

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
