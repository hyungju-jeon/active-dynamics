"""Check the appendix diagnostic against the implemented dynamics and known cases."""

import numpy as np
import torch

from actdyn.environment.vectorfield import jacobian_param_torch, jacobian_state_torch
from experiments.tnsre.figures.wilson_exploration import coverage, jacobians, parameter_information


def test_diagnostic_jacobians_match_simulator():
    z = np.random.default_rng(16).uniform(-3, 3, (40, 2))
    theta = np.array([12.0, 4.0, 13.0, 11.0])
    jz, jt = jacobians(z, theta)
    actual_z = jacobian_state_torch("wilson_cowan_1972", torch.tensor(z), theta, dynamics_alpha=1.0)
    actual_t = jacobian_param_torch(
        "wilson_cowan_1972", torch.tensor(z), np.tile(theta, (len(z), 1)), dynamics_alpha=1.0)
    np.testing.assert_allclose(jz, actual_z.detach().numpy(), atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(jt, actual_t.detach().numpy(), atol=3e-5, rtol=3e-5)


def test_coverage_counts_cells_once_and_does_not_clip_outside_points():
    z = np.array([[[-2., -2.], [-2., -2.], [2., 2.], [9., -2.]]])
    curve, visit, outside = coverage(z, 2)
    np.testing.assert_array_equal(curve, [[25, 25, 50, 50]])
    np.testing.assert_array_equal(visit, [[100, 0], [0, 100]])
    assert outside == 0.25


def test_down_state_has_no_local_weight_information():
    # Every weight multiplies a rate, so the drift is weight-insensitive at E = I = 0 (z = -2).
    meta = {"dt": .01, "state_noise": .1, "embedding_true": [12, 4, 13, 11],
            "observation_loading_matrix": [[1, 0], [0, 1]],
            "observation_loading_bias": [3, 3], "parameter_prior_covariance": 100.0}
    down = np.full((2, 15, 2), -2.0)
    np.testing.assert_allclose(parameter_information(down, meta), 0, atol=1e-12)
    informative = np.broadcast_to([.4, -.4], (2, 15, 2)).copy()  # rates (0.3, 0.2)
    gained = parameter_information(informative, meta)
    assert np.all(np.diff(gained, axis=1) > 0)
    np.testing.assert_array_equal(gained[0], gained[1])
