from __future__ import annotations

import torch

from actdyn.environment.vectorfield import pad_embedding_to_params
from actdyn.models.model import _kl_div_mc
from actdyn.utils.torch_utils import (
    attenuated_state_information,
    jacobian_wrt_param,
    make_uniform_sampler,
)


def test_make_uniform_sampler_shape_and_bounds():
    sampler = make_uniform_sampler(low=[-1.0, 0.0], high=[1.0, 2.0], dim=2)
    samples = sampler(128)

    assert samples.shape == (128, 2)
    assert torch.all(samples[:, 0] >= -1.0)
    assert torch.all(samples[:, 0] <= 1.0)
    assert torch.all(samples[:, 1] >= 0.0)
    assert torch.all(samples[:, 1] <= 2.0)


def test_jacobian_wrt_param_matches_expected_values():
    z = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    e = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    def fn(z_in: torch.Tensor, e_in: torch.Tensor) -> torch.Tensor:
        return torch.stack([z_in[:, 0] * e_in[:, 0], z_in[:, 1] + 2.0 * e_in[:, 1]], dim=-1)

    J = jacobian_wrt_param(fn, [z, e], argnum=1)
    assert J.shape == (2, 1, 2, 2)

    expected = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 2.0]]],
            [[[3.0, 0.0], [0.0, 2.0]]],
        ]
    )
    assert torch.allclose(J, expected)


def test_attenuated_state_information_matches_schur_complement_for_noncommuting_matrices():
    prior_cov = torch.tensor([[2.0, 0.4], [0.4, 0.8]], dtype=torch.float64)
    state_info = torch.tensor([[1.5, -0.7], [-0.7, 1.2]], dtype=torch.float64)
    precision = torch.linalg.inv(prior_cov)

    observed = attenuated_state_information(
        prior_cov.unsqueeze(0), state_info.unsqueeze(0)
    )[0]
    expected = state_info - state_info @ torch.linalg.solve(
        precision + state_info, state_info
    )
    wrong_order = torch.linalg.solve(
        torch.eye(2, dtype=torch.float64) + prior_cov @ state_info, state_info
    )

    assert not torch.allclose(wrong_order, expected)
    assert torch.allclose(observed, expected, atol=1e-10, rtol=1e-10)
    assert torch.allclose(observed, observed.T, atol=1e-12, rtol=1e-12)


def test_kl_div_mc_broadcasts_posterior_terms():
    z_prior = torch.zeros(4, 2, 3, 1)
    mu_q = torch.zeros(2, 3, 1)
    var_q = torch.ones(2, 3, 1)
    mu_p = torch.ones(2, 3, 1)
    var_p = torch.ones(2, 3, 1)

    kl = _kl_div_mc(mu_q, var_q, z_prior, mu_p, var_p)

    assert kl.shape == (2,)
    assert torch.allclose(kl, torch.full((2,), 1.5))


def test_pad_embedding_to_params_fills_fixed_tail():
    embedding = torch.tensor([[1.0, 2.0]])

    params = pad_embedding_to_params(
        embedding,
        full_params=torch.tensor([0.0, 0.0, 3.0]),
        min_embedding_dim=2,
    )

    assert torch.allclose(params, torch.tensor([[1.0, 2.0, 3.0]]))
