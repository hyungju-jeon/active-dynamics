from __future__ import annotations

import torch
import pytest

from actdyn.models.decoder import Decoder, GaussianNoise, LinearMapping
from actdyn.models.dynamics import FunctionDynamics
from actdyn.models.model import FilteringEmbedding
from actdyn.utils.torch_utils import safe_cholesky, symmetrize


def _fe_identity(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    if z.ndim == 2:
        z = z.unsqueeze(0)
    batch, time, _ = z.shape
    out = torch.zeros(batch, time, 2, 2, device=z.device)
    out[..., 0, 0] = 1.0
    out[..., 1, 1] = 1.0
    return out


def _fz_zero(z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    if z.ndim == 2:
        z = z.unsqueeze(0)
    batch, time, _ = z.shape
    return torch.zeros(batch, time, 2, 2, device=z.device)


def _zero_dynamics(z: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(z)


def _build_model(
    q_theta: float = 1e-4,
    k_theta: int = 2,
    *,
    adaptive_update: bool = False,
    adaptive_update_min_interval: int = 1,
    adaptive_update_eig_threshold: float | None = None,
    learning_sensitivity: str = "measurement_corrected",
) -> FilteringEmbedding:
    dynamics = FunctionDynamics(state_dim=2, dynamics_fn=_zero_dynamics, dt=0.1, device="cpu")
    dynamics.logvar = torch.nn.Parameter(torch.log(torch.ones(1, 2) * 0.01))

    mapping = LinearMapping(latent_dim=2, obs_dim=2, device="cpu")
    with torch.no_grad():
        mapping.network.weight.copy_(torch.eye(2))
        mapping.network.bias.zero_()
    decoder = Decoder(mapping=mapping, noise=GaussianNoise(obs_dim=2, sigma=0.1, device="cpu"))

    sigma0 = 0.1
    e_bel = {
        "m": torch.zeros(1, 2),
        "P": sigma0 * torch.eye(2).unsqueeze(0),
        "L": (1.0 / sigma0) * torch.eye(2).unsqueeze(0),
    }
    model = FilteringEmbedding(
        dynamics=dynamics,
        decoder=decoder,
        e=e_bel,
        Fe=_fe_identity,
        Fz=_fz_zero,
        q_theta=q_theta,
        k_theta=k_theta,
        adaptive_update=adaptive_update,
        adaptive_update_min_interval=adaptive_update_min_interval,
        adaptive_update_eig_threshold=adaptive_update_eig_threshold,
        learning_sensitivity=learning_sensitivity,
        device="cpu",
    )
    # Keep prediction deterministic so only filtering updates influence parameter belief.
    model.predict = lambda action=None: model.z["m"]
    model.z = {
        "m": torch.zeros(1, 1, 2),
        "P": torch.eye(2).unsqueeze(0).unsqueeze(0),
    }
    return model


@pytest.mark.parametrize("learning_sensitivity", ["dynamics_only", "measurement_corrected"])
def test_state_update_matches_dense_diagonal_observation_kalman_update(learning_sensitivity) -> None:
    model = _build_model(q_theta=1e-4, k_theta=4, learning_sensitivity=learning_sensitivity)
    model.z = {
        "m": torch.tensor([[[0.2, -0.1]]], dtype=torch.float32),
        "P": torch.tensor([[[[0.7, 0.2], [0.2, 0.5]]]], dtype=torch.float32),
    }
    y = torch.tensor([[[0.5, -0.4]]], dtype=torch.float32)
    u = torch.zeros(1, 1, 2)

    eye_latent = torch.eye(2).view(1, 1, 2, 2)
    z_pred_m = model.z["m"].clone()
    q = torch.nn.functional.softplus(model.dynamics.logvar).diag_embed().unsqueeze(0) * model.dt
    p_pred = model._project_spd(model.z["P"] + q + 1e-6 * eye_latent)
    h = model.decoder.jacobian(z_pred_m).unsqueeze(1)
    r_diag = model.decoder.var(z_pred_m).clamp_min(1e-6)
    r_cov = r_diag.diag_embed()
    eye_obs = torch.eye(2).view(1, 1, 2, 2)
    s = symmetrize(h @ p_pred @ h.transpose(-1, -2) + r_cov + 1e-6 * eye_obs)
    chol_s = safe_cholesky(s)
    k_gain = torch.cholesky_solve(h @ p_pred, chol_s).transpose(-1, -2)
    residual = y - model.decoder(z_pred_m)
    expected_m = z_pred_m + (k_gain @ residual.unsqueeze(-1)).squeeze(-1)
    kh = k_gain @ h
    expected_p = symmetrize(
        (eye_latent - kh) @ p_pred @ (eye_latent - kh).transpose(-1, -2)
        + k_gain @ r_cov @ k_gain.transpose(-1, -2)
    )

    model.update_posterior_embedding(y, u, update_theta=False)

    assert torch.allclose(model.z["m"], expected_m, atol=1e-5, rtol=1e-5)
    assert torch.allclose(model.z["P"], expected_p, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model._theta_sensitivity, torch.zeros_like(model._theta_sensitivity))
    assert model._theta_block_steps == 0


@pytest.mark.parametrize("learning_sensitivity", ["dynamics_only", "measurement_corrected"])
def test_block_update_applies_only_every_k_theta_steps(learning_sensitivity) -> None:
    model = _build_model(q_theta=1e-4, k_theta=3, learning_sensitivity=learning_sensitivity)
    y = torch.tensor([[[1.0, -0.5]]], dtype=torch.float32)
    u = torch.zeros(1, 1, 2)
    before = model.e["m"].clone()

    model.update_posterior_embedding(y, u)
    model.update_posterior_embedding(y, u)
    assert torch.allclose(model.e["m"], before)

    model.update_posterior_embedding(y, u)
    assert not torch.allclose(model.e["m"], before)
    torch.testing.assert_close(model._theta_sensitivity, torch.zeros_like(model._theta_sensitivity))
    assert model._theta_block_steps == 0


@pytest.mark.parametrize("learning_sensitivity", ["dynamics_only", "measurement_corrected"])
def test_adaptive_block_update_can_apply_before_k_theta(learning_sensitivity) -> None:
    model = _build_model(
        q_theta=1e-4,
        k_theta=5,
        adaptive_update=True,
        adaptive_update_min_interval=2,
        adaptive_update_eig_threshold=0.0,
        learning_sensitivity=learning_sensitivity,
    )
    y = torch.tensor([[[1.0, -0.5]]], dtype=torch.float32)
    u = torch.zeros(1, 1, 2)
    before = model.e["m"].clone()

    model.update_posterior_embedding(y, u)
    assert torch.allclose(model.e["m"], before)

    model.update_posterior_embedding(y, u)
    assert not torch.allclose(model.e["m"], before)
    assert model.last_information["parameter_update_reason"] == "block_eig"
    assert model._last_theta_block_steps_applied == 2
    torch.testing.assert_close(model._theta_sensitivity, torch.zeros_like(model._theta_sensitivity))


def test_q_theta_drift_increases_predictive_parameter_covariance() -> None:
    model = _build_model(q_theta=0.05, k_theta=1)
    p_before = model.e["P"].clone()
    m_before = model.e["m"].clone()
    model._theta_score_block.zero_()
    model._theta_info_block.zero_()

    model._apply_embedding_block_update()

    p_after = model.e["P"]
    expected_diag = torch.full((2,), 0.05)
    observed_diag = torch.diagonal((p_after - p_before)[0])
    assert torch.allclose(observed_diag, expected_diag, atol=1e-6)
    assert torch.allclose(model.e["m"], m_before)


def test_embedding_belief_remains_finite_and_spd_after_updates() -> None:
    model = _build_model(q_theta=1e-4, k_theta=2)
    u = torch.zeros(1, 1, 2)
    for _ in range(6):
        y = torch.randn(1, 1, 2) * 0.2
        model.update_posterior_embedding(y, u)
        assert torch.isfinite(model.e["m"]).all()
        assert torch.isfinite(model.e["P"]).all()
        assert torch.isfinite(model.e["L"]).all()
        eigvals = torch.linalg.eigvalsh(model.e["P"][0])
        assert torch.all(eigvals > 0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_learning_sensitivity_matches_fixed_observation_finite_difference(dtype) -> None:
    """Differentiate the actual multi-step linear Gaussian filter at fixed theta."""
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        a = torch.tensor([[0.85, 0.12], [-0.08, 0.9]])
        b = torch.tensor([[0.1, 0.02], [0.03, 0.08]])
        h = torch.tensor([[1.0, 0.4], [-0.2, 0.8]])
        observations = torch.tensor([[0.2, -0.1], [0.3, 0.05], [0.1, 0.2], [-0.1, 0.1]])

        def filtered(theta):
            model = _build_model(q_theta=0.0, k_theta=10)
            model.e["m"] = theta.reshape(1, 2).clone()
            model.Fe = lambda z, e: (b / model.dt).expand(*z.shape[:-1], 2, 2)
            model.Fz = lambda z, e: ((a - torch.eye(2)) / model.dt).expand(*z.shape[:-1], 2, 2)
            model.predict = lambda action=None: model.z["m"] @ a.T + (model.e["m"] @ b.T).unsqueeze(1)
            model.decoder.mapping.network.weight.data.copy_(h)
            states, sensitivities = [], []
            for y in observations:
                model.update_posterior_embedding(y.reshape(1, 1, 2))
                states.append(model.z["m"].squeeze().clone())
                sensitivities.append(model._theta_sensitivity.squeeze(0).clone())
            torch.testing.assert_close(model.e["m"], theta.reshape(1, 2))
            return torch.stack(states), torch.stack(sensitivities)

        theta = torch.tensor([0.15, -0.1])
        _, sensitivity = filtered(theta)
        delta = 1e-2 if dtype == torch.float32 else 1e-5
        finite_difference = torch.stack([
            (filtered(theta + delta * axis)[0] - filtered(theta - delta * axis)[0]) / (2 * delta)
            for axis in torch.eye(2)
        ], dim=-1)
        assert sensitivity.shape == (4, 2, 2)
        assert sensitivity.dtype == dtype
        tolerance = 3e-6 if dtype == torch.float32 else 1e-10
        torch.testing.assert_close(sensitivity, finite_difference, atol=tolerance, rtol=tolerance)
    finally:
        torch.set_default_dtype(previous_dtype)


def test_poisson_learning_scores_prediction_then_stores_local_corrected_sensitivity() -> None:
    """Check a dense Poisson local Gaussian update, without curvature derivatives."""
    from actdyn.models.decoder import LogLinearMapping, PoissonNoise

    model = _build_model(q_theta=0.0, k_theta=10)
    mapping = LogLinearMapping(latent_dim=2, obs_dim=3, dt=0.1, device="cpu")
    c = torch.tensor([[1.0, 0.4], [-0.3, 0.8], [0.2, -0.5]])
    with torch.no_grad():
        mapping.network[0].weight.copy_(c)
        mapping.network[0].bias.copy_(torch.tensor([1.2, 0.7, 1.0]))
    model.decoder = Decoder(mapping=mapping, noise=PoissonNoise(device="cpu"), device="cpu")
    model.z = {"m": torch.tensor([[[0.2, -0.1]]]), "P": torch.tensor([[[[0.7, 0.2], [0.2, 0.5]]]])}
    expected_s = torch.zeros(2, 2)
    expected_info = torch.zeros(2, 2)
    for observation in [torch.tensor([1., 0., 2.]), torch.tensor([0., 1., 0.])]:
        rate = model.decoder(model.z["m"]).squeeze()
        h = rate.unsqueeze(-1) * c
        r = torch.diag(rate)
        q = torch.diag(torch.nn.functional.softplus(model.dynamics.logvar).squeeze()) * model.dt
        p = model.z["P"].squeeze() + q + 1e-6 * torch.eye(2)
        innovation = h @ p @ h.T + r
        gain = torch.linalg.solve(innovation, h @ p).T
        predicted_s = expected_s + model.dt * torch.eye(2)
        expected_info += predicted_s.T @ h.T @ torch.linalg.solve(innovation, h @ predicted_s)
        expected_s = (torch.eye(2) - gain @ h) @ predicted_s
        model.update_posterior_embedding(observation.reshape(1, 1, 3))
        torch.testing.assert_close(model._theta_sensitivity.squeeze(0), expected_s, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(model._theta_info_block.squeeze(0), expected_info, atol=2e-6, rtol=2e-5)
    model._apply_embedding_block_update()
    assert model._theta_block_steps == 0
    assert torch.count_nonzero(model._theta_sensitivity) == 0


@pytest.mark.parametrize("k_theta", [1, 3])
def test_learning_sensitivity_resets_at_parameter_block_boundary(k_theta) -> None:
    model = _build_model(k_theta=k_theta)
    for step in range(k_theta):
        model.update_posterior_embedding(torch.tensor([[[0.1, -0.2]]]))
        if step < k_theta - 1:
            assert torch.count_nonzero(model._theta_sensitivity) > 0
    assert model._theta_block_steps == 0
    assert torch.count_nonzero(model._theta_sensitivity) == 0


def _linear_learning_problem(mode, *, block_steps=20, parameter_mean=None):
    """A fully specified linear model with correlated state and parameter priors."""
    a = torch.tensor([[.8, .25], [0., .9]])
    b = torch.tensor([[.3, .1], [-.1, .2]])
    c = torch.tensor([[1., .3], [-.2, .8]])
    q = torch.tensor([.02, .03])
    dynamics = FunctionDynamics(state_dim=2, dynamics_fn=_zero_dynamics, dt=1., device="cpu")
    dynamics.logvar = torch.nn.Parameter(torch.log(torch.expm1(q)).unsqueeze(0))
    mapping = LinearMapping(latent_dim=2, obs_dim=2, device="cpu")
    with torch.no_grad():
        mapping.network.weight.copy_(c)
        mapping.network.bias.zero_()
    decoder = Decoder(mapping=mapping, noise=GaussianNoise(obs_dim=2, sigma=.4, device="cpu"))
    mean = torch.tensor([[.1, -.2]]) if parameter_mean is None else parameter_mean.clone()
    p_theta = torch.tensor([[[.4, .1], [.1, .3]]])
    model = FilteringEmbedding(
        dynamics=dynamics, decoder=decoder,
        e={"m": mean, "P": p_theta, "L": torch.linalg.inv(p_theta)},
        Fe=lambda z, e: b.expand(*z.shape[:-1], 2, 2),
        Fz=lambda z, e: (a - torch.eye(2)).expand(*z.shape[:-1], 2, 2),
        q_theta=0., k_theta=block_steps, learning_sensitivity=mode, device="cpu",
    )
    model.z = {"m": torch.zeros(1, 1, 2), "P": torch.tensor([[[[.8, .2], [.2, .6]]]])}
    model.predict = lambda action=None: model.z["m"] @ a.T + (model.e["m"] @ b.T).unsqueeze(1)
    return model, a, b, c, q


@pytest.mark.parametrize("steps", [1, 4, 12])
def test_corrected_learning_matches_augmented_gaussian_parameter_posterior(steps):
    """Compare the complete block update with joint state-parameter conditioning."""
    model, a, b, c, q = _linear_learning_problem("measurement_corrected", block_steps=steps)
    transition = torch.cat([
        torch.cat([a, b], dim=1),
        torch.cat([torch.zeros(2, 2), torch.eye(2)], dim=1),
    ]).double()
    h = torch.cat([c, torch.zeros(2, 2)], dim=1).double()
    process = torch.block_diag(torch.diag(q) + 1e-6 * torch.eye(2), torch.zeros(2, 2)).double()
    r = model.decoder.var(torch.zeros(1, 1, 2)).flatten().diag().double()
    mean = torch.cat([model.z["m"].flatten(), model.e["m"].flatten()]).double()
    cov = torch.block_diag(model.z["P"][0, 0], model.e["P"][0]).double()
    initial_theta = model.e["m"].clone()
    for i in range(steps):
        observation = torch.tensor([.3 + .02 * i, -.15 + .01 * i])
        mean = transition @ mean
        cov = transition @ cov @ transition.T + process
        gain = torch.linalg.solve(h @ cov @ h.T + r, h @ cov).T
        mean += gain @ (observation.double() - h @ mean)
        cov -= gain @ h @ cov
        # Match the filter's tiny post-update covariance stabilization.
        cov[:2, :2] += 1e-8 * torch.eye(2)
        model.update_posterior_embedding(observation.view(1, 1, 2))
        if i < steps - 1:
            torch.testing.assert_close(model.e["m"], initial_theta)
    torch.testing.assert_close(model.e["m"].flatten().double(), mean[2:], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(model.e["P"][0].double(), cov[2:, 2:], atol=1e-5, rtol=1e-5)
    assert model._theta_block_steps == 0
    torch.testing.assert_close(model._theta_sensitivity, torch.zeros_like(model._theta_sensitivity))


def test_poisson_learning_scores_before_correcting_carried_sensitivity():
    from actdyn.models.decoder import LogLinearMapping, PoissonNoise, diagonal_observation_information

    models = []
    for mode in ("dynamics_only", "measurement_corrected"):
        model, a, b, c, q = _linear_learning_problem(mode)
        mapping = LogLinearMapping(latent_dim=2, obs_dim=2, dt=1., device="cpu")
        with torch.no_grad():
            mapping.network[0].weight.copy_(c)
            mapping.network[0].bias.zero_()
        model.decoder = Decoder(mapping=mapping, noise=PoissonNoise(device="cpu"), device="cpu")
        pred_mean = model.predict()
        pred_cov = a @ model.z["P"][0, 0] @ a.T + q.diag() + 1e-6 * torch.eye(2)
        _, information, _, _ = diagonal_observation_information(model.decoder, pred_mean)
        expected_sensitivity = torch.linalg.solve(torch.eye(2) + pred_cov @ information[0, 0], b)
        model.update_posterior_embedding(torch.tensor([[[3., 1.]]]))
        expected = b if mode == "dynamics_only" else expected_sensitivity
        torch.testing.assert_close(model._theta_sensitivity[0], expected, atol=1e-6, rtol=1e-5)
        models.append(model)
    # Changing the carried sensitivity must not change this first observation's score.
    for attr in ("_theta_score_block", "_theta_info_block"):
        torch.testing.assert_close(getattr(models[0], attr), getattr(models[1], attr))
    for field in ("m", "P"):
        torch.testing.assert_close(models[0].z[field], models[1].z[field])
        torch.testing.assert_close(models[0].e[field], models[1].e[field])


def test_learning_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="learning_sensitivity"):
        _build_model(learning_sensitivity="unknown")


def test_dynamics_only_learning_carries_nominal_multistep_sensitivity():
    model, a, b, *_ = _linear_learning_problem("dynamics_only")
    for step in range(1, 5):
        model.update_posterior_embedding(torch.tensor([[[.3, -.1]]]))
        # Closed-form derivative of the open-loop affine trajectory.
        expected = sum(torch.linalg.matrix_power(a, j) @ b for j in range(step))
        torch.testing.assert_close(model._theta_sensitivity[0], expected)
