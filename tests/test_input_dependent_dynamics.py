"""Contracts for learner drifts in which the input acts inside the drift, dz/dt = f(z, u; theta)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from actdyn.environment.vectorfield import (
    VectorFieldEnv,
    build_vectorfield,
    drift_torch,
    jacobian_embedding_torch,
    jacobian_state_torch,
    residual_torch,
)
from actdyn.metrics.information import drift_jacobians, planned_inputs
from actdyn.models.dynamics import FunctionDynamics
from actdyn.models.model import FilteringEmbedding

RAW = [1.2, 0.7, -0.9, 1.8, -0.4]  # (w_+, w_-, h_raw, gamma_raw, g_raw)


def _states(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 1.5 * torch.randn(n, 2, generator=g), torch.randn(n, 2, generator=g)


def test_inside_gain_field_is_the_documented_equation():
    z, u = _states()
    vf = build_vectorfield("wong_wang_inside_gain", torch.tensor([RAW]))
    w_plus, w_minus, h = RAW[0], RAW[1], RAW[2] / 6.0
    gamma = float(torch.nn.functional.softplus(torch.tensor(RAW[3])))
    g = float(torch.nn.functional.softplus(torch.tensor(RAW[4])))
    c = z.double() / 4.0  # s - 1/2
    x = w_plus * c - w_minus * c.flip(-1) + h + g * u.double()
    s = c + 0.5
    expected = 4.0 * (-s + (1.0 - s) * gamma * torch.sigmoid(6.0 * x))  # dz/dt = 4 ds/dt, tau = 1
    torch.testing.assert_close(vf.compute_with_input(z, u), expected.float(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(vf.compute(z), vf.compute_with_input(z, torch.zeros_like(z)))
    # Pool exchange symmetry.
    torch.testing.assert_close(vf.compute_with_input(z.flip(-1), u.flip(-1)), vf.compute_with_input(z, u).flip(-1))


def test_additive_fields_keep_their_drift_and_jacobians():
    z, u = _states(seed=1)
    theta = torch.tensor([[1.2, 0.8]])
    torch.testing.assert_close(drift_torch("wong_wang", z, theta, u, dynamics_alpha=1.0),
                               residual_torch("wong_wang", z, theta, dynamics_alpha=1.0) + u)
    for jac in (
        lambda uu: jacobian_state_torch("wong_wang", z, theta, dynamics_alpha=1.0, u=uu),
        lambda uu: jacobian_embedding_torch("wong_wang", z, theta.expand(6, 2), full_params=(1.2, 0.8),
                                            min_embedding_dim=2, dynamics_alpha=1.0, u=uu),
    ):
        torch.testing.assert_close(jac(u), jac(None), rtol=0, atol=0)  # an additive input has no effect


def test_inside_gain_jacobians_depend_on_the_input():
    z, u = _states(seed=2)
    e = torch.tensor(RAW).expand(6, 5)
    kw = dict(full_params=tuple(RAW), min_embedding_dim=5, dynamics_alpha=1.0)
    Fe_u = jacobian_embedding_torch("wong_wang_inside_gain", z, e, u=u, **kw)
    Fe_0 = jacobian_embedding_torch("wong_wang_inside_gain", z, e, **kw)
    assert Fe_u.shape == (6, 2, 5)
    assert torch.allclose(Fe_0[..., 4], torch.zeros_like(Fe_0[..., 4]))  # no input: g is not identified
    assert not torch.allclose(Fe_u, Fe_0)
    # Finite-difference check of d f / d z at the input.
    Fz = jacobian_state_torch("wong_wang_inside_gain", z, torch.tensor([RAW]), dynamics_alpha=1.0, u=u)
    h = 1e-3
    for j in range(2):
        dz = torch.zeros(2)
        dz[j] = h
        fd = (drift_torch("wong_wang_inside_gain", z + dz, torch.tensor([RAW]), u, dynamics_alpha=1.0)
              - drift_torch("wong_wang_inside_gain", z - dz, torch.tensor([RAW]), u, dynamics_alpha=1.0)) / (2 * h)
        torch.testing.assert_close(Fz[..., j], fd, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("dynamics_type,params", [("wong_wang", [1.2, 0.8]), ("wong_wang_inside_gain", RAW)])
def test_forward_step_and_env_step_use_the_input_as_the_field_defines(dynamics_type, params):
    env = VectorFieldEnv(dynamics_type, d_state=2, d_action=2, Q=0.0, dt=0.05, dyn_params=torch.tensor([params]))
    dyn = FunctionDynamics(state_dim=2, dt=0.05, dynamics_fn=env)
    dyn.logvar = torch.nn.Parameter(torch.zeros(1, 2))
    z, u = _states(n=1, seed=3)
    z, u = z.reshape(1, 1, 2), u.reshape(1, 1, 2)
    _s, mus, _v = dyn.sample_forward(init_z=z, action=u, k_step=1, add_noise=False, return_traj=True)
    expected = z + 0.05 * drift_torch(dynamics_type, z, torch.tensor([params]), u, dynamics_alpha=1.0)
    torch.testing.assert_close(mus[-1], expected, rtol=1e-6, atol=1e-6)
    env.state = z.reshape(2).clone()
    state, *_ = env.step(u.reshape(2))
    torch.testing.assert_close(state, expected.reshape(2), rtol=1e-6, atol=1e-6)


def test_filter_and_metric_pass_the_input_only_to_input_dependent_drifts():
    calls = []

    def Fz(z, e, u=None):
        calls.append(("z", u is not None))
        return torch.zeros(*z.shape, z.shape[-1])

    def Fe(z, e, u=None):
        calls.append(("e", u is not None))
        return torch.zeros(*z.shape, e.shape[-1])

    u = torch.ones(1, 1, 2)
    for input_dependent in (False, True):
        calls.clear()
        stub = SimpleNamespace(Fz=Fz, Fe=Fe, dynamics=SimpleNamespace(network=SimpleNamespace(input_dependent=input_dependent)))
        stub.input_dependent_dynamics = FilteringEmbedding.input_dependent_dynamics.fget(stub)
        FilteringEmbedding._jac_state(stub, torch.zeros(1, 1, 2), torch.zeros(1, 5), u)
        FilteringEmbedding._jac_embedding(stub, torch.zeros(1, 1, 2), torch.zeros(1, 5), u)
        assert calls == [("z", input_dependent), ("e", input_dependent)]
    rollout = {"model_state": torch.zeros(3, 4, 2), "env_action": torch.ones(3, 4, 2)}
    assert planned_inputs(SimpleNamespace(input_dependent_dynamics=False), rollout) is None
    assert planned_inputs(SimpleNamespace(input_dependent_dynamics=True), rollout).shape == (3, 4, 2)
    with pytest.raises(ValueError, match="env_action"):
        planned_inputs(SimpleNamespace(input_dependent_dynamics=True), {"model_state": torch.zeros(3, 4, 2)})
    calls.clear()
    drift_jacobians(Fe, Fz, torch.zeros(3, 4, 2), torch.zeros(3, 4, 5), rollout["env_action"])
    assert calls == [("e", True), ("z", True)]


def test_flex_inside_gain_model_matches_the_field_and_its_input_matrix():
    from actdyn.policy.baseline_flex import FlexWongWangInsideGainModel

    flex = FlexWongWangInsideGainModel(dt=0.05, dynamics_alpha=1.0, latent_dim=2, action_dim=2,
                                       initial_embedding=np.asarray(RAW), fixed_tail=np.zeros(0))
    z, u = _states(n=5, seed=4)
    vf = build_vectorfield("wong_wang_inside_gain", torch.tensor([RAW]))
    torch.testing.assert_close(flex(torch.cat([z, u], dim=1)), vf.compute_with_input(z, u), rtol=1e-5, atol=1e-5)
    x = z[0]
    B = flex.get_B(x)
    h = 1e-3
    fd = np.stack([((vf.compute_with_input(x[None], torch.eye(2)[j][None] * h)
                     - vf.compute_with_input(x[None], -torch.eye(2)[j][None] * h)) / (2 * h))[0].numpy()
                   for j in range(2)], axis=1)
    np.testing.assert_allclose(B, fd, rtol=2e-2, atol=2e-3)
    assert abs(B[0, 1]) < 1e-9 and abs(B[1, 0]) < 1e-9  # each input drives its own pool


def test_eval_learner_transition_is_the_euler_step_of_the_drift():
    from experiments.tnsre.eval_spiking_sessions import learner_transition

    z, u = _states(seed=5)
    step = learner_transition("wong_wang_inside_gain", np.asarray(RAW), np.asarray(RAW), 5, 0.05, torch.float32)
    expected = z + 0.05 * drift_torch("wong_wang_inside_gain", z, torch.tensor([RAW]), u, dynamics_alpha=1.0)
    torch.testing.assert_close(step(z, u), expected)


def test_reference_fit_reads_the_learner_parameters_of_the_record(tmp_path, monkeypatch):
    import json

    from experiments.tnsre import eval_spiking_sessions as ev

    path = tmp_path / "m2_fit.json"
    path.write_text(json.dumps({"learner_parameters": {"values": [1.0, 0.5, -1.2, 2.0, -1.5]}}))
    monkeypatch.setitem(ev.PROTOCOL, "reduced_fit", str(path))
    np.testing.assert_allclose(ev.reference_fit(), [1.0, 0.5, -1.2, 2.0, -1.5])


def test_planner_is_deterministic_budgeted_and_held_at_the_coarse_resolution():
    from actdyn.utils.validation import icem_reversal_plan
    from experiments.tnsre.eval_spiking_sessions import learner_transition, seed_pushes

    step = learner_transition("wong_wang_inside_gain", np.asarray(RAW), np.asarray(RAW), 5, 0.05, torch.float32)
    kwargs = dict(dt=0.05, window=60, coarse_factor=10, budget=2.0, action_max=1.0, noise_var=0.1,
                  objective="reach", success_gap=2.0, mc_samples=4, num_samples=16, num_elites=4,
                  num_iterations=3, seed_candidates=seed_pushes(60, 2.0, 0.05, 10))
    z0, drive = np.array([0.4, -1.6]), np.zeros((120, 2))
    a = icem_reversal_plan(step, z0, drive, **kwargs)
    b = icem_reversal_plan(step, z0, drive, **kwargs)
    np.testing.assert_array_equal(a["u"], b["u"])
    assert a["u"].shape == (60, 2) and a["energy"] <= 2.0 + 1e-5
    np.testing.assert_array_equal(a["u"][0], a["u"][9])  # values change only every coarse_factor bins
    with pytest.raises(ValueError, match="objective"):
        icem_reversal_plan(step, z0, drive, **{**kwargs, "objective": "terminal"})


def test_control_gap_variants_swap_one_parameter_or_the_filter():
    from experiments.tnsre.diag_control_gap import model_variants

    learned, fit = np.arange(5.0), 10.0 + np.arange(5.0)
    variants = {label: (a, b) for label, a, b in model_variants(learned, fit)}
    assert len(variants) == 12
    np.testing.assert_array_equal(variants["learned+fit:gamma_raw"][0], [0, 1, 2, 13, 4])
    np.testing.assert_array_equal(variants["fit+learned:g_raw"][0], [10, 11, 12, 13, 4])
    np.testing.assert_array_equal(variants["plan:learned/filter:fit"][0], learned)
    np.testing.assert_array_equal(variants["plan:learned/filter:fit"][1], fit)
    np.testing.assert_array_equal(learned, np.arange(5.0))  # inputs are not modified
