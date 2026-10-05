"""Contracts for the Wilson-Cowan and Wong-Wang benchmark systems."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from actdyn.environment.vectorfield import (
    build_vectorfield,
    jacobian_param_torch,
    jacobian_state_torch,
)
from actdyn.utils.experiment_runtime import read_trace_csv, write_trace_csv
from actdyn.utils.validation import basin_switch_cost_many
from experiments.experiment_definitions import (
    DEFAULT_MODEL_CATALOG_PATHS,
    DEFAULT_SUITE_CATALOG_PATHS,
    configure_catalogs,
    get_environment_preset,
)
from experiments.summarize import (
    BASIN_SWITCH_TRACE_FIELDS,
    basin_switch_enabled,
    recompute_basin_switch_traces,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TBME_ENV_PATHS = [
    REPO_ROOT / "experiments" / "experiment_env.yaml",
    REPO_ROOT / "experiments" / "tnsre" / "config" / "experiment_env.yaml",
]

# Attractors of the default parameterizations in latent coordinates, found by
# root finding on the drift (scripts under scratch, reproduced here as data).
# Equilibria (E, I) of the Wilson-Cowan (1972) multiple-steady-state example as listed by
# Akhmet & Cag (arXiv:1701.04015, Eq. 8): down state, up state, saddle.
WILSON_COWAN_1972_RATES = np.array(
    [[0.0, 0.0], [0.44234, 0.22751], [0.18816, 0.067243]], dtype=np.float32
)
WILSON_COWAN_1972_TRUE = np.array([12.0, 4.0, 13.0, 11.0], dtype=np.float32)
WONG_WANG_ATTRACTORS = np.array([[-1.8443, -0.1291], [-0.1291, -1.8443]], dtype=np.float32)
WONG_WANG_SADDLE = np.array([[-0.9336, -0.9336]], dtype=np.float32)
WONG_WANG_TRUE = np.array([1.2, 0.8], dtype=np.float32)


@pytest.fixture
def tbme_catalog():
    configure_catalogs(
        env_catalog_paths=TBME_ENV_PATHS,
        model_catalog_paths=DEFAULT_MODEL_CATALOG_PATHS,
        suite_catalog_paths=DEFAULT_SUITE_CATALOG_PATHS,
    )
    try:
        yield
    finally:
        configure_catalogs()


@pytest.mark.parametrize(
    ("dynamics_type", "equilibria", "param_dim"),
    [
        ("wilson_cowan_1972", 8.0 * (WILSON_COWAN_1972_RATES - 0.25), 4),
        ("wong_wang", np.concatenate([WONG_WANG_ATTRACTORS, WONG_WANG_SADDLE]), 2),
    ],
)
def test_default_parameters_have_documented_equilibria(dynamics_type, equilibria, param_dim):
    vf = build_vectorfield(dynamics_type)
    states = torch.as_tensor(equilibria)
    drift = vf(states)
    assert drift.shape == states.shape
    np.testing.assert_allclose(drift.numpy(), 0.0, atol=5e-3)

    params = vf.dyn_params.reshape(-1)
    assert params.shape == (param_dim,)
    jac_state = jacobian_state_torch(dynamics_type, states, params, dynamics_alpha=1.0)
    jac_param = jacobian_param_torch(
        dynamics_type, states, params.expand(states.shape[0], -1), dynamics_alpha=1.0
    )
    assert jac_state.shape == (states.shape[0], 2, 2)
    assert jac_param.shape == (states.shape[0], 2, param_dim)
    # Every learned weight moves the drift somewhere on the attractor set.
    assert torch.all(jac_param.abs().sum(dim=(0, 1)) > 1e-4)


def test_wong_wang_is_symmetric_under_pool_exchange():
    vf = build_vectorfield("wong_wang")
    z = torch.tensor([[-1.0, 0.4], [0.7, -1.5], [0.0, 0.0]])
    swapped = z.flip(-1)
    np.testing.assert_allclose(vf(z).flip(-1).numpy(), vf(swapped).numpy(), atol=1e-6)


def test_neural_presets_resolve_from_tbme_catalog(tbme_catalog):
    wc = get_environment_preset("tbme_wilson_cowan")
    assert wc.resolved_dynamics_type() == "wilson_cowan_1972"
    np.testing.assert_allclose(wc.resolved_true_params(), WILSON_COWAN_1972_TRUE)
    assert wc.parameter_scale == pytest.approx(10.0)
    assert wc.action_max == pytest.approx(3.0)
    assert wc.initial_parameter_nonnegative
    assert wc.initial_parameter_nonnegative
    assert wc.action_max == pytest.approx(3.0)
    assert wc.embedding_dim == 4
    assert wc.state_noise == pytest.approx(0.1)
    assert not basin_switch_enabled(wc)

    ww = get_environment_preset("tbme_wong_wang")
    assert ww.resolved_dynamics_type() == "wong_wang"
    np.testing.assert_allclose(ww.resolved_true_params(), WONG_WANG_TRUE)
    assert ww.embedding_dim == 2
    lo, hi = ww.resolved_state_bounds()
    # Trials start in the undecided low-activity corner below the saddle.
    assert np.all(hi < WONG_WANG_SADDLE[0])
    assert np.all(lo < hi)
    assert basin_switch_enabled(ww)
    np.testing.assert_allclose(ww.basin_switch_source, WONG_WANG_ATTRACTORS[1], atol=1e-4)
    np.testing.assert_allclose(ww.basin_switch_target, WONG_WANG_ATTRACTORS[0], atol=1e-4)
    assert ww.basin_switch_horizon == 300
    assert ww.basin_switch_eval_interval == 100


def _switch_kwargs(**overrides):
    kwargs = dict(
        estimator_dynamics_type="wong_wang",
        estimator_full_params=WONG_WANG_TRUE,
        estimator_min_embedding_dim=2,
        e_true=torch.as_tensor(WONG_WANG_TRUE),
        true_dynamics_type="wong_wang",
        true_full_params=WONG_WANG_TRUE,
        true_min_embedding_dim=2,
        source_state=WONG_WANG_ATTRACTORS[1],
        target_state=WONG_WANG_ATTRACTORS[0],
        dt=0.01,
        dynamics_alpha=1.0,
        horizon=150,
        action_max=1.0,
        iterations=40,
        learning_rate=0.1,
    )
    kwargs.update(overrides)
    return kwargs


def test_basin_switch_oracle_succeeds_and_zero_model_fails():
    estimates = torch.tensor([WONG_WANG_TRUE.tolist(), [0.0, 0.0]])
    out = basin_switch_cost_many(estimates, **_switch_kwargs())
    assert set(out) == {"energy", "success", "terminal_distance"}
    assert all(v.shape == (2,) for v in out.values())
    assert out["success"][0] == 1.0
    assert out["energy"][0] > 0.0
    # A connectivity-free model plans too little input to cross the separatrix.
    assert out["success"][1] == 0.0
    assert out["energy"][1] < out["energy"][0]

    repeat = basin_switch_cost_many(estimates, **_switch_kwargs())
    for key in out:
        np.testing.assert_array_equal(out[key], repeat[key])


def test_basin_switch_rejects_bad_shapes():
    with pytest.raises(ValueError, match=r"shape \(M, E\)"):
        basin_switch_cost_many(torch.as_tensor(WONG_WANG_TRUE), **_switch_kwargs())
    with pytest.raises(ValueError, match="horizon"):
        basin_switch_cost_many(
            torch.as_tensor(WONG_WANG_TRUE).reshape(1, -1), **_switch_kwargs(horizon=0)
        )


def test_recompute_basin_switch_traces_writes_rows_at_interval(
    tmp_path: Path, tbme_catalog, monkeypatch: pytest.MonkeyPatch
):
    import experiments.summarize as summarize_module

    monkeypatch.setattr(summarize_module, "BASIN_SWITCH_ITERATIONS", 5)
    run_dir = tmp_path / "wong_wang" / "adaptive" / "seed_0" / "repeat_01"
    run_dir.mkdir(parents=True)
    steps = [50, 100, 150, 200]
    write_trace_csv(
        run_dir / "embedding_estimate_trace.csv",
        [
            {"step": step, "cpu_time_sec": 0.01 * step, "e0": 1.2, "e1": 0.8}
            for step in steps
        ],
        ["step", "cpu_time_sec", "e0", "e1"],
    )
    record = {
        "policy_id": "adaptive",
        "seed": 0,
        "run_dir": run_dir,
        "metadata": {
            "env_preset_id": "tbme_wong_wang",
            "dynamics_type": "wong_wang",
            "embedding_true": WONG_WANG_TRUE.tolist(),
            "embedding_estimate": WONG_WANG_TRUE.tolist(),
            "true_params_full": WONG_WANG_TRUE.tolist(),
            "min_embedding_dim": 2,
            "status": "completed",
        },
    }
    assert recompute_basin_switch_traces([record]) == 1
    rows = read_trace_csv(run_dir / "basin_switch_trace.csv")
    assert [int(float(row["step"])) for row in rows] == [100, 200]
    assert set(rows[0]) == set(BASIN_SWITCH_TRACE_FIELDS)
    assert float(rows[0]["basin_switch_horizon"]) == 300
    assert float(rows[0]["basin_switch_iterations"]) == 5
    assert float(rows[0]["basin_switch_energy"]) == pytest.approx(
        float(rows[0]["basin_switch_energy_oracle"])
    )
    # Existing traces are reused unless forced.
    assert recompute_basin_switch_traces([record]) == 0
    assert recompute_basin_switch_traces([record], force=True) == 1


def test_wilson_cowan_1972_stability_and_published_weights():
    from actdyn.utils.vectorfields_eqn import WilsonCowan1972

    vf = WilsonCowan1972()
    # The learned coordinates are the published weights (12, 4, 13, 11) as is.
    np.testing.assert_allclose(vf.dyn_params.numpy()[0], WILSON_COWAN_1972_TRUE)
    states = torch.as_tensor(8.0 * (WILSON_COWAN_1972_RATES - 0.25))
    jac = jacobian_state_torch("wilson_cowan_1972", states, vf.dyn_params.reshape(-1), dynamics_alpha=1.0)
    real = torch.linalg.eigvals(jac).real
    assert torch.all(real[:2] < 0)  # down and up states are stable
    assert real[2].min() < 0 < real[2].max()  # saddle
    # The shifted logistic makes E = I = 0 an exact equilibrium.
    np.testing.assert_allclose(vf(states[:1]).numpy(), 0.0, atol=1e-6)


def test_flex_wilson_cowan_1972_model_matches_vector_field():
    from actdyn.environment.vectorfield import residual_torch
    from actdyn.policy.baseline_flex import FlexWilsonCowan1972Model

    params = np.array([10.0, 6.0, 15.0, 9.0], dtype=np.float32)
    model = FlexWilsonCowan1972Model(
        dt=0.01, dynamics_alpha=1.0, latent_dim=2, action_dim=2,
        initial_embedding=params, fixed_tail=np.zeros(0, dtype=np.float32),
    )
    z = torch.tensor([[-1.5, 0.3], [0.8, -0.9], [2.0, 0.6]])
    u = torch.tensor([[0.2, -0.1], [0.0, 0.0], [-0.5, 0.4]])
    out = model(torch.cat([z, u], dim=1))
    expected = residual_torch("wilson_cowan_1972", z, torch.as_tensor(params), dynamics_alpha=1.0) + u
    np.testing.assert_allclose(out.detach().numpy(), expected.detach().numpy(), atol=1e-5)


def test_flex_wong_wang_model_matches_vector_field():
    from actdyn.environment.vectorfield import residual_torch
    from actdyn.policy.baseline_flex import FlexWongWangModel

    params = np.array([1.7, 0.6], dtype=np.float32)
    model = FlexWongWangModel(
        dt=0.05, dynamics_alpha=1.0, latent_dim=2, action_dim=2,
        initial_embedding=params, fixed_tail=np.zeros(0, dtype=np.float32),
    )
    z = torch.tensor([[-1.8, -1.7], [0.9, -1.8], [-0.4, 0.2]])
    u = torch.tensor([[0.3, 0.0], [0.0, -0.2], [-0.5, 0.5]])
    out = model(torch.cat([z, u], dim=1))
    expected = residual_torch("wong_wang", z, torch.as_tensor(params), dynamics_alpha=1.0) + u
    np.testing.assert_allclose(out.detach().numpy(), expected.detach().numpy(), atol=1e-6)


def _wong_wang_trajectories(theta, n_traj=3, n_steps=80, seed=0):
    from actdyn.environment.vectorfield import residual_torch

    rng = np.random.default_rng(seed)
    u = rng.choice([-1.0, 0.0, 1.0], size=(n_traj, n_steps, 2)).astype(np.float32)
    z = np.zeros((n_traj, n_steps + 1, 2), dtype=np.float32)
    z[:, 0] = -1.8
    th = torch.as_tensor(theta, dtype=torch.float32).expand(n_traj, -1)
    for t in range(n_steps):
        zt = torch.as_tensor(z[:, t])
        z[:, t + 1] = (zt + 0.05 * (residual_torch("wong_wang", zt, th, dynamics_alpha=1.0) + torch.as_tensor(u[:, t]))).numpy()
    return z, u


def test_rollout_r2_is_one_for_the_generating_model_and_lower_otherwise():
    from actdyn.utils.validation import rollout_r2_on_trajectories

    z, u = _wong_wang_trajectories([1.9, 1.1])
    r2 = rollout_r2_on_trajectories(
        torch.tensor([[1.9, 1.1], [1.2, 0.8]]), dynamics_type="wong_wang", full_params=WONG_WANG_TRUE,
        min_embedding_dim=2, states=z, inputs=u, dt=0.05, horizon=30, stride=10,
    )
    assert r2.shape == (2,)
    np.testing.assert_allclose(r2[0], 1.0, atol=1e-6)
    assert r2[1] < 0.99
    with pytest.raises(ValueError, match="states"):
        rollout_r2_on_trajectories(torch.tensor([[1.9, 1.1]]), dynamics_type="wong_wang", full_params=WONG_WANG_TRUE,
                                   min_embedding_dim=2, states=z[:, :-1], inputs=u, dt=0.05, horizon=30, stride=10)


def test_project_to_budget_enforces_step_and_energy_bounds():
    from actdyn.utils.validation import project_to_budget

    rng = np.random.default_rng(0)
    u = torch.as_tensor(rng.normal(scale=2.0, size=(5, 30, 2)), dtype=torch.float32)
    out = project_to_budget(u, torch.tensor(1.5), dt=0.1, action_max=1.0)
    assert torch.all(torch.linalg.norm(out, dim=-1) <= 1.0 + 1e-6)
    assert torch.all(0.1 * torch.sum(out**2, dim=(-2, -1)) <= 1.5 + 1e-5)
    small = 0.01 * torch.ones(1, 30, 2)
    torch.testing.assert_close(project_to_budget(small, torch.tensor(1.5), dt=0.1, action_max=1.0), small)


def test_parameter_scale_sets_flex_units(tbme_catalog):
    from types import SimpleNamespace

    from experiments.run import _flex_parameter_settings

    spec = SimpleNamespace(flex_regularization=0.1, flex_parameter_step_clip=0.25,
                           flex_parameter_min=-5.0, flex_parameter_max=5.0)
    settings = _flex_parameter_settings(spec, get_environment_preset("tbme_wilson_cowan"))
    assert settings == pytest.approx({"regularization": 1e-3, "parameter_step_clip": 2.5,
                                      "parameter_min": -50.0, "parameter_max": 50.0})
    # Unit scale leaves the other systems unchanged.
    assert _flex_parameter_settings(spec, get_environment_preset("tbme_duffing")) == pytest.approx(
        {"regularization": 0.1, "parameter_step_clip": 0.25, "parameter_min": -5.0, "parameter_max": 5.0})
