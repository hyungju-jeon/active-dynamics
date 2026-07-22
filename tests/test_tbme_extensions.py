from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    module_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(name, None)


def test_planar_systems_include_tbme_dynamics():
    module = _load_module("cosyne_planar_tbme", "experiments/cosyne/planar_systems.py")
    assert module.has_planar_system_defaults("damped_pendulum")
    assert module.has_planar_system_defaults("double_integrator")
    assert module.has_planar_system_defaults("two_basin_bridge")

    init_state = module.sample_initial_state("two_basin_bridge", seed=0)
    assert init_state.shape == (2,)
    assert -2.4 <= float(init_state[0]) <= -0.8

    emb = module.true_embedding("two_basin_bridge")
    assert emb.shape == (4,)
    jac = module.jacobian_param_torch(
        "two_basin_bridge",
        torch.tensor([[[-1.5, 0.2]]], dtype=torch.float32),
        torch.tensor(emb.reshape(1, 1, -1), dtype=torch.float32),
        dynamics_alpha=1.0,
    )
    assert jac.shape == (1, 1, 2, 4)
    assert float(jac[0, 0, 1, 0]) != 0.0
    assert abs(float(jac[0, 0, 1, 2])) < abs(float(jac[0, 0, 1, 0]))

    state = np.asarray([[0.2, -0.1]], dtype=np.float32)
    pend = module.residual_np(
        "damped_pendulum",
        state,
        module.true_embedding("damped_pendulum"),
        dynamics_alpha=1.0,
    )
    integ = module.residual_np(
        "double_integrator",
        state,
        module.true_embedding("double_integrator"),
        dynamics_alpha=1.0,
    )
    assert pend.shape == (1, 2)
    assert integ.shape == (1, 2)


def test_realdata_loader_accepts_standardized_npz(tmp_path: Path):
    module = _load_module(
        "cosyne_realdata_tbme", "experiments/cosyne/realdata_spiking.py"
    )
    behavior = np.stack(
        [
            np.linspace(-1.0, 1.0, 32),
            np.linspace(0.5, -0.5, 32),
            np.sin(np.linspace(0.0, 1.0, 32)),
            np.cos(np.linspace(0.0, 1.0, 32)),
        ],
        axis=1,
    ).astype(np.float32)
    spikes = np.tile(np.arange(12, dtype=np.float32), (32, 1))
    path = tmp_path / "replay_fixture.npz"
    np.savez(
        path, behavior=behavior, spikes=spikes, dt=np.asarray([0.02], dtype=np.float32)
    )

    dataset = module.load_replay_dataset(
        dataset_id="fixture",
        dataset_path=path,
        state_key="behavior",
        observation_key="spikes",
        latent_dim=2,
        max_observation_dim=6,
        time_bin_ms=20.0,
    )
    assert dataset.states.shape == (32, 2)
    assert dataset.spikes.shape == (32, 6)
    assert dataset.dt == pytest.approx(0.02)
    assert dataset.metadata["num_units"] == 6


def test_realdata_split_and_ridge_helpers(tmp_path: Path):
    module = _load_module(
        "cosyne_realdata_tbme_helpers", "experiments/cosyne/realdata_spiking.py"
    )
    x = np.stack(
        [
            np.linspace(-1.0, 1.0, 40),
            np.linspace(1.0, -1.0, 40),
        ],
        axis=1,
    )
    a_true = np.asarray([[0.85, 0.10], [-0.05, 0.90]], dtype=np.float64)
    states = np.zeros((41, 2), dtype=np.float64)
    states[:-1] = x
    states[1:] = x @ a_true
    spikes = np.abs(np.random.default_rng(0).normal(size=(41, 16))).astype(np.float32)
    path = tmp_path / "linear_replay.npz"
    np.savez(
        path,
        behavior=states.astype(np.float32),
        spikes=spikes,
        dt=np.asarray([0.02], dtype=np.float32),
    )

    dataset = module.load_replay_dataset(
        dataset_id="linear",
        dataset_path=path,
        state_key="behavior",
        observation_key="spikes",
        latent_dim=2,
        max_observation_dim=8,
        time_bin_ms=20.0,
    )
    train_idx, eval_idx = module.split_replay_dataset(dataset, train_fraction=0.7)
    x_all, y_all = module.build_transition_matrices(dataset)
    coef = module.fit_linear_dynamics_ridge(
        x_all[train_idx], y_all[train_idx], ridge=1e-4
    )
    mse = module.evaluate_prediction_mse(x_all[eval_idx], y_all[eval_idx], coef)
    r2 = module.evaluate_prediction_r2(x_all[eval_idx], y_all[eval_idx], coef)
    assert mse < 0.2
    assert r2 > 0.6


def test_video_renderer_skips_realdata_experiment(tmp_path: Path):
    module = _load_module("tbme_video_tbme", "experiments/tbme/render_videos.py")
    exit_code = module.main(
        [
            "--base-dir",
            str(tmp_path / "results"),
            "--exp-id",
            "tbme_exp3_realdata_policy",
        ]
    )
    assert exit_code == 0


def test_prepare_realdata_binning_helpers():
    module = _load_module(
        "tbme_prepare_realdata", "experiments/tbme/prepare_exp3_data.py"
    )
    behavior = np.arange(24, dtype=np.float32).reshape(12, 2)
    binned = module.bin_regular_timeseries(behavior, sample_rate_hz=100.0, bin_ms=20.0)
    assert binned.shape == (6, 2)
    assert np.allclose(binned[0], np.asarray([1.0, 2.0], dtype=np.float32))
    assert np.allclose(binned[-1], np.asarray([21.0, 22.0], dtype=np.float32))

    spike_trains = [
        np.asarray([0.001, 0.019, 0.020, 0.041], dtype=np.float64),
        np.asarray([0.0, 0.039], dtype=np.float64),
    ]
    spike_counts = module.bin_spike_trains(spike_trains, num_bins=3, dt_sec=0.02)
    assert spike_counts.shape == (3, 2)
    assert np.array_equal(
        spike_counts,
        np.asarray(
            [
                [2.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_prepare_realdata_repairs_missing_behavior_samples():
    module = _load_module(
        "tbme_prepare_realdata_fill", "experiments/tbme/prepare_exp3_data.py"
    )
    behavior = np.asarray(
        [
            [0.0, 10.0],
            [np.nan, np.nan],
            [2.0, 14.0],
            [3.0, 16.0],
            [np.nan, 18.0],
            [5.0, 20.0],
        ],
        dtype=np.float32,
    )
    repaired = module.fill_nan_timeseries(behavior)
    assert repaired.shape == behavior.shape
    assert not np.isnan(repaired).any()
    assert np.allclose(
        repaired,
        np.asarray(
            [
                [0.0, 10.0],
                [1.0, 12.0],
                [2.0, 14.0],
                [3.0, 16.0],
                [4.0, 18.0],
                [5.0, 20.0],
            ],
            dtype=np.float32,
        ),
    )


def test_tbme_exp1_launcher_includes_two_basin_schedule_ablation():
    module = _load_module(
        "tbme_run_exp1_schedule_addition", "experiments/tbme/run_exp1.py"
    )
    assert "tbme_exp1_two_basin_schedule_ablation" in module.EXP1_SUITES


def test_timevarying_hotspots_prototype_smoke(tmp_path: Path):
    module = _load_module(
        "tbme_timevarying_hotspots",
        "experiments/tbme/run_circular_timevarying_hotspots_experiment.py",
    )
    out_dir = tmp_path / "hotspots"
    exit_code = module.main(
        [
            "--output-dir",
            str(out_dir),
            "--seeds",
            "0",
            "--total-steps",
            "12",
            "--skip-figures",
        ]
    )
    assert exit_code == 0
    summary_path = out_dir / "summary.json"
    trace_path = out_dir / "trace.csv"
    assert summary_path.exists()
    assert trace_path.exists()
    payload = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
    assert payload["config"]["total_steps"] == 12
    assert set(payload["policies"].keys()) == {
        "radius_greedy",
        "hotspot_greedy",
        "hotspot_intercept",
    }


def test_current_tbme_catalog_configures_initial_parameter_distribution():
    from experiments.experiment_definitions import (
        DEFAULT_MODEL_CATALOG_PATHS,
        DEFAULT_SUITE_CATALOG_PATHS,
        configure_catalogs,
        get_environment_preset,
    )

    env_paths = [
        REPO_ROOT / "experiments" / "experiment_env.yaml",
        REPO_ROOT / "experiments" / "tbme" / "config" / "experiment_env.yaml",
    ]
    configure_catalogs(
        env_catalog_paths=env_paths,
        model_catalog_paths=DEFAULT_MODEL_CATALOG_PATHS,
        suite_catalog_paths=DEFAULT_SUITE_CATALOG_PATHS,
    )
    try:
        preset = get_environment_preset("tbme_gated_duffing")
        assert np.allclose(preset.initial_parameter_mean_vector(), np.zeros(4))
        assert preset.initial_parameter_variance == pytest.approx(0.25)
    finally:
        configure_catalogs()


def test_confounded_gate_matches_hidden_nuisance_design():
    from actdyn.environment.vectorfield import (
        jacobian_param_torch,
        jacobian_state_torch,
    )

    params = torch.tensor([0.5], dtype=torch.float32)
    states = torch.tensor([[-0.5, 0.0, 0.25], [-0.32, 0.0, 0.25]], dtype=torch.float32)

    param_jacobian = jacobian_param_torch(
        "confounded_gate",
        states,
        params.expand(states.shape[0], -1),
        dynamics_alpha=1.0,
    )
    state_jacobian = jacobian_state_torch(
        "confounded_gate", states, params, dynamics_alpha=1.0
    )

    assert param_jacobian.shape == (2, 3, 1)
    assert torch.allclose(param_jacobian[:, (0, 2), 0], torch.zeros(2, 2))
    assert param_jacobian[0, 1, 0] == pytest.approx(20.0, abs=1e-3)
    assert param_jacobian[1, 1, 0] == pytest.approx(10.0, abs=2e-3)
    assert state_jacobian.shape == (2, 3, 3)
    assert float(state_jacobian[0, 1, 2].detach()) == pytest.approx(20.0, abs=1e-5)
    assert abs(float(state_jacobian[1, 1, 2].detach())) < 2e-3
    assert torch.allclose(state_jacobian[:, 2], torch.zeros(2, 3))


def test_confounded_gate_canonical_rollouts_separate_objectives():
    from actdyn.environment.vectorfield import (
        jacobian_param_torch,
        jacobian_state_torch,
    )
    from actdyn.utils.torch_utils import attenuated_state_information

    params = torch.tensor([0.5], dtype=torch.float32)
    state_by_gate = {
        "ambiguity": torch.tensor([[-0.5, 0.0, 0.0]], dtype=torch.float32),
        "informative": torch.tensor([[-0.32, 0.0, 0.0]], dtype=torch.float32),
    }
    dt = 0.2
    horizon = 40
    state_information = torch.diag(torch.tensor([0.0, 1.0, 0.0]))
    eye = torch.eye(3)
    scores = {}

    for gate, state in state_by_gate.items():
        f_theta = jacobian_param_torch(
            "confounded_gate", state, params, dynamics_alpha=1.0
        )[0].detach()
        f_state = jacobian_state_torch(
            "confounded_gate", state, params, dynamics_alpha=1.0
        )[0].detach()
        sensitivity = torch.zeros(3, 1)
        covariance = 4.0 * eye
        gate_scores = {
            "paldi": 0.0,
            "fully_observable": 0.0,
            "dynamics": 0.0,
            "state_information": 0.0,
        }
        for _ in range(horizon):
            transition = eye + f_state * dt
            sensitivity = transition @ sensitivity + f_theta * dt
            attenuated = attenuated_state_information(covariance, state_information)
            gate_scores["paldi"] += float(
                (sensitivity.T @ attenuated @ sensitivity).item()
            )
            gate_scores["fully_observable"] += float(
                (sensitivity.T @ state_information @ sensitivity).item()
            )
            gate_scores["dynamics"] += float(
                (sensitivity.T @ covariance @ sensitivity).item()
            )
            gate_scores["state_information"] += float(
                0.5 * torch.logdet(eye + covariance @ state_information).item()
            )
            covariance = transition @ covariance @ transition.T + 0.01 * eye
        scores[gate] = gate_scores

    assert scores["informative"]["paldi"] > scores["ambiguity"]["paldi"]
    for ablation in ("fully_observable", "dynamics", "state_information"):
        assert scores["ambiguity"][ablation] > scores["informative"][ablation]


def test_rank_imbalanced_gate_separates_logdet_and_e_optimality():
    from actdyn.environment.vectorfield import (
        jacobian_param_torch,
        jacobian_state_torch,
    )
    from actdyn.utils.torch_utils import attenuated_state_information

    params = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    state_by_gate = {
        "balanced": torch.tensor([[-0.5, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "main": torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    }
    dt = 0.2
    horizon = 40
    eye_state = torch.eye(4)
    eye_parameter = torch.eye(3)
    state_information = torch.diag(torch.tensor([0.0, 1.0, 1.0, 1.0]))
    scores = {}

    for gate, state in state_by_gate.items():
        f_theta = jacobian_param_torch(
            "rank_imbalanced_gate", state, params, dynamics_alpha=1.0
        )[0].detach()
        f_state = jacobian_state_torch(
            "rank_imbalanced_gate", state, params, dynamics_alpha=1.0
        )[0].detach()
        sensitivity = torch.zeros(4, 3)
        covariance = 0.25 * eye_state
        information = torch.zeros(3, 3)
        for _ in range(horizon):
            transition = eye_state + f_state * dt
            sensitivity = transition @ sensitivity + f_theta * dt
            attenuated = attenuated_state_information(covariance, state_information)
            information += sensitivity.T @ attenuated @ sensitivity
            covariance = transition @ covariance @ transition.T + 0.01 * eye_state

        scores[gate] = {
            "paldi": float(0.5 * torch.logdet(eye_parameter + information).item()),
            "e_optimality": float(torch.linalg.eigvalsh(information)[0].item()),
        }

    assert scores["main"]["paldi"] > scores["balanced"]["paldi"]
    assert scores["balanced"]["e_optimality"] > scores["main"]["e_optimality"]


def test_compound_tri_gate_canonical_fixed_gate_rankings():
    """Each objective selects its intended gate under the configured state Fisher."""
    from actdyn.environment.vectorfield import (
        jacobian_param_torch,
        jacobian_state_torch,
    )
    from actdyn.utils.torch_utils import attenuated_state_information

    params = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
    state_by_gate = {
        "ambiguity": torch.tensor([[-0.5, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "balanced": torch.tensor([[-0.32, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "main": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    }
    dt = 0.1
    horizon = 40
    eye_state = torch.eye(5)
    eye_parameter = torch.eye(3)
    state_information = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0, 0.01]))
    scores = {}

    for gate, state in state_by_gate.items():
        f_theta = jacobian_param_torch(
            "compound_tri_gate", state, params, dynamics_alpha=1.0
        )[0].detach()
        f_state = jacobian_state_torch(
            "compound_tri_gate", state, params, dynamics_alpha=1.0
        )[0].detach()
        sensitivity = torch.zeros(5, 3)
        covariance = eye_state.clone()
        information = torch.zeros(3, 3)
        fully_observed_information = torch.zeros(3, 3)
        dynamics_score = 0.0
        state_information_score = 0.0
        for _ in range(horizon):
            transition = eye_state + f_state * dt
            sensitivity = transition @ sensitivity + f_theta * dt
            attenuated = attenuated_state_information(covariance, state_information)
            information += sensitivity.T @ attenuated @ sensitivity
            fully_observed_information += (
                sensitivity.T @ state_information @ sensitivity
            )
            dynamics_score += float(
                torch.trace(sensitivity.T @ covariance @ sensitivity)
            )
            state_information_score += float(
                0.5 * torch.logdet(eye_state + covariance @ state_information)
            )
            covariance = transition @ covariance @ transition.T + 0.01 * eye_state

        scores[gate] = {
            "paldi": float(0.5 * torch.logdet(eye_parameter + information)),
            "fully_observable": float(
                0.5 * torch.logdet(eye_parameter + fully_observed_information)
            ),
            "e_optimality": float(torch.linalg.eigvalsh(information)[0]),
            "dynamics": dynamics_score,
            "state_information": state_information_score,
            "variance": float(torch.trace(f_theta.T @ f_theta)),
        }

    assert scores["main"]["paldi"] > scores["ambiguity"]["paldi"]
    assert scores["main"]["paldi"] > scores["balanced"]["paldi"]
    assert scores["ambiguity"]["fully_observable"] > scores["main"]["fully_observable"]
    assert scores["balanced"]["e_optimality"] > scores["main"]["e_optimality"]
    for objective in ("dynamics", "state_information", "variance"):
        assert scores["ambiguity"][objective] > scores["main"][objective]

    ambiguity_param = jacobian_param_torch(
        "compound_tri_gate", state_by_gate["ambiguity"], params, dynamics_alpha=1.0
    )[0]
    ambiguity_state = jacobian_state_torch(
        "compound_tri_gate", state_by_gate["ambiguity"], params, dynamics_alpha=1.0
    )[0]
    # theta_1 and h are collinear in the same response equation, with h amplified.
    assert float(ambiguity_state[1, 4].detach()) > 7.0 * float(
        ambiguity_param[1, 0].detach()
    )
    assert torch.linalg.svdvals(ambiguity_param)[1] < 0.01


def test_three_gate_diagnostic_fixed_gate_rankings_and_full_rank_main_gate():
    """Every nonzero true parameter remains identifiable at the full-rank M gate."""
    from actdyn.environment.vectorfield import (
        jacobian_param_torch,
        jacobian_state_torch,
    )
    from actdyn.utils.torch_utils import attenuated_state_information

    params = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    state_by_gate = {
        "ambiguity": torch.tensor([[-0.5, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "balanced": torch.tensor([[-0.1, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "main": torch.tensor([[0.3, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    }
    dt = 0.1
    horizon = 40
    eye_state = torch.eye(5)
    eye_parameter = torch.eye(3)
    # Ten paired Poisson neurons at z=0: two neurons per coordinate.
    state_information = torch.diag(torch.tensor([0.09, 0.09, 0.09, 0.09, 0.0025]))
    scores = {}

    for gate, state in state_by_gate.items():
        f_theta = jacobian_param_torch(
            "three_gate_diagnostic", state, params, dynamics_alpha=1.0
        )[0].detach()
        f_state = jacobian_state_torch(
            "three_gate_diagnostic", state, params, dynamics_alpha=1.0
        )[0].detach()
        sensitivity = torch.zeros(5, 3)
        covariance = eye_state.clone()
        information = torch.zeros(3, 3)
        fully_observed_information = torch.zeros(3, 3)
        dynamics_score = 0.0
        state_information_score = 0.0
        for _ in range(horizon):
            transition = eye_state + f_state * dt
            sensitivity = transition @ sensitivity + f_theta * dt
            attenuated = attenuated_state_information(covariance, state_information)
            information += sensitivity.T @ attenuated @ sensitivity
            fully_observed_information += (
                sensitivity.T @ state_information @ sensitivity
            )
            dynamics_score += float(
                torch.trace(sensitivity.T @ covariance @ sensitivity)
            )
            state_information_score += float(
                0.5 * torch.logdet(eye_state + covariance @ state_information)
            )
            covariance = transition @ covariance @ transition.T + 0.01 * eye_state

        scores[gate] = {
            "paldi": float(0.5 * torch.logdet(eye_parameter + information)),
            "fully_observable": float(
                0.5 * torch.logdet(eye_parameter + fully_observed_information)
            ),
            "e_optimality": float(torch.linalg.eigvalsh(information)[0]),
            "dynamics": dynamics_score,
            "state_information": state_information_score,
            "variance": float(torch.trace(f_theta.T @ f_theta)),
        }

    assert scores["main"]["paldi"] > scores["ambiguity"]["paldi"]
    assert scores["main"]["paldi"] > scores["balanced"]["paldi"]
    assert scores["ambiguity"]["fully_observable"] > scores["main"]["fully_observable"]
    assert scores["balanced"]["e_optimality"] > scores["main"]["e_optimality"]
    for objective in ("dynamics", "state_information", "variance"):
        assert scores["ambiguity"][objective] > scores["main"][objective]

    ambiguity_param = jacobian_param_torch(
        "three_gate_diagnostic", state_by_gate["ambiguity"], params, dynamics_alpha=1.0
    )[0]
    ambiguity_state = jacobian_state_torch(
        "three_gate_diagnostic", state_by_gate["ambiguity"], params, dynamics_alpha=1.0
    )[0]
    main_param = jacobian_param_torch(
        "three_gate_diagnostic", state_by_gate["main"], params, dynamics_alpha=1.0
    )[0]
    assert float(ambiguity_state[1, 4].detach()) == pytest.approx(100.0, abs=1e-4)
    nuisance_ratio = float(ambiguity_state[1, 4].detach()) / float(
        ambiguity_param[1, 0].detach()
    )
    assert nuisance_ratio == pytest.approx(5.0, abs=1e-4)
    assert torch.linalg.svdvals(main_param)[2] > 0.73
