from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
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


def test_tbme_runner_parser_accepts_expected_args():
    module = _load_module("tbme_run_exp1_v1", "experiments/tbme/run_exp1.py")
    captured: dict[str, object] = {}

    def _fake_run_family(*, argv, suite_ids, default_base_dir):
        captured["argv"] = list(argv)
        captured["suite_ids"] = list(suite_ids)
        captured["default_base_dir"] = default_base_dir
        return 17

    module.run_family = _fake_run_family
    exit_code = module.main(["--mode", "summary", "--seeds", "0,10", "--base-dir", "results/tbme"])
    assert exit_code == 17
    assert captured["argv"] == ["--mode", "summary", "--seeds", "0,10", "--base-dir", "results/tbme"]
    assert captured["suite_ids"] == [
        "tbme_exp1_duffing_policy",
        "tbme_exp1_duffing_policy_sota",
        "tbme_exp1_pendulum_policy",
        "tbme_exp1_pendulum_policy_sota",
        "tbme_exp1_double_integrator_policy",
        "tbme_exp1_double_integrator_policy_sota",
        "tbme_exp1_objective_duffing",
        "tbme_exp1_duffing_challenge_policy",
        "tbme_exp1_duffing_challenge_sota",
        "tbme_exp1_duffing_budget_ablation_short",
        "tbme_exp1_duffing_budget_ablation_medium",
        "tbme_exp1_duffing_ig_ablation",
        "tbme_exp1_duffing_schedule_ablation",
        "tbme_exp1_duffing_competitor_compare",
        "tbme_exp1_two_basin_policy",
        "tbme_exp1_two_basin_budget_ablation",
    ]
    assert captured["default_base_dir"] == "results/tbme/exp1"


def test_tbme_catalog_define_expected_matrices():
    module = _load_module("tbme_specs_v1", "experiments/tbme/catalog.py")
    assert Path(module.DEFAULT_ENV_CATALOG_PATH).exists()
    assert Path(module.DEFAULT_MODEL_CATALOG_PATH).exists()
    assert Path(module.DEFAULT_SUITE_CATALOG_PATH).exists()
    assert module.ENVIRONMENT_PRESETS["tbme_pendulum_damped"].system_id == "damped_pendulum"
    assert module.ENVIRONMENT_PRESETS["tbme_double_integrator"].system_id == "double_integrator"
    assert module.ENVIRONMENT_PRESETS["tbme_duffing_planning_challenge"].observation_dim == 24
    assert module.ENVIRONMENT_PRESETS["tbme_two_basin_bridge"].system_id == "two_basin_bridge"
    assert module.ENVIRONMENT_PRESETS["tbme_two_basin_bridge"].embedding_dim == 4
    assert module.ENVIRONMENT_PRESETS["tbme_two_basin_bridge"].action_max == pytest.approx(0.35)
    assert (
        module.ENVIRONMENT_PRESETS["tbme_duffing_family_mismatch"].estimator_system_id
        == "damped_pendulum"
    )
    assert module.ENVIRONMENT_PRESETS["tbme_duffing_family_mismatch"].asymmetric_loading is False
    assert (
        module.ENVIRONMENT_PRESETS["tbme_duffing_parameter_mismatch"].estimator_system_id
        == "single_attractor_cubic_0p2"
    )
    assert module.ENVIRONMENT_PRESETS["tbme_mcrtt_spikes"].real_data is True
    from actdyn.environment import get_planar_system_defaults, env_params_from_embedding, true_embedding

    mismatch_defaults = get_planar_system_defaults("single_attractor_cubic_0p2")
    assert mismatch_defaults["dynamics_type"] == "duffing"
    assert mismatch_defaults["true_params"][2] == pytest.approx(0.2)
    env_params = env_params_from_embedding("single_attractor_cubic_0p2", np.asarray([0.1, -0.2], dtype=np.float32))
    assert np.allclose(env_params, np.asarray([0.1, -0.2, 0.2], dtype=np.float32))
    env_params_3d = env_params_from_embedding(
        "single_attractor", np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    )
    assert np.allclose(env_params_3d, np.asarray([0.1, -0.2, 0.3], dtype=np.float32))
    assert np.allclose(
        true_embedding("single_attractor", embedding_dim=2),
        np.asarray([-0.55, 1.0], dtype=np.float32),
    )
    assert np.allclose(
        true_embedding("single_attractor", embedding_dim=3),
        np.asarray([-0.55, 1.0, 0.1], dtype=np.float32),
    )
    assert module.POLICY_SPECS["baseline_prbs"].policy_type == "baseline-prbs"
    assert module.POLICY_SPECS["flex"].policy_type == "flex"
    assert module.POLICY_SPECS["rhc"].policy_type == "rhc"
    assert module.POLICY_SPECS["rhc_mvr"].policy_type == "rhc"
    assert module.POLICY_SPECS["ensemble"].objective_kind == "state_variance"
    assert module.POLICY_SPECS["e_optimality"].objective_kind == "e_optimality"
    assert module.SCHEDULE_SPECS["tbme_u1_r1_h40"].planning_horizon == 40
    assert module.POLICY_SPECS["ig_full_observable"].objective_kind == "fully_observable_parameter_eig"
    assert module.POLICY_SPECS["sched_h20_u5_r5"].schedule_id == "u5_r5_h20"
    assert module.EXPERIMENT_SPECS["tbme_exp1_duffing_policy"].policy_ids == (
        "active_myopic",
        "active_planning",
        "baseline_prbs",
        "random",
        "off_policy",
    )
    assert module.EXPERIMENT_SPECS["tbme_exp1_duffing_challenge_policy"].env_preset_id == (
        "tbme_duffing_planning_challenge"
    )
    assert module.EXPERIMENT_SPECS["tbme_exp1_duffing_policy_sota"].policy_ids == (
        "active_myopic",
        "active_planning",
        "flex",
        "rhc",
        "baseline_prbs",
        "random",
    )
    assert module.EXPERIMENT_SPECS["tbme_exp1_pendulum_policy_sota"].policy_ids == (
        "active_myopic",
        "active_planning",
        "flex",
        "rhc",
        "baseline_prbs",
        "random",
    )
    assert module.EXPERIMENT_SPECS["tbme_exp1_double_integrator_policy_sota"].policy_ids == (
        "active_myopic",
        "active_planning",
        "flex",
        "rhc",
        "baseline_prbs",
        "random",
    )
    assert module.EXPERIMENT_SPECS["tbme_exp1_duffing_challenge_sota"].total_steps == 2000
    assert module.EXPERIMENT_SPECS["tbme_exp1_duffing_budget_ablation_short"].total_steps == 200
    assert module.EXPERIMENT_SPECS["tbme_exp1_duffing_ig_ablation"].policy_ids == (
        "ig_parameter",
        "ig_full_observable",
        "ig_e_optimality",
        "ig_state_information",
        "ig_dynamics",
        "ig_sampling_variance",
    )
    assert module.EXPERIMENT_SPECS["tbme_exp1_duffing_schedule_ablation"].policy_ids[-1] == "baseline_prbs"
    assert module.EXPERIMENT_SPECS["tbme_exp1_two_basin_policy"].env_preset_id == "tbme_two_basin_bridge"
    assert module.EXPERIMENT_SPECS["tbme_exp1_two_basin_policy"].policy_ids == (
        "active_myopic",
        "active_planning",
        "baseline_prbs",
        "random",
    )
    assert module.EXPERIMENT_SPECS["tbme_exp1_two_basin_budget_ablation"].total_steps == 200
    assert (
        module.EXPERIMENT_SPECS["tbme_exp2_robustness_duffing"].env_preset_id
        == "tbme_duffing_family_mismatch"
    )
    assert module.EXPERIMENT_SPECS["tbme_exp2_robustness_duffing_sota"].policy_ids == (
        "active_planning",
        "flex",
        "rhc",
        "baseline_prbs",
        "random",
    )
    assert (
        module.EXPERIMENT_SPECS["tbme_exp2_robustness_duffing_parameter"].env_preset_id
        == "tbme_duffing_parameter_mismatch"
    )
    assert module.EXPERIMENT_SPECS["tbme_exp2_robustness_duffing_parameter_sota"].policy_ids == (
        "active_planning",
        "flex",
        "rhc",
        "baseline_prbs",
        "random",
    )
    assert module.EXPERIMENT_SPECS["tbme_exp3_realdata_policy"].experiment_kind == "realdata"


def test_tbme_runtime_config_respects_catalog_policy_type_for_prbs(tmp_path: Path):
    specs = _load_module("tbme_specs_runtime_prbs", "experiments/tbme/catalog.py")
    runner = _load_module("experiment_runner_runtime_prbs", "experiments/run.py")
    env_preset = specs.get_environment_preset("tbme_duffing_easy")
    schedule_spec = specs.get_schedule_spec("u1_r1_h1")
    policy_spec = specs.get_policy_spec("baseline_prbs")
    cfg = runner._build_runtime_experiment_config(
        run_dir=tmp_path / "run",
        seed=3,
        total_steps=50,
        experiment_kind="duffing",
        policy_id="baseline_prbs",
        env_preset=env_preset,
        schedule_spec=schedule_spec,
        policy_spec=policy_spec,
    )
    assert cfg.policy.policy_type == "baseline-prbs"


def test_tbme_runner_instantiates_flex_policy_as_exact_flex():
    specs = _load_module("tbme_specs_runtime_flex", "experiments/tbme/catalog.py")
    runner = _load_module("experiment_runner_runtime_flex", "experiments/run.py")
    import actdyn

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    fake_env = SimpleNamespace(action_space=action_space)
    fake_model = SimpleNamespace(action_encoder=SimpleNamespace(action_space=action_space))
    env_preset = specs.get_environment_preset("tbme_duffing_easy")
    policy_spec = specs.get_policy_spec("flex")
    schedule_spec = specs.get_schedule_spec(policy_spec.schedule_id)

    policy = runner._instantiate_synthetic_policy(
        actdyn_module=actdyn,
        env=fake_env,
        env_preset=env_preset,
        model=fake_model,
        metric=None,
        device="cpu",
        policy_id="flex",
        policy_spec=policy_spec,
        schedule_spec=schedule_spec,
        seed=0,
    )

    assert policy.__class__.__name__ == "FLEXPolicy"
    assert policy.chunk == 1


def test_tbme_runner_instantiates_exact_rhc_policy():
    pytest.importorskip("casadi")
    specs = _load_module("tbme_specs_runtime_rhc", "experiments/tbme/catalog.py")
    runner = _load_module("experiment_runner_runtime_rhc", "experiments/run.py")
    import actdyn

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    fake_env = SimpleNamespace(action_space=action_space)
    fake_model = SimpleNamespace()
    policy_spec = specs.get_policy_spec("rhc")
    schedule_spec = specs.get_schedule_spec(policy_spec.schedule_id)

    policy = runner._instantiate_synthetic_policy(
        actdyn_module=actdyn,
        env=fake_env,
        model=fake_model,
        metric=None,
        device="cpu",
        policy_id="rhc",
        policy_spec=policy_spec,
        schedule_spec=schedule_spec,
        seed=0,
    )

    assert policy.__class__.__name__ == "RecedingHorizonCuriosityPolicy"
    assert policy.chunk == schedule_spec.planning_horizon
    assert policy.objective == "rhc_us"
    assert policy.prior_precision == pytest.approx(1e-8)
    assert policy.beta == pytest.approx(1.0)
    assert policy.optimize_hyperparams is True
    assert policy.warm_start is False


def test_exact_rhc_core_plans_and_updates_one_episode():
    pytest.importorskip("casadi")
    import actdyn.policy.baseline_rhc as baseline_rhc

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    policy = baseline_rhc.RecedingHorizonCuriosityPolicy(
        action_space=action_space,
        horizon=2,
        num_features=8,
        planner_maxiter=10,
        optimize_hyperparams=False,
        device="cpu",
        seed=0,
    )

    action_seq, cost = policy.get_action(torch.zeros(1, 1), observed_state=torch.zeros(1, 1))
    assert action_seq.shape == (1, 2, 1)
    assert torch.isfinite(cost)

    first_info = policy.update(
        {
            "env_state": torch.tensor([[[0.0]]], dtype=torch.float32),
            "next_env_state": torch.tensor([[[0.1]]], dtype=torch.float32),
            "env_action": action_seq[:, :1],
        }
    )
    assert first_info["parameter_posterior_updated"] is False
    info = policy.update(
        {
            "env_state": torch.tensor([[[0.1]]], dtype=torch.float32),
            "next_env_state": torch.tensor([[[0.15]]], dtype=torch.float32),
            "env_action": action_seq[:, 1:2],
        }
    )
    assert policy._internal_model is not None
    assert policy._internal_model.num_samples == 2
    assert policy._internal_model.prior_precision == pytest.approx(1e-8)
    assert info["parameter_posterior_updated"] is True
    assert info["rhc_episode_index"] == 1.0


def test_tbme_family_scripts_define_expected_suite_sets():
    exp2 = _load_module("tbme_run_exp2_v1", "experiments/tbme/run_exp2.py")
    assert exp2.EXP2_SUITES == [
        "tbme_exp2_robustness_duffing",
        "tbme_exp2_robustness_duffing_sota",
        "tbme_exp2_robustness_duffing_parameter",
        "tbme_exp2_robustness_duffing_parameter_sota",
    ]


def test_duffing_parameter_mismatch_uses_fixed_non_inferred_cubic():
    module = _load_module("planar_systems_duffing_param_mismatch_v1", "experiments/_OLD/cosyne/planar_systems.py")
    params = module.env_params_from_embedding(
        "single_attractor_cubic_0p2",
        np.asarray([-0.55, 1.0], dtype=np.float32),
    )
    assert np.allclose(params, np.asarray([-0.55, 1.0, 0.2], dtype=np.float32))


def test_trajectory_r2_accounts_for_estimator_system_mismatch():
    runner = _load_module("experiment_runner_traj_r2_param_mismatch_v1", "experiments/run.py")
    r2 = runner._trajectory_r2(
        e_est=torch.tensor([-0.55, 1.0], dtype=torch.float32),
        e_true=torch.tensor([-0.55, 1.0], dtype=torch.float32),
        system_id="single_attractor",
        estimator_system_id="single_attractor_cubic_0p2",
        dt=0.01,
        dynamics_alpha=1.0,
        horizon=50,
        n_starts=8,
        rng=np.random.default_rng(0),
        device="cpu",
    )
    assert r2 < 0.999


def test_tbme_exp3_runner_parser_accepts_expected_args(tmp_path: Path):
    module = _load_module("tbme_run_exp3_v2", "experiments/tbme/run_exp3.py")
    config = module.load_config(str(REPO_ROOT / "experiments" / "tbme" / "exp3_digital_twin.yaml"))
    captured: dict[str, object] = {}

    module.load_config = lambda path: config
    module.resolve_session_root = lambda base_dir, create: tmp_path / "session_1"

    def _fake_run_workflow(*, config, session_root, mode):
        captured["config"] = config
        captured["session_root"] = session_root
        captured["mode"] = mode
        return 19

    module.run_workflow = _fake_run_workflow
    exit_code = module.main(
        [
            "--mode",
            "benchmark",
            "--base-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--latent-dim",
            "4",
            "--control-dim",
            "3",
            "--benchmark-steps",
            "12",
            "--seeds",
            "0,7",
            "--policy-ids",
            "active_myopic,random",
        ]
    )
    assert exit_code == 19
    assert captured["mode"] == "benchmark"
    assert captured["session_root"] == tmp_path / "session_1"
    cfg = captured["config"]
    assert cfg.runtime.device == "cpu"
    assert cfg.generator.latent_dim == 4
    assert cfg.twin.control_dim == 3
    assert cfg.benchmark.total_steps == 12
    assert cfg.benchmark.seeds == [0, 7]
    assert cfg.benchmark.policy_ids == ["active_myopic", "random"]


def test_root_runner_catalog_preparse_does_not_treat_mode_as_model_catalog():
    module = _load_module("experiment_runner_catalog_preparse", "experiments/run.py")
    captured: dict[str, object] = {}

    def _fake_configure_catalogs(*, env_catalog_paths=None, model_catalog_paths=None, suite_catalog_paths=None):
        captured["env_catalog_paths"] = env_catalog_paths
        captured["model_catalog_paths"] = model_catalog_paths
        captured["suite_catalog_paths"] = suite_catalog_paths
        raise RuntimeError("stop after catalog capture")

    module.configure_catalogs = _fake_configure_catalogs
    with pytest.raises(RuntimeError, match="stop after catalog capture"):
        module.main(
            [
                "--env-catalog",
                "experiments/experiment_env.yaml",
                "--env-catalog",
                "experiments/tbme/experiment_env.yaml",
                "--model-catalog",
                "experiments/experiment_model.yaml",
                "--model-catalog",
                "experiments/tbme/experiment_model.yaml",
                "--suite-catalog",
                "experiments/tbme/experiment_suite.yaml",
                "--mode",
                "summary",
            ]
        )

    assert captured["env_catalog_paths"] == [
        "experiments/experiment_env.yaml",
        "experiments/tbme/experiment_env.yaml",
    ]
    assert captured["model_catalog_paths"] == [
        "experiments/experiment_model.yaml",
        "experiments/tbme/experiment_model.yaml",
    ]
    assert captured["suite_catalog_paths"] == ["experiments/tbme/experiment_suite.yaml"]


def test_summary_reports_first_trajectory_r2_threshold_crossing():
    module = _load_module("tbme_summary_thresholds", "experiments/summarize.py")
    rows = [
        {
            "policy_id": "active_planning",
            "step": 10,
            "value_mean": 0.90,
            "cpu_time_sec_mean": 1.0,
            "n_points": 4,
        },
        {
            "policy_id": "active_planning",
            "step": 20,
            "value_mean": 0.96,
            "cpu_time_sec_mean": 2.0,
            "n_points": 4,
        },
        {
            "policy_id": "active_planning",
            "step": 30,
            "value_mean": 0.995,
            "cpu_time_sec_mean": 3.0,
            "n_points": 4,
        },
        {
            "policy_id": "random",
            "step": 10,
            "value_mean": 0.80,
            "cpu_time_sec_mean": 0.5,
            "n_points": 4,
        },
    ]

    summary = module.summarize_trajectory_r2_thresholds(rows)
    by_policy = {row["policy_id"]: row for row in summary}

    assert by_policy["active_planning"]["step_to_r2_0p90"] == 10
    assert by_policy["active_planning"]["cpu_time_sec_to_r2_0p90"] == 1.0
    assert by_policy["active_planning"]["step_to_r2_0p95"] == 20
    assert by_policy["active_planning"]["cpu_time_sec_to_r2_0p95"] == 2.0
    assert by_policy["active_planning"]["step_to_r2_0p99"] == 30
    assert by_policy["active_planning"]["cpu_time_sec_to_r2_0p99"] == 3.0
    assert by_policy["random"]["step_to_r2_0p90"] is None
    assert by_policy["random"]["step_to_r2_0p95"] is None
    assert by_policy["random"]["step_to_r2_0p99"] is None
