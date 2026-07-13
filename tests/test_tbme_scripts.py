from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from actdyn.utils.experiment_runtime import write_trace_csv


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


def test_tbme_runner_parser_accepts_expected_args(monkeypatch: pytest.MonkeyPatch):
    module = _load_module("tbme_run_current", "experiments/tbme/run_tbme_experiments.py")
    captured: dict[str, object] = {}

    def _fake_main(argv, *, suite_entries=None):
        captured["argv"] = list(argv)
        captured["suite_entries"] = suite_entries
        return 17

    monkeypatch.setattr(module.shared_run, "main", _fake_main)
    exit_code = module.main(
        [
            "exp_simple_system_identification",
            "--exp-ids",
            "duffing",
            "--mode",
            "summary",
            "--seeds",
            "0,10",
            "--base-dir",
            "results/tbme",
        ]
    )

    paths = module.tbme_catalog_paths()
    expected_argv: list[str] = []
    for path in paths["env_catalog_paths"]:
        expected_argv.extend(["--env-catalog", str(path)])
    for path in paths["model_catalog_paths"]:
        expected_argv.extend(["--model-catalog", str(path)])
    expected_argv.extend(
        [
            "--exp-ids",
            "duffing",
            "--base-dir",
            "results/tbme",
            "--seeds",
            "0,10",
            "--path-layout",
            "tbme_tracks",
            "--mode",
            "summary",
        ]
    )

    assert exit_code == 17
    assert captured["argv"] == expected_argv
    assert sorted(captured["suite_entries"]) == [
        "damped_pendulum",
        "duffing",
        "gated_duffing",
    ]


def _write_r2_ceiling_metadata(suite_dir: Path, *, state_noise: float) -> None:
    run_dir = suite_dir / "adaptive" / "seed_0" / "repeat_01"
    run_dir.mkdir(parents=True)
    metadata = {
        "dynamics_alpha": 1.0,
        "dynamics_type": "gated_duffing",
        "embedding_true": [-1.2, -0.8, 0.5, 1.1],
        "env_preset_id": "tbme_gated_duffing",
        "min_embedding_dim": 2,
        "state_noise": state_noise,
        "status": "completed",
        "trajectory_eval_horizon": 4,
        "trajectory_eval_samples": 32,
        "true_params_full": [-1.2, -0.8, 0.5, 1.1],
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_asset_true_model_r2_ceiling_is_one_without_process_noise(tmp_path: Path):
    from experiments.tbme import tbme_figures_assets as module

    _write_r2_ceiling_metadata(tmp_path, state_noise=0.0)

    assert module._asset_true_model_r2_ceiling(tmp_path) == 1.0


def test_asset_true_model_r2_ceiling_reflects_process_noise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from experiments.tbme import tbme_figures_assets as module

    monkeypatch.setattr(module, "_ASSET_R2_CEILING_REPEATS", 4)
    _write_r2_ceiling_metadata(tmp_path, state_noise=0.8)

    ceiling = module._asset_true_model_r2_ceiling(tmp_path)

    assert ceiling is not None
    assert ceiling < 1.0


def test_asset_median_iqr_uses_summary_quantiles_and_seed_final_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.tbme import tbme_figures_assets as module

    summary_dir = tmp_path / "summary"
    write_trace_csv(
        summary_dir / "trajectory_r2_over_steps.csv",
        [
            {
                "policy_id": "adaptive",
                "step": 10,
                "trajectory_r2_mean": 3.0,
                "value_sem": 1.0,
                "value_median": 1.5,
                "value_q25": 0.75,
                "value_q75": 3.75,
                "cpu_time_sec_mean": 2.0,
                "n_points": 4,
            }
        ],
        [
            "policy_id",
            "step",
            "trajectory_r2_mean",
            "value_sem",
            "value_median",
            "value_q25",
            "value_q75",
            "cpu_time_sec_mean",
            "n_points",
        ],
    )
    write_trace_csv(
        summary_dir / "metrics.csv",
        [
            {
                "policy_id": "adaptive",
                "seed": seed,
                "status": "completed",
                "trajectory_r2_final_mean": value,
            }
            for seed, value in enumerate((0.0, 1.0, 2.0, 9.0))
        ],
        ["policy_id", "seed", "status", "trajectory_r2_final_mean"],
    )

    mean_curve = module._asset_r2_curve_rows(tmp_path, r2_summary="mean_sem")["adaptive"][0]
    curve = module._asset_r2_curve_rows(tmp_path, r2_summary="median_iqr")["adaptive"][0]
    mean_final = module._asset_final_r2_summary(
        tmp_path,
        "adaptive",
        r2_summary="mean_sem",
    )
    final = module._asset_final_r2_summary(
        tmp_path,
        "adaptive",
        r2_summary="median_iqr",
    )

    assert (mean_curve["center"], mean_curve["lower"], mean_curve["upper"]) == pytest.approx(
        (3.0, 2.0, 4.0)
    )
    assert mean_final[0] == pytest.approx(3.0)
    assert mean_final[0] - mean_final[1] == pytest.approx(mean_final[2] - mean_final[0])
    assert mean_final[3] == 4
    assert curve["center"] == pytest.approx(1.5)
    assert curve["lower"] == pytest.approx(0.75)
    assert curve["upper"] == pytest.approx(3.75)
    assert final == pytest.approx((1.5, 0.75, 3.75, 4))

    monkeypatch.setattr(module, "_asset_true_model_r2_ceiling", lambda *args, **kwargs: None)
    source = SimpleNamespace(exp_id="condition", label="Condition", suite_dir=tmp_path)
    metric_rows = module._asset_method_metric_rows(
        [source],
        ["adaptive"],
        r2_summary="median_iqr",
    )
    csv_path = tmp_path / "median_iqr" / "metrics.csv"
    module._asset_write_method_csv(
        csv_path,
        metric_rows,
        r2_summary="median_iqr",
    )
    recovery_path = tmp_path / "median_iqr" / "recovery.pdf"
    bar_path = tmp_path / "median_iqr" / "final.pdf"

    assert "_trajectory_r2_center" not in csv_path.read_text(encoding="utf-8")
    assert module._asset_plot_recovery_curves(
        recovery_path,
        sources=[source],
        policy_ids=["adaptive"],
        r2_summary="median_iqr",
    ) == recovery_path
    assert module._asset_plot_final_bar(
        bar_path,
        sources=[source],
        policy_ids=["adaptive"],
        metric_rows=metric_rows,
    ) == bar_path
    assert recovery_path.exists()
    assert bar_path.exists()


def test_asset_r2_summary_selection_is_explicit() -> None:
    from experiments.tbme import tbme_figures_assets as module

    assert module._asset_parse_r2_summaries("mean_sem,median_iqr") == [
        "mean_sem",
        "median_iqr",
    ]
    with pytest.raises(ValueError, match="Unknown R2 summary"):
        module._asset_parse_r2_summaries("median_sem")


def test_objective_ablation_plot_handles_three_sources(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from experiments.tbme.tbme_figures_experiment import plot_objective_ablation

    policy_id = "active_dynamics"
    sources = [
        SimpleNamespace(exp_id=f"condition_{idx}", label=f"Condition {idx}")
        for idx in range(3)
    ]
    metric_rows = [
        {
            "experiment": source.exp_id,
            "policy_id": policy_id,
            "step_to_r2_0p95": 10.0 + idx,
        }
        for idx, source in enumerate(sources)
    ]
    curves_by_source = {
        source.exp_id: {
            policy_id: [
                {"step": 0.0, "value": 0.1, "sem": 0.0},
                {"step": 1.0, "value": 0.96, "sem": 0.01},
            ]
        }
        for source in sources
    }
    output_path = tmp_path / "objective_ablation.pdf"

    written = plot_objective_ablation(
        output_path,
        sources=sources,
        metric_rows=metric_rows,
        curves_by_source=curves_by_source,
        policy_ids=[policy_id],
        policy_label=lambda _policy_id: "Active dynamics",
        policy_color=lambda _policy_id: "#1f77b4",
        apply_style=None,
        style_axis=lambda _ax: None,
        stroke_color="#000000",
        neutral_light="#cccccc",
    )

    assert written == output_path
    assert output_path.exists()


def test_tbme_catalog_define_expected_matrices():
    module = _load_module("tbme_catalogs_current", "experiments/tbme/run_tbme_experiments.py")
    bundle = module.configure_tbme_catalogs()
    paths = module.tbme_catalog_paths()

    assert bundle.environment_catalog_paths == tuple(path.resolve() for path in paths["env_catalog_paths"])
    assert bundle.model_catalog_paths == tuple(path.resolve() for path in paths["model_catalog_paths"])
    assert bundle.suite_catalog_paths == ()
    assert bundle.environment_presets["tbme_damped_pendulum"].system_id == "damped_pendulum"
    assert bundle.environment_presets["tbme_gated_duffing"].system_id == "gated_duffing"
    assert bundle.environment_presets["tbme_gated_duffing"].embedding_dim == 4
    assert np.allclose(
        bundle.environment_presets["tbme_duffing_parameter_mismatch"].resolved_true_params(estimator=True),
        np.asarray([-0.5, -0.75, 0.2], dtype=np.float32),
    )

    policies = bundle.policy_specs
    assert "baseline_random" not in policies
    assert "baseline_prbs" not in policies
    assert "off-policy" not in policies
    assert policies["prbs"].policy_type == "baseline-prbs"
    assert policies["random"].policy_type == "random"
    assert policies["random"].schedule_id == "u1_r1_h1"
    assert policies["off_policy"].policy_type == "off-policy"
    assert policies["off_policy"].schedule_id == "u1_r1_h1"
    assert policies["flex"].policy_type == "flex"
    assert policies["rhc"].policy_type == "rhc"
    assert "rhc_mvr" not in policies
    assert policies["active_state_variance"].objective_kind == "state_variance"
    assert policies["active_observation_variance"].objective_kind == "observation_variance"
    assert policies["active_e_optimality"].objective_kind == "e_optimality"
    assert policies["active_fully_observable"].objective_kind == (
        "fully_observable_parameter_eig"
    )
    assert bundle.schedule_specs["active_planning_u1_r1_h40"].planning_horizon == 40
    assert policies["active_planning_u5_r5_h40"].schedule_id == "active_planning_u5_r5_h40"

    duffing = bundle.experiment_specs["duffing"]
    assert duffing.env_preset_id == "tbme_duffing"
    assert duffing.policy_ids == (
        "adaptive",
        "adaptive_async_anytime",
        "adaptive_async_realtime",
        "active_planning",
        "active_myopic",
        "prbs",
        "random",
        "flex",
        "flex_true_state",
        "rhc",
        "off_policy",
    )
    assert "baseline_prbs" not in duffing.policy_ids
    assert "active_planning_u5_r5_h40" not in duffing.policy_ids
    assert "active_e_optimality" in bundle.experiment_specs["gated_duffing"].policy_ids
    objective_ablation = bundle.experiment_specs["gated_duffing_asymmetric"]
    assert "active_observation_variance" in objective_ablation.policy_ids
    assert "active_state_variance" in objective_ablation.policy_ids


def test_tbme_runtime_config_respects_catalog_policy_type_for_prbs(tmp_path: Path):
    catalogs = _load_module("tbme_catalogs_runtime_prbs", "experiments/tbme/run_tbme_experiments.py")
    specs = catalogs.configure_tbme_catalogs(suite_entries={})
    runner = _load_module("experiment_runner_runtime_prbs", "experiments/run.py")
    env_preset = specs.environment_presets["tbme_duffing"]
    policy_spec = specs.policy_specs["prbs"]
    schedule_spec = specs.schedule_specs[policy_spec.schedule_id]
    cfg = runner._build_runtime_experiment_config(
        run_dir=tmp_path / "run",
        seed=3,
        total_steps=50,
        experiment_kind="duffing",
        policy_id="prbs",
        env_preset=env_preset,
        schedule_spec=schedule_spec,
        policy_spec=policy_spec,
    )
    assert cfg.policy.policy_type == "baseline-prbs"


def test_tbme_runner_instantiates_flex_policy_as_exact_flex():
    catalogs = _load_module("tbme_catalogs_runtime_flex", "experiments/tbme/run_tbme_experiments.py")
    specs = catalogs.configure_tbme_catalogs(suite_entries={})
    runner = _load_module("experiment_runner_runtime_flex", "experiments/run.py")
    import actdyn

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    fake_env = SimpleNamespace(action_space=action_space)
    fake_model = SimpleNamespace(action_encoder=SimpleNamespace(action_space=action_space))
    env_preset = specs.environment_presets["tbme_duffing"]
    policy_spec = specs.policy_specs["flex"]
    schedule_spec = specs.schedule_specs[policy_spec.schedule_id]

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
    catalogs = _load_module("tbme_catalogs_runtime_rhc", "experiments/tbme/run_tbme_experiments.py")
    specs = catalogs.configure_tbme_catalogs(suite_entries={})
    runner = _load_module("experiment_runner_runtime_rhc", "experiments/run.py")
    import actdyn

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    fake_env = SimpleNamespace(action_space=action_space)
    fake_model = SimpleNamespace()
    env_preset = specs.environment_presets["tbme_duffing"]
    policy_spec = specs.policy_specs["rhc"]
    schedule_spec = specs.schedule_specs[policy_spec.schedule_id]

    policy = runner._instantiate_synthetic_policy(
        actdyn_module=actdyn,
        env=fake_env,
        env_preset=env_preset,
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
    assert policy.optimize_hyperparams is False
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
    module = _load_module("tbme_run_family_current", "experiments/tbme/run_tbme_experiments.py")
    suites, groups = module._shared_tbme_data()
    assert set(groups) == {
        "simple_system_identification",
        "observation_action_bottleneck",
        "model_mismatch",
        "objective_ablation",
        "scheduling",
    }
    assert [entry["suite_id"] for entry in groups["simple_system_identification"]] == [
        "duffing",
        "damped_pendulum",
        "gated_duffing",
    ]
    assert "gated_duffing_parameter_mismatch_mild" in suites
    assert "gated_duffing_observation_bottleneck_mild" in suites


def test_duffing_parameter_mismatch_uses_fixed_non_inferred_cubic():
    catalogs = _load_module("tbme_catalogs_param_mismatch", "experiments/tbme/run_tbme_experiments.py")
    specs = catalogs.configure_tbme_catalogs(suite_entries={})
    preset = specs.environment_presets["tbme_duffing_parameter_mismatch"]
    params = preset.params_from_embedding(np.asarray([-0.5, -0.75], dtype=np.float32), estimator=True)
    assert np.allclose(params, np.asarray([-0.5, -0.75, 0.2], dtype=np.float32))


def test_trajectory_r2_accounts_for_estimator_system_mismatch():
    catalogs = _load_module("tbme_catalogs_traj_r2_param_mismatch", "experiments/tbme/run_tbme_experiments.py")
    specs = catalogs.configure_tbme_catalogs(suite_entries={})
    true_preset = specs.environment_presets["tbme_duffing"]
    mismatch_preset = specs.environment_presets["tbme_duffing_parameter_mismatch"]
    from actdyn.utils.validation import trajectory_r2_vectorfield

    r2 = trajectory_r2_vectorfield(
        e_est=torch.as_tensor(mismatch_preset.true_embedding_vector(estimator=True), dtype=torch.float32),
        e_true=torch.as_tensor(true_preset.true_embedding_vector(), dtype=torch.float32),
        true_dynamics_type=str(true_preset.resolved_dynamics_type()),
        true_full_params=true_preset.resolved_true_params(),
        estimator_dynamics_type=str(mismatch_preset.resolved_dynamics_type(estimator=True)),
        estimator_full_params=mismatch_preset.resolved_true_params(estimator=True),
        true_min_embedding_dim=int(true_preset.resolved_min_embedding_dim()),
        estimator_min_embedding_dim=int(mismatch_preset.resolved_min_embedding_dim()),
        dt=0.01,
        dynamics_alpha=1.0,
        horizon=50,
        n_starts=8,
        rng=np.random.default_rng(0),
        device="cpu",
    )
    assert r2 < 0.999


def test_trajectory_r2_many_matches_repeated_single_call():
    from actdyn.utils.validation import trajectory_r2_vectorfield, trajectory_r2_vectorfield_many

    e_true = torch.tensor([-0.55, 1.0], dtype=torch.float32)
    estimates = torch.tensor(
        [
            [-0.55, 1.0],
            [-0.45, 0.85],
            [-0.65, 1.15],
        ],
        dtype=torch.float32,
    )
    kwargs = {
        "true_dynamics_type": "duffing",
        "true_full_params": np.asarray([-0.55, 1.0, 0.1], dtype=np.float32),
        "estimator_dynamics_type": "duffing",
        "estimator_full_params": np.asarray([-0.55, 1.0, 0.1], dtype=np.float32),
        "true_min_embedding_dim": 2,
        "estimator_min_embedding_dim": 2,
        "dt": 0.01,
        "dynamics_alpha": 1.0,
        "horizon": 8,
        "n_starts": 5,
        "device": "cpu",
        "state_noise": 0.1,
    }

    sequential_rng = np.random.default_rng(123)
    expected = np.asarray(
        [
            trajectory_r2_vectorfield(
                e_est=estimate,
                e_true=e_true,
                rng=sequential_rng,
                **kwargs,
            )
            for estimate in estimates
        ],
        dtype=np.float32,
    )
    observed = trajectory_r2_vectorfield_many(
        e_estimates=estimates,
        e_true=e_true,
        rng=np.random.default_rng(123),
        **kwargs,
    )

    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)


def test_tbme_all_runner_parser_accepts_expected_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_module("tbme_run_all_current", "experiments/tbme/run_tbme_experiments.py")
    captured: dict[str, object] = {}

    def _fake_main(argv, *, suite_entries=None):
        captured["argv"] = list(argv)
        captured["suite_entries"] = suite_entries
        return 19

    monkeypatch.setattr(module.shared_run, "main", _fake_main)
    exit_code = module.main(
        [
            "all",
            "--exp-ids",
            "duffing",
            "--base-dir",
            str(tmp_path),
            "--seeds",
            "0",
            "--mode",
            "summary",
            "--policy-ids",
            "random",
        ]
    )
    assert exit_code == 19
    assert captured["argv"][-4:] == ["--mode", "summary", "--policy-ids", "random"]
    assert captured["suite_entries"]["duffing"]["env_preset_id"] == "tbme_duffing"
    assert "random" in captured["suite_entries"]["duffing"]["model_ids"]
    assert "baseline_prbs" not in captured["suite_entries"]["duffing"]["model_ids"]


def test_root_runner_catalog_preparse_does_not_treat_mode_as_model_catalog():
    module = _load_module("experiment_runner_catalog_preparse", "experiments/run.py")
    captured: dict[str, object] = {}

    def _fake_configure_catalogs(
        *,
        env_catalog_paths=None,
        model_catalog_paths=None,
        suite_catalog_paths=None,
        suite_entries=None,
    ):
        captured["env_catalog_paths"] = env_catalog_paths
        captured["model_catalog_paths"] = model_catalog_paths
        captured["suite_catalog_paths"] = suite_catalog_paths
        captured["suite_entries"] = suite_entries
        raise RuntimeError("stop after catalog capture")

    module.configure_catalogs = _fake_configure_catalogs
    with pytest.raises(RuntimeError, match="stop after catalog capture"):
        module.main(
            [
                "--env-catalog",
                "experiments/experiment_env.yaml",
                "--env-catalog",
                "experiments/tbme/config/experiment_env.yaml",
                "--model-catalog",
                "experiments/experiment_model.yaml",
                "--model-catalog",
                "experiments/tbme/config/experiment_model.yaml",
                "--suite-catalog",
                "experiments/tbme/config/experiment_suite.yaml",
                "--mode",
                "summary",
            ]
        )

    assert captured["env_catalog_paths"] == [
        "experiments/experiment_env.yaml",
        "experiments/tbme/config/experiment_env.yaml",
    ]
    assert captured["model_catalog_paths"] == [
        "experiments/experiment_model.yaml",
        "experiments/tbme/config/experiment_model.yaml",
    ]
    assert captured["suite_catalog_paths"] == ["experiments/tbme/config/experiment_suite.yaml"]
    assert captured["suite_entries"] is None


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
