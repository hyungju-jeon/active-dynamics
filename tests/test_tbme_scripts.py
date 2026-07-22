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

from actdyn.utils.experiment_runtime import read_trace_csv, write_trace_csv


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
    module = _load_module(
        "tbme_run_current", "experiments/tbme/run_tbme_experiments.py"
    )
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

    assert module._ASSET_PREDICTIVE_R2_LABEL == "Predictive R²"
    assert module._ASSET_FINAL_R2_LABEL == "Final predictive R²"
    assert module._asset_parse_r2_summaries("mean_sem,median_iqr") == [
        "mean_sem",
        "median_iqr",
    ]
    with pytest.raises(ValueError, match="Unknown R2 summary"):
        module._asset_parse_r2_summaries("median_sem")


def test_flex_comparison_asset_writes_mean_and_median_r2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.tbme import tbme_figures_assets as module

    policy_ids = ("flex", "flex_filter", "flex_true", "flex_rollback")
    refs = []
    for idx, exp_id in enumerate(
        (
            "duffing",
            "damped_pendulum",
            "gated_duffing",
            "gated_duffing_asymmetric",
            "gated_duffing_challenging",
            "gated_duffing_observation_bottleneck_mild",
            "gated_duffing_observation_bottleneck_strong",
        )
    ):
        suite_dir = tmp_path / "tracks" / exp_id
        summary_dir = suite_dir / "summary"
        write_trace_csv(
            summary_dir / "trajectory_r2_over_steps.csv",
            [
                {
                    "policy_id": policy_id,
                    "step": step,
                    "trajectory_r2_mean": 0.2 + 0.3 * step + 0.01 * policy_idx,
                    "value_sem": 0.02,
                    "value_median": 0.25 + 0.3 * step + 0.01 * policy_idx,
                    "value_q25": 0.20 + 0.3 * step + 0.01 * policy_idx,
                    "value_q75": 0.30 + 0.3 * step + 0.01 * policy_idx,
                    "cpu_time_sec_mean": float(step),
                }
                for policy_idx, policy_id in enumerate(policy_ids)
                for step in (0, 1, 2)
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
            ],
        )
        write_trace_csv(
            summary_dir / "metrics.csv",
            [
                {
                    "policy_id": policy_id,
                    "seed": seed,
                    "status": "completed",
                    "trajectory_r2_final_mean": (
                        ""
                        if exp_id == "gated_duffing"
                        and policy_id == "flex_filter"
                        and seed == 1
                        else 0.75 + 0.01 * seed
                    ),
                    "value_final_mean": 0.1 + 0.01 * seed,
                }
                for policy_id in policy_ids
                for seed in (0, 1)
            ],
            [
                "policy_id",
                "seed",
                "status",
                "trajectory_r2_final_mean",
                "value_final_mean",
            ],
        )
        refs.append(
            SimpleNamespace(
                suite_id=exp_id,
                label=f"Condition {idx}",
                session_root=tmp_path,
            )
        )

    from experiments.tbme.figures import groups as groups_mod

    monkeypatch.setitem(groups_mod.groups(), "flex_comparison", refs)
    mean_path = tmp_path / "assets" / "tbme_fig_flex_comparison.pdf"
    median_path = tmp_path / "assets" / "median_iqr" / "tbme_fig_flex_comparison.pdf"
    # One bar figure plus one recovery figure for each of the three condition groups.
    expected_stems = [
        f"tbme_fig_flex_comparison_{suffix}{recovery}"
        for suffix in ("baseline", "hard", "snr")
        for recovery in ("", "_recovery")
    ]

    mean_written = module._asset_plot_flex_comparison(mean_path, r2_summary="mean_sem")
    median_written = module._asset_plot_flex_comparison(
        median_path,
        r2_summary="median_iqr",
    )
    assert [path.stem for path in mean_written] == expected_stems
    assert [path.stem for path in median_written] == expected_stems
    assert all(path.exists() for path in mean_written + median_written)
    for suffix in ("baseline", "hard", "snr"):
        for base in (mean_path, median_path):
            bar_path = base.with_name(f"{base.stem}_{suffix}{base.suffix}")
            assert bar_path.with_suffix(".csv").exists()
    assert module._ASSET_FLEX_POLICIES == ("flex_true", "flex_filter", "flex_rollback")
    assert "flex_rollback" in module._ASSET_MATCHED_POLICIES
    assert "flex" not in module._ASSET_MATCHED_POLICIES
    # The variant labels are local to this figure; elsewhere flex_rollback is FLEX.
    assert module._asset_policy_label("flex_rollback") == "FLEX"
    assert (
        module._asset_policy_label("flex_rollback", module._ASSET_FLEX_LABELS)
        == "FLEX (EKF+stable)"
    )
    mean_rows = read_trace_csv(
        mean_path.with_name(f"{mean_path.stem}_baseline.csv")
    )
    failed_row = next(
        row
        for row in mean_rows
        if row["experiment"] == "gated_duffing" and row["policy_id"] == "flex_filter"
    )
    assert failed_row["policy_label"] == "FLEX (EKF)"
    assert failed_row["n_total"] == "2"
    assert failed_row["n_r2"] == "1"
    assert failed_row["n_r2_nonfinite"] == "1"
    assert failed_row["r2_nonfinite_rate"] == "0.5"


def _write_tri_gate_run(
    root: Path, *, policy_id: str, seed: int, final_error: float, final_r2: float
) -> None:
    run_dir = root / policy_id / f"seed_{seed}" / "repeat_01"
    run_dir.mkdir(parents=True)
    metadata = {
        "exp_id": "three_gate_diagnostic",
        "policy_id": policy_id,
        "seed": seed,
        "embedding_error_final": final_error,
        "dt": 0.01,
        "total_steps": 40,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    write_trace_csv(
        run_dir / "trajectory_r2_trace.csv",
        [
            {"step": 0, "trajectory_r2": 0.0},
            {"step": 20, "trajectory_r2": final_r2 / 2.0},
            {"step": 40, "trajectory_r2": final_r2},
        ],
        ["step", "trajectory_r2"],
    )
    # Selector dwells at rest, then gate A, then gate M.
    selector = [-1.0] * 10 + [-0.5] * 10 + [0.3] * 20
    write_trace_csv(
        run_dir / "state_action_trace.csv",
        [{"step": idx, "true_x": value} for idx, value in enumerate(selector)],
        ["step", "true_x"],
    )


def test_gate_diagnostic_asset_writes_figure_and_summary(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from experiments.tbme import tbme_figures_assets as module

    for seed in (0, 1):
        _write_tri_gate_run(
            tmp_path,
            policy_id="compound_active_planning",
            seed=seed,
            final_error=0.5 + 0.1 * seed,
            final_r2=0.9,
        )
        _write_tri_gate_run(
            tmp_path,
            policy_id="random",
            seed=seed,
            final_error=1.7,
            final_r2=0.2,
        )
    output_path = tmp_path / "assets" / "tbme_fig_gate_diagnostic.pdf"

    written = module._asset_plot_gate_diagnostic(
        output_path,
        r2_summary="median_iqr",
        result_roots=(tmp_path,),
        exemplar_seed=0,
    )

    assert written == output_path
    assert output_path.exists()
    # The suite ships with the objective_ablation group, so the assets CLI can
    # resolve its session tracks directory as the default result root.
    assert any(
        ref.suite_id == module._ASSET_TRI_GATE_EXP_ID
        for ref in __import__("experiments.tbme.figures.groups", fromlist=["groups"]).groups()["objective_ablation"]
    )
    rows = read_trace_csv(output_path.with_suffix(".csv"))
    assert [row["policy_id"] for row in rows] == [
        "compound_active_planning",
        "random",
    ]
    paldi_row = rows[0]
    assert paldi_row["label"] == "PALDI"
    assert paldi_row["n_seeds"] == "2"
    occupancy = [
        float(paldi_row[key])
        for key in ("rest_fraction", "gate_A_fraction", "gate_B_fraction", "gate_M_fraction")
    ]
    assert occupancy == pytest.approx([0.25, 0.25, 0.0, 0.5])

    trajectories_path = tmp_path / "assets" / "tbme_fig_gate_diagnostic_trajectories.pdf"
    assert (
        module._asset_plot_gate_diagnostic_trajectories(
            trajectories_path,
            result_roots=(tmp_path,),
            exemplar_seed=0,
        )
        == trajectories_path
    )
    assert trajectories_path.exists()

    with pytest.raises(RuntimeError, match="No trajectory R2 curves available"):
        module._asset_plot_gate_diagnostic(
            tmp_path / "assets" / "missing.pdf",
            r2_summary="median_iqr",
            result_roots=(tmp_path / "does_not_exist",),
        )


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
    module = _load_module(
        "tbme_catalogs_current", "experiments/tbme/run_tbme_experiments.py"
    )
    bundle = module.configure_tbme_catalogs()
    paths = module.tbme_catalog_paths()

    assert bundle.environment_catalog_paths == tuple(
        path.resolve() for path in paths["env_catalog_paths"]
    )
    assert bundle.model_catalog_paths == tuple(
        path.resolve() for path in paths["model_catalog_paths"]
    )
    assert bundle.suite_catalog_paths == ()
    assert (
        bundle.environment_presets["tbme_damped_pendulum"].system_id
        == "damped_pendulum"
    )
    assert bundle.environment_presets["tbme_gated_duffing"].system_id == "gated_duffing"
    assert bundle.environment_presets["tbme_gated_duffing"].embedding_dim == 4
    confounded = bundle.environment_presets["tbme_confounded_gate"]
    assert confounded.resolved_dynamics_type() == "confounded_gate"
    assert confounded.latent_dim == 3
    assert confounded.action_dim == 2
    assert confounded.resolved_state_bounds()[0].shape == (3,)
    assert np.allclose(
        confounded.filter_initial_state_mean_vector(),
        np.asarray([-0.5, 0.0, 0.0], dtype=np.float32),
    )
    assert confounded.observation_nuisance_scale == pytest.approx(0.02)
    rank_imbalanced = bundle.environment_presets["tbme_rank_imbalanced_gate"]
    assert rank_imbalanced.resolved_dynamics_type() == "rank_imbalanced_gate"
    assert rank_imbalanced.latent_dim == 4
    assert rank_imbalanced.action_dim == 1
    assert rank_imbalanced.embedding_dim == 3
    compound = bundle.environment_presets["tbme_compound_tri_gate"]
    assert compound.resolved_dynamics_type() == "compound_tri_gate"
    assert compound.latent_dim == 5
    assert compound.action_dim == 1
    assert compound.embedding_dim == 3
    assert compound.observation_model == "linear"
    assert compound.observation_noise_type == "gaussian"
    assert compound.observation_information_diag == pytest.approx((1, 1, 1, 1, 0.01))
    assert compound.trajectory_eval_state_noise == pytest.approx(0.0)
    assert compound.trajectory_eval_state_low[-1] == pytest.approx(0.0)
    assert compound.trajectory_eval_state_high[-1] == pytest.approx(0.0)
    compound_poisson = bundle.environment_presets["tbme_compound_tri_gate_poisson"]
    assert compound_poisson.resolved_dynamics_type() == "compound_tri_gate"
    assert compound_poisson.observation_model == "log_linear"
    assert compound_poisson.observation_noise_type == "poisson"
    assert compound_poisson.observation_dim == 160
    assert compound_poisson.observation_loading_design == "paired_diagonal"
    assert compound_poisson.observation_loading_gains == pytest.approx(
        (0.35, 0.35, 0.35, 0.35, 0.035)
    )
    assert compound_poisson.observation_loading_repeats_per_sign == 16
    simple = bundle.environment_presets["tbme_three_gate_diagnostic"]
    assert simple.resolved_dynamics_type() == "three_gate_diagnostic"
    assert simple.latent_dim == 5
    assert simple.action_dim == 1
    assert simple.embedding_dim == 3
    assert simple.resolved_true_params() == pytest.approx((1.0, 1.0, 1.0))
    assert simple.initial_parameter_mean == pytest.approx((0.0, 0.0, 0.0))
    assert np.all(
        simple.initial_parameter_mean_vector() != simple.resolved_true_params()
    )
    assert simple.filter_initial_state_mean_vector() == pytest.approx(
        (-1.0, 0.0, 0.0, 0.0, 0.0)
    )
    assert simple.action_max == pytest.approx(1.5)
    assert simple.observation_model == "log_linear"
    assert simple.observation_noise_type == "poisson"
    assert simple.observation_dim == 10
    assert simple.observation_loading_gains == pytest.approx((0.3, 0.3, 0.3, 0.3, 0.05))
    assert simple.observation_loading_repeats_per_sign == 1
    assert simple.state_init_uncertainty == pytest.approx(1.0)
    assert simple.trajectory_eval_state_indices == (1, 2, 3)
    assert simple.trajectory_eval_coordinate_balanced is True
    assert np.allclose(
        bundle.environment_presets[
            "tbme_duffing_parameter_mismatch"
        ].resolved_true_params(estimator=True),
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
    assert policies["compound_active_planning"].action_cost_weight == pytest.approx(0.0)
    assert policies["off_policy"].schedule_id == "u1_r1_h1"
    assert policies["flex"].policy_type == "flex"
    assert policies["flex"].flex_regularization == pytest.approx(0.01)
    assert policies["flex"].flex_parameter_step_clip == pytest.approx(0.25)
    assert policies["flex_filter"].policy_type == "flex-upstream"
    assert policies["flex_filter"].use_true_state is False
    assert policies["flex_filter"].flex_parameter_step_clip is None
    assert policies["flex_true"].policy_type == "flex-upstream"
    assert policies["flex_true"].use_true_state is True
    assert policies["flex_rollback"].policy_type == "flex-rollback"
    assert policies["flex_rollback"].flex_regularization == pytest.approx(0.1)
    assert policies["flex_rollback"].flex_parameter_step_clip == pytest.approx(0.25)
    assert policies["rhc"].policy_type == "rhc"
    assert "rhc_mvr" not in policies
    assert policies["active_state_variance"].objective_kind == "state_variance"
    assert (
        policies["active_observation_variance"].objective_kind == "observation_variance"
    )
    assert policies["active_e_optimality"].objective_kind == "e_optimality"
    assert policies["active_fully_observable"].objective_kind == (
        "fully_observable_parameter_eig"
    )
    assert bundle.schedule_specs["active_planning_u1_r1_h40"].planning_horizon == 40
    assert (
        policies["active_planning_u5_r5_h40"].schedule_id == "active_planning_u5_r5_h40"
    )

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
        "flex_filter",
        "flex_true",
        "flex_rollback",
    )
    assert "baseline_prbs" not in duffing.policy_ids
    assert "active_planning_u5_r5_h40" not in duffing.policy_ids
    assert "active_e_optimality" in bundle.experiment_specs["gated_duffing"].policy_ids
    objective_ablation = bundle.experiment_specs["gated_duffing_asymmetric"]
    assert "active_observation_variance" in objective_ablation.policy_ids
    assert "active_state_variance" in objective_ablation.policy_ids
    confounded_suite = _load_module(
        "tbme_confounded_gate_suite",
        "experiments/tbme/exp_objective_ablation.py",
    ).EXPERIMENT_SUITES["confounded_gate"]
    assert confounded_suite["env_preset_id"] == "tbme_confounded_gate"
    assert "active_planning" in confounded_suite["model_ids"]
    assert "random" in confounded_suite["model_ids"]
    rank_suite = _load_module(
        "tbme_rank_imbalanced_gate_suite",
        "experiments/tbme/exp_objective_ablation.py",
    ).EXPERIMENT_SUITES["rank_imbalanced_gate"]
    assert rank_suite["env_preset_id"] == "tbme_rank_imbalanced_gate"
    assert rank_suite["model_ids"] == [
        "active_planning",
        "active_e_optimality",
        "prbs",
        "random",
    ]
    compound_suite = _load_module(
        "tbme_compound_tri_gate_suite",
        "experiments/tbme/exp_objective_ablation.py",
    ).EXPERIMENT_SUITES["compound_tri_gate"]
    assert compound_suite["env_preset_id"] == "tbme_compound_tri_gate"
    assert compound_suite["total_steps"] == 2000
    assert compound_suite["model_ids"] == [
        "compound_active_planning",
        "compound_active_fully_observable",
        "compound_active_e_optimality",
        "compound_active_state_information",
        "compound_active_dynamics",
        "compound_active_observation_variance",
        "compound_active_state_variance",
        "prbs",
        "random",
    ]
    simple_suite = _load_module(
        "tbme_three_gate_diagnostic_suite",
        "experiments/tbme/exp_objective_ablation.py",
    ).EXPERIMENT_SUITES["three_gate_diagnostic"]
    assert simple_suite["env_preset_id"] == "tbme_three_gate_diagnostic"
    assert simple_suite["total_steps"] == 2000
    assert simple_suite["model_ids"] == compound_suite["model_ids"]


def test_tbme_runtime_config_respects_catalog_policy_type_for_prbs(tmp_path: Path):
    catalogs = _load_module(
        "tbme_catalogs_runtime_prbs", "experiments/tbme/run_tbme_experiments.py"
    )
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


def test_tbme_runner_seeds_sync_icem_action_sampling():
    catalogs = _load_module(
        "tbme_catalogs_runtime_icem",
        "experiments/tbme/run_tbme_experiments.py",
    )
    specs = catalogs.configure_tbme_catalogs(suite_entries={})
    runner = _load_module("experiment_runner_runtime_icem", "experiments/run.py")
    import actdyn

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    fake_env = SimpleNamespace(action_space=action_space)
    fake_model = SimpleNamespace(
        action_encoder=SimpleNamespace(action_space=action_space)
    )
    env_preset = specs.environment_presets["tbme_duffing"]
    policy_spec = specs.policy_specs["active_planning"]
    schedule_spec = specs.schedule_specs[policy_spec.schedule_id]

    policy = runner._instantiate_synthetic_policy(
        actdyn_module=actdyn,
        env=fake_env,
        env_preset=env_preset,
        model=fake_model,
        metric=None,
        device="cpu",
        policy_id="active_planning",
        policy_spec=policy_spec,
        schedule_spec=schedule_spec,
        seed=29,
    )

    assert policy._action_rng_seed == 29


def test_tbme_runner_instantiates_flex_policy_as_exact_flex():
    catalogs = _load_module(
        "tbme_catalogs_runtime_flex", "experiments/tbme/run_tbme_experiments.py"
    )
    specs = catalogs.configure_tbme_catalogs(suite_entries={})
    runner = _load_module("experiment_runner_runtime_flex", "experiments/run.py")
    import actdyn

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    fake_env = SimpleNamespace(action_space=action_space)
    fake_model = SimpleNamespace(
        action_encoder=SimpleNamespace(action_space=action_space)
    )
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


@pytest.mark.parametrize(
    ("policy_id", "class_name", "use_true_state", "rollback"),
    [
        ("flex_filter", "FLEXUpstreamPolicy", False, False),
        ("flex_true", "FLEXUpstreamPolicy", True, False),
        ("flex_rollback", "FLEXPolicy", False, True),
    ],
)
def test_tbme_runner_instantiates_additive_flex_variants(
    policy_id: str,
    class_name: str,
    use_true_state: bool,
    rollback: bool,
) -> None:
    catalogs = _load_module("tbme_catalogs_runtime_flex_variants", "experiments/tbme/run_tbme_experiments.py")
    specs = catalogs.configure_tbme_catalogs(suite_entries={})
    runner = _load_module("experiment_runner_runtime_flex_variants", "experiments/run.py")
    import actdyn

    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    fake_env = SimpleNamespace(action_space=action_space)
    fake_model = SimpleNamespace(action_encoder=SimpleNamespace(action_space=action_space))
    env_preset = specs.environment_presets["tbme_duffing"]
    policy_spec = specs.policy_specs[policy_id]
    schedule_spec = specs.schedule_specs[policy_spec.schedule_id]

    policy = runner._instantiate_synthetic_policy(
        actdyn_module=actdyn,
        env=fake_env,
        env_preset=env_preset,
        model=fake_model,
        metric=None,
        device="cpu",
        policy_id=policy_id,
        policy_spec=policy_spec,
        schedule_spec=schedule_spec,
        seed=0,
    )

    assert policy.__class__.__name__ == class_name
    assert policy.use_observed_state is use_true_state
    assert policy.rollback_unstable_update is rollback


def test_flex_comparison_suite_has_requested_environments_and_models() -> None:
    suite = _load_module("tbme_flex_comparison", "experiments/tbme/exp_flex_comparison.py")
    assert tuple(suite.EXPERIMENT_SUITES) == (
        "duffing",
        "damped_pendulum",
        "gated_duffing",
        "gated_duffing_asymmetric",
        "gated_duffing_challenging",
        "gated_duffing_observation_bottleneck_mild",
        "gated_duffing_observation_bottleneck_strong",
    )
    assert suite.MODEL_IDS == ["flex", "flex_filter", "flex_true", "flex_rollback"]
    assert all(
        spec["env_preset_id"] == f"tbme_{exp_id}"
        for exp_id, spec in suite.EXPERIMENT_SUITES.items()
    )


def test_tbme_runner_instantiates_exact_rhc_policy():
    pytest.importorskip("casadi")
    catalogs = _load_module(
        "tbme_catalogs_runtime_rhc", "experiments/tbme/run_tbme_experiments.py"
    )
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

    action_seq, cost = policy.get_action(
        torch.zeros(1, 1), observed_state=torch.zeros(1, 1)
    )
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
    module = _load_module(
        "tbme_run_family_current", "experiments/tbme/run_tbme_experiments.py"
    )
    suites, groups = module._shared_tbme_data()
    assert set(groups) == {
        "simple_system_identification",
        "observation_action_bottleneck",
        "model_mismatch",
        "objective_ablation",
        "scheduling",
        "flex_comparison",
    }
    assert [entry["suite_id"] for entry in groups["simple_system_identification"]] == [
        "duffing",
        "damped_pendulum",
        "gated_duffing",
    ]
    assert "gated_duffing_parameter_mismatch_mild" in suites
    assert "gated_duffing_observation_bottleneck_mild" in suites
    assert [entry["suite_id"] for entry in groups["flex_comparison"]] == [
        "duffing",
        "damped_pendulum",
        "gated_duffing",
        "gated_duffing_asymmetric",
        "gated_duffing_challenging",
        "gated_duffing_observation_bottleneck_mild",
        "gated_duffing_observation_bottleneck_strong",
    ]
    assert all(
        entry["policy_ids"] == ("flex_filter", "flex_true", "flex_rollback")
        for entry in groups["flex_comparison"]
    )


def test_duffing_parameter_mismatch_uses_fixed_non_inferred_cubic():
    catalogs = _load_module(
        "tbme_catalogs_param_mismatch", "experiments/tbme/run_tbme_experiments.py"
    )
    specs = catalogs.configure_tbme_catalogs(suite_entries={})
    preset = specs.environment_presets["tbme_duffing_parameter_mismatch"]
    params = preset.params_from_embedding(
        np.asarray([-0.5, -0.75], dtype=np.float32), estimator=True
    )
    assert np.allclose(params, np.asarray([-0.5, -0.75, 0.2], dtype=np.float32))


def test_trajectory_r2_accounts_for_estimator_system_mismatch():
    catalogs = _load_module(
        "tbme_catalogs_traj_r2_param_mismatch",
        "experiments/tbme/run_tbme_experiments.py",
    )
    specs = catalogs.configure_tbme_catalogs(suite_entries={})
    true_preset = specs.environment_presets["tbme_duffing"]
    mismatch_preset = specs.environment_presets["tbme_duffing_parameter_mismatch"]
    from actdyn.utils.validation import trajectory_r2_vectorfield

    r2 = trajectory_r2_vectorfield(
        e_est=torch.as_tensor(
            mismatch_preset.true_embedding_vector(estimator=True), dtype=torch.float32
        ),
        e_true=torch.as_tensor(
            true_preset.true_embedding_vector(), dtype=torch.float32
        ),
        true_dynamics_type=str(true_preset.resolved_dynamics_type()),
        true_full_params=true_preset.resolved_true_params(),
        estimator_dynamics_type=str(
            mismatch_preset.resolved_dynamics_type(estimator=True)
        ),
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
    from actdyn.utils.validation import (
        trajectory_r2_vectorfield,
        trajectory_r2_vectorfield_many,
    )

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


def test_trajectory_r2_many_supports_three_state_vectorfields():
    from actdyn.utils.validation import trajectory_r2_vectorfield_many

    observed = trajectory_r2_vectorfield_many(
        e_estimates=torch.tensor([[0.5], [0.0]], dtype=torch.float32),
        e_true=torch.tensor([0.5], dtype=torch.float32),
        true_dynamics_type="confounded_gate",
        true_full_params=np.asarray([0.5], dtype=np.float32),
        estimator_dynamics_type="confounded_gate",
        estimator_full_params=np.asarray([0.5], dtype=np.float32),
        true_min_embedding_dim=1,
        estimator_min_embedding_dim=1,
        dt=0.2,
        dynamics_alpha=1.0,
        horizon=40,
        n_starts=5,
        rng=np.random.default_rng(123),
        device="cpu",
        state_dim=3,
    )

    assert observed.shape == (2,)
    assert observed[0] == pytest.approx(1.0, abs=1e-6)
    assert observed[1] < observed[0]


def test_trajectory_r2_many_supports_targeted_compound_gate_starts():
    from actdyn.utils.validation import trajectory_r2_vectorfield_many

    observed = trajectory_r2_vectorfield_many(
        e_estimates=torch.tensor(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
        ),
        e_true=torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        true_dynamics_type="compound_tri_gate",
        true_full_params=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        estimator_dynamics_type="compound_tri_gate",
        estimator_full_params=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        true_min_embedding_dim=3,
        estimator_min_embedding_dim=3,
        dt=0.01,
        dynamics_alpha=1.0,
        horizon=100,
        n_starts=8,
        rng=np.random.default_rng(123),
        device="cpu",
        state_dim=5,
        state_noise=0.0,
        state_low=np.asarray([-0.55, -0.25, -0.25, -0.25, 0.0]),
        state_high=np.asarray([0.05, 0.25, 0.25, 0.25, 0.0]),
    )

    assert observed.shape == (2,)
    assert observed[0] == pytest.approx(1.0, abs=1e-6)
    assert observed[1] < 0.9


def test_trajectory_r2_many_scores_all_three_gate_diagnostic_parameters():
    """The fixed validation metric must expose errors in theta_1, theta_2, and theta_3."""
    from actdyn.utils.validation import trajectory_r2_vectorfield_many

    observed = trajectory_r2_vectorfield_many(
        e_estimates=torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        e_true=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        true_dynamics_type="three_gate_diagnostic",
        true_full_params=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        estimator_dynamics_type="three_gate_diagnostic",
        estimator_full_params=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        true_min_embedding_dim=3,
        estimator_min_embedding_dim=3,
        dt=0.01,
        dynamics_alpha=1.0,
        horizon=100,
        n_starts=8,
        rng=np.random.default_rng(123),
        device="cpu",
        state_dim=5,
        state_noise=0.0,
        state_low=np.asarray([-1.05, -0.25, -0.25, -0.25, 0.0]),
        state_high=np.asarray([0.35, 0.25, 0.25, 0.25, 0.0]),
        state_indices=(1, 2, 3),
        coordinate_balanced=True,
    )

    assert observed.shape == (4,)
    assert observed[0] == pytest.approx(1.0, abs=1e-6)
    assert np.all(observed[1:] < 0.99)


def test_three_gate_diagnostic_reach_hold_baseline_separates_transit_from_dwell():
    from experiments.tbme.tbme_figures_experiment import (
        _reach_hold_selector_occupancy,
    )

    occupancy = _reach_hold_selector_occupancy(
        rest_center=-1.0,
        target_center=0.3,
        gate_centers=(-0.5, -0.1, 0.3),
        rest_cutoff=-0.75,
        selector_contraction=1.0,
        dt=0.01,
        total_steps=2000,
    )

    np.testing.assert_allclose(
        occupancy,
        np.asarray([0.011, 0.0275, 0.055, 0.9065]),
        rtol=0.0,
        atol=1e-8,
    )


def test_tbme_all_runner_parser_accepts_expected_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_module(
        "tbme_run_all_current", "experiments/tbme/run_tbme_experiments.py"
    )
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
    assert captured["suite_catalog_paths"] == [
        "experiments/tbme/config/experiment_suite.yaml"
    ]
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
