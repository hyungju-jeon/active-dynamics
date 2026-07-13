from pathlib import Path

import pytest

from actdyn.utils.experiment_runtime import read_trace_csv, write_trace_csv
from experiments.experiment_io import (
    experiment_env_slug,
    experiment_run_dir,
    experiment_summary_dir,
    write_json,
)
from experiments.experiment_definitions import get_environment_preset, get_experiment_spec
from experiments.summarize import (
    aggregate_trace,
    aggregate_trajectory_r2_trace,
    collect_track_records,
    main as summarize_main,
)
from experiments.tbme import tbme_figures_summary as tbme_summary
from experiments.tbme.run_tbme_experiments import (
    _shared_tbme_data,
    configure_tbme_catalogs,
)
from experiments.tbme.tbme_figures_summary import (
    SUMMARY_POLICY_FAMILIES,
    _get_policy_families,
)


def test_tbme_env_slug_uses_gated_duffing_name() -> None:
    assert experiment_env_slug("tbme_gated_duffing") == "gated_duffing"
    assert (
        experiment_env_slug("tbme_gated_duffing_parameter_mismatch_mild")
        == "gated_duffing_parameter_mismatch_mild"
    )
    assert experiment_env_slug("tbme_duffing") == "duffing"


def test_shared_tbme_suites_dedupe_and_merge_methods() -> None:
    suites, groups = _shared_tbme_data()
    assert "gated_duffing" in suites
    assert all(not suite_id.startswith("exp") for suite_id in suites)
    assert all(spec["env_preset_id"] == f"tbme_{suite_id}" for suite_id, spec in suites.items())
    assert all(
        not source_id.startswith("exp")
        for spec in suites.values()
        for source_id in spec["source_exp_ids"]
    )

    methods = tuple(suites["gated_duffing"]["model_ids"])
    assert methods.count("active_planning") == 1
    assert "active_myopic" in methods
    assert "active_e_optimality" in methods
    assert "active_planning_u1_r1_h40" in methods
    assert suites["gated_duffing_parameter_mismatch_mild"]["source_exp_ids"] == [
        "gated_duffing_parameter_mismatch_mild"
    ]
    assert suites["gated_duffing_parameter_mismatch_mild"]["source_modules"] == ["exp_model_mismatch"]
    assert suites["gated_duffing_parameter_mismatch_strong"]["source_exp_ids"] == [
        "gated_duffing_parameter_mismatch_strong"
    ]
    assert suites["gated_duffing_parameter_mismatch_strong"]["source_modules"] == ["exp_model_mismatch"]

    scheduling = {
        item["suite_id"]: set(item["policy_ids"])
        for item in groups["scheduling"]
    }
    assert "active_planning_u1_r1_h40" in scheduling["gated_duffing"]
    assert "active_planning" not in scheduling["gated_duffing"]

    objective = {
        item["suite_id"]: set(item["policy_ids"])
        for item in groups["objective_ablation"]
    }
    assert "active_e_optimality" in objective["gated_duffing"]
    assert "prbs" not in objective["gated_duffing"]


def test_summary_policy_families_skip_incomplete_family() -> None:
    baselines = tuple(SUMMARY_POLICY_FAMILIES["baselines"])
    policy_ids = set(baselines) | {"adaptive_async_realtime"}

    families = dict(_get_policy_families(policy_ids))

    assert families["baselines"] == list(baselines)
    assert "adaptive" not in families


def test_summary_final_trajectory_r2_uses_trajectory_r2_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    (summary_dir / "metrics.csv").write_text(
        "\n".join(
            [
                "policy_id,seed,n_repeats,status,value_final_mean,trajectory_r2_final_mean,runtime_sec_mean",
                "active_planning,0,1,completed,0.3,0.7,1.0",
            ]
        ),
        encoding="utf-8",
    )
    captured_rows = {}

    def _capture_final_plot(_figures_dir, **kwargs):
        captured_rows[str(kwargs["output_stem"])] = kwargs["rows"]

    def _skip_plot(*_args, **_kwargs):
        return None

    monkeypatch.setattr(tbme_summary, "plot_final_value_by_policy", _capture_final_plot)
    monkeypatch.setattr(tbme_summary, "plot_metric_over_steps", _skip_plot)
    monkeypatch.setattr(tbme_summary, "plot_metric_over_cpu_time", _skip_plot)
    monkeypatch.setattr(tbme_summary, "_get_action_magnitude", _skip_plot)

    tbme_summary.make_summary_metric_figures(
        tmp_path,
        [".pdf"],
        policy_ids=["active_planning"],
        family_id="adaptive",
    )

    rows = captured_rows["final_trajectory_r2_by_policy_adaptive"]
    assert rows[0]["value_final_mean"] == "0.7"


def test_tbme_tracks_layout_paths_use_env_method_seed_repeat(tmp_path: Path) -> None:
    configure_tbme_catalogs()
    exp_spec = get_experiment_spec("gated_duffing")

    assert experiment_run_dir(
        tmp_path,
        exp_spec,
        "random",
        seed=2,
        repeat=3,
        layout="tbme_tracks",
    ) == tmp_path / "tracks" / "gated_duffing" / "random" / "seed_2" / "repeat_03"
    assert experiment_summary_dir(
        tmp_path,
        exp_spec,
        layout="tbme_tracks",
    ) == tmp_path / "tracks" / "gated_duffing" / "summary"


def test_summary_collects_records_from_tbme_tracks_layout(tmp_path: Path) -> None:
    configure_tbme_catalogs()
    run_dir = tmp_path / "tracks" / "gated_duffing" / "random" / "seed_0" / "repeat_01"
    write_json(
        run_dir / "run_metadata.json",
        {
            "status": "completed",
            "policy_id": "random",
            "seed": 0,
        },
    )

    records, missing = collect_track_records(
        tmp_path,
        "gated_duffing",
        [0],
        policy_filter={"random"},
        layout="tbme_tracks",
    )

    assert not missing
    assert len(records) == 1
    assert records[0]["run_dir"] == run_dir


def test_summary_cpu_time_uses_cumulative_loop_compute(tmp_path: Path) -> None:
    run_dir = tmp_path / "tracks" / "duffing" / "adaptive_async_realtime" / "seed_0" / "repeat_01"
    write_json(
        run_dir / "run_metadata.json",
        {
            "policy_id": "adaptive_async_realtime",
            "seed": 0,
            "parameter_error_trace_path": str(run_dir / "parameter_error_trace.csv"),
            "information_trace_path": str(run_dir / "information_trace.csv"),
        },
    )
    write_trace_csv(
        run_dir / "parameter_error_trace.csv",
        [
            {"step": 1, "cpu_time_sec": 100.0, "parameter_error": 2.0},
            {"step": 2, "cpu_time_sec": 200.0, "parameter_error": 1.0},
        ],
        ["step", "cpu_time_sec", "parameter_error"],
    )
    write_trace_csv(
        run_dir / "information_trace.csv",
        [
            {"step": 1, "cpu_time_sec": 100.0, "loop_compute_sec": 0.01},
            {"step": 2, "cpu_time_sec": 200.0, "loop_compute_sec": 0.02},
        ],
        ["step", "cpu_time_sec", "loop_compute_sec"],
    )
    records = [
        {
            "policy_id": "adaptive_async_realtime",
            "seed": 0,
            "run_dir": run_dir,
            "metadata": {},
        }
    ]

    rows = aggregate_trace(
        records,
        metadata_key="parameter_error_trace_path",
        fallback_name="parameter_error_trace.csv",
        value_col="parameter_error",
    )

    assert [row["cpu_time_sec_mean"] for row in rows] == pytest.approx([0.01, 0.03])


def test_trajectory_r2_summary_reports_median_and_interquartile_range(
    tmp_path: Path,
) -> None:
    records = []
    for seed, value in enumerate((0.0, 1.0, 2.0, 9.0)):
        run_dir = tmp_path / f"seed_{seed}"
        write_trace_csv(
            run_dir / "trajectory_r2_trace.csv",
            [{"step": 10, "cpu_time_sec": float(seed), "trajectory_r2": value}],
            ["step", "cpu_time_sec", "trajectory_r2"],
        )
        records.append(
            {
                "policy_id": "adaptive",
                "seed": seed,
                "run_dir": run_dir,
                "metadata": {},
            }
        )

    rows = aggregate_trajectory_r2_trace(records, exp_spec=None)

    assert len(rows) == 1
    assert rows[0]["value_mean"] == pytest.approx(3.0)
    assert rows[0]["value_median"] == pytest.approx(1.5)
    assert rows[0]["value_q25"] == pytest.approx(0.75)
    assert rows[0]["value_q75"] == pytest.approx(3.75)
    assert rows[0]["n_points"] == 4


def test_summary_recomputes_trajectory_r2_from_embedding_trace(tmp_path: Path) -> None:
    configure_tbme_catalogs()
    exp_spec = get_experiment_spec("duffing")
    env_preset = get_environment_preset("tbme_duffing")
    run_dir = tmp_path / "tracks" / "duffing" / "random" / "seed_0" / "repeat_01"
    e_true = env_preset.true_embedding_vector(embedding_dim=2)
    metadata = {
        "status": "completed",
        "exp_id": "duffing",
        "env_preset_id": "tbme_duffing",
        "policy_id": "random",
        "seed": 0,
        "runtime_sec": 1.0,
        "embedding_error_final": 0.0,
        "embedding_true": [float(x) for x in e_true.tolist()],
        "embedding_estimate": [float(x) for x in e_true.tolist()],
        "dynamics_type": env_preset.resolved_dynamics_type(),
        "estimator_dynamics_type": env_preset.resolved_dynamics_type(estimator=True),
        "true_params_full": [float(x) for x in env_preset.resolved_true_params()],
        "estimator_true_params_full": [
            float(x) for x in env_preset.resolved_true_params(estimator=True)
        ],
        "min_embedding_dim": env_preset.resolved_min_embedding_dim(),
        "state_noise": 0.0,
        "trajectory_eval_interval": exp_spec.trajectory_eval_interval,
        "trajectory_eval_horizon": 3,
        "trajectory_eval_samples": 4,
        "embedding_estimate_trace_path": str(run_dir / "embedding_estimate_trace.csv"),
        "information_trace_path": str(run_dir / "information_trace.csv"),
    }
    write_json(run_dir / "run_metadata.json", metadata)
    write_trace_csv(
        run_dir / "embedding_estimate_trace.csv",
        [
            {
                "step": 0,
                "cpu_time_sec": 0.0,
                "embedding_dim": 2,
                "e0": float(e_true[0]),
                "e1": float(e_true[1]),
                "cov_diag_mean": 1.0,
            },
            {
                "step": exp_spec.trajectory_eval_interval,
                "cpu_time_sec": 0.1,
                "embedding_dim": 2,
                "e0": float(e_true[0]),
                "e1": float(e_true[1]),
                "cov_diag_mean": 0.5,
            },
        ],
        ["step", "cpu_time_sec", "embedding_dim", "e0", "e1", "cov_diag_mean"],
    )
    write_trace_csv(
        run_dir / "information_trace.csv",
        [
            {"step": 0, "cpu_time_sec": 10.0, "loop_compute_sec": 0.01},
            {
                "step": exp_spec.trajectory_eval_interval,
                "cpu_time_sec": 20.0,
                "loop_compute_sec": 0.02,
            },
        ],
        ["step", "cpu_time_sec", "loop_compute_sec"],
    )

    exit_code = summarize_main(
        [
            "--base-dir",
            str(tmp_path),
            "--exp-id",
            "duffing",
            "--summary-dir",
            str(tmp_path / "summary"),
            "--policy-ids",
            "random",
            "--seeds",
            "0",
            "--path-layout",
            "tbme_tracks",
        ]
    )

    assert exit_code == 0
    traj_rows = read_trace_csv(run_dir / "trajectory_r2_trace.csv")
    assert [int(row["step"]) for row in traj_rows] == [0, exp_spec.trajectory_eval_interval]
    assert [float(row["cpu_time_sec"]) for row in traj_rows] == pytest.approx([0.01, 0.03])
    summary_traj_rows = read_trace_csv(tmp_path / "summary" / "trajectory_r2_over_steps.csv")
    assert [float(row["cpu_time_sec_mean"]) for row in summary_traj_rows] == pytest.approx(
        [0.01, 0.03]
    )
    for row in summary_traj_rows:
        assert float(row["value_median"]) == pytest.approx(float(row["trajectory_r2_mean"]))
        assert float(row["value_q25"]) == pytest.approx(float(row["trajectory_r2_mean"]))
        assert float(row["value_q75"]) == pytest.approx(float(row["trajectory_r2_mean"]))
    metrics_rows = read_trace_csv(tmp_path / "summary" / "metrics.csv")
    assert float(metrics_rows[0]["trajectory_r2_final_mean"]) >= 0.999
