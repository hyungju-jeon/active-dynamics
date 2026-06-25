from pathlib import Path

import pytest

from experiments.experiment_io import (
    experiment_env_slug,
    experiment_run_dir,
    experiment_summary_dir,
    write_json,
)
from experiments.experiment_definitions import get_experiment_spec
from experiments.summarize import collect_track_records
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
