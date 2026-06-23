from pathlib import Path

from experiments.experiment_io import (
    experiment_env_slug,
    experiment_run_dir,
    experiment_summary_dir,
    write_json,
)
from experiments.summarize import collect_track_records
from experiments.tbme.run_tbme_experiments import (
    configure_tbme_catalogs,
    shared_tbme_experiment_suites,
    shared_tbme_group_suites,
)
from experiments.experiment_definitions import get_experiment_spec


def test_tbme_env_slug_uses_gated_duffing_name() -> None:
    assert experiment_env_slug("tbme_asymmetric_basin") == "gated_duffing"
    assert (
        experiment_env_slug("tbme_asymmetric_basin_parameter_mismatch_mild")
        == "gated_duffing_parameter_mismatch_mild"
    )
    assert experiment_env_slug("tbme_duffing") == "duffing"


def test_shared_tbme_suites_dedupe_and_merge_methods() -> None:
    suites = shared_tbme_experiment_suites()
    assert "gated_duffing" in suites
    assert all(not suite_id.startswith("exp") for suite_id in suites)
    assert all(
        not source_id.startswith("exp")
        for spec in suites.values()
        for source_id in spec["source_exp_ids"]
    )

    methods = tuple(suites["gated_duffing"]["model_ids"])
    assert methods.count("active_planning_u20_r20_h40") == 1
    assert "active_myopic" in methods
    assert "active_e_optimality_u20_r20_h40" in methods
    assert "active_planning_u1_r1_h40" in methods
    assert suites["duffing_parameter_mismatch"]["source_exp_ids"] == ["duffing_parameter_mismatch"]
    assert suites["duffing_parameter_mismatch"]["source_modules"] == ["exp_model_mismatch"]
    assert suites["gated_duffing_parameter_mismatch"]["source_exp_ids"] == [
        "gated_duffing_parameter_mismatch"
    ]
    assert suites["gated_duffing_parameter_mismatch"]["source_modules"] == ["exp_model_mismatch"]
    assert suites["duffing_parameter_mismatch_mild"]["source_exp_ids"] == [
        "duffing_parameter_mismatch_mild"
    ]
    assert suites["duffing_parameter_mismatch_mild"]["source_modules"] == [
        "exp_parameter_mismatch_stress"
    ]

    groups = shared_tbme_group_suites()
    scheduling = {
        item["suite_id"]: set(item["policy_ids"])
        for item in groups["scheduling"]
    }
    assert "active_planning_u1_r1_h40" in scheduling["gated_duffing"]
    assert "active_planning_u20_r20_h40" not in scheduling["gated_duffing"]

    objective = {
        item["suite_id"]: set(item["policy_ids"])
        for item in groups["objective_ablation"]
    }
    assert "active_e_optimality_u20_r20_h40" in objective["gated_duffing"]
    assert "prbs" not in objective["gated_duffing"]


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
    ) == tmp_path / "gated_duffing" / "random" / "seed_2" / "repeat_03"
    assert experiment_summary_dir(
        tmp_path,
        exp_spec,
        layout="tbme_tracks",
    ) == tmp_path / "gated_duffing" / "summary"


def test_summary_collects_records_from_tbme_tracks_layout(tmp_path: Path) -> None:
    configure_tbme_catalogs()
    run_dir = tmp_path / "gated_duffing" / "random" / "seed_0" / "repeat_01"
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
