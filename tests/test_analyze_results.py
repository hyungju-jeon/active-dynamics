from __future__ import annotations

import json
from pathlib import Path

import actdyn.utils.training_log_analysis as training_log_analysis


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_all_models_and_summary(tmp_path: Path):
    base = tmp_path / "results"
    _write_json(
        base / "model_a" / "seed_1" / "logs" / "log_0.json",
        [
            {"step": 0, "train_elbo": -1.0},
            {"step": 1, "train_elbo": -0.8},
        ],
    )
    _write_json(
        base / "model_a" / "seed_2" / "logs" / "log_0.json",
        [
            {"step": 0, "train_elbo": -1.2},
            {"step": 1, "train_elbo": -1.0},
        ],
    )

    results = training_log_analysis.analyze_all_models(str(base))
    assert "model_a" in results
    assert "log" in results["model_a"]

    metric_data = results["model_a"]["log"]
    assert sorted(set(metric_data["seed"])) == [1, 2]

    plot_data = training_log_analysis.prepare_metric_plot_data(metric_data, "train_elbo")
    assert plot_data is not None
    assert plot_data["n_seeds"] == 2
    assert len(plot_data["time_steps"]) == 2

    summary = training_log_analysis.summarize_results(results)
    assert "model_a" in summary
    assert "log" in summary["model_a"]
    assert "train_elbo_last_mean" in summary["model_a"]["log"]
