from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    module_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_track_metadata(
    base_dir: Path,
    model_tag: str,
    exp_id: str,
    seed: int,
    repeat: int = 1,
    latent_error_final: float = 0.1,
) -> None:
    run_dir = (
        base_dir
        / "tracks"
        / model_tag
        / exp_id
        / f"seed_{seed}"
        / f"repeat_{repeat:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_tag": model_tag,
        "commit": "deadbee",
        "seed": seed,
        "exp_id": exp_id,
        "total_steps": 1000,
        "base_dir": str(run_dir),
        "status": "completed",
        "start_time": "2026-02-27T00:00:00Z",
        "end_time": "2026-02-27T00:01:00Z",
        "runtime_sec": 60.0,
        "rollout_steps": 1000,
        "state_error_final": latent_error_final,
        "state_error_mean": latent_error_final + 0.05,
        "embedding_error_final": latent_error_final,
        "embedding_error_mean": latent_error_final + 0.05,
        "q_theta": 1e-4,
        "k_theta": 10,
        "eig_gamma": 1.0,
        "writing_ref": "docs/active-dynamics-writing/methods.tex",
        "nan_detected": False,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(payload), encoding="utf-8")


def test_run_ciss_tracks_parser_accepts_expected_args():
    module = _load_module("cosyne_runner", "experiments/cosyne/run_ciss_tracks.py")
    parser = module.build_parser()
    mode_choices = parser._option_string_actions["--mode"].choices
    assert "rbf" not in mode_choices
    args = parser.parse_args(
        [
            "--exp-ids",
            "active_short,active_long,RND,random",
            "--seeds",
            "0,10",
            "--total-steps",
            "1000",
            "--model-tag",
            "updated",
            "--base-dir",
            "results/cosyne",
            "--q-theta",
            "1e-4",
            "--k-theta",
            "10",
            "--eig-gamma",
            "1.0",
        ]
    )

    assert args.exp_ids == "active_short,active_long,RND,random"
    assert args.seeds == "0,10"
    assert args.total_steps == 1000
    assert args.model_tag == "updated"
    assert args.base_dir == "results/cosyne"
    assert args.q_theta == 1e-4
    assert args.k_theta == 10
    assert args.eig_gamma == 1.0


def test_summarizer_fail_on_missing_expected_matrix(tmp_path: Path):
    module = _load_module("cosyne_summary", "experiments/cosyne/summarize_cosyne_results.py")
    base_dir = tmp_path / "results"
    summary_dir = tmp_path / "summary"
    _write_track_metadata(base_dir, model_tag="updated", exp_id="active_short", seed=0)

    exit_code = module.main(
        [
            "--base-dir",
            str(base_dir),
            "--summary-dir",
            str(summary_dir),
            "--exp-ids",
            "active_short,active_long,RND,random",
            "--seeds",
            "0,10",
            "--model-tags",
            "updated",
            "--fail-on-missing",
        ]
    )
    assert exit_code == 1


def test_summarizer_writes_expected_track_row_count(tmp_path: Path):
    module = _load_module("cosyne_summary_ok", "experiments/cosyne/summarize_cosyne_results.py")
    base_dir = tmp_path / "results"
    summary_dir = tmp_path / "summary"
    exp_ids = ["active_short", "active_long", "RND", "random"]
    seeds = [0, 10]
    model_tags = ["updated"]

    for model_tag in model_tags:
        for exp_id in exp_ids:
            for seed in seeds:
                _write_track_metadata(
                    base_dir,
                    model_tag=model_tag,
                    exp_id=exp_id,
                    seed=seed,
                    latent_error_final=0.12,
                )

    exit_code = module.main(
        [
            "--base-dir",
            str(base_dir),
            "--summary-dir",
            str(summary_dir),
            "--exp-ids",
            ",".join(exp_ids),
            "--seeds",
            "0,10",
            "--model-tags",
            "updated",
            "--fail-on-missing",
        ]
    )

    assert exit_code == 0
    metrics_csv = summary_dir / "metrics.csv"
    assert metrics_csv.exists()

    with metrics_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(exp_ids) * len(seeds) * len(model_tags)
    assert (summary_dir / "metrics.md").exists()
