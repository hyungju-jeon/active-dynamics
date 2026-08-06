"""Command line interface for Active Dynamics."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from actdyn.config import ExperimentConfig
import actdyn.utils.training_log_analysis as training_log_analysis

_setup_experiment_fn = None


def _resolve_config_path(config: str | None, config_path: str, config_name: str) -> Path:
    if config is not None:
        path = Path(config).expanduser().resolve()
    else:
        path = (Path(config_path) / f"{config_name}.yaml").expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def _default_results_dir(config_file: Path, run_label: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_label is None:
        run_label = config_file.stem
    return str(Path("results") / run_label / timestamp)


def _load_config(config_file: Path, results_dir: str | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig.from_yaml(str(config_file))

    if isinstance(cfg.results_dir, str) and "${hydra:" in cfg.results_dir:
        cfg.results_dir = _default_results_dir(config_file)

    if results_dir is not None:
        cfg.results_dir = results_dir

    return cfg


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _configure_device(cfg: ExperimentConfig) -> None:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"


def _run_single_config(cfg: ExperimentConfig) -> None:
    global _setup_experiment_fn
    if _setup_experiment_fn is None:
        from actdyn.utils.experiment_setup import setup_experiment as _setup_experiment_fn_local

        _setup_experiment_fn = _setup_experiment_fn_local

    _seed_everything(int(cfg.seed))
    _configure_device(cfg)

    experiment, _, _, _ = _setup_experiment_fn(cfg)

    if cfg.run_online:
        experiment.run()

    if cfg.run_analysis and hasattr(experiment, "post_run"):
        experiment.post_run()

    if cfg.run_offline:
        _seed_everything(int(cfg.seed))
        offline_experiment, _, _, _ = _setup_experiment_fn(cfg)
        offline_experiment.offline_run()


def _iter_sweep_configs(conf_dir: Path, selected: str | None = None) -> Iterable[Path]:
    if selected is not None:
        selected_path = conf_dir / f"{selected}.yaml"
        if not selected_path.exists():
            raise FileNotFoundError(f"Sweep config not found: {selected_path}")
        return [selected_path]

    return sorted(p for p in conf_dir.glob("*.yaml") if p.stem != "config")


def cmd_run(args: argparse.Namespace) -> int:
    config_file = _resolve_config_path(args.config, args.config_path, args.config_name)

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = _default_results_dir(config_file)

    cfg = _load_config(config_file=config_file, results_dir=results_dir)

    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.device is not None:
        cfg.device = args.device

    if args.online is not None:
        cfg.run_online = args.online
    if args.offline is not None:
        cfg.run_offline = args.offline
    if args.analysis is not None:
        cfg.run_analysis = args.analysis

    _run_single_config(cfg)
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    conf_dir = Path(args.config_path).expanduser().resolve()
    if not conf_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {conf_dir}")

    config_files = list(_iter_sweep_configs(conf_dir, selected=args.config))
    if not config_files:
        print("No sweep configs found.")
        return 0

    failures: list[tuple[str, str]] = []

    for index, config_file in enumerate(config_files, start=1):
        print(f"[{index}/{len(config_files)}] {config_file.name}")

        if args.dry_run:
            continue

        run_results_dir = None
        if args.results_dir is not None:
            run_results_dir = str(Path(args.results_dir).expanduser().resolve() / config_file.stem)

        cfg = _load_config(config_file=config_file, results_dir=run_results_dir)

        if args.seed is not None:
            cfg.seed = int(args.seed)
        if args.device is not None:
            cfg.device = args.device

        try:
            _run_single_config(cfg)
        except Exception as exc:  # pragma: no cover - run-time integration behavior
            failures.append((config_file.name, str(exc)))
            print(f"FAILED {config_file.name}: {exc}")

    if failures:
        print("\\nSweep finished with failures:")
        for name, message in failures:
            print(f"- {name}: {message}")
        return 1

    print("\\nSweep finished successfully.")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    base_dir = Path(args.exp_folder).expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {base_dir}")

    results = training_log_analysis.analyze_all_models(str(base_dir), is_offline=args.offline)
    if not results:
        print("No analysis results found.")
        return 1

    if args.summary:
        training_log_analysis.print_summary(results)

    if args.plot:
        saved_paths = training_log_analysis.plot_metrics_over_time(
            results, output_dir=args.output_dir
        )
        print(f"Saved {len(saved_paths)} plot(s).")

    if args.compare:
        saved_paths = training_log_analysis.plot_all_models_metric_comparison(
            results, output_dir=args.output_dir
        )
        print(f"Saved {len(saved_paths)} comparison plot(s).")

    if args.save_summary:
        training_log_analysis.save_summary_results(results, exp_folder=str(base_dir))
        print("Saved analysis summary.")

    return 0


def _add_boolean_override_flags(parser: argparse.ArgumentParser, flag: str, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{flag}", dest=flag, action="store_true", help=help_text)
    group.add_argument(
        f"--no-{flag}",
        dest=flag,
        action="store_false",
        help=f"Disable {help_text.lower()}",
    )
    parser.set_defaults(**{flag: None})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="actdyn", description="Active Dynamics CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a single experiment config")
    run_parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    run_parser.add_argument("--config-path", type=str, default="conf", help="Config directory")
    run_parser.add_argument("--config-name", type=str, default="config", help="Config file name")
    run_parser.add_argument("--results-dir", type=str, default=None, help="Override results dir")
    run_parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    run_parser.add_argument("--device", type=str, default=None, help="Override device")
    _add_boolean_override_flags(run_parser, "online", "Run online training")
    _add_boolean_override_flags(run_parser, "offline", "Run offline training")
    _add_boolean_override_flags(run_parser, "analysis", "Run post analysis")
    run_parser.set_defaults(func=cmd_run)

    sweep_parser = subparsers.add_parser("sweep", help="Run multiple configs from a conf directory")
    sweep_parser.add_argument("--config-path", type=str, required=True, help="Directory with YAML configs")
    sweep_parser.add_argument("--config", type=str, default=None, help="Single config name to run")
    sweep_parser.add_argument("--results-dir", type=str, default=None, help="Base results directory")
    sweep_parser.add_argument("--seed", type=int, default=None, help="Override seed for all runs")
    sweep_parser.add_argument("--device", type=str, default=None, help="Override device for all runs")
    sweep_parser.add_argument("--dry-run", action="store_true", help="Print selected configs only")
    sweep_parser.set_defaults(func=cmd_sweep)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze saved experiment outputs")
    analyze_parser.add_argument("exp_folder", type=str, help="Result directory to analyze")
    analyze_parser.add_argument("--offline", action="store_true", help="Analyze offline logs")
    analyze_parser.add_argument("--summary", action="store_true", help="Print summary table")
    analyze_parser.add_argument("--plot", action="store_true", help="Generate training metric plots")
    analyze_parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate training metric comparison plots across models",
    )
    analyze_parser.add_argument(
        "--save-summary",
        action="store_true",
        help="Write analysis_summary.json under exp_folder",
    )
    analyze_parser.add_argument("--output-dir", type=str, default=None, help="Plot output dir")
    analyze_parser.set_defaults(func=cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
