#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from exp3_digital_twin import (
        DEFAULT_BASE_DIR,
        DEFAULT_CONFIG_PATH,
        load_config,
        run_workflow,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiment_common import resolve_session_root
else:
    from .exp3_digital_twin import (
        DEFAULT_BASE_DIR,
        DEFAULT_CONFIG_PATH,
        load_config,
        run_workflow,
    )
    from ..experiment_common import resolve_session_root


def _parse_csv_ints(raw: str | None) -> list[int]:
    if raw is None:
        return []
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _parse_csv_strs(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the MC_RTT neural digital twin and run TBME Experiment 3 active identification.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--mode", choices=["fit", "benchmark", "summary", "all"], default="all")
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--sequence-stride", type=int, default=None)
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-eval-sequences", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--control-dim", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--benchmark-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--policy-ids", type=str, default=None)
    parser.add_argument("--myopic-horizon", type=int, default=None)
    parser.add_argument("--planning-horizon", type=int, default=None)
    parser.add_argument("--mpc-num-samples", type=int, default=None)
    parser.add_argument("--mpc-num-iterations", type=int, default=None)
    parser.add_argument("--mpc-num-elite", type=int, default=None)
    parser.add_argument("--figure-format", action="append", dest="figure_formats")
    return parser


def _apply_overrides(args: argparse.Namespace):
    config = load_config(args.config)
    if args.seed is not None:
        config.runtime.seed = int(args.seed)
    if args.device is not None:
        config.runtime.device = str(args.device)
    if args.dataset_path is not None:
        config.dataset.dataset_path = str(args.dataset_path)
    if args.sequence_length is not None:
        config.dataset.sequence_length = int(args.sequence_length)
    if args.sequence_stride is not None:
        config.dataset.sequence_stride = int(args.sequence_stride)
    if args.max_units is not None:
        config.dataset.max_units = int(args.max_units)
    if args.max_train_sequences is not None:
        config.dataset.max_train_sequences = int(args.max_train_sequences)
    if args.max_eval_sequences is not None:
        config.dataset.max_eval_sequences = int(args.max_eval_sequences)
    if args.latent_dim is not None:
        config.generator.latent_dim = int(args.latent_dim)
    if args.control_dim is not None:
        config.twin.control_dim = int(args.control_dim)
    if args.n_epochs is not None:
        config.generator.n_epochs = int(args.n_epochs)
    if args.benchmark_steps is not None:
        config.benchmark.total_steps = int(args.benchmark_steps)
    if args.eval_every is not None:
        config.benchmark.eval_every = int(args.eval_every)
    if args.seeds is not None:
        config.benchmark.seeds = _parse_csv_ints(args.seeds)
    if args.policy_ids is not None:
        config.benchmark.policy_ids = _parse_csv_strs(args.policy_ids)
    if args.myopic_horizon is not None:
        config.benchmark.myopic_horizon = int(args.myopic_horizon)
    if args.planning_horizon is not None:
        config.benchmark.planning_horizon = int(args.planning_horizon)
    if args.mpc_num_samples is not None:
        config.benchmark.mpc_num_samples = int(args.mpc_num_samples)
    if args.mpc_num_iterations is not None:
        config.benchmark.mpc_num_iterations = int(args.mpc_num_iterations)
    if args.mpc_num_elite is not None:
        config.benchmark.mpc_num_elite = int(args.mpc_num_elite)
    if args.figure_formats:
        config.summary.figure_formats = list(args.figure_formats)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _apply_overrides(args)
    session_root = resolve_session_root(
        Path(args.base_dir),
        create=args.mode in {"fit", "all"},
    )
    return int(run_workflow(config=config, session_root=session_root, mode=args.mode))


if __name__ == "__main__":
    raise SystemExit(main())
