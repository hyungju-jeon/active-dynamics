#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from seqvae_mcrtt import DEFAULT_BASE_DIR, DEFAULT_CONFIG_PATH, load_config, run_suite, summarize_session

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiment_common import resolve_session_root
else:
    from .seqvae_mcrtt import (
        DEFAULT_BASE_DIR,
        DEFAULT_CONFIG_PATH,
        load_config,
        run_suite,
        summarize_session,
    )
    from ..experiment_common import resolve_session_root


def _parse_csv_ints(raw: str | None) -> list[int]:
    if raw is None:
        return []
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and summarize SeqVAE-with-MLP-dynamics models on MC_RTT replay data.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--mode", choices=["run", "summary", "all"], default="all")
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--observation-key", type=str, default=None)
    parser.add_argument("--latent-dims", type=str, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--sequence-stride", type=int, default=None)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-eval-sequences", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--recovery-epochs", type=int, default=None)
    parser.add_argument("--synthetic-num-sequences", type=int, default=None)
    parser.add_argument("--sample-decoder-noise", action="store_true")
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
    if args.observation_key is not None:
        config.dataset.observation_key = str(args.observation_key)
    if args.sequence_length is not None:
        config.dataset.sequence_length = int(args.sequence_length)
    if args.sequence_stride is not None:
        config.dataset.sequence_stride = int(args.sequence_stride)
    if args.max_train_sequences is not None:
        config.dataset.max_train_sequences = int(args.max_train_sequences)
    if args.max_eval_sequences is not None:
        config.dataset.max_eval_sequences = int(args.max_eval_sequences)
    if args.latent_dims is not None:
        config.seqvae.latent_dims = _parse_csv_ints(args.latent_dims)
    if args.n_epochs is not None:
        config.seqvae.n_epochs = int(args.n_epochs)
    if args.recovery_epochs is not None:
        config.recovery.refit_n_epochs = int(args.recovery_epochs)
    if args.synthetic_num_sequences is not None:
        config.recovery.synthetic_num_sequences = int(args.synthetic_num_sequences)
    if args.sample_decoder_noise:
        config.recovery.sample_decoder_noise = True
    if args.figure_formats:
        config.summary.figure_formats = list(args.figure_formats)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _apply_overrides(args)
    session_root = resolve_session_root(
        Path(args.base_dir),
        create=args.mode in {"run", "all"},
        exp_ids=["seqvae_mcrtt"],
    )
    if args.mode == "summary":
        return int(summarize_session(session_root=session_root, config=config))
    return int(run_suite(config=config, session_root=session_root, summarize=args.mode == "all"))


if __name__ == "__main__":
    raise SystemExit(main())
