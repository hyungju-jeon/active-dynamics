"""Thin runner for the CISS RBF workflow.

Usage:
  python experiments/ciss/RBF_video.py --mode process
  python experiments/ciss/RBF_video.py --mode analysis
  python experiments/ciss/RBF_video.py --mode all
"""

from __future__ import annotations

import argparse

from rbf_analysis import run_analysis
from rbf_process import run_processing


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CISS RBF workflow runner")
    parser.add_argument("--mode", choices=["process", "analysis", "all"], default="all")
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-dir", type=str, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.mode in {"process", "all"}:
        run_processing(total_steps=args.total_steps, seed=args.seed, base_dir=args.base_dir)

    if args.mode in {"analysis", "all"}:
        run_analysis(base_dir=args.base_dir)


if __name__ == "__main__":
    main()
