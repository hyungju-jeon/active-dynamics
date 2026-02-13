"""Thin orchestrator for active embedding processing + analysis."""

import argparse

from analyze_active_embedding import (
    DEFAULT_RESULTS_DIR,
    analyze_embedding_results,
)
from process_active_embedding import main as run_processing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run active embedding processing and/or analysis.")
    parser.add_argument(
        "--mode",
        choices=["process", "analysis", "all"],
        default="all",
        help="Which stage to run.",
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Result directory for analysis inputs/outputs.",
    )
    parser.add_argument(
        "--unknown-file",
        default="unknown_comparison.pkl",
        help="Filename for unknown-observation comparison pickle.",
    )
    parser.add_argument(
        "--known-file",
        default="active_comparison.pkl",
        help="Filename for known-observation comparison pickle.",
    )
    parser.add_argument(
        "--output",
        default="embedding_error_comparison.png",
        help="Output figure path (absolute) or filename (relative to base-dir).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum x-axis value for the comparison plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode in ("process", "all"):
        run_processing()

    if args.mode in ("analysis", "all"):
        output_path = analyze_embedding_results(
            base_dir=args.base_dir,
            unknown_filename=args.unknown_file,
            known_filename=args.known_file,
            output_filename=args.output,
            max_steps=args.max_steps,
        )
        print(f"Saved analysis plot: {output_path}")


if __name__ == "__main__":
    main()
