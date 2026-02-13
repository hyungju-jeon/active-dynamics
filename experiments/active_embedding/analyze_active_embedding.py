"""Analyze active embedding outputs saved under results/active_embedding."""

import argparse
import os
import pickle

import matplotlib.pyplot as plt

from actdyn.utils.visualize import plot_embedding_error_comparison, set_matplotlib_style

DEFAULT_RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../results", "active_embedding")
)


def analyze_embedding_results(
    base_dir: str = DEFAULT_RESULTS_DIR,
    unknown_filename: str = "unknown_comparison.pkl",
    known_filename: str = "active_comparison.pkl",
    output_filename: str = "embedding_error_comparison.png",
    max_steps: int = 500,
) -> str:
    set_matplotlib_style()
    unknown_path = os.path.join(base_dir, unknown_filename)
    known_path = os.path.join(base_dir, known_filename)

    if not os.path.exists(unknown_path):
        raise FileNotFoundError(f"Missing unknown-results file: {unknown_path}")
    if not os.path.exists(known_path):
        raise FileNotFoundError(f"Missing known-results file: {known_path}")

    with open(unknown_path, "rb") as f:
        unknown_results = pickle.load(f)
    with open(known_path, "rb") as f:
        known_results = pickle.load(f)

    fig, _ = plot_embedding_error_comparison(
        unknown_results=unknown_results,
        known_results=known_results,
        max_steps=max_steps,
    )

    output_path = output_filename
    if not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze active embedding experiment outputs.")
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory that contains active_embedding result pickle files.",
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
