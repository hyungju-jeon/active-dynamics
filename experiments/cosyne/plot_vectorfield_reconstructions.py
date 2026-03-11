from __future__ import annotations

import argparse
import json

from mixed_family_lib import default_results_root, run_vectorfield_figure_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate family-wise true vs reconstructed vector-field figures.")
    parser.add_argument("--system-bank", choices=["mixed80", "mixed40", "legacy4"], default="mixed80")
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--embedding-mode", choices=["fixed", "learned_system_id"], default="learned_system_id")
    parser.add_argument("--train-samples-per-system", type=int, default=1500)
    parser.add_argument("--train-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-embed", type=int, default=2)
    parser.add_argument("--d-hidden-dynamics", type=int, default=64)
    parser.add_argument("--d-hidden-hypernet-dynamics", type=int, default=16)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--dynamics-scale", type=float, default=10.0)
    parser.add_argument("--grid-n", type=int, default=25)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results-root", type=str, default=default_results_root())
    parser.add_argument("--results-subdir", default="mixed_family_metadynamics_figures")
    parser.add_argument("--figure-filename", default="vectorfield_family_comparison.png")
    parser.add_argument("--metadata-filename", default="vectorfield_family_comparison.json")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_vectorfield_figure_experiment(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
