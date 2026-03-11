from __future__ import annotations

import argparse
import json

from mixed_family_lib import (
    CANONICAL_VECTORFIELD_GRID_N,
    CANONICAL_VECTORFIELD_GRID_RANGE,
    CANONICAL_VECTORFIELD_LAYOUT,
    canonical_vectorfield_system_names,
    default_results_root,
    run_vectorfield_figure_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the canonical family-wise true vs reconstructed vector-field figure.")
    parser.add_argument("--system-bank", choices=["mixed80", "mixed40", "legacy4"], default="mixed80")
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="Explicit representative systems. Defaults to the canonical per-family representatives for the selected bank.",
    )
    parser.add_argument("--embedding-mode", choices=["fixed", "learned_system_id"], default="learned_system_id")
    parser.add_argument("--train-samples-per-system", type=int, default=1500)
    parser.add_argument("--train-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-embed", type=int, default=2)
    parser.add_argument("--d-hidden-dynamics", type=int, default=64)
    parser.add_argument("--d-hidden-hypernet-dynamics", type=int, default=16)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--dynamics-scale", type=float, default=10.0)
    parser.add_argument("--grid-n", type=int, default=CANONICAL_VECTORFIELD_GRID_N)
    parser.add_argument("--grid-min", type=float, default=CANONICAL_VECTORFIELD_GRID_RANGE[0])
    parser.add_argument("--grid-max", type=float, default=CANONICAL_VECTORFIELD_GRID_RANGE[1])
    parser.add_argument("--figure-layout", default=CANONICAL_VECTORFIELD_LAYOUT)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results-root", type=str, default=default_results_root())
    parser.add_argument("--results-subdir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--figure-filename", default="vectorfield_family_comparison_official.png")
    parser.add_argument("--metadata-filename", default="vectorfield_family_comparison_official.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.systems is None:
        args.systems = list(canonical_vectorfield_system_names(args.system_bank))
    return args


def main() -> None:
    args = parse_args()
    payload = run_vectorfield_figure_experiment(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
