from __future__ import annotations

import argparse
import json
import os

from mixed_family_lib import (
    CANONICAL_VECTORFIELD_GRID_N,
    CANONICAL_VECTORFIELD_GRID_RANGE,
    CANONICAL_VECTORFIELD_LAYOUT,
    default_results_root,
    run_online_identification_experiment,
    run_pretrain_eval_experiment,
    run_vectorfield_figure_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for mixed-family meta-dynamics workflows.")
    parser.add_argument("--mode", choices=["pretrain_eval", "identify", "vectorfield_figures"], default="pretrain_eval")
    parser.add_argument("--system-bank", choices=["mixed80", "mixed40", "legacy4"], default="mixed80")
    parser.add_argument("--embedding-mode", choices=["fixed", "learned_system_id"], default="learned_system_id")
    parser.add_argument("--train-samples-per-system", type=int, default=1500)
    parser.add_argument("--train-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-embed", type=int, default=2)
    parser.add_argument("--d-hidden-dynamics", type=int, default=64)
    parser.add_argument("--d-hidden-hypernet-dynamics", type=int, default=16)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--policies", nargs="*", default=["active_short", "random"])
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--results-root", type=str, default=default_results_root())
    parser.add_argument("--results-subdir", default="mixed_family_metadynamics")
    parser.add_argument("--rollout-horizon", type=int, default=200)
    parser.add_argument("--rollout-inits", type=int, default=32)
    parser.add_argument("--rollout-dt", type=float, default=0.005)
    parser.add_argument("--rollout-dt-sweep", nargs="*", type=float, default=[])
    parser.add_argument("--rollout-init-low", type=float, default=-1.2)
    parser.add_argument("--rollout-init-high", type=float, default=1.2)
    parser.add_argument("--dynamics-scale", type=float, default=10.0)
    parser.add_argument("--active-horizon", type=int, default=3)
    parser.add_argument("--active-num-iterations", type=int, default=2)
    parser.add_argument("--active-num-samples", type=int, default=12)
    parser.add_argument("--active-num-elite", type=int, default=4)
    parser.add_argument("--active-chunk", type=int, default=2)
    parser.add_argument("--active-action-cost-weight", type=float, default=0.01)
    parser.add_argument("--active-action-strength", type=float, default=0.3)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--summary-markdown", default=os.path.join(os.path.dirname(__file__), "mixed_family_metadynamics_summary.md"))
    parser.add_argument("--figure-filename", default="vectorfield_family_comparison_official.png")
    parser.add_argument("--metadata-filename", default="vectorfield_family_comparison_official.json")
    parser.add_argument("--grid-n", type=int, default=CANONICAL_VECTORFIELD_GRID_N)
    parser.add_argument("--grid-min", type=float, default=CANONICAL_VECTORFIELD_GRID_RANGE[0])
    parser.add_argument("--grid-max", type=float, default=CANONICAL_VECTORFIELD_GRID_RANGE[1])
    parser.add_argument("--figure-layout", default=CANONICAL_VECTORFIELD_LAYOUT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "pretrain_eval":
        payload = run_pretrain_eval_experiment(args)
        print(json.dumps(payload, indent=2))
        return
    if args.mode == "vectorfield_figures":
        payload = run_vectorfield_figure_experiment(args)
        print(json.dumps(payload, indent=2))
        return
    payload = run_online_identification_experiment(args, print_progress=True)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
