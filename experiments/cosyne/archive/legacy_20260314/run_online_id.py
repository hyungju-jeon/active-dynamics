from __future__ import annotations

import argparse
import json
import os

from mixed_family_lib import default_results_root, run_online_identification_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rollout-centered online system identification."
    )
    parser.add_argument(
        "--system-bank",
        choices=["mixed200", "mixed80", "mixed40", "legacy4", "known_duffing40"],
        default="mixed80",
    )
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument(
        "--embedding-mode",
        choices=["fixed", "learned_system_id", "family_param"],
        default="learned_system_id",
    )
    parser.add_argument("--train-samples-per-system", type=int, default=1500)
    parser.add_argument("--train-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-embed", type=int, default=2)
    parser.add_argument("--d-hidden-dynamics", type=int, default=64)
    parser.add_argument("--d-hidden-hypernet-dynamics", type=int, default=16)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--geometry-reg-weight", type=float, default=0.05)
    parser.add_argument("--geometry-anchor-samples", type=int, default=512)
    parser.add_argument("--geometry-neighbor-k", type=int, default=4)
    parser.add_argument("--interpolation-aug-weight", type=float, default=0.25)
    parser.add_argument("--interpolation-aug-samples", type=int, default=128)
    parser.add_argument("--train-state-low", type=float, default=-3.0)
    parser.add_argument("--train-state-high", type=float, default=3.0)
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--policies", nargs="*", default=["active_short", "random", "no_policy"])
    parser.add_argument("--rollout-horizon", type=int, default=200)
    parser.add_argument("--rollout-inits", type=int, default=32)
    parser.add_argument("--rollout-dt", type=float, default=0.01)
    parser.add_argument("--rollout-init-low", type=float, default=-1.2)
    parser.add_argument("--rollout-init-high", type=float, default=1.2)
    parser.add_argument("--dynamics-scale", type=float, default=5)
    parser.add_argument("--active-horizon", type=int, default=10)
    parser.add_argument("--active-num-iterations", type=int, default=10)
    parser.add_argument("--active-num-samples", type=int, default=24)
    parser.add_argument("--active-num-elite", type=int, default=4)
    parser.add_argument("--active-chunk", type=int, default=2)
    parser.add_argument("--active-action-cost-weight", type=float, default=0.00)
    parser.add_argument("--active-action-strength", type=float, default=1)
    parser.add_argument(
        "--save-acq-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save acquisition-objective map traces for active-learning sessions.",
    )
    parser.add_argument("--acq-map-interval", type=int, default=5)
    parser.add_argument("--acq-map-grid", type=int, default=61)
    parser.add_argument("--acq-map-lim", type=float, default=3.0)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(
            default_results_root(), "metadynamics_training", "meta_dynamics_checkpoint.pt"
        ),
    )
    parser.add_argument("--results-root", type=str, default=default_results_root())
    parser.add_argument("--results-subdir", default="metadynamics_online_id")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse existing online_id_record.json files in the target results directory.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_online_identification_experiment(args, print_progress=True)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
