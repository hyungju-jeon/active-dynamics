from __future__ import annotations

import argparse
import json

from mixed_family_lib import default_results_root, run_online_identification_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rollout-centered online system identification.")
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
    parser.add_argument("--total-steps", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--policies", nargs="*", default=["active_short", "random"])
    parser.add_argument("--rollout-horizon", type=int, default=200)
    parser.add_argument("--rollout-inits", type=int, default=32)
    parser.add_argument("--rollout-dt", type=float, default=0.005)
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
    parser.add_argument("--results-root", type=str, default=default_results_root())
    parser.add_argument("--results-subdir", default="mixed_family_metadynamics_online_id")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_online_identification_experiment(args, print_progress=True)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
