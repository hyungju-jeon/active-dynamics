from __future__ import annotations

import argparse
import json

from mixed_family_lib import default_results_root, run_embedding_cluster_figure_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate embedding-cluster visualization from a trained checkpoint."
    )
    parser.add_argument(
        "--system-bank", choices=["mixed80", "mixed40", "legacy4"], default="mixed80"
    )
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument(
        "--embedding-mode", choices=["fixed", "learned_system_id"], default="learned_system_id"
    )
    parser.add_argument("--train-samples-per-system", type=int, default=1500)
    parser.add_argument("--train-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-embed", type=int, default=2)
    parser.add_argument("--d-hidden-dynamics", type=int, default=64)
    parser.add_argument("--d-hidden-hypernet-dynamics", type=int, default=16)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--dynamics-scale", type=float, default=10.0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results-root", type=str, default=default_results_root())
    parser.add_argument("--results-subdir", default="cosyne/metadynamics_training")
    parser.add_argument("--figure-filename", default="embedding_family_clusters.png")
    parser.add_argument("--metadata-filename", default="embedding_family_clusters.json")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_embedding_cluster_figure_experiment(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
