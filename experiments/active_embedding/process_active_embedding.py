"""Run active embedding data generation/processing workflows.

This keeps experiment execution separate from post-processing analysis.
"""

from exp_active import main as run_active_embedding_experiment


def main() -> None:
    run_active_embedding_experiment(run_analysis=False)


if __name__ == "__main__":
    main()
