#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_experiments import main as run_main
else:
    from ..run_experiments import main as run_main


TBME_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = TBME_DIR.parent


EXP1_SUITES = [
    "tbme_exp1_duffing_main",
    "tbme_exp1_damped_pendulum_main",
    "tbme_exp1_asymmetric_basin_main",
    "tbme_exp1_multi_stable_main",
]
DEFAULT_BASE_DIR = "results/tbme/exp1"


def _catalog_args() -> list[str]:
    return [
        "--env-catalog",
        str(EXPERIMENTS_DIR / "experiment_env.yaml"),
        "--env-catalog",
        str(TBME_DIR / "experiment_env.yaml"),
        "--model-catalog",
        str(EXPERIMENTS_DIR / "experiment_model.yaml"),
        "--model-catalog",
        str(TBME_DIR / "experiment_model.yaml"),
        "--suite-catalog",
        str(TBME_DIR / "experiment_suite.yaml"),
    ]


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    return int(
        run_main(
            [
                *_catalog_args(),
                "--exp-ids",
                ",".join(EXP1_SUITES),
                "--base-dir",
                DEFAULT_BASE_DIR,
                *argv_list,
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
