#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _run_family import run_family
else:
    from ._run_family import run_family


EXP1_SUITES = [
    "tbme_exp1_duffing_policy",
    "tbme_exp1_duffing_policy_sota",
    "tbme_exp1_pendulum_policy",
    "tbme_exp1_pendulum_policy_sota",
    "tbme_exp1_double_integrator_policy",
    "tbme_exp1_double_integrator_policy_sota",
    "tbme_exp1_objective_duffing",
    "tbme_exp1_duffing_challenge_policy",
    "tbme_exp1_duffing_challenge_sota",
    "tbme_exp1_duffing_budget_ablation_short",
    "tbme_exp1_duffing_budget_ablation_medium",
    "tbme_exp1_duffing_ig_ablation",
    "tbme_exp1_duffing_schedule_ablation",
    "tbme_exp1_duffing_competitor_compare",
]


def main(argv: list[str] | None = None) -> int:
    return run_family(argv=argv, suite_ids=EXP1_SUITES, default_base_dir="results/tbme/exp1")


if __name__ == "__main__":
    raise SystemExit(main())
