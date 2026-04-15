#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _run_family import run_family
else:
    from ._run_family import run_family


EXP2_SUITES = [
    "tbme_exp2_robustness_duffing",
    "tbme_exp2_robustness_duffing_sota",
    "tbme_exp2_robustness_pendulum",
    "tbme_exp2_robustness_pendulum_sota",
    "tbme_exp2_robustness_double_integrator",
    "tbme_exp2_robustness_double_integrator_sota",
]


def main(argv: list[str] | None = None) -> int:
    return run_family(argv=argv, suite_ids=EXP2_SUITES, default_base_dir="results/tbme/exp2")


if __name__ == "__main__":
    raise SystemExit(main())
