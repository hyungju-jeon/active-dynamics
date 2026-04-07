#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _run_family import run_family
else:
    from ._run_family import run_family


EXP3_SUITES = ["tbme_exp3_realdata_policy"]


def main(argv: list[str] | None = None) -> int:
    return run_family(argv=argv, suite_ids=EXP3_SUITES, default_base_dir="results/tbme/exp3")


if __name__ == "__main__":
    raise SystemExit(main())
