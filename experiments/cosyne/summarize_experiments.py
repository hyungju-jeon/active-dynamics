#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from catalog_runner_delegate import load_entrypoint_module
else:
    from ..catalog_runner_delegate import load_entrypoint_module


CATALOG_DIR = Path(__file__).resolve().parent
SHARED_CATALOG_DIR = CATALOG_DIR.parent


def main(argv: list[str] | None = None) -> int:
    module = load_entrypoint_module(
        "summarize_experiments.py",
        env_catalogs=[SHARED_CATALOG_DIR / "experiment_env.yaml"],
        model_catalogs=[SHARED_CATALOG_DIR / "experiment_model.yaml"],
        suite_catalogs=[CATALOG_DIR / "experiment_suite.yaml"],
        alias="cosyne_summarize_experiments_delegate",
    )
    return int(module.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
