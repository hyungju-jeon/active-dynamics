from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiment_common import resolve_session_root
    from run_experiments import main as run_main
else:
    from ..experiment_common import resolve_session_root
    from ..run_experiments import main as run_main


TBME_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = TBME_DIR.parent


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


def run_family(
    *,
    argv: list[str] | None,
    suite_ids: list[str],
    default_base_dir: str,
) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--base-dir", type=str, default=default_base_dir)
    parser.add_argument("--mode", choices=["run", "summary", "video", "all"], default="run")
    parser.add_argument("--exp-id", type=str, default=None)
    parser.add_argument("--exp-ids", type=str, default=None)
    parser.add_argument("--env-catalog", action="append", dest="env_catalogs")
    parser.add_argument("--model-catalog", action="append", dest="model_catalogs")
    parser.add_argument("--suite-catalog", action="append", dest="suite_catalogs")
    known, remaining = parser.parse_known_args(argv_list)
    if known.exp_id is not None or known.exp_ids is not None:
        raise SystemExit("This script manages experiment selection internally; do not pass --exp-id or --exp-ids.")
    if known.env_catalogs or known.model_catalogs or known.suite_catalogs:
        raise SystemExit("This script manages catalog selection internally; do not pass catalog arguments.")

    session_root = resolve_session_root(
        Path(known.base_dir),
        create=known.mode in {"run", "all"},
        exp_ids=suite_ids,
    )
    delegated_argv = [
        *_catalog_args(),
        "--exp-ids",
        ",".join(suite_ids),
        "--mode",
        str(known.mode),
        "--base-dir",
        str(session_root),
        *remaining,
    ]
    return int(run_main(delegated_argv))
