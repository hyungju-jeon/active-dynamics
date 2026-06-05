#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


TBME_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = TBME_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
BASE_ENV_CATALOG = EXPERIMENTS_DIR / "experiment_env.yaml"
BASE_MODEL_CATALOG = EXPERIMENTS_DIR / "experiment_model.yaml"
TBME_CONFIG_DIR = TBME_DIR / "config"
TBME_ENV_CATALOG = TBME_CONFIG_DIR / "experiment_env.yaml"
TBME_MODEL_CATALOG = TBME_CONFIG_DIR / "experiment_model.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from actdyn.utils.experiment_runtime import seed_range_csv


def tbme_catalog_paths() -> dict[str, list[Path]]:
    """Return the env/model catalog stack used by TBME experiment entrypoints."""
    return {
        "env_catalog_paths": [BASE_ENV_CATALOG, TBME_ENV_CATALOG],
        "model_catalog_paths": [BASE_MODEL_CATALOG, TBME_MODEL_CATALOG],
    }


@dataclass(frozen=True)
class TbmeExperimentFamily:
    exp_ids: tuple[str, ...]
    base_dir: str
    default_seeds: str
    experiment_suites: dict[str, dict[str, Any]]


def _load_suite_module(module_ref: str) -> ModuleType:
    module_name = str(module_ref).strip()
    if not module_name:
        raise ValueError("TBME suite module name is empty")
    if "." not in module_name:
        module_name = f"experiments.tbme.{module_name}"
    return importlib.import_module(module_name)


def _module_family_name(module: ModuleType) -> str:
    if module.__name__ == "__main__" and getattr(module, "__file__", None):
        return Path(str(module.__file__)).stem
    return module.__name__.rsplit(".", 1)[-1]


def _iter_tbme_suite_modules():
    for path in sorted(TBME_DIR.glob("exp*.py")):
        module = importlib.import_module(f"experiments.tbme.{path.stem}")
        if hasattr(module, "EXPERIMENT_SUITES"):
            yield module


def _csv_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part) for part in value)


def _family_from_suite_data(
    experiment_suites: Mapping[str, Any],
    family_name: str,
    *,
    default_exp_ids: Any = None,
    default_base_dir: str | None = None,
    default_seeds: str | None = None,
    default_seed_count: int | None = None,
) -> TbmeExperimentFamily:
    suites = {str(exp_id): dict(spec) for exp_id, spec in experiment_suites.items()}
    exp_ids = _csv_tuple(default_exp_ids) or tuple(suites)
    unknown = sorted(set(exp_ids) - set(suites))
    if unknown:
        raise ValueError(
            f"Family {family_name!r} default suites are undefined: {', '.join(unknown)}"
        )
    seeds = default_seeds
    if seeds is None:
        seeds = seed_range_csv(int(default_seed_count)) if default_seed_count is not None else "0"
    return TbmeExperimentFamily(
        exp_ids=exp_ids,
        base_dir=default_base_dir or "results/tbme",
        default_seeds=str(seeds),
        experiment_suites=suites,
    )


def _family_from_module(
    module: ModuleType,
    *,
    default_exp_ids: Any = None,
    default_base_dir: str | None = None,
    default_seeds: str | None = None,
) -> TbmeExperimentFamily:
    family_name = _module_family_name(module)
    return _family_from_suite_data(
        getattr(module, "EXPERIMENT_SUITES"),
        family_name,
        default_exp_ids=default_exp_ids if default_exp_ids is not None else getattr(module, "DEFAULT_EXP_IDS", None),
        default_base_dir=(
            default_base_dir if default_base_dir is not None else getattr(module, "BASE_DIR", None)
        ),
        default_seeds=default_seeds,
        default_seed_count=getattr(module, "DEFAULT_SEED_COUNT", None),
    )


def all_tbme_experiment_suites() -> dict[str, dict[str, Any]]:
    """Return all TBME suite definitions declared by local suite modules."""
    suites: dict[str, dict[str, Any]] = {}
    for module in _iter_tbme_suite_modules():
        family = _family_from_module(module)
        for exp_id, spec in family.experiment_suites.items():
            if exp_id in suites:
                raise ValueError(f"Duplicate TBME suite id: {exp_id}")
            suites[exp_id] = dict(spec)
    return suites


def configure_tbme_catalogs(suite_entries: dict[str, dict[str, Any]] | None = None):
    """Configure `experiments.experiment_definitions` with TBME catalogs and suites."""
    from experiments.experiment_definitions import configure_catalogs

    paths = tbme_catalog_paths()
    return configure_catalogs(
        env_catalog_paths=paths["env_catalog_paths"],
        model_catalog_paths=paths["model_catalog_paths"],
        suite_catalog_paths=(),
        suite_entries=all_tbme_experiment_suites() if suite_entries is None else suite_entries,
    )


if __package__ in {None, ""}:
    from experiments import run as shared_run
else:
    from .. import run as shared_run


def _add_entrypoint_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--exp-ids",
        type=str,
        default=None,
        help="Comma-separated suite ids. Defaults to the selected TBME family.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Output root. Defaults to the selected TBME family directory.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds. Defaults to the selected TBME family seed set.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one TBME suite module through experiments.run.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "suite_module",
        help="Suite module, for example exp01_baseEnv or experiments.tbme.exp01_baseEnv.",
    )
    return _add_entrypoint_arguments(parser)


def build_experiment_parser(
    description: str,
    *,
    prog: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, prog=prog, allow_abbrev=False)
    return _add_entrypoint_arguments(parser)


def _run_family(family: TbmeExperimentFamily, passthrough: Sequence[str]) -> int:
    paths = tbme_catalog_paths()
    run_argv: list[str] = []
    for catalog_key, flag in (
        ("env_catalog_paths", "--env-catalog"),
        ("model_catalog_paths", "--model-catalog"),
    ):
        for path in paths[catalog_key]:
            run_argv.extend([flag, str(path)])
    run_argv.extend(
        [
            "--exp-ids",
            ",".join(family.exp_ids),
            "--base-dir",
            family.base_dir,
            "--seeds",
            family.default_seeds,
            *passthrough,
        ]
    )
    return int(shared_run.main(run_argv, suite_entries=family.experiment_suites))


def run_experiment_entrypoint(
    module_globals: Mapping[str, Any],
    argv: Sequence[str] | None = None,
) -> int:
    """Run the calling TBME experiment module through the shared experiment runner."""
    module_name = str(module_globals["__name__"])
    module = sys.modules[module_name]

    family_name = _module_family_name(module)
    parser = build_experiment_parser(
        f"Run {family_name} through experiments.run.",
        prog=family_name,
    )
    args, passthrough = parser.parse_known_args(None if argv is None else list(argv))
    family = _family_from_module(
        module,
        default_exp_ids=args.exp_ids,
        default_base_dir=args.base_dir,
        default_seeds=args.seeds,
    )
    return _run_family(family, passthrough)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    module = _load_suite_module(str(args.suite_module))
    family = _family_from_module(
        module,
        default_exp_ids=args.exp_ids,
        default_base_dir=args.base_dir,
        default_seeds=args.seeds,
    )
    return _run_family(family, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
