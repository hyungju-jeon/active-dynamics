from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType


EXPERIMENTS_DIR = Path(__file__).resolve().parent
ENV_CATALOGS_ENVVAR = "ACTDYN_ENV_CATALOGS"
MODEL_CATALOGS_ENVVAR = "ACTDYN_MODEL_CATALOGS"
SUITE_CATALOGS_ENVVAR = "ACTDYN_SUITE_CATALOGS"


def _serialize_catalogs(paths: list[Path]) -> str:
    return os.pathsep.join(str(path.resolve()) for path in paths)


def load_entrypoint_module(
    filename: str,
    *,
    env_catalogs: list[Path],
    model_catalogs: list[Path],
    suite_catalogs: list[Path],
    alias: str,
) -> ModuleType:
    cached_modules = {
        "experiment_specs": sys.modules.pop("experiment_specs", None),
        "experiment_common": sys.modules.pop("experiment_common", None),
    }
    previous = {
        ENV_CATALOGS_ENVVAR: os.environ.get(ENV_CATALOGS_ENVVAR),
        MODEL_CATALOGS_ENVVAR: os.environ.get(MODEL_CATALOGS_ENVVAR),
        SUITE_CATALOGS_ENVVAR: os.environ.get(SUITE_CATALOGS_ENVVAR),
    }
    os.environ[ENV_CATALOGS_ENVVAR] = _serialize_catalogs(env_catalogs)
    os.environ[MODEL_CATALOGS_ENVVAR] = _serialize_catalogs(model_catalogs)
    os.environ[SUITE_CATALOGS_ENVVAR] = _serialize_catalogs(suite_catalogs)

    module_path = EXPERIMENTS_DIR / filename
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load entrypoint from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    sys.path.insert(0, str(EXPERIMENTS_DIR))
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(alias, None)
        for module_name, previous_module in cached_modules.items():
            sys.modules.pop(module_name, None)
            if previous_module is not None:
                sys.modules[module_name] = previous_module
        try:
            sys.path.remove(str(EXPERIMENTS_DIR))
        except ValueError:
            pass
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
