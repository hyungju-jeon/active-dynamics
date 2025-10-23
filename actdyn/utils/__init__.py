"""Utilities for actdyn experiments.

Expose a small set of commonly used helpers. Hydra integration utilities are
re-exported when Hydra is available; otherwise they are skipped so importing
the package does not require Hydra.
"""

from .save_load import save_model, load_model, save_rollout, load_rollout

__all__ = ["setup_experiment", "save_model", "load_model", "save_rollout", "load_rollout"]


def setup_experiment(*args, **kwargs):
    """Lazy import wrapper for setup_experiment from actdyn.utils.helpers.

    This avoids importing heavy dependencies (torch, gym) at package import
    time. The real function will be imported on first call.
    """
    from .experiment_helpers import setup_experiment as _setup

    return _setup(*args, **kwargs)


# Optional hydra integration helpers
try:
    from .hydra_integration import (
        hydra_experiment,
        HydraExperimentConfig,
        register_actdyn_configs,
        setup_hydra_experiment,
    )

    __all__.extend(
        [
            "hydra_experiment",
            "HydraExperimentConfig",
            "register_actdyn_configs",
            "setup_hydra_experiment",
        ]
    )
except Exception:
    # Missing hydra or optional imports - keep package importable
    pass
