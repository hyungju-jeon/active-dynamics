"""Active Dynamics package."""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Hyungju Jeon"

__all__ = ["__version__", "__author__"]


def _try_export(import_stmt: str, symbol: str) -> None:
    try:
        module = __import__(import_stmt, fromlist=[symbol])
        globals()[symbol] = getattr(module, symbol)
        __all__.append(symbol)
    except Exception:
        # Keep top-level import lightweight when optional runtime deps are missing.
        pass


_try_export("actdyn.config", "ExperimentConfig")
_try_export("actdyn.utils.experiment_setup", "setup_experiment")
_try_export("actdyn.core.agent", "Agent")
_try_export("actdyn.core.agent", "AsyncAgent")
_try_export("actdyn.core.experiment", "Experiment")
_try_export("actdyn.models.model", "SeqVae")
_try_export("actdyn.models.model_wrapper", "ModelWrapper")
_try_export("actdyn.environment.vectorfield", "VectorFieldEnv")
_try_export("actdyn.environment.env_wrapper", "EnvWrapper")
_try_export("actdyn.policy.mpc", "MpcICem")
_try_export("actdyn.metrics.information", "FisherInformationMetric")
