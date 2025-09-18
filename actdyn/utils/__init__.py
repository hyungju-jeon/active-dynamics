# Utils package for actdyn

# Import commonly used utilities
from .helpers import setup_experiment

# Hydra integration utilities (optional import to avoid hydra dependency if not needed)
try:
    from .hydra_integration import (
        hydra_experiment,
        HydraExperimentConfig,
        register_actdyn_configs,
        setup_hydra_experiment,
    )

    __all__ = [
        "setup_experiment",
        "Logger",
        "hydra_experiment",
        "HydraExperimentConfig",
        "register_actdyn_configs",
        "setup_hydra_experiment",
    ]
except ImportError:
    # Hydra not available - skip hydra utilities
    __all__ = ["setup_experiment", "Logger"]
