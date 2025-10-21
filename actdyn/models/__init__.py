"""Models package for Active Dynamics.

This package exposes base classes and a few core implementations. Factory
functions are provided to dynamically resolve implementations by short
string keys. Dynamic imports keep top-level imports light-weight.
"""

from typing import Type

from .base import (
    BaseModel,
    BaseEncoder,
    BaseDynamics,
    BaseMapping,
    BaseNoise,
    BaseDynamicsEnsemble,
)
from .decoder import Decoder
from .model_wrapper import VAEWrapper
from .model import SeqVae

__all__ = [
    # Base classes
    "BaseModel",
    "BaseEncoder",
    "BaseDynamics",
    "BaseMapping",
    "BaseNoise",
    "BaseDynamicsEnsemble",
    # Concrete implementations
    "Decoder",
    "VAEWrapper",
    "SeqVae",
    # Factory functions
    "mapping_from_str",
    "noise_from_str",
    "encoder_from_str",
    "dynamics_from_str",
    "model_from_str",
]

import importlib

# Factory tables
_model_map = {"seq-vae": (".model", "SeqVae")}
_encoder_map = {"mlp": (".encoder", "MLPEncoder"), "rnn": (".encoder", "RNNEncoder")}
_mapping_map = {
    "identity": (".decoder", "IdentityMapping"),
    "linear": (".decoder", "LinearMapping"),
    "loglinear": (".decoder", "LogLinearMapping"),
    "mlp": (".decoder", "MLPMapping"),
}
_noise_map = {"gaussian": (".decoder", "GaussianNoise"), "poisson": (".decoder", "PoissonNoise")}
_dynamics_map = {
    "linear": (".dynamics", "LinearDynamics"),
    "mlp": (".dynamics", "MLPDynamics"),
    "rbf": (".dynamics", "RBFDynamics"),
}


def _resolve(map_table, key: str):
    if key not in map_table:
        raise ImportError(f"Unknown key: {key}. Available: {list(map_table.keys())}")
    module_name, class_name = map_table[key]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)


def mapping_from_str(mapping_str: str) -> Type[BaseMapping]:
    return _resolve(_mapping_map, mapping_str)


def noise_from_str(noise_str: str) -> Type[BaseNoise]:
    return _resolve(_noise_map, noise_str)


def encoder_from_str(encoder_str: str) -> Type[BaseEncoder]:
    return _resolve(_encoder_map, encoder_str)


def dynamics_from_str(dynamics_str: str) -> Type[BaseDynamics]:
    return _resolve(_dynamics_map, dynamics_str)


def model_from_str(model_str: str) -> Type[BaseModel]:
    return _resolve(_model_map, model_str)
